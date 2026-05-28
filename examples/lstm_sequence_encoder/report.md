# LSTM Sequence Encoder — GPU Optimization Report

**This optimization achieved a 2.0× speedup on the dominant recurrent gate GEMM of LSTMSequenceEncoder (B=32, NVIDIA RTX PRO 6000 Blackwell) by promoting the FP32 SIMT matmuls onto the idle tensor cores.**

The single highest-leverage transformation — bf16 dtype promotion of the per-timestep gate GEMMs — moved the dominant operator off the FP32 SIMT path (tensor cores 0% utilized) and onto the Blackwell HMMA tensor-core path, while eliminating the paired split-K reduction kernels. The two stacked LSTM layers' recurrent gate matmuls sped up **2.02×**; the batched input-projection GEMMs sped up **4.03×**.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture family | Blackwell |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Compile mode | inductor (cuDNN RNN unrolled into per-timestep Aten ops) |
| Model | LSTMSequenceEncoder (2-layer LSTM, hidden=512, input=256) → mean-pool → Linear(512→10) |
| Batch size | 32 |
| Sequence length | 128 |
| Iterations | warmup=2, measure=2 *(ncu replay — relative timing only)* |

> **Caveat on all durations in this report.** Values are ncu application-replay timings, which run 2–5× longer than real wall-clock execution and are inflated by counter collection. Treat every `ns` value as a *relative within-/across-profile comparison*, never as an absolute latency.

---

## 2. Operator Summary (baseline)

Because `torch.compile` **unrolls** the cuDNN RNN, the same logical recurrent gate GEMM appears as ~256 per-timestep records (128 steps × 2 layers). Operators are grouped by kernel/shape family — the meaningful unit here. Bottleneck class is derived from tensor-core activity, SM throughput, and occupancy (Blackwell exposes `tensor_core_active_pct`).

| Operator (shape family) | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `layer::unique::prologue` (NVTX catch-all + warmup) | 42.8% | 5,861,731 | 1567 | Mixed / warmup-inflated |
| `aten::mm [32,512]×[512,2048]` recurrent gate GEMM | 31.3% | 4,279,596 | 1024 | Compute-bound, **tensor-core idle (0%)** |
| `aten::mm` (generic NVTX catch-all) | 20.8% | 2,842,364 | 755 | **tensor-core idle (0%)** |
| `aten::mm [4096,512]×[512,2048]` input projection | 3.1% | 425,508 | 2 | **tensor-core idle (0%)** |
| `aten::mm [4096,256]×[256,2048]` input projection | 1.8% | 247,714 | 2 | **tensor-core idle (0%)** |
| `aten::addmm [32,512]×[512,10]` classifier | 0.1% | 12,800 | 4 | Small GEMM, tensor-core idle |

**Core finding:** `tensor_core_active_pct == 0.0` across *every* GEMM. The inductor decomposition replaced the single fused cuDNN `elemWiseRNNcell` with a serial chain of tiny FP32 SIMT GEMMs (`Kernel2`) plus `splitKreduce` reductions. With M=32 (batch), the gate GEMM has almost no row-dimension parallelism, so cuBLAS routes it to the small-tile SIMT kernel and the Blackwell tensor cores sit completely idle (SM throughput ~15%, achieved occupancy ~12%).

---

## 3. Reading the Metrics

Only the metrics that drive this workload's bottleneck are explained.

- **`tensor_core_active_pct = 0.0` (not null)** — the highest-ROI signal in this profile. A GEMM running at 0.0% executed entirely on the FP32 SIMT path with tensor cores idle. On a tensor-core-rich Blackwell part this is pure left-on-the-table throughput. (A *null* value is expected for non-GEMM kernels and is not a problem.)
- **`sm_throughput_pct` ~15%** on the gate GEMM — the SMs are starved. A 32-row GEMM cannot fill 188 SMs; combined with 0% tensor-core use, this confirms the small-M SIMT path, not a memory wall.
- **`achieved_occupancy` ~12%** — corroborates the under-utilization; the kernel is latency/launch-bound on the serial timestep dependency (h_t → h_{t+1}), not occupancy-limited by registers.
- **`warp_cycles_per_instruction = null`** everywhere — this counter was removed on Blackwell; expected, not a defect. Latency diagnosis uses `eligible_cycles_pct` (~22%) + occupancy instead.
- **Kernel count as evidence** — the gate-GEMM family carries 2 kernels per timestep-op: the `Kernel2` SIMT GEMM **and** its paired `splitKreduce_kernel`. Halving this count (see §6) is direct evidence the bf16 path dropped the split-K reduction.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype promotion (fp32→bf16) | recurrent gate GEMM `aten.addmm` (258) + input-proj `aten.mm` (1) | `tensor_core_active_pct=0.0`, `sm_throughput=14.6%`, paired `splitKreduce` (60.2% of attributed time) | medium | **APPLIED** |
| OPT-2 | memory layout (weight pre-transpose / bias fold) | recurrent gate weight `[512,2048]` | `l2_hit_rate=57.5%`, `dram_throughput=26.4%` | low | **NOT_APPLIED** (stub) |
| OPT-3 | fusion (view/cat noop elimination) | `triton_poi_fused__unsafe_view` / `add_addmm_cat` (512) | `sm_throughput=0.36%`, grid=64 blocks, launch-bound | medium | **NOT_APPLIED** (no-op) |

Statuses are read from `profiler_output/validation_report.json`. Only **OPT-1 transforms the graph**; OPT-2 and OPT-3 degraded gracefully (one INFO stub, one guarded WARNING no-op) with no exceptions — see §5/§8 for why.

---

## 5. Implementation Notes

# Implementation Notes — LSTMSequenceEncoder optimized backend

Backend registration name: `lstm_sequence_encoder_opt`
Target: NVIDIA RTX PRO 6000 Blackwell Server Edition · compile_mode `inductor` · dtype fp32
torch 2.11.0+cu128. `torch._dynamo.config.allow_rnn = True` is set at module import (required to trace `nn.LSTM`).

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 — BF16 dtype promotion on recurrent gate GEMMs | `_aten_fw_compiler` (`_aten_inner_compile`) | Casts FP32 operands of all 258 `aten.addmm.default` gate GEMMs + the 1 `aten.mm.default` input projection to bf16 (FP32 accumulate), casts result back to fp32. Flips cuBLAS off the FP32 SIMT "Kernel2" onto the Blackwell HMMA tensor-core path and typically drops the paired `splitKreduce_kernel`. The single highest-leverage change. |
| OPT-2 — recurrent weight pre-transpose / bias fold | stub — not applied | Bias is already folded into `aten.addmm.default` by the decomposition (no `mm -> add` chain to re-fuse); the `register_buffer` weight hoist conflicts with the OPT-1 bf16 cast on the same operand (register_buffer fixes dtype at registration). optimizations.json rates this LOW. Detection-only: reports the 258 addmm nodes. |
| OPT-3 — view/cat layout-noop elimination | stub — not applied (guarded) | The actual graph has ZERO `aten._unsafe_view` and ZERO identity-shape `aten.view` nodes; all 516 views genuinely reshape between the `[32,2048]` gate tensor and four `[32,512]` slices (256 `aten.split.Tensor`). No safe layout-noop exists to erase; Inductor already fuses the gate activations (`triton_poi_fused_add_addmm_cat`). The pass keeps a real, guarded transform (`_is_identity_view`) but degrades to a no-op here. |

## Key Design Decisions

**optimizations.json named the wrong op for the bottleneck.** It described the recurrent gate GEMM as `aten::mm [[32,512]x[512,2048]]` and wrote all three passes against `torch.ops.aten.mm.default`. Tracing the actual post-AOTAutograd Aten IR shows the gate GEMM is realized as `aten.addmm.default(bias[2048], h[32,512], W[512,2048])` (258 nodes) with the bias pre-folded, and there is exactly **one** `aten.mm.default` (the `[4096,256]x[256,2048]` input projection). OPT-1 was therefore broadened to cover both `aten.mm` and `aten.addmm`; targeting only `mm` would have promoted 1 of 259 GEMMs and missed the entire ~60%-of-time recurrent bottleneck. Verified end-to-end: 259 GEMMs promoted, no NaN/Inf, max abs logit drift 2.3e-4 vs eager FP32 across the 128-step recurrence.

**OPT-1 uses `prims.convert_element_type` rather than the `aten._to_copy` named in the fx_steps.** On torch 2.11 `aten._to_copy.default` carries both a fallback and a decomp registration; inserting it post-AOTAutograd makes Inductor raise "both a fallback and a decomp for same op". `convert_element_type` is the cast primitive Inductor itself emits, lowers cleanly to a fused Triton cast, and folds into neighbouring elementwise kernels.

**OPT-2 downgraded to a stub for two independent reasons.** (1) The bias-fold half is already done by the decomposition (`addmm`, not `mm + add`). (2) The weight-hoist half requires `register_buffer`, which fixes the buffer dtype at registration time and directly conflicts with the OPT-1 bf16 cast applied to that same weight operand — exactly the register_buffer-after-dtype ordering hazard the optimizations.json OPT-2 notes flag. bf16 from OPT-1 already halves the weight read traffic that OPT-2 targeted, so the residual win does not justify the fragile 258-node buffer rewrite (rated LOW confidence in the proposal).

**OPT-3 downgraded to a guarded no-op because the premise does not hold for this graph.** The proposal assumed `aten._unsafe_view` layout-noops between gate slices; the real decomposition emits genuine `aten.view` reshapes (input shape != output shape on all 516) paired with `aten.split.Tensor`. Erasing a genuine reshape corrupts downstream shapes, so the pass only erases views where `output_shape == input_shape` (`_is_identity_view`) and reports that none qualify. Inductor already performs the activation fusion the proposal sought.

**Flat compile path, not dedup.** `UniqueSubgraphRegistry` reports 1 partition / 1 signature: the LSTM unroll collapses into a single flat Aten graph with no repeated block structure, so `build_partition_equivalence_map()` is empty and the backend takes the flat `compile_fx(..., inner_compile=...)` path, preserving cross-timestep Inductor fusion. The dedup branch is retained for parity/robustness but is not exercised.

**Out-of-scope structural fix (noted, not implemented).** The fundamental inefficiency is that torch.compile decomposes the cuDNN fused RNN into hundreds of tiny tensor-core-idle GEMMs — the eager cuDNN `elemWiseRNNcell` path is faster. The highest-value action beyond FX-pass scope is to keep the LSTM in its fused cuDNN form (do not compile the `nn.LSTM`; compile only the mean-pool + classifier head). The three passes here are the best available improvements *within* the already-decomposed graph.

## Validation

`py_compile` clean on both generated files. All 4 tests pass (`test_import`, `test_backend_registration`, `test_get_model_and_input`, `test_compiled_forward_pass`) with `TORCHINDUCTOR_FX_GRAPH_CACHE=0`. The compiled forward emits the expected `(32, 10)` logits, no NaN/Inf, and the captured INFO logs confirm OPT-1 applied to 259 GEMMs with OPT-2/OPT-3 reporting their no-op rationale.

---

## 6. Before/After Results

Both captures used identical iteration counts (warmup=2, measure=2) and batch size (B=32). **Operator matching caveat:** the baseline used the built-in dedup backend, which groups many kernels under a `layer::unique::prologue` NVTX range (and captures the eager warmup cuDNN path there); the optimized capture used the custom `lstm_sequence_encoder_opt` backend, which attributes per-op. Raw `operator_name` therefore does not match across captures, so operators are compared by **shape family** (op + tensor sizes), which is stable across both.

### Matched operators (by shape family)

| Operator (shape family) | Baseline (ns) | Optimized (ns) | Speedup | Tensor-core % (before → after) |
|---|---|---|---|---|
| Recurrent gate GEMM `[32,512]×[512,2048]` (×256 timestep-ops) | 4,279,596 | 2,116,648 | **2.02×** | 0.0 → 10.9 |
| Input-projection GEMM `[4096,*]×[*,2048]` | 673,222 | 167,232 | **4.03×** | 0.0 → 47.6 |
| Classifier `addmm [32,512]×[512,10]` | 12,800 | 7,520 | 1.70× | 0.0 → 9.5 |
| Weight transpose `aten::t` (new — bf16-path overhead) | — | 336,963 | — | n/a |
| **Matched GEMM subtotal** | **4,965,618** | **2,291,400** | **2.17×** | — |

**Kernel-count evidence for OPT-1:** the recurrent gate-GEMM family dropped from **1024 → 512 kernels** — exactly half. The bf16 tensor-core path no longer needs the paired `splitKreduce_kernel`, so one of the two kernels per timestep-op disappeared, precisely as OPT-1 predicted.

### Raw total-profile ratio (reported, not headlined)

Total profiled ncu duration: **13,681,233 ns → 2,634,571 ns (5.19×)**. This figure is **not** a reliable wall-clock proxy: the baseline total includes the dedup backend's `layer::unique::prologue` group (1567 kernels, 42.8%), which folds in the eager-cuDNN warmup path that the custom-backend capture attributes differently. The matched shape-family speedups above (2.0–4.0×) are the defensible numbers.

### Speedup attribution

The gate-GEMM and input-projection speedups are attributed to **OPT-1** — all three attribution criteria hold: (1) `status == APPLIED`; (2) `tensor_core_active_pct` moved 0.0 → 10.9% / 47.6% in the expected direction; (3) the containing operators show the measured speedup. OPT-2 and OPT-3 are `NOT_APPLIED` and contributed nothing.

---

## 7. What Drove Each Speedup

**BF16 dtype promotion on recurrent gate GEMMs (OPT-1, +2.02× on the recurrent gate GEMM family):** casting the `aten.addmm`/`aten.mm` gate-projection operands to bf16 (with fp32 accumulate) routes cuBLAS off the small-tile FP32 SIMT `Kernel2` onto the Blackwell HMMA tensor-core path. The evidence is twofold: `tensor_core_active_pct` rose from 0.0 to 10.9% on the recurrent family (and to 47.6% / 74% on the larger-M projection GEMMs), and the per-family kernel count halved from 1024 to 512 as the paired `splitKreduce_kernel` reduction was eliminated.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | memory layout (weight pre-transpose / bias fold) | recurrent gate weight `[512,2048]` | Bias already folded into `addmm`; `register_buffer` weight hoist conflicts with OPT-1's bf16 cast on the same operand (dtype fixed at registration). bf16 already halves the weight read traffic OPT-2 targeted. | ~7% (low conf.) |
| OPT-3 | fusion (view/cat noop elimination) | per-timestep `view`/`cat` reshapes | Graph has zero `aten._unsafe_view` and zero identity-shape views; all 516 views genuinely reshape. Inductor already fuses the gate activations. Nothing safe to erase. | ~11% (medium conf., already partly realized by Inductor) |

**Second-order bottleneck exposed:** after OPT-1, a new `aten::t` weight-transpose group accounts for **12.8%** of optimized time — overhead introduced/exposed by the bf16 operand handling. Additionally, the recurrent gate GEMM still reaches only ~11% tensor-core activity at M=32 (vs 74% for the M=4096 batched projections): the small batch dimension limits tensor-core tiling even on the HMMA path, exactly the medium-confidence risk OPT-1 flagged.

**Largest remaining win is structural, not an FX pass.** The root inefficiency is that `torch.compile` decomposes the fused cuDNN RNN (`elemWiseRNNcell`) into a serial chain of hundreds of tiny GEMMs in the first place. The highest-value action beyond FX-pass scope is to **keep the LSTM in its fused cuDNN form** — do not compile/decompose the `nn.LSTM`; compile only the mean-pool + classifier head. Within the already-decomposed graph, OPT-1 is the best available win, and OPT-2/OPT-3 offer marginal residual gains discounted heavily by confidence and by what Inductor already does.

---

## Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py \
    --workload examples/lstm_sequence_encoder/lstm_sequence_encoder.py \
    --correlation-pass --output-prefix profiler_output/lstm_sequence_encoder ...
# (full nsys+ncu pipeline driven by /optimize Stage 0)

# Optimized capture (custom backend)
#   uses @register_backend name: lstm_sequence_encoder_opt
/optimize examples/lstm_sequence_encoder/lstm_sequence_encoder.py --from=validate
```

**Environment note for RNN/GRU/LSTM workloads:** `run_workload.py` now sets `torch._dynamo.config.allow_rnn = True`. Without it, Dynamo refuses to trace `nn.LSTM`, graph-breaks around the entire RNN, and never invokes the profiling backend (manifests as `KeyError: 'run_fn'`). This fix applies to any recurrent workload.
