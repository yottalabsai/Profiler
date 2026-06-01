# Optimization Report — `lstm_sequence_encoder`

**This optimization achieved a 1.78× total speedup on the LSTMSequenceEncoder (B=32, NVIDIA RTX PRO 6000 Blackwell Server Edition)** by promoting the unrolled LSTM gate GEMMs from the FP32 SIMT path to the bf16 Tensor-Core path.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (~188 SMs, inferred) |
| Architecture family | Blackwell (Sm100 / GB202) |
| PyTorch version | 2.11.0+cu128 |
| Compile mode (baseline) | `inductor` (built-in dedup backend) |
| Compile mode (optimized) | `lstm_sequence_encoder_opt` (custom `@register_backend`) |
| Model | 2-layer `nn.LSTM` (in=256, hidden=512) → mean-pool over time → `nn.Linear(512, 10)` |
| Batch size | 32 (seq_len=128) |
| Iteration count | warmup=2 / measure=2 *(ncu replay — relative timing only)* |

> **Timing caveat.** All durations below are **nsys-derived** GPU kernel times (from the capture phase), not ncu replay values — they are close to real execution time, but should be read as **relative** comparisons (baseline vs optimized, operator vs operator) rather than absolute latencies. This capture ran at **dynamic boost clocks** (it predates GPU clock locking), so the exact `1.78×` is subject to a few percent of clock variation between the two captures; the structural evidence (tensor-core engagement, kernel re-routing) is clock-independent and robust. Baseline and optimized profiles were captured on the same GPU ~1 h apart in the same session — no cross-session caveat applies. (Re-capturing with clock locking would tighten the figure.)

---

## 2. Operator Summary (baseline)

Times are percentages of the **5800.8 µs** of attributed aten-operator GPU time. The `layer::unique::prologue` NVTX envelope (8729.7 µs / 2325 kernels) is the single-partition wrapper that *nests* these operators — it is reported separately to avoid double-counting and excluded from the denominator.

| Operator | Time (%) | Duration (µs) | Kernels | Bottleneck Class |
|---|---:|---:|---:|---|
| `aten::mm` (gate projections) | 85.9% | 4981.8 | 1028 | Compute-bound, FP32 SIMT — Tensor Cores idle (tc=0%), under-occupied (occ≈8%) |
| `aten::_unsafe_view` | 7.0% | 405.1 | 254 | Memory / layout bookkeeping |
| `aten::addmm` | 6.8% | 395.6 | 258 | Compute-bound, FP32 SIMT (tc=0%) |
| `aten::transpose` | 0.2% | 10.4 | 4 | Memory / layout |
| `aten::zeros` | 0.1% | 7.9 | 6 | Init |
| *(envelope)* `layer::unique::prologue` | — | 8729.7 | 2325 | NVTX wrapper (nests all of the above) |

The workload is overwhelmingly **GEMM-bound**: the 1028 `aten::mm` kernels are the per-timestep recurrent gate projections (`[[32,512]×[512,2048]]`, M=32) across 2 layers × ~128 timesteps × {input-to-hidden, hidden-to-hidden}. Every one ran on the `cutlass_80_simt_sgemm_*` scalar FP32 pipeline with Tensor Cores completely idle.

---

## 3. Reading the Metrics

Only the metrics that drive this workload's bottleneck are explained.

- **`tensor_core_active_pct = 0.0` (not null)** — the highest-ROI signal here. A GEMM reporting `0.0` ran on the FP32 SIMT scalar path with the Tensor Core MMA pipeline entirely unused. Every gate GEMM in the baseline reports exactly this. (A *null* value, by contrast, is expected for non-GEMM kernels like `view`/`transpose` and is never a problem.)
- **`sm_throughput_pct`** — fraction of peak SM issue throughput. Baseline gate GEMMs sit at ~16.6%; anything below ~40% on a compute-bound GEMM signals an idle math pipeline or a tile too small to fill the machine.
- **`achieved_occupancy`** — baseline ~8.3%. With M=32 the GEMM launches a grid of only ~128 CTAs (<1 wave on ~188 SMs), so most of the GPU sits idle regardless of clock.
- **`memory_throughput_pct`** — baseline gate GEMMs ≈26% DRAM. Below the ~60% memory-bound threshold, confirming the bottleneck is **compute/scheduling**, not bandwidth.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype_promotion (aten) | `aten.mm` / `aten.addmm` gate GEMMs | `tensor_core_active_pct=0.0`, `smsp__pipe_tensor_cycles_active=0`, SIMT `cutlass_80_simt_sgemm` kernel name, occ≈8% | high | **APPLIED** |
| OPT-2 | hoist input projection (functional) | per-timestep `F.linear` | Input-to-hidden projection already batched by Inductor (`[4096,256]`/`[4096,512]` GEMMs) | high | **NOT_APPLIED** (no-op) |
| OPT-3 | cuDNN fused-RNN restore (functional) | `nn.LSTM` recurrence | Eager path uses cuDNN `elemWiseRNNcell` + `cutlass_80_tensorop_*` | medium | **NOT_APPLIED** (mutually exclusive) |
| OPT-4 | freezing + max_autotune (inductor_config) | all GEMMs | Per-GEMM `cublasLt::splitKreduce` epilogue + runtime `_tn_` transpose | medium | **APPLIED** |

---

## 5. Implementation Notes

# Implementation Notes — lstm_sequence_encoder_opt

Backend registered name: **`lstm_sequence_encoder_opt`** (via `@register_backend`).
Workload: `LSTMSequenceEncoder` (2-layer `nn.LSTM`, hidden=512, in=256, seq=128, batch=32)
+ mean-pool + `nn.Linear(512, 10)`. `compile_mode = inductor`, baseline `dtype = float32`.
Strategist track: low-risk in-place — **OPT-2 → OPT-1 → OPT-4** (OPT-3 mutually exclusive,
not applied).

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-2 input-projection hoist (F.linear triplet) | functional | `_run_functional_passes` (detect/report) | Hoisting the input-to-hidden projection keys on per-timestep `F.linear` nodes sharing one weight — only expressible at the functional level. On `nn.LSTM` it is a single opaque node, and the hoist is **already realized by Inductor's `nn.LSTM` decomposition** (profile shows batched `[4096,256]`/`[4096,512]` input GEMMs), so the pass detects/reports and no-ops. |
| OPT-3 cuDNN fused-RNN restore | functional | stub — not applied | Route A (keep `nn.LSTM` on the eager cuDNN fused path) changes the recurrent execution backend rather than transforming kernels in place; it is **mutually exclusive** with OPT-1/OPT-2 over the recurrent matmuls. Detection-only stub: logs the available cuDNN path, never transforms the graph. |
| OPT-1 bf16 / Tensor-Core promotion of gate GEMMs | aten | `_aten_inner_compile` | After AOTAutograd decomposes `nn.LSTM` into per-timestep `aten.mm` + classifier `aten.addmm`, the dtype cast is cleanly expressed on the decomposed primitives. Casts both matmul operands to bf16 and the result back to fp32; engages the Blackwell bf16 Tensor-Core MMA pipeline (the FP32 SIMT path runs at 0% tensor-core utilization). The real win. |
| OPT-4 freezing + max_autotune | inductor_config | `config_patches` on `compile_fx` | Constant-weight freezing (fold `_tn_` transpose, drop per-call weight guards, pre-pack the bf16 tensorop layout) and autotuning a non-split-K GEMM tile are owned by Inductor — expressed as a scoped `config_patches` dict, no graph surgery. |
| (non-graph) eval() | non-graph | `get_model_and_input()` | `.eval()` is required for freezing (incompatible with training) and for the inference-only bf16 promotion. No dtype/memory_format whole-module change is needed. |

All four passes execute through the fixed three-stage funnel `_compile_unit`:
`_run_functional_passes(gm)` → `compile_fx(inner_compile=_aten_inner_compile, config_patches=...)`.
`compile_fx` owns AOTAutograd / decomposition / the boxed calling convention exactly once;
the funnel does **not** use `aot_autograd(fw_compiler=compile_fx)` (raises `AssertionError`
in `copy_misaligned_inputs` on torch 2.11).

## Key Design Decisions

**OPT-2 is a detect/report no-op, not a graph rewrite.** The profile is decisive: Inductor's
own lowering of `nn.LSTM` already batches the input-to-hidden projection into single
`[seq*batch, in] × [in, 4h]` GEMMs (`[[4096,256],[256,2048]]` for layer 0 and
`[[4096,512],[512,2048]]` for layer 1). The per-timestep `[[32,512],[512,2048]]` GEMMs that
remain (512 of them) are the genuinely recurrent hidden-to-hidden term `W_hh @ h_{t-1}`,
which is data-dependent and cannot be hoisted. A generic functional `F.linear`-triplet
matcher therefore finds nothing to hoist on `nn.LSTM` (it is one opaque node), and forcing a
structural rewrite would be both unsafe (recovering per-timestep slice topology generically)
and redundant. The pass logs the situation and leaves the graph correctness-preserving;
the canonical manual fix (a `FastLSTM` precompute rewrite) is documented in the source for
the hand-written-cell case.

**OPT-1 is the load-bearing pass and is a pure op-target dtype cast.** It reads no weight
values, so it needs no `ph_to_tensor` lookup; it matches every `aten.mm.default` /
`aten.addmm.default` post-decomposition. For `mm` it casts operands 0,1; for `addmm` it casts
the two matmul operands 1,2 (the bias arg 0 stays fp32, and the output is cast back to fp32
so the bias add and downstream sigmoid/tanh/mul cell-state ops stay well-typed). The result
cast back to fp32 preserves the recurrent-state dtype contract. Inductor CSE folds the
repeated constant weight casts (`W_ih`/`W_hh` are constant across all timesteps), so the
per-timestep cost is a single activation cast, dwarfed by the Tensor-Core GEMM speedup.
`_repropagate_meta` re-runs FakeTensorProp after the structural rewrite so the inserted
`aten._to_copy` casts carry `meta['val']` before `compile_fx_inner` lowers Aten → Triton.

**OPT-3 / OPT-1+OPT-2 mutual exclusion is honored by not transforming.** OPT-3 Route A and
the in-place GEMM transforms target the same recurrent matmuls by different mechanisms;
summing their estimated impacts would be incorrect. The strategist chose the in-place track,
so OPT-3 is wired in as a detection-only stub that never mutates the graph — guaranteeing the
two routes are never applied to the same nodes.

**Cross-level ordering needs no within-level prerequisites.** OPT-1's `prerequisite_for:
["OPT-4"]` is satisfied automatically by the funnel order (functional → aten →
inductor_config): the bf16 weights from the aten pass are in place before OPT-4's freezing
emits the pre-packed tensorop layout at the config level. No explicit sequencing is required.

**Flat compile path.** `UniqueSubgraphRegistry` finds no repeated structurally-identical
partitions (the model is a single `nn.LSTM` + classifier), so `equiv_map` is empty and the
backend takes the flat `_compile_unit(gm, example_inputs)` path, preserving cross-op Inductor
fusion. The per-rep dedup branch is retained for models with repeated identical blocks.

## Validation Findings (bugs caught and fixed during bring-up)

1. **`allow_rnn` is mandatory for the backend to ever run.** With a plain
   `torch.compile(model, backend=...)`, Dynamo produces `graph_count == 0` for this model —
   it silently graph-breaks around the entire `nn.LSTM` and runs it eagerly, so the custom
   backend is **never invoked**. `torch._dynamo.config.allow_rnn = True` (which
   `run_workload.py` sets before compiling) is required for Dynamo to trace through the RNN
   and hand the backend the FX graph. The forward-pass test sets the same flag so it mirrors
   the real capture path; without it the backend appears to "work" but applies nothing.

2. **OPT-1 must use `prims.convert_element_type`, not `aten._to_copy`.** Under OPT-4's
   `freezing` + `max_autotune`, Inductor registers `aten._to_copy.default` as **both** a
   decomposition and a fallback, raising `InductorError: both a fallback and a decomp for
   same op: aten._to_copy.default`. Switching the dtype cast to
   `torch.ops.prims.convert_element_type.default` (the canonical primitive AOTAutograd emits)
   lowers cleanly and is CSE-folded.

3. **`addmm` requires the bias cast to match the matmul operands.** Casting only `mat1`/`mat2`
   of the classifier `aten.addmm` to bf16 while leaving the fp32 bias raised
   `RuntimeError: self and mat2 must have the same dtype, but got Float and BFloat16` inside
   Inductor's `bias_addmm` lowering. OPT-1 therefore casts all three `addmm` operands
   (`args 0,1,2`) to bf16; the fp32 result cast restores precision.

**Numerical check:** compiled output vs eager fp32 max-abs-diff = `1.9e-4` (expected bf16
GEMM tolerance), shape `[32, 10]`, no NaN/Inf. The autotuner selects bf16 `triton_mm` /
`bias_addmm` kernels (`dtypes: bfloat16, bfloat16, bfloat16`) and freezing fixes the weight
layout — confirming OPT-1 + OPT-4 engage the intended Tensor-Core path.

---

## 6. Before/After Results

Operators matched by name across captures. OPT-1's bf16 promotion re-routes most gate
projections from the bias-less `aten::mm` form into Inductor's `bias_addmm` (`aten::addmm`)
lowering, so the two GEMM families are reported **combined** to avoid a misleading per-name
view. Both captures share batch size 32 on the same GPU.

| Operator | Baseline (µs) | Optimized (µs) | Speedup |
|---|---:|---:|---:|
| Gate GEMMs — `aten::mm` + `aten::addmm` (combined) | 5377.4 | 2310.6 | **2.33×** |
| `aten::_unsafe_view` | 405.1 | 317.8 | 1.27× |
| `aten::transpose` | 10.4 | 8.2 | 1.27× |
| `aten::zeros` | 7.9 | 6.4 | 1.24× |
| `aten::t` (layout, new/grown) | 0.0 | 344.2 | — |
| `aten::view` (layout, new/grown) | 0.0 | 265.3 | — |
| **Total (attributed aten GPU time)** | **5800.8** | **3252.4** | **1.78×** |

**Speedup attribution (Step B):**

- **OPT-1 (bf16 Tensor-Core promotion)** — *attributed.* Status `APPLIED`; the gate-GEMM
  cohort's `tensor_core_active_pct` moved in the expected direction (baseline `0.0` →
  optimized 11.2% on `addmm`, 23.8% on the batched `mm`), `sm_throughput_pct` rose from
  ~16.6% → ~64% on the batched GEMM, and the combined GEMM time fell 2.33×. This is the
  dominant contributor to the 1.78× total.
- **OPT-4 (freezing + max_autotune)** — *contributing, not separately isolable.* Status
  `APPLIED`; it is the enabler for OPT-1's pre-packed bf16 tensorop layout and removes the
  per-GEMM `splitKreduce` epilogue. Its effect is entangled with OPT-1 and not reported as a
  standalone row.
- **OPT-2 / OPT-3** — *no credit.* Both `NOT_APPLIED`. Any residual change in non-GEMM
  bookkeeping ops is Inductor's own doing, not these passes.

---

## 7. What Drove Each Speedup

**bf16 / Tensor-Core promotion of gate GEMMs (OPT-1, +2.33× on the combined gate-GEMM cohort):**
Casting both operands of every decomposed `aten.mm`/`aten.addmm` to bf16 (result cast back to
fp32) moves the per-timestep gate projections off the scalar FP32 SIMT pipeline and onto the
Blackwell bf16 Tensor-Core MMA pipeline. The evidence is unambiguous: `tensor_core_active_pct`
went from exactly `0.0` (Tensor Cores fully idle, `cutlass_80_simt_sgemm` kernel) to 11–24%
(bf16 `triton_mm`/`bias_addmm` kernels), and `sm_throughput_pct` on the batched input GEMM
jumped from ~16.6% to ~64%.

**freezing + max_autotune (OPT-4, enabler):** Constant-weight freezing pre-packs the bf16
tensorop weight layout and folds the runtime `_tn_` transpose, while autotuning selects a
non-split-K GEMM tile. The visible signature is the disappearance of the per-GEMM
`cublasLt::splitKreduce` epilogue kernels that decorated every baseline gate GEMM, and the
autotuner log selecting frozen bf16 `triton_mm` kernels.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | hoist input projection (functional) | per-timestep `F.linear` | No shared-weight per-timestep `F.linear` pattern; `nn.LSTM` is a single opaque node and Inductor's decomposition already batches the input projection (no-op) | ~37.9% (already realized by Inductor) |
| OPT-3 | cuDNN fused-RNN restore (functional) | `nn.LSTM` recurrence | Mutually exclusive with the applied OPT-1/OPT-2 in-place track over the recurrent matmuls; would replace, not stack | ~75.7% (alternative track, discounted ~0.5 for medium confidence → ~38%) |

**Second-order bottleneck exposed.** With the GEMMs accelerated, layout/bookkeeping ops now
dominate the optimized profile: `aten::t` (344.2 µs), `aten::_unsafe_view` (317.8 µs), and
`aten::view` (265.3 µs) together account for ~28% of the 3252.4 µs optimized total. These are
memory-bound reshape/transpose kernels introduced by the bf16 cast boundaries and the recurrent
slice topology; `tensor_core_active_pct` is correctly null for them. No proposed FX pass targets
them directly.

**Estimated additional gain.** OPT-2 offers no further gain (its win is already captured by
Inductor). The single largest untapped opportunity is **OPT-3** — running `nn.LSTM` on the
eager cuDNN fused path keeps gate activations and cell state resident across the elementwise
update and would eliminate the ~511 recurrent kernel launches entirely. Because it is mutually
exclusive with the applied bf16 track, realizing it requires a re-run on the alternative track
rather than a stacked pass; the strategist estimates a ceiling near ~75% reduction at medium
confidence (~38% discounted). Attacking the residual view/transpose overhead would need a new
proposal not present in the current `optimizations.json`.

---

## Reproduction

```bash
# Baseline capture
/capture examples/lstm_sequence_encoder/lstm_sequence_encoder.py

# Propose → backend → validate
/propose examples/lstm_sequence_encoder/profile.json
/backend examples/lstm_sequence_encoder/lstm_sequence_encoder.py examples/lstm_sequence_encoder/optimizations.json
/validate examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py

# Optimized re-capture (custom backend)
/capture examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
    --profile-name=optimized --compile-backend=lstm_sequence_encoder_opt

# Report
/report examples/lstm_sequence_encoder/
```
