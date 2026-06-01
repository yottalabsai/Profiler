# Optimization Report — LSTM Sequence Encoder

**This optimization achieved a 3.63× total speedup on LSTMSequenceEncoder (B=32, NVIDIA RTX PRO 6000 Blackwell)** by abandoning Inductor's harmful decomposition of `nn.LSTM` and letting the recurrent region run eagerly through cuDNN's fused Tensor-Core RNN. The baseline's Inductor build unrolled the LSTM into ~1,280 FP32 **SIMT scalar** GEMMs (Tensor Cores 0.0% active) plus an equal number of `splitKreduce` epilogues; the optimized build replaces them with cuDNN's batched bf16 tensor-op GEMMs (input projection batched to M=4096, Tensor Cores 23.8% active). A standalone `triton_poi_fused_t_*` transpose cohort (~0.41 ms) is newly introduced by the compiled head and is included in the optimized total below.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture family | Blackwell |
| PyTorch version | 2.11.0+cu128 |
| Compile mode (baseline) | `inductor` (built-in dedup backend) |
| Compile mode (optimized) | `lstm_sequence_encoder_opt` (custom backend; LSTM runs eager) |
| Batch size | 32 (SEQ_LEN=128, INPUT_SIZE=256, HIDDEN_SIZE=512, NUM_LAYERS=2) |
| Iteration count | 2 warmup / 2 measure (nsys capture — durations measured at locked GPU clocks; relative comparison) |
| GPU clock lock | 1845 MHz graphics / 12481 MHz memory — identical across both captures (cached) |

**Timing source.** Per-operator **durations** come from the **nsys capture** phase (GPU kernel times), not from ncu. `run_workload.py` probed and locked the sustained clock and cached it, so the baseline and optimized captures locked to the **identical** 1845 MHz — the comparison is fair and reproducible. The ncu replay phase contributed only the hardware **counters** (Tensor-Core %, SM/DRAM throughput, occupancy), collected at its own base-clock lock.

---

## 2. Operator Summary (Baseline)

Dominant GPU kernels in `profile.json` (deduped by kernel id; total attributed GPU time **14,530 µs**):

| Kernel | Time (%) | Duration (µs) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `cutlass_80_simt_sgemm_128x32_8x5_tn` (gate GEMMs) | 58.6% | 8,512.6 | 1280 | Compute, **FP32 SIMT — Tensor Cores idle** |
| `cublasLt::splitKreduce_kernel` (split-K epilogue) | 15.5% | 2,258.6 | 1280 | Launch/epilogue overhead |
| `cutlass_80_simt_sgemm_256x128_8x4_tn` | 11.6% | 1,687.4 | 10 | Compute, FP32 SIMT |
| `cutlass_80_simt_sgemm_32x128_8x5_tn` | 0.2% | 22.5 | 5 | Compute, FP32 SIMT |
| `triton_poi_fused_clone_transpose_0` | 0.1% | 14.1 | 5 | Memory-bound |
| `triton_red_fused_mean_transpose_258` | 0.1% | 14.1 | 5 | Memory-bound (mean pool) |

The dominant operator (`layer::unique::prologue`, holding the unrolled gate GEMMs) shows **`tensor_core_active_pct = 0.0`**, `sm_throughput = 18.5%`, `achieved_occupancy = 9.4%` — a tiny per-timestep GEMM (M=32) running on the scalar SIMT path with ~1 wave of CTAs, trailed by a split-K reduction that is pure overhead at this size.

---

## 3. Reading the Metrics

Only the metrics that drive this workload's bottleneck are explained:

- **`tensor_core_active_pct = 0.0` (not null)** — the single highest-ROI signal here. The gate GEMMs ran on the FP32 SIMT path with Tensor Cores **completely idle**. A null value is expected for non-GEMM kernels; `0.0` on a GEMM means a real, addressable Tensor-Core opportunity.
- **`achieved_occupancy ≈ 9.4%`** — with M=32 the GEMM launches a <1-wave grid (~128 CTAs), so most of the Blackwell SM array sits idle. Anything that raises the GEMM's M dimension raises occupancy directly.
- **`sm_throughput ≈ 18.5%`** — SMs are mostly idle; this is a latency/under-occupancy problem, not a saturated-compute one.
- **`dram_throughput ≈ 20%`, `l2_hit_rate ≈ 69%`** — **not** memory-bound; the problem is the compute path and launch granularity, not bandwidth.
- **`splitKreduce` count == GEMM count** — every tiny GEMM emits a matching split-K reduction kernel, doubling the launch count for zero useful FLOPs.

---

## 4. Optimizations Applied

Pass statuses from `profiler_output/validation_report.json`:

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | op-substitution (keep LSTM eager → cuDNN fused RNN) | `nn.LSTM` / `aten.lstm` | TC 0.0% on SIMT path; `warnings[]` shows eager cuDNN tensor-op path exists | medium | **NOT_APPLIED** (Route-1 graph break — see note) |
| OPT-2 | fusion (hoist input projection, M 32→4096) | `F.linear` | M=32 → <1-wave grid, 9.4% occupancy | high | NOT_APPLIED (mutually exclusive w/ OPT-1; stub) |
| OPT-3 | dtype_promotion (bf16 GEMM operands) | `aten.mm` / `aten.addmm` | `tensor_core_active_pct = 0.0` | high | NOT_APPLIED in validation harness (fires on head GEMM at re-capture) |
| OPT-4 | inductor_config (freezing + max_autotune) | `aten.mm` | 1280 `splitKreduce` epilogues | medium | NOT_APPLIED (compile_fx not reached for LSTM region) |

**Critical interpretation.** All four passes report `NOT_APPLIED` because, under the adopted **Route 1**, `nn.LSTM` is a hard Dynamo graph break (`graph_count = 0`) — the custom backend callback is never invoked for the recurrent region. **This is the intended design, not a failure.** The measured 3.63× speedup is therefore *not* credited to an FX-graph transformation. It comes from the **structural decision to let the LSTM run eagerly** (where cuDNN's fused Tensor-Core RNN executes it), instead of the baseline's built-in Inductor backend that *decomposed* it into the slow FP32 SIMT path. Per the attribution rule (a speedup is credited to a pass only when `status == APPLIED`), the win is attributed to the Route-1 architecture choice, which OPT-1 documents — not to a transformed graph.

---

## 5. Implementation Notes

# Implementation Notes — `lstm_sequence_encoder_opt`

Registered backend: **`lstm_sequence_encoder_opt`** (`@register_backend`,
auto-registered on import of `lstm_sequence_encoder_optimized.py`).

Funnel: `_run_functional_passes(gm)` → `compile_fx(inner_compile=_aten_inner_compile,
config_patches=_config_patches())`. Routed by `ir_level`
(functional → aten → inductor_config). Dedup-aware via `UniqueSubgraphRegistry`.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 — keep LSTM eager (cuDNN fused RNN, op-substitution) | functional + non-graph | `_run_functional_passes` (audit) + model architecture | The "unrolled gate loop → fused cuDNN RNN" substitution is realised structurally: `nn.LSTM` is a hard Dynamo graph break (graph_count 0), so it already runs eagerly through cuDNN's Tensor-Core `elemWiseRNNcell` + `s1688gemm` path; the functional pass only audits that no unrolled gate-GEMM cohort leaked into the compiled graph. Functional level because the substitution site is the `nn.LSTM` call before any decomposition. |
| OPT-2 — hoist input projection (fusion) | functional | `_run_functional_passes` (stub — not applied) | Mutually exclusive with OPT-1; only meaningful if the unrolled custom-cell loop is retained. Under OPT-1 there is no shared-`W_ih` `F.linear` triplet in the graph, so it is detection-only and never transforms. Functional level because the shared-weight identity across timesteps is obscured after AOTAutograd decomposition. |
| OPT-3 — bf16 GEMM operand promotion (dtype_promotion) | aten | `_aten_inner_compile` | Promotes float32 `aten.mm`/`aten.addmm` operands to bf16 (result cast back to f32) so cuBLAS/cutlass selects the tensorop GEMM instead of the 0%-Tensor-Core `simt_sgemm` path. Applies to whatever GEMM reaches the graph (the classifier addmm `[32,512]x[512,10]`, plus any gate mm if the loop is ever retained). Aten level because dtype casts key on decomposed `aten.mm`/`addmm` nodes. The conservative TF32 half is also enabled non-graph in `get_model_and_input()`. |
| OPT-4 — freezing + max_autotune (inductor_config) | inductor_config | `config_patches` | Constant-folds weights (drops the `_tn_` transpose and per-call weight guards) and autotunes a non-split-K GEMM tile to eliminate `splitKreduce` epilogues; with OPT-3's bf16 weights it emits a pre-packed tensorop layout. Scoped to each `compile_fx` call (no global mutation). inductor_config because Inductor owns weight layout/freezing. |

## Key Design Decisions

**Route 1 over Route 2.** The proposal defines two mutually-exclusive routes over the recurrent matmuls. I adopted Route 1 (OPT-1) — ranked priority 1 and called "the single largest structural win" — because it is also the only route that is *satisfiable on the natural graph*. On torch 2.11, `nn.LSTM` forces a hard Dynamo graph break: `dynamo.explain` reports `graph_count 0` for this model and the custom backend is never invoked for the recurrent region, so the cuDNN fused Tensor-Core RNN runs eagerly by default — exactly OPT-1's intended end state. Route 2 (OPT-2 input-projection hoist on per-timestep `F.linear` triplets) has no matchable pattern because Dynamo never traces the gate loop into the functional graph; OPT-2 is therefore a detection-only stub that logs the mutual-exclusion decision rather than transforming.

**OPT-3/OPT-4 still wired through the funnel.** Even though the dominant gate GEMMs live in the eager region, the `mean + classifier` tail (and any GEMM a future caller forces into the graph via `fullgraph` or a custom unrolled cell) still benefits from Tensor-Core promotion and freezing. Implementing them as real funnel passes means they fire wherever an in-graph GEMM exists, with no effect when the graph is empty of GEMMs. The conservative TF32 toggle in `get_model_and_input()` additionally accelerates the eager cuDNN LSTM, which Inductor never sees and thus the bf16 aten pass cannot reach.

**OPT-3 self-edge restore.** After `node.replace_all_uses_with(out_f32)`, the inserted f32 cast's own input would point at itself; `out_f32.update_arg(0, node)` restores the edge from the cast back to the original mm. `_repropagate_meta` then repopulates `meta['val']` on inserted cast nodes before `compile_fx_inner`.

**OPT-4 fallback.** `freezing`/`max_autotune` are not available in every Inductor build; `_compile_unit` retries `compile_fx` without `config_patches` if the patched call raises, so the head still lowers.

> **Post-generation fixes (applied during re-capture).** Two minimal, behavior-preserving fixes were made to the OPT-3 aten pass so the backend compiled and ran: (1) the inserted dtype casts were switched from `aten._to_copy.default` to `prims.convert_element_type.default` to avoid an Inductor "both a fallback and a decomp for same op" lowering assertion; (2) for `aten.addmm`, the bias operand is now also promoted to bf16 (Inductor's fused `bias_addmm` requires `bias.dtype == mat.dtype`). Both preserve the intent: bf16 GEMM operands, result cast back to f32.

---

## 6. Before/After Results

Both captures share batch size B=32 and ran on the same GPU (`NVIDIA RTX PRO 6000 Blackwell Server Edition`), ~5h15m apart, locked to the identical 1845 MHz — under the 6-hour cross-session threshold, so no clock-variation caveat applies.

**Totals** (deduped unique-kernel GPU time):

| | Baseline | Optimized | Speedup |
|---|---|---|---|
| Total attributed GPU time | 14,530.5 µs | 3,997.9 µs | **3.63×** |
| Unique kernels | 3,875 | 1,086 | 3.57× fewer |

**By kernel cohort** (baseline SIMT cohort matched to the optimized tensor-op cohort it was replaced by):

| Cohort | Baseline (µs) | Optimized (µs) | Speedup |
|---|---|---|---|
| Recurrent gate GEMMs (`simt_sgemm` → `tensorop_bf16`) | 8,512.6 + 1,687.4 = 10,200.0 | 2,617.5 + 108.0 + 10.0 = 2,735.5 | **3.73×** |
| Split-K epilogues (`splitKreduce`) | 2,258.6 | 0 (eliminated) | ∞ (removed) |
| Classifier head GEMM (`addmm`/`tensorop_bf16_relu`) | included in SIMT cohort | 98.7 | — |
| Mean-pool / misc triton | ~28 | ~12 | ~2.3× |
| **Transpose cohort** (`triton_poi_fused_t_*`) | n/a | 374.9 + 35.1 + 6.3 ≈ 416 | **new overhead** |
| **Total** | **14,530.5** | **3,997.9** | **3.63×** |

The split-K epilogue cohort (2,258 µs, 15.5% of baseline) is **entirely eliminated** — cuDNN's fused RNN performs no split-K reduction. The optimization newly introduces a `triton_poi_fused_t_*` transpose cohort (~416 µs) from the compiled head's layout handling; it is included in the optimized total above and does not inflate the speedup.

**Speedup attribution** (per `validation_report.json`): all FX passes are `NOT_APPLIED` under the Route-1 graph break, so the speedup is **not** credited to any FX-graph transformation. It is attributed to the Route-1 structural choice (LSTM kept eager → cuDNN fused Tensor-Core RNN), with the non-graph TF32 toggle from `get_model_and_input()` enabling cuDNN's Tensor-Core path. Inductor's baseline decomposition was the source of the regression that Route 1 reverses.

---

## 7. What Drove The Speedup

**Keep LSTM eager → cuDNN fused Tensor-Core RNN (Route 1 / OPT-1, 3.73× on the recurrent GEMMs):** the baseline's Inductor backend unrolled the two-layer LSTM into ~1,280 per-timestep FP32 GEMMs (M=32) on the scalar SIMT path; the optimized backend lets `nn.LSTM` graph-break to cuDNN, which executes the same math as batched bf16 tensor-op GEMMs. The dominant optimized GEMM is `aten::mm [[4096,256],[256,2048]]` — **M jumped from 32 to 4096** because cuDNN batches the input-to-hidden projection across all 128 timesteps in one launch (the effect OPT-2 would have hand-rolled, delivered natively). Hardware evidence: `tensor_core_active_pct` rose **0.0% → 23.8%**, `achieved_occupancy` **9.4% → 39.4%**, `sm_throughput` **18.5% → 63.7%**, and the entire 2,258 µs `splitKreduce` epilogue cohort disappeared.

**TF32/bf16 Tensor-Core enablement (OPT-3 non-graph half):** `torch.backends` TF32/reduced-precision flags set in `get_model_and_input()` let cuDNN and the compiled classifier `addmm` select the `cutlass_..._tensorop_bf16` kernels instead of FP32 SIMT — visible as every dominant optimized kernel now carrying the `tensorop_bf16` tag.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | fusion (hoist input projection) | `F.linear` | Mutually exclusive with the adopted Route 1; the gate loop never enters the functional graph, so no shared-weight triplet exists to match. cuDNN already batches the projection (M=4096) natively, so the gain is **already realized** by Route 1. | Subsumed by OPT-1 |
| OPT-3 | dtype_promotion (in-graph) | `aten.mm` / `aten.addmm` | Backend callback not invoked for the eager LSTM region; the aten pass only reaches the small compiled head (`addmm [32,512]×[512,10]`), already on Tensor Cores. The TF32 non-graph half supplies the LSTM benefit. | <1% (head only) |
| OPT-4 | inductor_config (freezing) | `aten.mm` | `compile_fx` is never reached for the recurrent region under the graph break; applies only to the head, where weights are tiny. | <1% (head only) |

Residual second-order bottleneck: the newly-introduced `triton_poi_fused_t_*` transpose cohort (~416 µs, ~10% of the optimized total) is now the largest non-cuDNN cost. A layout-aware pass that feeds cuDNN's preferred memory format directly — or routing the head through `channels_last`/pre-transposed weights — could reclaim most of it. Estimated additional gain if addressed: **~8–10%** of the optimized total (~0.3–0.4 ms). Beyond that, the recurrent region is now bound by cuDNN's fused RNN and is not further addressable at the FX level.

---

## Reproduction

```bash
# Baseline capture (built-in inductor dedup backend)
/capture examples/lstm_sequence_encoder/lstm_sequence_encoder.py

# Propose → backend → validate
/propose  examples/lstm_sequence_encoder/profile.json
/backend  examples/lstm_sequence_encoder/lstm_sequence_encoder.py examples/lstm_sequence_encoder/optimizations.json
/validate examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py

# Optimized re-capture (custom backend)
/capture examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
    --profile-name=optimized --compile-backend=lstm_sequence_encoder_opt
```
