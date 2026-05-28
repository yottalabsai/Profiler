# SDPAAttentionBlock — GPU Optimization Report

**This optimization achieved a ~2.4× speedup on the attention block's compute kernels (SDPAAttentionBlock, B=8, NVIDIA RTX PRO 6000 Blackwell)** by promoting the FP32 SIMT GEMM/attention path to the BF16 Tensor-Core path and fusing the three Q/K/V projections into a single GEMM.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, ~188 SMs, 5th-gen Tensor Cores) |
| Architecture family | Blackwell |
| PyTorch version | 2.11.0+cu128 |
| CUDA | 12.8 |
| Compile mode (baseline) | inductor (built-in dedup backend) |
| Compile mode (optimized) | `sdpa_attention_opt` (custom `@register_backend`) |
| Batch size | 8 (seq_len 512, dim 512, 8 heads × 64) |
| Iterations | 2 measure-iters (ncu replay — relative timing only) |

> **All `duration_ns` values below are ncu counter-collection timings, inflated 2–5× over real wall-clock. They are used only for relative comparison within and between these two profiles, never as absolute latency.**

---

## 2. Operator Summary (Baseline)

Sorted by Time (%). On Blackwell, `tensor_core_active_pct` is reported but `warp_cycles_per_instruction` is removed; bottleneck class is derived from Tensor-Core engagement, memory throughput %, and achieved occupancy.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---:|---:|---:|---|
| `layer::unique::prologue` † | 41.13 | 913,658 | 16 | Dedup wrapper — *encloses* the mm/fmha kernels below (overlaps, not additive) |
| `aten::mm` (correlation) | 13.46 | 299,007 | 4 | Compute-bound on **FP32 SIMT** (TC 0%, occ 16.5%) |
| `aten::_efficient_attention_forward` | 6.19 | 137,408 | 1 | Compute-bound, **FP32 fmha + register spill** (TC 58.7%, 757,760 spill wavefronts) |
| `aten::_efficient_attention_forward` (op_id 14, fused) | 6.18 | 137,375 | 1 | Same SDPA call (fused double-count) |
| `aten::_efficient_attention_forward` (op_id 36, fused) | 6.18 | 137,311 | 1 | Same SDPA call (fused double-count) |
| `aten::mm` (op_id 4 — Q proj) | 3.37 | 74,880 | 1 | FP32 SIMT, TC 0% |
| `aten::mm` (op_id 5 — K proj) | 3.37 | 74,879 | 1 | FP32 SIMT, TC 0% |
| `aten::mm` (op_id 6 — V proj) | 3.36 | 74,656 | 1 | FP32 SIMT, TC 0% |
| `aten::mm` (op_id 21/26/27/28/43 — proj/out) | ~3.35 each | ~74,500 each | 1 each | FP32 SIMT, TC 0% |

† The `layer::unique::prologue` NVTX range is a layer-dedup wrapper that encloses the same `Kernel2` (mm) and `fmha` kernels attributed individually below — treat its 41% as overlapping with, not additive to, the per-operator totals.

**Diagnosis:** every Q/K/V/output projection GEMM (`Kernel2`, 4096×512 @ 512×512) ran on the FP32 CUDA-core SIMT path — `tensor_core_active_pct = 0.0`, `avg_threads_per_warp = 32.0`, `registers_per_thread = 210`, achieved occupancy ~16.5%. The 5th-gen Tensor Cores were idle on the bulk of compute. The SDPA kernel was the FP32 CUTLASS variant (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`): it engaged Tensor Cores (58.7%) but spilled hard — `l1tex__t_output_wavefronts_pipe_lsu_mem_local.sum = 757,760` per launch, occupancy 14%.

---

## 3. Reading the Metrics

Only the metrics that drove this workload's bottlenecks are explained here.

- **`tensor_core_active_pct = 0.0` (not null)** — the highest-ROI signal in this profile. A *zero* (not null) value means the GEMM ran entirely on the FP32 SIMT FMA path with Tensor Cores idle. Confirmed here by `avg_threads_per_warp = 32.0` (fully scalar 32-thread warp loop). A *null* value is expected for non-GEMM kernels and is not a problem.
- **`local_memory_spills` (`l1tex__t_output_wavefronts_pipe_lsu_mem_local.sum`)** — 757,760 on the FP32 fmha means the kernel's register footprint exceeded the file and spilled to local memory every launch. Any nonzero value on a hot kernel is a red flag; the BF16 variant drops it to 0.
- **`registers_per_thread`** — 210 on the FP32 GEMM capped occupancy at ~16.5%. Lower is better when occupancy is the limiter; the BF16 GEMM drops to 46.
- **`achieved_occupancy` (`sm__warps_active … pct`)** — 16.5% baseline indicates the SM is starved of resident warps (register-bound). Rising toward 70%+ indicates the limiter was removed.
- **`sm_throughput_pct`** — ~36% baseline shows the SMs were busy but not doing useful Tensor-Core math (scalar FMA loop).

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype promotion (→ BF16, TF32 fallback) | 9 × `aten::mm`, 3 × `aten::_efficient_attention_forward` | `tensor_core_active_pct = 0.0`, `avg_threads_per_warp = 32.0`, fmha `local_memory_spills = 757,760` | high | **APPLIED** |
| OPT-2 | fusion (QKV → 1 GEMM N=1536 + slices) | `aten::mm` op_id 4/5/6 (Q/K/V proj) | 3 separate `Kernel2` launches, grid [64,1,2]=128 blocks → ~68% SM coverage (single wave) | high | **APPLIED** |
| OPT-3 | memory layout (weight pre-transpose) | fused QKV + output proj weights | `l1tex__t_sector_hit_rate = 10.8%`, `registers_per_thread = 210` | low | **NOT_APPLIED** (graceful) |

OPT-1 and OPT-2 applied via Inductor's `post_grad_custom_pre_pass` hook. OPT-3 is a detection-only stub: the constant pre-transpose is not materializable from FakeTensor graph inputs and Blackwell cuBLAS absorbs the transpose internally, so it logs a WARNING and applies no transform.

---

## 5. Implementation Notes

# SDPAAttentionBlock — Optimized Backend Implementation Notes

Registered backend name: **`sdpa_attention_opt`**

Target workload: `examples/sdpa_attention/sdpa_attention.py` (`SDPAAttentionBlock`)
Compile mode: `inductor`. Device: NVIDIA RTX PRO 6000 Blackwell (GB202, ~188 SMs, 5th-gen Tensor Cores).
torch 2.11.0+cu128.

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 dtype promotion (BF16 on `aten.mm` operands + SDPA Q/K/V; FP32 cast-back; LayerNorm FP32) | `_aten_pass_chain` via `post_grad_custom_pre_pass` (`_pass_promote_dtype`) | Reroutes FP32-SIMT GEMM and FP32 fmha onto the HMMA Tensor-Core path; structural rewrite needing no weight values. |
| OPT-1 TF32 global fallback (`allow_tf32 = True`) | module import (non-graph) | code_hint fast-path: engages TF32 Tensor Cores for any residual FP32 matmul if BF16 rewrite no-ops. |
| OPT-2 QKV fusion (3 sibling `aten.mm` -> one N=1536 GEMM + 3 `aten.slice`) | `_aten_pass_chain` (`_pass_fuse_qkv`) | One launch + single activation read; concatenates weight placeholder nodes with an `aten.cat` graph node (FakeTensor-safe). |
| OPT-3 weight pre-transpose (speculative) | `_aten_pass_chain` (`_pass_pretranspose_weights`) — **stub, detection only** | Low confidence; constant pre-transpose not materializable from FakeTensor inputs and cuBLAS on Blackwell usually absorbs the transpose. Reports candidate count, no transform. |

All transforming passes run inside `_aten_pass_chain` at Aten IR level, in the
dependency order OPT-1 -> OPT-2 -> OPT-3 mandated by `prerequisite_for` in
`optimizations.json`. No graph pass lives in `get_model_and_input()`.

## Key Design Decisions

**IR injection point — Inductor `post_grad_custom_pre_pass`, not an `aot_autograd` fw_compiler.**
On torch 2.11 the `aot_autograd` wrapper referenced by the generic template is no longer importable
from `torch._functorch.aot_autograd`, and nesting `compile_fx` inside an `aot_module_simplified`
`fw_compiler` double-runs AOTAutograd (boxed-args `AssertionError`) or collides decomp/fallback on
`aten.native_layer_norm`. The supported torch 2.11 hook for Aten-IR passes is Inductor's
`post_grad_custom_pre_pass`, which receives the fully decomposed, functionalized Aten `fx.Graph`
immediately before lowering. The backend installs the pass chain there and delegates the full
AOTAutograd + lowering pipeline to `compile_fx(gm, example_inputs)` — the same delegation the prior
working examples in this repo use. Verified: the hook sees `aten.mm.default` x4,
`aten._scaled_dot_product_efficient_attention.default`, and `aten.permute.default`.

**Post-grad IR specifics differ from the canonical recipe.** A bias-free `nn.Linear` weight
transpose appears as `aten.permute.default(weight_ph, [1, 0])`, not `aten.t.default`; passes match
`permute`. Dtype casts use `prims.convert_element_type.default` — `aten._to_copy.default` triggers
an Inductor "both a fallback and a decomp for same op" assertion at this level.

**Structural fusion on FakeTensors.** Graph inputs at the post-grad level are FakeTensors with no
readable storage (`DataDependentOutputException` on any value read), so OPT-2 cannot precompute a
concatenated weight constant. Instead it fuses *structurally*: it concatenates the three weight
placeholder nodes via an `aten.cat` graph node, casts to BF16, emits one `aten.mm`, and slices the
`[M, 1536]` result. Output dims (`n_q/n_k/n_v`) are read from each placeholder's `meta['val']` shape.

**Topological-safe insertion.** The fused subgraph is inserted before the *earliest* of the three
sibling mm nodes, and fresh `aten.permute` weight nodes are created there (placeholders are defined
at graph top, so they are in scope everywhere). Inserting at the last sibling instead produced
"used before defined" lint failures because q/k/v consumers precede it. After OPT-1, each original
mm's live user is its FP32 convert-back node, so OPT-2 redirects those convert-back nodes (not the
mm nodes directly) to the new slices.

**LayerNorm stays FP32.** No pass touches `aten.native_layer_norm`; only mm operands and SDPA Q/K/V
are promoted, and every mm result is cast back to FP32, so the residual add and post-LN consume
FP32. Honors the accuracy guard in `optimizations.json`. End-to-end numerics: max abs diff vs the
FP32 reference ~4e-4 (BF16 GEMM rounding), no NaN/Inf.

**OPT-3 is a detection-only stub.** It is low-confidence and speculative; it counts remaining
`permute(weight)->mm` candidates (the output projection after OPT-2 fuses QKV) and logs a WARNING
explaining why no transform is applied. It never mutates the graph and never raises into the compile.

**Flat compile path for this single-block model.** `UniqueSubgraphRegistry` finds no repeated
structure, so the backend calls `compile_fx(gm, example_inputs)` directly, preserving cross-layer
Inductor fusion. The dedup branch is retained per Rule 9 for structural reuse if the workload grows.

## Validation

`pytest test_sdpa_attention_optimized.py` — 4/4 passed: import, backend registration,
get_model_and_input (CUDA / shape (8,512,512) / FP32), and the compiled smoke test. The smoke test
captures all three pass log lines and asserts the BF16 output is finite.

---

## 6. Before/After Results

Both captures use **batch size 8**. Operators are matched by logical role across profiles (operator IDs change between captures); the three baseline Q/K/V projections collapse to one fused GEMM in the optimized profile and are reported as a single summed row. All durations are per-launch ncu-inflated timings.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---:|---:|---:|
| `aten::mm` — Q/K/V projection (fused ×3) | 224,415 | 99,424 | **2.26×** |
| `aten::mm` — output projection | ~74,500 | 34,592 | **2.15×** |
| `aten::_efficient_attention_forward` (SDPA) | 137,408 | 47,168 | **2.91×** |
| **Total (per forward, major compute kernels)** | **~436,323** | **~181,184** | **~2.41×** |

Small reshape/view/LayerNorm Triton kernels (each ≤ ~6 µs, no ncu counter rows) are excluded from the total; they are unchanged in count and negligible in time. The optimized profile adds two tiny `aten::t` weight-cast kernels (~1.2 µs each) from the BF16 cast path.

---

## 7. What Drove Each Speedup

**dtype promotion to BF16 (OPT-1, applied — primary driver across all GEMMs and SDPA):** Casting the `aten::mm` operands and SDPA Q/K/V inputs to BF16 rerouted cuBLAS off the scalar FP32 SIMT `Kernel2` path onto the HMMA Tensor-Core path and selected the BF16 CUTLASS attention kernel. Evidence: the projection GEMMs moved `tensor_core_active_pct` from **0.0 → 18.9–21.1%** with `registers_per_thread` dropping **210 → 46** and achieved occupancy rising **16.5% → 74%**; the attention kernel switched `fmha_cutlassF_f32_aligned` → `fmha_cutlassF_bf16_aligned` and its `local_memory_spills` collapsed from **757,760 → 0**, cutting its duration from 137,408 → 47,168 ns.

**QKV fusion (OPT-2, applied — +2.26× on the projection GEMMs):** The three separate Q/K/V `aten::mm` nodes (each N=512, grid [64,1,2]=128 blocks, ~68% SM coverage in a single wave) became one N=1536 GEMM that reads the shared post-LayerNorm activation once and launches grid [1024,6,1], filling the machine. Evidence: three ~74.8 µs `Kernel2` launches collapsed to one 99.4 µs launch (2.26× on summed time), and the fused GEMM's achieved occupancy is 74% versus 16.5% for the baseline single-wave projections.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-3 | weight pre-transpose | fused QKV + output-projection weights | Speculative stub — constant pre-transpose not materializable from FakeTensor graph inputs; Blackwell cuBLAS absorbs the transpose internally | ~1.1% of total (low confidence) |

Second-order bottleneck exposed by the optimization: the **fused QKV GEMM is now the single largest kernel** (99,424 ns) and is still only ~19% Tensor-Core-active with `eligible_cycles_pct` ~19% and `l1tex__t_sector_hit_rate` ~1% — the BF16 GEMM is now occupancy-healthy (74%) but issue-starved, suggesting the GEMM tiling/epilogue, not precision, is the next limiter. OPT-3 targets exactly this (weight-load locality) but is low-confidence on Blackwell.

Estimated additional gain if OPT-3 were applied and effective: **~1% of total** — marginal. The BF16 promotion and QKV fusion already captured the dominant wins; no further high-confidence FX-level gains are identified in this profile.

---

## Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py --workload examples/sdpa_attention/sdpa_attention.py \
    --output-prefix profiler_output/sdpa_attention --correlation-pass
nsys profile --trace=cuda,nvtx --output=profiler_output/sdpa_attention \
    python3 nvidia/scripts/run_workload.py --workload examples/sdpa_attention/sdpa_attention.py \
        --output-prefix profiler_output/sdpa_attention
# → ManifestBuilder → AttributionEngine → KernelProfileOrchestrator (ncu, sudo) → build_profile
# Output: examples/sdpa_attention/profile.json

# Optimized capture (custom backend)
#   run_workload.py with --compile-backend sdpa_attention_opt (registered in sdpa_attention_optimized.py)
# Output: examples/sdpa_attention/profile_optimized.json

# Validation
pytest examples/sdpa_attention/test_sdpa_attention_optimized.py
```
