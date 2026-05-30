# SDPA Attention — Optimization Report

This optimization achieved a **2.23× total speedup** on `SDPAAttentionBlock` (B=8, T=512, D=512, H=8) on an NVIDIA RTX PRO 6000 Blackwell Server Edition, driven almost entirely by promoting the FP32 SIMT compute path to bf16 Tensor Cores.

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, ~188 SMs assumed) |
| Architecture family | Blackwell |
| PyTorch | 2.11.0+cu128 |
| Compile mode (baseline) | `inductor` (built-in dedup backend) |
| Compile mode (optimized) | `sdpa_attention_opt` (custom FX backend) |
| Batch size | 8 (T=512, D=512, NUM_HEADS=8, HEAD_DIM=64) |
| Iterations | 2 warmup / 2 measure (ncu replay — relative timing only) |

> All durations in this report come from ncu application-mode replay and are inflated 2–5× relative to wall-clock. Treat every absolute `ns` value as **relative**, valid only for before/after comparison within this report.

## 2. Operator Summary (baseline)

Durations exclude the `layer::unique::prologue` dedup-wrapper partition, which re-attributes the same 21 kernels already counted in the granular operators below (the `fused_kernel_double_count` edge-case flag).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::mm` (×8) | 66.6% | 479,168 | 8 | Compute-bound, **Tensor Cores idle** (FP32 SIMT) |
| `aten::_efficient_attention_forward` (×2) | 30.7% | 220,608 | 2 | Compute-bound (FP32 mem-efficient fmha) |
| `aten::_unsafe_view` (×2) | 1.6% | 11,168 | 2 | Memory-bound (reshape) |
| `aten::native_layer_norm` (×2) | 1.2% | 8,768 | 2 | Memory-bound (DRAM ~46%) |
| **Total (attributed, granular)** | 100% | **719,712** | 14 | — |

Every projection matmul ran as `cutlass_80_simt_sgemm_128x256_8x4_tn_align1` and the two attention ops as `fmha_cutlassF_f32_aligned_64x64_rf_sm80` — both FP32 paths that bypass the Tensor Cores entirely.

## 3. Reading the Metrics

Only the counters that drive this workload's bottleneck are explained here.

- **`smsp__pipe_tensor_cycles_active` / `tensor_core_active_pct` = 0.0 (not null)** — the single highest-ROI signal in this profile. A literal `0.0` means the GEMM executed on the FP32 **SIMT/CUDA-core** datapath with the Tensor Cores completely idle. (A *null* value, by contrast, is expected for non-GEMM kernels and is not a problem.) Every baseline `aten::mm` reported `0.0`, confirmed by the `simt_sgemm` substring in the kernel name.
- **`sm__throughput.avg.pct_of_peak` ≈ 36%** — the SIMT GEMMs only reach about a third of peak SM throughput; combined with idle Tensor Cores this is the textbook "wrong datapath" fingerprint.
- **`sm__warps_active` (achieved occupancy) ≈ 16.6%** — very low. The FP32 kernel burns 210 registers/thread, capping occupancy; small grids (128 CTAs) under-fill the ~188-SM device.
- **`gpu__dram_throughput` ≈ 7.5%** on the GEMMs — confirms these kernels are compute/datapath-limited, not bandwidth-limited, so the fix is the compute path (dtype), not memory layout.

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype promotion (fp32→bf16) | `aten::mm` (7), `aten::_efficient_attention_forward` (2) | `tensor_core_active_pct=0.0`, `simt_sgemm` kernel, 36% SM throughput, 16.6% occupancy across ~89% of attributed time | high | **APPLIED** |
| OPT-2 | QKV fusion | 3× bias-free Q/K/V `aten::mm` per block (6 nodes) | 128-CTA grids at 16.6% occupancy; 3 serial sub-one-wave launches sharing one activation | high | **NOT_APPLIED** |
| OPT-3 | weight pre-transpose / alignment | `aten::mm` weight operands (7) | `align1` tile, `l1tex__t_sector_hit_rate ≈ 10.8%`, 210 regs/thread | low | **NOT_APPLIED** |

Validation source: `profiler_output/validation_report.json`. OPT-1 promoted 5 mm/SDPA operand groups to bf16. OPT-2 found no 3-way shared-activation `mm` triplet in the lowered graph and degraded gracefully; OPT-3 found no surviving `aten.t → mm` weight chain and degraded gracefully. Both no-ops were clean (no exceptions).

## 5. Implementation Notes

# Implementation Notes — sdpa_attention_opt

Backend name: `sdpa_attention_opt` (registered via `@register_backend` at module import).
Workload: `SDPAAttentionBlock` (multi-head self-attention, FP32, B=8 T=512 D=512 H=8).
Compile mode: `inductor`. All three optimizations are graph-level Aten-IR passes.

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 dtype promotion (fp32 -> bf16) | `_aten_inner_compile` (`_pass_bf16_promotion`) | Casts every `aten.mm` operand pair and the SDPA q/k/v operands to bf16 so CUTLASS picks an HMMA Tensor-Core tile instead of the idle `simt_sgemm` path (tensor_core_active_pct was 0.0); a single output down-cast restores the fp32 contract. Op-target pass — needs no weight values. |
| OPT-2 QKV fusion | `_aten_inner_compile` (`_pass_fuse_qkv`) | Three bias-free projections share the ln_pre activation; replaces three `aten.mm(act, W)` with one `aten.cat` + wide `aten.mm` + three `aten.slice` views, turning three sub-one-wave launches into one wider GEMM. |
| OPT-3 weight pre-transpose / alignment | `_aten_inner_compile` (`_pass_pretranspose_weights`) | Folds `aten.t.default(weight)` feeding an `aten.mm` into a contiguous pre-transposed bf16 buffer so CUTLASS can lift the `align1` -> `align8` 128-bit load path. Weight-VALUE-reading — uses the threaded `real_inputs` `ph_to_tensor` lookup. Confidence low: graceful no-op when no surviving `t->mm` chain exists. |
| (non-graph levers) | none | This workload has no conv/layout/batch-shape lever; dtype is promoted selectively in-graph (OPT-1), not by casting the whole module, so `get_model_and_input()` returns the unmodified FP32 model. |

## Key Design Decisions

**Strategy D (`inner_compile` hook), not `aot_autograd(fw_compiler=...)`.** All passes run inside `_aten_inner_compile`, installed via `compile_fx(gm, example_inputs, inner_compile=...)`. `compile_fx` keeps ownership of AOTAutograd, the decomposition table, the boxed calling convention, and the fwd/bwd partitioner; we only swap the leaf compiler (Aten -> Triton) by delegating to `compile_fx_inner` after the passes. The `aot_autograd` fw_compiler path raises `AssertionError: Expected tensors only, but got list` in `copy_misaligned_inputs` on torch 2.11, so it is avoided.

**Prerequisite ordering OPT-1 -> OPT-2 -> OPT-3 is load-bearing.** OPT-1 runs first because the fused/pre-transposed buffers built by OPT-2/OPT-3 must inherit the bf16 runtime dtype; OPT-2 runs before OPT-3 because fusion changes the surviving weight node set that OPT-3's pre-transpose acts on. `FakeTensorProp` is re-run between passes so each downstream pass (and Inductor) can read fresh `meta['val']` — OPT-2 in particular reads weight `shape[1]` from meta to size its slices.

**FakeTensor vs. real weights.** Inductor traces `_aten_inner_compile` under FakeTensorMode, so `example_inputs` there may be FakeTensors. OPT-1 and OPT-2 are purely structural (operate on nodes / read shape meta) and are FakeTensor-safe. OPT-3 reads actual weight values to materialize the pre-transposed buffer, so the backend threads genuine parameter tensors as `real_inputs` (via `functools.partial`) and builds the `ph_to_tensor` map from those, not from the Fake `example_inputs`.

**Dedup-aware but flat in practice.** `UniqueSubgraphRegistry` is built unconditionally per Rule 9. A single attention block produces no repeated structural partitions, so `build_partition_equivalence_map()` is empty and the flat compile path is taken — this preserves cross-op Inductor fusion (e.g. the residual-add + LayerNorm Triton fusion). The per-rep dedup branch is retained for models with repeated identical blocks (e.g. multi-layer transformers).

**Graceful degradation.** Every pass is independently try/guarded and emits a `logger.warning` no-op when its pattern is absent (OPT-1: no mm/SDPA; OPT-2: no 3-way shared-activation triplet; OPT-3: no `t->mm` chain). A single pass failure degrades to the unmodified subgraph rather than crashing compilation.

## 6. Before/After Results

Both profiles share batch size 8 and were captured on the same GPU ~11 minutes apart — no cross-session caveat applies. Operators are matched by `operator_name`; the new `aten::t` row in the optimized profile is the set of transpose/cast materializations introduced by OPT-1's bf16 conversion.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| `aten::mm` (×8) | 479,168 | 220,703 | **2.17×** |
| `aten::_efficient_attention_forward` (×2) | 220,608 | 73,152 | **3.02×** |
| `aten::_unsafe_view` (×2) | 11,168 | 10,112 | 1.10× |
| `aten::native_layer_norm` (×2) | 8,768 | 11,744 | 0.75× |
| `aten::t` (×8, new) | 0 | 7,392 | — (cast overhead) |
| **Total** | **719,712** | **323,103** | **2.23×** |

**Speedup attribution.** Both the `aten::mm` (2.17×) and `aten::_efficient_attention_forward` (3.02×) gains are attributed to **OPT-1**: its status is `APPLIED`, the expected hardware metric moved in the right direction (see §7), and both operators improved. No speedup is attributed to OPT-2 or OPT-3 — they are `NOT_APPLIED`.

The small `native_layer_norm` regression (0.75×) and the new ~7.4k ns `aten::t` row are second-order costs of inserting bf16 casts at the FP32 boundary; they are dwarfed by the GEMM and attention gains.

## 7. What Drove Each Speedup

**dtype promotion fp32→bf16 (OPT-1, +2.17× on `aten::mm`, +3.02× on attention):** Casting the mm operands and SDPA q/k/v inputs to bf16 caused CUTLASS to re-select a Tensor-Core HMMA tile in place of the CUDA-core SIMT kernel. The kernel name changed from `cutlass_80_simt_sgemm_128x256_8x4_tn_align1` to `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_...`, and `smsp__pipe_tensor_cycles_active` rose from **0.0% → 21.0%** (Tensor Cores now engaged), with achieved occupancy climbing **16.6% → 60.9%**, SM throughput **36% → 45%**, and per-GEMM DRAM read traffic roughly halving (**9.5 → 4.7 units**) from the narrower bf16 operands. On the attention side the kernel switched from `fmha_cutlassF_f32_aligned` to `fmha_cutlassF_bf16_aligned`, the faster bf16 mem-efficient backend.

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | QKV fusion | 3× Q/K/V `aten::mm` per block | "No 3-way shared-activation mm triplet found" — after bf16 lowering the three projections did not surface as a matchable sibling triplet in the Aten graph | ~30k ns (~4.2% of baseline); eliminates 4 launches |
| OPT-3 | weight pre-transpose / alignment | fused/qkv weight operand | "No aten.t→mm weight chain found" — the transpose was already folded upstream once bf16 was in place | ~15k ns (~2.1%), low confidence; likely redundant after OPT-1 |

If both remaining passes were applied and matched, the discounted projected additional gain is roughly **6% of the original baseline** (~45k ns) — modest, since OPT-1 already captured the dominant FP32→Tensor-Core win. OPT-2 is the higher-value follow-up: re-expressing the projections so the fusion pass can match the triplet (or fusing Q/K/V at the module level) would cut launch overhead and present a wider GEMM tile to better fill the 188-SM device.

## Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py --workload examples/sdpa_attention/sdpa_attention.py \
    --correlation-pass --output-prefix profiler_output/sdpa_attention \
    --inductor-debug-dir profiler_output/sdpa_attention_inductor_debug
nsys profile --trace=cuda,nvtx --output=profiler_output/sdpa_attention \
    python3 nvidia/scripts/run_workload.py --workload examples/sdpa_attention/sdpa_attention.py \
    --output-prefix profiler_output/sdpa_attention
# (manifest → attribution → ncu replay → build_profile → profile.json)

# Optimized capture (registered backend)
python3 nvidia/scripts/run_workload.py --workload examples/sdpa_attention/sdpa_attention_optimized.py \
    --compile-backend sdpa_attention_opt   # → profile_optimized.json
```

For a clock-locked, fully apples-to-apples comparison, capture both profiles back-to-back with `nvidia-smi -lgc <freq>`.
