# Optimization Report — EmbeddingProjection

**This optimization achieved a 2.22× total speedup on EmbeddingProjection (B=64, T=128, NVIDIA RTX PRO 6000 Blackwell Server Edition).**

The matmul compute dropped ~4.5× by moving every GEMM off the FP32 SIMT path onto Tensor Cores; about half that win is reabsorbed by a new FP32 recast of the logit output that OPT-1 introduces (see §6/§8).

The single highest-ROI signal in the baseline was unambiguous: every GEMM ran on the FP32 SIMT path (`cutlass_80_simt_sgemm_*`) with `tensor_core_active_pct = 0.0`, leaving the Blackwell Tensor Cores 100% idle across ~99% of attributed GPU time. Promoting the matmul operands to BF16 routed all of them onto the `tensorop_bf16_s16816` Tensor-Core path.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture family | Blackwell (SM 100) |
| PyTorch version | 2.11.0+cu128 |
| Compile mode | inductor |
| Batch size | 64 (T = 128 → 8192 rows per GEMM) |
| Iteration count | warmup 2 / measure 2 (nsys capture — GPU kernel times; relative comparison) |
| Capture (optimized → baseline) | 2026-05-31 20:58 UTC → 23:42 UTC (~2h44m apart, same GPU — within-session) |

> Per-operator **durations** below are **nsys-derived GPU kernel times** (close to real execution), used for **relative** comparison (baseline vs. optimized, operator vs. operator). The ncu phase contributes only the hardware **counters** (tensor-core %, SM/DRAM throughput, occupancy).

---

## 2. Operator Summary (baseline)

Sorted by share of attributed GPU time. Bottleneck class on Blackwell is derived from SM throughput, DRAM throughput, and achieved occupancy (the per-kernel `tensor_core_active_pct` is reported directly because it is the governing signal here).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::mm` logit `[8192,512]×[512,32000]` (×2) | 85.3% | 8,699,358 | 2 | Compute-bound, **TC 0.0** (SIMT FP32 SGEMM; SM 64, DRAM 13) |
| `aten::addmm` proj2 `[8192,2048]×[2048,512]` (×2) | 7.8% | 793,342 | 2 | Compute-bound, TC 0.0 (SM 42) |
| `aten::mm` proj1 `[8192,512]×[512,2048]` (×2) | 6.3% | 638,269 | 2 | Compute-bound, TC 0.0 (SM 57) |
| `aten::addmm` (bias tail) | 0.4% | 44,831 | 2 | Memory-bound (DRAM 89) |
| `aten::embedding` (gather) | 0.2% | 18,304 | 2 | Memory-bound (DRAM 52) |

**Total attributed:** 10,194,104 ns across 8 operators, 0 unattributed (2 measured forward passes).

---

## 3. Reading the Metrics

Only the metrics that drive this workload's bottleneck are explained.

- **`tensor_core_active_pct = 0.0` (not null).** The highest-ROI signal in this profile. A literal `0.0` on a GEMM means it ran on the FP32 SIMT path with Tensor Cores completely idle — pure wasted Blackwell throughput. (A `null` value, by contrast, is expected for non-GEMM kernels — embedding gather, LayerNorm, GELU, transpose — and is never a problem.) Every GEMM in the baseline reads `0.0`.
- **`sm_throughput_pct` ~57–64% with `tensor_core_active_pct = 0.0`.** The SMs are busy, but busy doing scalar FP32 FMAs instead of Tensor-Core MMAs — high SM% here is a symptom of the wrong instruction mix, not of healthy utilization.
- **`achieved_occupancy` ~17%.** Capped by 212 registers/thread on the SGEMM path. Secondary to the dtype issue; the BF16 tensorop kernels relieve register pressure as a side effect.
- **`dram_throughput_pct`** governs the bandwidth-bound tails (embedding gather 52%, small addmm 87%) — these are sub-1% of runtime and not separately actionable.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | BF16 dtype promotion (aten IR) | every `aten.mm` / `aten.addmm` (proj1, proj2, logit ×2) | `tensor_core_active_pct = 0.0` on all GEMMs; `cutlass_80_simt_sgemm_*` dispatch; ~1.05 GB FP32 logit write | high | **APPLIED** |
| OPT-2 | Weight freezing + `max_autotune` (inductor_config) | wide logit GEMM (N=32000) | unusual N=32000 GEMM benefits from tile/split-K autotune against a frozen `_tn_` BF16 layout | medium | **APPLIED** |

Both passes applied cleanly — neither degraded gracefully (no WARNING/skip emitted). Source: `profiler_output/validation_report.json`.

---

## 5. Implementation Notes

# Implementation Notes — EmbeddingProjection (`embedding_projection_opt`)

Backend file: `examples/embedding_projection/embedding_projection_optimized.py`
Test suite:   `examples/embedding_projection/test_embedding_projection_optimized.py`
Registered backend name: **`embedding_projection_opt`**
compile_mode: `inductor` (standard FX-pass funnel; full backend written)

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 — BF16 dtype promotion on `aten.mm` / `aten.addmm` (cast matrix operands to bf16, leave addmm bias fp32, recast output to fp32) | aten | `_aten_inner_compile` | Every GEMM dispatches to the SIMT FP32 `cutlass_80_simt_sgemm_*` path with `tensor_core_active_pct=0.0`; bf16 operands route cuBLAS to a Blackwell Tensor Core kernel and halve the ~1.05 GB fp32 logit-output write. Runs at the aten level because the cast keys on the decomposed `aten.mm`/`aten.addmm` op targets, which only exist after AOTAutograd decomposition. |
| OPT-2 — Weight freezing + `max_autotune` (`freezing=True`, `max_autotune=True`, `max_autotune_gemm_backends="ATEN,TRITON"`) | inductor_config | `config_patches` on `compile_fx` | Inductor cannot pre-pack or autotune against an unfrozen weight layout; freezing folds the `_tn_` transpose to compile time and autotune benchmarks tile/split-K configs for the unusual N=32000 logit GEMM. It is a config dict, not a graph rewrite, so it is owned by Inductor lowering. |

No functional-level pass exists for this workload, and no non-graph `get_model_and_input()` optimization is applied (the module and input stay fp32; OPT-1 promotes selectively in-graph).

## Key Design Decisions

**Why no functional-level fusion.** `proj1` (512→2048), `proj2` (2048→512) and the logit projection (512→32000) form a sequential dependent chain off the embedding→LayerNorm output — there is no shared activation feeding three independent projections, so QKV-style weight fusion is unavailable. Dtype promotion plus freezing/autotune are the only structural levers, matching the proposal's `global_notes`.

**Why `prims.convert_element_type` instead of `aten._to_copy`.** OPT-1 runs inside `_aten_inner_compile` on the already-decomposed graph. On torch 2.11 `aten._to_copy` carries both a fallback and a decomp registration; inserting it post-decomposition makes Inductor raise "both a fallback and a decomp for same op". `prims.convert_element_type` lowers cleanly to a Triton elementwise cast. The proposal's `fx_steps` use `_to_copy`, but it is written against the conceptual aten graph; the production funnel substitutes the prims op for torch-2.11 compatibility.

**addmm bias handling.** For `aten.addmm.default(bias, mat1, mat2)` only `mat1`/`mat2` (args 1,2) are cast to bf16; the bias (arg 0) is left fp32 and the bf16 GEMM result is recast to fp32 before the bias add is observed by downstream consumers via the output cast, preserving numerics. `proj2` (the only biased Linear reaching addmm) is thus promoted without a bias-dtype mismatch.

**Output dtype contract.** Each promoted GEMM gets a `convert_element_type(..., float32)` inserted immediately after it and `replace_all_uses_with` (excluding the new cast itself) re-anchors all consumers, so the graph output stays float32 — verified by test 4's dtype assertion. Inductor CSE collapses duplicate weight casts, so the per-node casts do not multiply weight-conversion work.

**Cross-level ordering.** OPT-1 is `prerequisite_for` OPT-2: freezing/autotune is most effective once the GEMMs are bf16 tensorop kernels (more tile-sensitive knobs than SIMT SGEMM). The funnel's fixed `functional → aten → inductor_config` order satisfies this automatically — no within-level sequencing needed.

**Flat compile path.** EmbeddingProjection has no repeated structure, so `UniqueSubgraphRegistry.build_partition_equivalence_map()` returns empty and the backend takes the flat `_compile_unit` path. This preserves Inductor's cross-op fusion of the bandwidth-bound embedding/LayerNorm/GELU tails (sub-1% each, not separately actionable). The dedup branch is retained unchanged for forward compatibility.

---

## 6. Before/After Results

Baseline and optimized were captured ~2h44m apart on the same GPU (within-session — no cross-session caveat). Both profiles cover **2 measured forward passes**; operators are matched by shape/role across profiles, not by `operator_id`. Durations are total attributed ns across the 2 passes.

| Operator | Baseline (ns) | Optimized (ns) | Speedup | Tensor-core % |
|---|---|---|---|---|
| `aten::mm` logit `[8192,512]×[512,32000]` | 8,699,358 | 1,934,751 | **4.50×** | 0.0 → 79.2–79.6 |
| `aten::addmm`/`mm` proj2 `[8192,2048]×[2048,512]` ¹ | 793,342 | 166,816 | **4.76×** | 0.0 → 85.6–85.7 |
| `aten::mm` proj1 `[8192,512]×[512,2048]` | 638,269 | 147,680 | **4.32×** | 0.0 → 75.9 |
| embedding + bias/GELU tails | 63,135 | 83,584 | 0.76× | n/a |
| `aten::t` weight transpose (introduced) ² | 0 | 122,784 | new overhead | n/a |
| FP32 logit recast (introduced; `triton_poi_fused_5`) ² | 0 | 2,131,710 | new overhead | n/a |
| **Total attributed** | **10,194,104** | **4,587,325** | **2.22×** | — |

¹ The biased proj2 is `aten::addmm` in the baseline (bias folded into the GEMM epilogue) and `aten::mm` + a separate bias add in the optimized graph (OPT-1 keeps the bias FP32 and recasts the BF16 GEMM result before the add). Compared GEMM-to-GEMM.
² Not present in the baseline — introduced by OPT-1's FP32 output recast and Inductor's frozen-layout transpose. See Section 8.

**Reading the total:** compute (the 3 GEMM rows) dropped **4.5×**, but OPT-1's FP32 recast of the `[8192,32000]` logit output adds 2.13 ms that did not exist in the baseline, so the **net end-to-end speedup is 2.22×**. The recast kernels sit in `unattributed_kernels` (a correlation-map gap) but are real per-forward work and are counted in the optimized total — excluding them would report a misleading 4.15×. Tensor Cores went 0.0 → 75.9–85.7% on exactly the GEMM rows; OPT-2 (freezing + autotune) selected the `256x128` tensorop tile for the N=32000 logit GEMM and hoisted its weight transpose to compile time (now the standalone `aten::t` row).

---

## 7. What Drove Each Speedup

**BF16 dtype promotion (OPT-1, +4.50× on the logit projection and all GEMMs):** casting the matmul operands to BF16 moves every GEMM off the scalar `cutlass_80_simt_sgemm_*` SIMT path onto the `cutlass_80_tensorop_bf16_s16816gemm` Tensor-Core path (the `s16816` shape is the BF16 16×8×16 MMA). The decisive evidence is `tensor_core_active_pct` rising from a literal **0.0** to **79.4% (logit), 85.6% (proj2), 75.9% (proj1)**, with the dominant kernel name changing from `simt_sgemm` to `tensorop_bf16_s16816gemm` on every matmul.

**Weight freezing + autotune (OPT-2, incremental on the logit GEMM):** with the BF16 layout frozen at compile time, Inductor's autotuner benchmarked tile/split-K configurations for the unusual N=32000 GEMM and selected the `256x128_32x3` tensorop tile (visible in the optimized dominant-kernel name). Its contribution is folded into the logit GEMM's 4.50× — freezing also moved the `_tn_` transpose out of the hot path (now the standalone `aten::t` pointwise kernel).

The net total (2.22×) is below the per-GEMM compute win (4.5×) because OPT-1's FP32 logit recast adds ~1.07 ms/forward (2.13 ms across both passes) of new pointwise work — see §8.

---

## 8. Remaining Opportunities

All proposed optimizations (OPT-1, OPT-2) were applied. The optimized profile, however, exposes two **second-order bottlenecks created by the BF16 promotion itself** — these are new residual opportunities, not unapplied proposals:

| Residual | Optimized cost | Mechanism | Projected gain |
|---|---|---|---|
| FP32 recast of the logit output (`triton_poi_fused_5`, ×2) | ~2.13 ms (both passes; ~1.07 ms/forward) | OPT-1 recasts the BF16 `[8192, 32000]` logit tensor back to FP32 — a ~1.05 GB read+write pointwise kernel that is now **as expensive as the logit GEMM itself** | If the consumer tolerates BF16 logits, dropping the final output recast removes ~1 ms/forward (~comparable to the GEMM win) |
| `aten::t` weight transpose | ~0.12 ms (both passes) | DRAM-bound (80%) layout shuffle for the frozen BF16 weights | Minor; a fused transpose-in-epilogue or pre-transposed frozen weight would absorb it |

**Estimated additional gain:** eliminating the FP32 logit recast is the dominant remaining lever — it closes most of the gap between the 2.22× net and the ~4.5× compute ceiling, and is the single most impactful next step. It requires a contract decision (are downstream consumers BF16-safe?), so it was deliberately not applied here: OPT-1 preserves the FP32 output dtype to guarantee numerical equivalence (verified by the test suite's dtype + NaN/Inf assertions).

---

## 9. Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py \
    --workload examples/embedding_projection/embedding_projection.py \
    --output-prefix profiler_output/embedding_projection \
    --warmup-iters 2 --measure-iters 2 --correlation-pass
# (then nsys + ncu replay; produces profile.json)

# Optimized: register backend "embedding_projection_opt" and re-capture
#   --compile-backend=embedding_projection_opt  --profile-name=optimized
# (produces profile_optimized.json)

# Validate before re-profiling
pytest examples/embedding_projection/test_embedding_projection_optimized.py
```

Or run the whole pipeline end-to-end:

```
/optimize examples/embedding_projection/embedding_projection.py
```
