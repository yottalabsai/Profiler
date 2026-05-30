# MLPActivations — GPU Optimization Report

**This optimization achieved ≈8.7× total speedup on MLPActivations (B=256, NVIDIA RTX PRO 6000 Blackwell Server Edition)** by moving every Linear-layer GEMM off the FP32 CUDA-core (SIMT) path and onto the bf16 Tensor-Core path, then folding each activation into the GEMM epilogue.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture family | Blackwell |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Baseline compile mode | `inductor` (built-in dedup + Inductor backend) |
| Optimized compile mode | `mlp_activations_opt` (custom `torch.compile` backend) |
| Batch size | 256 |
| Input | `(256, 512)` fp32 |
| Iteration count | warmup 2 / measure 2 *(ncu replay — relative timing only)* |
| Model | 4× `nn.Linear` (512→2048→2048→2048→512) + ReLU / GELU / SiLU / Tanh |

> **Timing caveat.** All durations below are ncu application-replay values, which run 2–5× longer than real wall-clock execution. They are valid for **relative** before/after comparison (both profiles captured the same way, same GPU, ~13 min apart in one session), not as absolute latencies.

---

## 2. Operator Summary (Baseline)

Sorted by GPU time. `layer::unique::prologue` is the NVTX range enclosing the entire fused forward partition (all 24 kernels); the individual `aten::mm` rows are kernels *inside* that range — they are not added to it (doing so would double-count).

| Operator | Time (%) | Duration (ns) | Kernels | Tensor-Core % | Bottleneck Class |
|---|---|---|---|---|---|
| `layer::unique::prologue` (full forward) | 100.0 | 1,348,613 | 24 | 0.0 | Compute-bound, SIMT (no Tensor Cores) |
| `aten::mm` [256,2048]×[2048,512] (fc4) | — | 137,664 | 1 | 0.0 | Wave-starved skinny GEMM (occ 8.3%) |
| `aten::mm` [256,2048]×[2048,512] | — | 135,585 | 1 | 0.0 | Wave-starved skinny GEMM (occ 8.4%) |
| `aten::mm` [256,2048]×[2048,2048] (hidden) | — | 127,457 | 1 | 0.0 | Compute-bound SIMT (sm 23%, occ 17%) |
| `aten::mm` [256,2048]×[2048,2048] | — | 126,048 | 1 | 0.0 | Compute-bound SIMT |
| `aten::mm` [256,2048]×[2048,2048] | — | 125,632 | 1 | 0.0 | Compute-bound SIMT |
| `aten::mm` [256,2048]×[2048,2048] | — | 124,417 | 1 | 0.0 | Compute-bound SIMT |
| `aten::mm` [256,512]×[512,2048] (fc1) | — | 51,520 | 1 | 0.0 | Compute-bound SIMT |
| `aten::mm` [256,512]×[512,2048] | — | 51,232 | 1 | 0.0 | Compute-bound SIMT |
| `aten::addmm` (bias folds) | — | 14,912 | 8 | 0.0 | Memory-bound (mem 24%) |

**De-duplicated GEMM budget** (8 distinct `aten::mm` + `aten::addmm`, prologue excluded): **894,467 ns**. The remaining ~454 k ns inside the prologue is activation, transpose, and dtype-cast pointwise work.

The single dominant inefficiency: **every GEMM dispatched to `cutlass_80_simt_sgemm_*`** with `smsp__pipe_tensor_cycles_active = 0` — the Tensor Cores sat completely idle on a Tensor-Core-class GPU.

---

## 3. Reading the Metrics

Only metrics that drive this workload's bottleneck are explained.

- **`smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`** (reported as "Tensor-Core %"). **0.0 (not null)** means the GEMM ran on the FP32 SIMT datapath with Tensor Cores entirely idle — the highest-ROI signal in a profile. Every baseline GEMM reads 0.0 here. After optimization the fused GEMM templates read 30–58%, confirming the HMMA path is active. (A *null* value, by contrast, is normal for genuinely elementwise kernels and is not a problem.)
- **`sm__throughput.avg.pct_of_peak_sustained_elapsed`** ("sm %"). Compute-pipe utilization. Baseline hidden GEMMs sit at ~23%; the skinny 2048→512 GEMMs at only ~6%, indicating they cannot fill the machine.
- **`sm__warps_active.avg.pct_of_peak_sustained_active`** ("occupancy"). The skinny GEMMs achieve only ~8% occupancy — too few thread blocks (176) to hide latency, the classic wave-starved small-GEMM symptom that motivates per-shape autotuning (OPT-2).
- **`gpu__dram_throughput`** ("mem %"). Memory-pipe utilization. Low (<10%) on the GEMMs confirms they are compute/SIMT-bound, not bandwidth-bound — so the fix is the math datapath, not data movement.

---

## 4. Optimizations Applied

Statuses from `profiler_output/validation_report.json` (all four validation steps passed: syntax, import, registration, pytest 4/4).

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype promotion fp32 → bf16 (Tensor-Core path) | all `aten.mm` / `aten.addmm` | `tensor_core_active_pct = 0.0` on every GEMM; `cutlass_80_simt_sgemm` kernels | HIGH | **APPLIED** |
| OPT-2 | max-autotune GEMM templates | skinny 2048→512 GEMMs (op 9/19) | occ 8.3%, sm 6.3%, 176 blocks — wave-starved | MEDIUM | **APPLIED** |
| OPT-3 | bias + activation epilogue fusion | `addmm` + ReLU/GELU/SiLU/Tanh | 4 standalone activation launches + 1 DRAM round-trip per pass | MEDIUM | **APPLIED** |

---

## 5. Implementation Notes

# MLPActivations — Optimized Backend Implementation Notes

Backend name: `mlp_activations_opt` (registered via `@register_backend`).
Workload: four `nn.Linear` layers with ReLU / GELU / SiLU / Tanh activations.
Prerequisite DAG (strict order): OPT-1 → OPT-2 → OPT-3.

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 — dtype promotion to bf16 / Tensor-Core path | `_aten_inner_compile` (`_pass_gemm_bf16_casts`) | Casts both operands (and addmm bias) of every `aten.mm`/`aten.addmm` to bf16 and back-casts the result to fp32; only OPT-1 is a true node-level graph transform, and it is the dtype root the other two depend on. |
| OPT-2 — max-autotune GEMM templates | backend (`_apply_inductor_config`, Inductor config) | Autotuning is a compile-config toggle (`max_autotune`, `max_autotune_gemm`, `coordinate_descent_tuning`, broad `max_autotune_gemm_backends`), not node surgery — it changes which kernel the GEMM lowering selects for the wave-starved skinny/small GEMMs. |
| OPT-3 — bias+activation epilogue fusion | backend (`_apply_inductor_config`, Inductor config) | Epilogue fusion is enabled implicitly by a Triton GEMM template; we set `epilogue_fusion=True` and pin `max_autotune_gemm_backends="TRITON"` (extern cutlass cannot host a fused Triton epilogue). No `addmm`+activation node rewrite is needed — Inductor's pointwise scheduler folds the activation into the template. |

## Key Design Decisions

**Why OPT-2 and OPT-3 are config passes, not FX node passes.** The proposal's own `fx_steps[]` for both express them as `torch._inductor.config` assignments (`max_autotune`, `epilogue_fusion`, `max_autotune_gemm_backends`) plus a recompile, explicitly stating "no manual node surgery." Max-autotune changes *kernel selection* during GEMM lowering and epilogue fusion is performed by Inductor's pointwise scheduler once the GEMM is a Triton template node; neither is observable as an Aten-IR node edit. They are applied in `_apply_inductor_config()`, called from the backend before any `compile_fx` so the config is live for every lowering. Each `setattr` is individually `hasattr`-guarded and try-wrapped so a missing attribute on the local torch build degrades to a no-op (WARNING) rather than breaking compilation.

**Prerequisite ordering inside one process.** OPT-1 must precede OPT-2/OPT-3 because the eligible autotune template set (Tensor-Core vs SIMT) and the fusible Triton epilogue both require the bf16 operands OPT-1 inserts. OPT-1 runs inside `_aten_inner_compile`; the OPT-2/OPT-3 config is set in the backend before the `compile_fx` call that ultimately invokes `_aten_inner_compile`, so the bf16 casts are present in the graph the autotuner benchmarks. OPT-3 sets `max_autotune_gemm_backends="TRITON"` *after* OPT-2's broader `"TRITON,CUTLASS,ATEN"`, deliberately overriding it so the final backend list is Triton-only (the precondition for the fused epilogue).

**Why `prims.convert_element_type` instead of the proposal's `aten.to.dtype`.** On torch 2.11 `aten._to_copy.default` (what `aten.to.dtype` decomposes to) carries both a fallback and a decomp registration; inserting it post-AOTAutograd makes Inductor raise "both a fallback and a decomp for same op." `prims.convert_element_type.default` is the form Inductor itself emits for dtype conversions, lowers cleanly to a Triton cast, and lets OPT-1's casts fuse into neighbouring kernels (and into the OPT-3 epilogue).

**No dtype guard in OPT-1.** aten rejects a GEMM whose two operands disagree in dtype. Rather than guarding on fp32, the pass forces *every* tensor operand (including the addmm bias) to bf16, skipping the cast only when an operand is provably already bf16 (`meta['val'].dtype`). This is correct regardless of the operands' incoming dtype and keeps the cutlass/Triton epilogue homogeneous.

**Flat compile path.** MLPActivations' four layers have distinct shapes and distinct activations, so `UniqueSubgraphRegistry.build_partition_equivalence_map()` returns empty and the backend takes the flat path — compiling the whole graph through `compile_fx(..., inner_compile=_aten_inner_compile)`. This preserves cross-op Inductor fusion of the bf16 casts and activation epilogues into neighbouring kernels. The dedup branch is retained for protocol compliance and falls back to the flat path on any per-partition compile failure.

**`inner_compile` seam over `aot_autograd(fw_compiler=...)`.** Installing the Aten-IR pass via `compile_fx`'s `inner_compile` hook lets `compile_fx` own AOTAutograd, the decomp table, the boxed calling convention, and the partitioner. The `aot_autograd(fw_compiler=compile_fx)` alternative raises `AssertionError: Expected tensors only, but got list` in `copy_misaligned_inputs` on torch 2.11.

---

## 6. Before/After Results

Both profiles share batch size 256, the same GPU, and were captured ~13 minutes apart in one session — no cross-session caveat applies.

The optimization restructured the graph: the FP32 SIMT GEMMs and their separate activation kernels were fused into bf16 Tensor-Core Triton templates (`triton_tem_fused_addmm_{relu,gelu,silu,tanh}`), which the attribution engine labels by the activation operator they end in. Because the operator decomposition changed completely, the only honest comparison is **full-forward to full-forward**.

| Measure | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| Full forward (all kernels) | 1,348,613 | 154,465 | **8.73×** |
| — of which: de-dup GEMM work (SIMT → Tensor-Core fused templates) | 894,467 | 147,457¹ | 6.07× |
| — leftover pointwise (transpose / cast) | ~454,146 | 7,008² | — |

¹ The four fused `triton_tem_fused_addmm_*` template operators (`aten::gelu` 49,344 + `aten::t` 39,648 + `aten::relu` 39,264 + `aten::silu` 19,201). These are the GEMM-with-epilogue kernels now carrying the matmul, bias, and activation in one launch.
² 10 small unattributed `triton_poi_fused_*` pointwise kernels.

**Step B — Speedup attribution** (all three passes `APPLIED`):

- The 8.73× full-forward speedup is attributed primarily to **OPT-1**: tensor-core activity went from `0.0` on every baseline GEMM to **57.5% (GELU template), 37.7% (SiLU), 29.7% (ReLU)** — the bf16 HMMA path is now driving the GEMMs, the expected direction, and the operators carrying those kernels all shrank.
- **OPT-3** is corroborated by kernel-count collapse: baseline had separate GEMM + standalone activation + `addmm` bias kernels; optimized shows single `triton_tem_fused_addmm_{act}` kernels (matmul+bias+activation in one launch), eliminating per-pass activation DRAM round-trips.
- **OPT-2** is credited with the disproportionate shrink of the skinny 2048→512 GEMMs (baseline ~137 k ns each at 8% occupancy) now folded into the same fused templates; it is not separately isolable from OPT-1/OPT-3 in this profile, so its individual contribution is reported as bundled.

**Step C — Residual bottlenecks (re-ranked optimized profile):**

| Operator | Optimized (ns) | New bottleneck |
|---|---|---|
| `aten::gelu` template | 49,344 | Tensor-Core 57.5%, sm 18% — now the largest single op; headroom to ~100% TC |
| `aten::t` (transpose) | 39,648 | Memory/layout-bound (TC 17.7%, sm 4.7%) — pure data movement, no math |
| `aten::relu` template | 39,264 | Balanced (TC 29.7%, mem 22%, sm 33%) |
| `aten::silu` template | 19,201 | TC 37.7%, occupancy only 8.5% — still wave-starved |

The largest residual is the standalone **transpose work (`aten::t`, ~40 k ns)** — layout conversions not yet folded into the GEMM operands.

---

## 7. What Drove Each Speedup

**dtype promotion fp32 → bf16 (OPT-1, primary driver of the 8.7× total):** Casting every `aten.mm`/`aten.addmm` operand to bf16 switches cutlass/Triton kernel selection from the `cutlass_80_simt_sgemm` CUDA-core path to the HMMA Tensor-Core path. Evidence: `smsp__pipe_tensor_cycles_active` rose from a flat `0.0` across all baseline GEMMs to 30–58% across the fused templates, and Inductor autotune logs confirm the matmuls ran with `torch.bfloat16` inputs.

**bias + activation epilogue fusion (OPT-3, +launch reduction):** Once each GEMM is a Triton template, `epilogue_fusion=True` folds the bias add and the elementwise activation into the matmul kernel. Evidence: the baseline's separate `aten::addmm` (8 kernels) and standalone activation kernels disappeared, replaced by single `triton_tem_fused_addmm_{relu,gelu,silu,tanh}` kernels — removing four kernel launches and one full activation DRAM round-trip per forward pass.

**max-autotune GEMM templates (OPT-2, occupancy fix for skinny GEMMs):** Per-shape tile benchmarking replaces the one default cutlass tile that left the 2048→512 GEMMs at 8% occupancy / 176 blocks. Evidence: those two ~137 k ns SIMT GEMMs no longer appear as standalone wave-starved kernels — they were absorbed into the autotuned bf16 fused templates.

---

## 8. Remaining Opportunities

All three proposed optimizations were applied and validated. The following are *new* second-order opportunities exposed by the optimized profile (not in the original `optimizations.json`, listed as future work — not yet implemented):

| ID | Type | Target | Reason | Projected Gain |
|---|---|---|---|---|
| FUT-1 | transpose elision / operand pre-layout | `aten::t` (~40 k ns, 26% of optimized forward) | Standalone layout conversion not folded into GEMM operands; could be absorbed by storing weights pre-transposed or fusing the transpose into the template's load | not yet implemented |
| FUT-2 | larger autotune tiles / occupancy tuning for SiLU template | `aten::silu` template (occ 8.5%) | Still wave-starved despite Tensor-Core path | not yet implemented |
| FUT-3 | push Tensor-Core utilization toward peak | `aten::gelu` template (TC 57.5%) | Largest single op; ~40% TC headroom remains | not yet implemented |

If the transpose work (FUT-1) were fully elided, the optimized forward could drop from ~154 k ns toward ~115 k ns (a further ~1.3×), making the realistic combined ceiling on this workload ≈11× over baseline. These are estimates, not measured.

---

## 9. Reproduction

```bash
# Re-run the whole pipeline end-to-end:
/optimize examples/mlp_activations/mlp_activations.py

# Baseline capture only (built-in inductor dedup backend) → profile.json
# Optimized capture only:
#   run_workload.py --compile-backend mlp_activations_opt
#   with matched warmup/measure iters (2/2) → profile_optimized.json
```

Artifacts: `profile.json`, `optimizations.json`, `mlp_activations_optimized.py`, `test_mlp_activations_optimized.py`, `profile_optimized.json`, `profiler_output/{validation_report.json,implementation_notes.md}`.
