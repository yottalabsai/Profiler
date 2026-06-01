# GPT-2 Optimization Report

**This optimization achieved a 3.6× total GPU-time speedup on GPT-2 small (B=4, seq=128, NVIDIA RTX PRO 6000 Blackwell Server Edition)** by routing all transformer-block GEMMs off the FP32 SIMT path onto Blackwell Tensor Cores and eliminating an Ampere-ISA attention kernel that was running under backward compatibility on a Blackwell device.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture family | Blackwell (Sm100) |
| PyTorch version | 2.11.0+cu128 |
| Compile mode (baseline) | `inductor` (built-in layer-dedup backend) |
| Compile mode (optimized) | `gpt2_opt` (custom `@register_backend`) |
| Model | GPT-2 small (117M), 12 identical transformer blocks |
| Batch size | 4 |
| Sequence length | 128 |
| Iteration count | warmup=3, measure=10 (ncu replay — **relative timing only**) |

> All `duration_ns` values come from ncu counter-collection (application-mode replay), which inflates absolute kernel durations roughly 2–5× over true wall-clock. **Every number below is a relative comparison within or between these two profiles, not an absolute wall-clock prediction.**

---

## 2. Operator Summary (Baseline)

Built from the granular per-block aten operators aggregated by class (representative per-call duration × 12 blocks), per the strategist's `fused_kernel_double_count` handling — the `layer::unique::modules` dedup roll-up is **not** summed here to avoid double counting.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::mm` (FFN down-proj, [512×3072]×[3072×768]) | 30.7% | 1,050,000 | 12 | Compute / **Tensor Cores idle** (SIMT FP32, occ 8.5%) |
| `aten::mm` (FFN up-proj, [512×768]×[768×3072]) | 29.4% | 1,005,600 | 12 | Compute / **Tensor Cores idle** (SIMT FP32, occ 16.6%) |
| `aten::addmm` (fused QKV proj, [512×768]×[768×2304]) | 20.7% | 706,800 | 12 | Compute / **Tensor Cores idle** (SIMT FP32, occ 20.1%) |
| `aten::mm` (attn output-proj, [512×768]×[768×768]) | 11.4% | 388,800 | 12 | Compute / **Tensor Cores idle** (SIMT FP32) |
| `aten::_efficient_attention_forward` (SDPA) | 7.0% | 240,000 | 12 | **Sm80 ISA on Sm100** (FP32 xFormers fallback) |

GEMMs total ~92% of attributed time and **all** run on the CUTLASS SIMT FP32 path (`cutlass_80_simt_sgemm_*`) with Tensor Cores completely idle. LayerNorm and tanh-GELU Triton kernels are DRAM-bound, already Inductor-fused, and individually below the 1% actionable threshold — omitted from the table.

---

## 3. Reading the Metrics

Only the metrics that drive the bottlenecks in this workload:

- **`tensor_core_active_pct = 0.0` (not null)** — the single highest-ROI signal here. A value of exactly `0.0` means the GEMM ran on the FP32 SIMT (scalar multiply-accumulate) pipeline with the Tensor Core MMA units completely idle. Every one of the 840 baseline GEMM kernels reports `0.0`. A *null* value, by contrast, is expected for non-GEMM kernels and is **not** a problem.
- **`achieved_occupancy`** — fraction of resident-warp capacity used. The FFN-down GEMM sits at **8.5%** (FP32 SIMT burns 128–210 registers/thread, starving occupancy). Below ~30% on a compute kernel signals the scheduler can't hide latency; a lower-register Tensor Core kernel lifts this directly.
- **`sm_throughput_pct`** — SM pipe utilization. FFN-down is at **19.95%**: the SM is mostly stalled, confirming the kernel is bound by pipeline selection (SIMT) rather than useful arithmetic.
- **`dram_throughput_pct` (5.5–9%) + `l2_hit_rate` (~89%)** — weights are L2-resident and memory is nearly idle, so this is **compute/pipeline-bound, not memory-bound**. Halving operand bytes (BF16) is a side benefit; the real lever is the arithmetic pipeline.
- **Kernel-name ISA token** — `fmha_cutlassF_f32_aligned_64x64_rf_sm80` carries `f32` (FP32, no Tensor Cores inside attention) and `sm80` (Ampere ISA running on Sm100 via backward compatibility). ncu returns an **empty metrics object** for these launches — the diagnostic fingerprint of an Sm80→Sm100 SASS mismatch.

---

## 4. Optimizations Applied

Statuses from `profiler_output/validation_report.json`.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | `dtype_promotion` (aten) | all `aten.mm` / `aten.addmm` (48 block GEMMs) | `tensor_core_active_pct=0.0`, occ 8–20%, SIMT `cutlass_80_simt_sgemm` | high | **APPLIED** |
| OPT-2 | `op_substitution` (functional) | `F.scaled_dot_product_attention` (12 blocks) | empty ncu metrics, `sm80` kernel name on Sm100 | high | **APPLIED** |
| OPT-3 | `inductor_config` | 48 frozen GEMM weights + embeddings | FFN-down occ 8.5%, sm_throughput 19.95% | medium | **APPLIED** |
| OPT-4 | `common_subexpression_hoist` (aten) | per-block causal-mask reconstruction | l2_hit 5%, sm 1%, 11 redundant launches | low | **NOT_APPLIED** (stub; subsumed by OPT-2 `is_causal=True`) |

---

## 5. Implementation Notes

# gpt2_opt — Implementation Notes

Backend registered name: **`gpt2_opt`** (via `@register_backend` in
`examples/gpt2/gpt2_optimized.py`, fired at module import).

Target device: NVIDIA RTX PRO 6000 Blackwell Server Edition (Sm100). `compile_mode = "inductor"`.
Workload: GPT-2 small (117M), 12 identical transformer blocks, batch=4, seq=128, FP32.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-2 — Flash SDPA backend selection | functional | `_run_functional_passes` (`_fpass_enable_flash_sdpa`) | Set `enable_flash_sdp(True)` / `enable_mem_efficient_sdp(False)` before `compile_fx` so Dynamo traces `F.scaled_dot_product_attention` to `aten._scaled_dot_product_flash_attention` (native Sm100) instead of the Sm80 xFormers `_efficient_attention`. Flag is read at Dynamo trace time, so it must run at the functional level. GPT-2 is causal → `is_causal=True` natively, dropping the explicit mask. |
| OPT-1 — BF16 dtype promotion | aten | `_aten_inner_compile` (`_apass_bf16_promotion`) | Cast operands of every `aten.mm.default` and `aten.addmm.default` to BF16 and the result back to FP32. Routes all 48 block GEMMs off the SIMT FP32 `cutlass_80_simt_sgemm` path (`tensor_core_active_pct=0.0`) onto the Blackwell BF16 Tensor Core path. Runs at aten because the GEMM ops only exist post-AOTAutograd decomposition. The `addmm` branch covers GPT-2's already-fused QKV projection. |
| OPT-4 — Causal-mask CSE/hoist | aten | `_aten_inner_compile` (`_apass_mask_hoist_stub`) — **stub, not applied** | Low confidence, ~0.5% of attributed time, and fully subsumed by OPT-2 (Flash with `is_causal=True` never materializes the additive `[4,12,128,128]` mask). Detect-only: logs whether any residual `_efficient_attention` node with an explicit `attn_bias` survives; no graph mutation. |
| OPT-3 — Weight freezing + max_autotune | inductor_config | `config_patches` on `compile_fx` (`_cfg_freezing`) | Scoped `{"freezing": True, "max_autotune": True, "max_autotune_gemm_backends": "ATEN,TRITON"}`. Inductor constant-folds and pre-packs the 48 frozen (eval, `requires_grad=False`) GEMM weights into the BF16 Tensor Core layout and autotunes tiling. Inductor owns weight layout, so this is a config dict, not a graph pass. |
| QKV projection fusion | — (N/A) | not implemented | GPT-2's QKV is already a single fused `[768->2304]` `addmm` (HuggingFace Conv1D packs Q/K/V into one weight). No fusion pass is applicable; OPT-1's `addmm` branch promotes that wide GEMM directly. |

## Key Design Decisions

**Why OPT-1 handles both `aten.mm` and `aten.addmm`.** Unlike the `sdpa_attention` example (all bias-free `mm`), GPT-2's projections come from HuggingFace `Conv1D`, which decomposes to `addmm(bias, x, weight)`. The fused-QKV projection (20.7% of time) and the attention output / FFN projections appear as `addmm` when a bias is present and `mm` otherwise. The pass promotes `addmm` operands `args[1]`/`args[2]` plus the bias `args[0]`, leaving the bias-add fused into the Tensor Core GEMM.

**Why the funnel ordering matters here.** OPT-2 (functional) must select the Flash op before AOTAutograd traces SDPA; OPT-1 (aten) then casts q/k/v to BF16 inside `inner_compile`, so the BF16 operands are in place when the Flash kernel dispatches. This is the prerequisite chain `OPT-1 -> OPT-3` and `OPT-2 -> OPT-1` from the proposal, satisfied automatically by the cross-level funnel order (functional → aten → inductor_config) with no within-level sequencing.

**Why `prims.convert_element_type` instead of `aten._to_copy`.** On torch 2.11, `aten._to_copy` carries both a fallback and a decomp registration; inserting it into an already-decomposed Aten graph triggers Inductor's "both a fallback and a decomp for same op" assertion. `prims.convert_element_type` lowers cleanly to a Triton elementwise cast and is CSE-folded by Inductor for shared weight casts.

**Why per-rep compilation (dedup) instead of `replace_pattern`.** GPT-2 has 12 structurally identical blocks. `UniqueSubgraphRegistry` splits the graph, compiles one representative per equivalence class through the full funnel, and shares the compiled callable with the duplicates — the same mechanism the capture pipeline uses for ~12× ncu/compile reuse. Functional passes run per-rep inside `_compile_unit`, never on the pre-split graph. If no repeats are detected the flat compile path is used, preserving cross-layer Inductor fusion.

**Why OPT-4 is a stub.** It is sub-threshold on its own and made redundant by OPT-2: once Flash with `is_causal=True` is selected there is no explicit additive mask subgraph to hoist. The stub remains as a detector — if it logs residual `attn_bias` producers, that signals OPT-2 did not take and the mem-efficient backend was retained.

**Non-graph state.** `get_model_and_input()` keeps the original `gpt2.py` contract (FP32 model, int64 `(4,128)` input_ids, `.eval()`). BF16 promotion is selective and in-graph (GEMM operands only) rather than `model.bfloat16()`, so LayerNorm and softmax stay FP32 for numerical stability per the proposal. `.eval()` is required for OPT-3 freezing.

---

## 6. Before/After Results

Both profiles were captured on the **same GPU** (RTX PRO 6000 Blackwell), 64 minutes apart (20:31 → 21:35 UTC) — within the same session window, so no cross-session clock caveat applies.

**Attribution-tier note (why this is an aggregate comparison, not a per-`operator_name` table).** The baseline used the built-in dedup backend, so Dynamo correlation succeeded and GEMMs are attributed per-`aten::mm`/`aten::addmm`. The optimized workload uses the custom `gpt2_opt` backend, which calls the precompiled callable directly and bypasses Dynamo's `aten::` RecordFunction scopes — so its `.corr.json` is empty and kernels fall to the Inductor-fusion tier, where the GEMM work appears as fused `triton_tem_fused_*` templates grouped under generic operator names. Per-`operator_name` matching is therefore unreliable; the comparison below is at the **work-class level**, which is robust to the attribution-tier change.

| Work class | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| Dense GEMM (FFN + QKV + attn-out projections) | 34,962,000 | 8,218,000 | **4.25×** |
| Attention (SDPA → fused softmax+bmm) | 3,124,000 | 821,000 | **3.80×** |
| **Total measured-region GPU time (10 iters)** | **49,566,000** | **13,828,000** | **3.58×** |

- **GEMM:** 840 baseline `cutlass_80_simt_sgemm` launches (`tensor_core_active_pct=0.0` on every one) collapse to 720 `triton_tem_fused_*` Tensor Core GEMM templates. The optimized GEMM group reports `tensor_core_active_pct` up to **~31%** (peak 54.5%), confirming the MMA units are now engaged.
- **Attention:** the Sm80 xFormers `fmha_cutlassF_f32_aligned_64x64_rf_sm80` kernel (156 launches, FP32, empty ncu metrics) is **completely eliminated** — zero `fmha`/`sm80` kernels remain. Attention is now realized as Inductor-fused Triton softmax + Tensor-Core `bmm` templates.

**Speedup attribution** (all three conditions — APPLIED status, expected metric change, operator improvement — verified):

- The GEMM speedup is attributed to **OPT-1** (`APPLIED`): `tensor_core_active_pct` moved `0.0 → ~31%` and the dense-GEMM class shrank 4.25×. **OPT-3** (`APPLIED`, freezing + max_autotune) contributed by selecting the `triton_tem` autotuned templates that now carry the work; its share is not separable from OPT-1's within this profile.
- The attention speedup is attributed to **OPT-2** (`APPLIED`): the `sm80` FP32 kernel disappeared (expected direction) and the attention class shrank 3.8×.
- **OPT-4** (`NOT_APPLIED`) contributed nothing — consistent with its stub status; the causal mask was never materialized because OPT-2's `is_causal=True` removed it.

> Caveat: the optimized profile's unattributed bucket (5.14 ms) is larger than the baseline's (0.48 ms) because the custom backend produced an empty correlation map; part of that bucket is pre-NVTX warm-up init. The total-row figures are the measured-region kernel sums each capture-agent reported, and the 3.58× headline is robust to this — the GEMM-class comparison (4.25×), which is unaffected by the unattributed bucket, independently corroborates it.

---

## 7. What Drove Each Speedup

**BF16 Tensor Core GEMM promotion (OPT-1, +4.25× on the dense-GEMM class):** Casting both operands of every `aten.mm`/`aten.addmm` to BF16 routes cuBLAS/Inductor off the scalar SIMT FP32 pipeline onto the Blackwell Tensor Core MMA path. The decisive evidence is `tensor_core_active_pct` moving from exactly `0.0` on all 840 baseline `cutlass_80_simt_sgemm` launches to ~31% (peak 54.5%) on the 720 `triton_tem_fused_*` templates that replaced them, with the dense-GEMM class dropping from 34.96 ms to 8.22 ms.

**Flash SDPA backend selection (OPT-2, +3.80× on the attention class):** Setting `enable_flash_sdp(True)` / `enable_mem_efficient_sdp(False)` before compilation steered SDPA away from the Sm80 xFormers `_efficient_attention` fallback; Inductor then lowered attention into fused Triton softmax + Tensor-Core `bmm` kernels. The evidence is the disappearance of the `fmha_cutlassF_f32_aligned_64x64_rf_sm80` kernel entirely (156 launches → 0), resolving the Sm80-on-Sm100 ISA mismatch that had produced empty ncu metrics.

**Weight freezing + max_autotune (OPT-3, contributes within the GEMM speedup):** Marking the 48 eval-mode weights as constants let Inductor constant-fold, pre-pack them into the Tensor Core layout, and autotune GEMM tiling — which is what materialized the `triton_tem` autotuned templates now carrying the GEMM work. Its contribution is real but not separable from OPT-1's in this profile, since both act on the same kernels.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-4 | `common_subexpression_hoist` | per-block causal-mask reconstruction | STUB no-op; no explicit `attn_bias` mask producers found — subsumed by OPT-2 `is_causal=True` | ~0.5% (16,000 ns) |

OPT-4 is the only unapplied proposal, and it was intentionally left as a detect-only stub because OPT-2 already eliminates the additive causal mask it would hoist. Applying it would yield no measurable gain on this profile.

**Second-order bottleneck (exposed post-optimization):** with GEMMs now ~4× faster, `aten::native_layer_norm` rises in relative cost (≈1.26 ms across the measured region, ~8.3% occupancy, DRAM-bound) and the fused-softmax reduction (`triton_red_fused__safe_softmax_*`, ~0.56 ms) become the next-largest residuals. Both are memory/launch-bound Triton kernels with no GEMM/attention FX transform available; a future pass would target LayerNorm fusion or a larger batch to amortize launch overhead. No further FX-level GEMM/attention gains are identified in this profile.

---

## Reproduction

```bash
# Baseline capture (built-in dedup backend) → profile.json
#   /capture examples/gpt2/gpt2.py

# Optimized capture (custom backend) → profile_optimized.json
#   /capture examples/gpt2/gpt2_optimized.py --profile-name=optimized --compile-backend=gpt2_opt

# Validate the backend before re-profiling
#   /validate examples/gpt2/gpt2_optimized.py

# Full pipeline end-to-end
#   /optimize examples/gpt2/gpt2.py
```
