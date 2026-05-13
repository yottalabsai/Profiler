# GPT-2 Optimization Comparison

## Summary

| | Baseline | Optimized |
|---|---|---|
| Device | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB |
| Batch / Seq | B=4, seq_len=128 | B=4, seq_len=128 |
| dtype | FP32 | BF16 |
| Backend | inductor + layer_dedup | gpt2_opt (BF16 + max-autotune) |
| Total attributed time | 103,930,896 ns | ~34–42 M ns (est.) |
| **Overall speedup** | 1.0x | **~2.5–3.0x** |

---

## Operator-Level Comparison

| Operator | Baseline kernel | Baseline (ns) | Optimized kernel | Optimized (ns) | Speedup | Bottleneck Resolved? |
|---|---|---|---|---|---|---|
| FFN up-proj [512×768]×[768×3072] mm | `ampere_sgemm_32x128_nn` | 197,952 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_nn` | 21,120 | **9.4x** | YES — HMMA Tensor Cores active |
| Attn output [512×768]×[768×768] mm | `ampere_sgemm_64x32_sliced1x4_nn` | 54,848 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_nn` | 12,769 | **4.3x** | YES — HMMA Tensor Cores active |
| `aten::_efficient_attention_forward` | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 30,720 /kernel | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | ~17,200 /kernel | **1.79x** | PARTIAL — wave_starvation remains |
| `aten::mm_0` (aggregate) | `ampere_sgemm_*` | 53,966,937 | `ampere_bf16_s16816gemm_*` | — | ~6–7x (est.) | YES |
| `aten::addmm_0` (aggregate) | `ampere_sgemm_128x32_nn` | 16,301,364 | `ampere_bf16_s16816gemm_*` | — | ~6–7x (est.) | YES |

---

## Hardware Counter Evidence

### GEMM operators (mm / addmm)
| Counter | Baseline | Optimized |
|---|---|---|
| `tensor_core_active_pct` | **0.0%** | **55.9%** |
| Dominant kernel family | `ampere_sgemm_*` (FP32 SIMT) | `ampere_bf16_s16816gemm_*` (HMMA BF16) |
| FFN up-proj per-kernel | 197,952 ns | 21,120 ns |
| Attn output per-kernel | 54,848 ns | 12,769 ns |

### Flash attention
| Counter | Baseline | Optimized |
|---|---|---|
| Kernel | `fmha_cutlassF_f32_*` | `fmha_cutlassF_bf16_*` |
| `registers_per_thread` | **168** | **128** |
| `local_memory_spills` | **24,468,480 bytes** | **0 bytes** |
| `local_memory_wavefronts` | 226,560 /kernel | 0 /kernel |
| `achieved_occupancy` | 6.30% | 6.34% |
| Per-kernel duration | 30,720 ns | ~17,200 ns |

---

## Pass Attribution

| Pass | Status | Evidence |
|---|---|---|
| OPT-1: BF16 dtype promotion | **APPLIED** | All GEMM kernels switched from `ampere_sgemm_*` (FP32 SIMT) to `ampere_bf16_s16816gemm_*` (HMMA); attention register spills eliminated (24.5 MB → 0) |
| OPT-2: Pre-transpose weights | **NOT APPLIED** (graceful degradation) | Pattern `aten.t(get_attr)` not found in pre-Inductor partition graph; pass runs too early in dedup path |
| OPT-3: max-autotune | **APPLIED** | Inductor selected `128x128_stages_64x3` tile variant; fused `triton_tem_fused_addmm_native_layer_norm_view_*` kernels produced (benchmark-time selection) |

---

## Residual Opportunities

1. **Wave starvation on fmha** — `achieved_occupancy = 6.34%`, grid = `[2, 12, 4]` = 96 CTAs on 108 SMs. Increasing batch size to ≥16 would fill ~3.5 waves and reduce attention time by ~50%. Alternatively: Flash Attention v2 backend. *(medium confidence)*

2. **OPT-2 not yet realized** — Pre-transpose weights pass needs to be relocated to run post-Inductor lowering, or implemented as weight preprocessing at model-load time. Estimated 2–4% additional gain. *(medium confidence)*

3. **tanh GELU in fused FFN kernel** — `triton_tem_fused_add_addmm_mul_pow_tanh_view_7` at 26,752 ns shows `tensor_core_active_pct = 43.26%`, `sm_throughput = 34.38%`. GELU approximation complexity limits throughput. Limited improvement possible without architecture change. *(low confidence)*
