# conv_block — Baseline vs. Optimized Comparison

**GPU:** NVIDIA A100-SXM4-80GB  
**Model:** VGG-style ConvBlock (3× Conv2d→BN→ReLU + pooling + Linear head)  
**Batch size:** 16 (identical across both profiles)

---

## Overall Result

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Total attributed time | 4,119,109 ns | ~1,404,657 ns | **2.93× faster** |
| Attributed operators | 28 | 27 | −1 (BN eliminated) |
| Kernel launches | 273 | 214 | −59 (−22%) |
| Unattributed kernels | 0% | 0% | — |

---

## Operator Time Budget

| Operator Group | Baseline (ns) | % | Optimized (ns) | % | Speedup |
|---|---|---|---|---|---|
| `aten::cudnn_convolution` (all conv nodes) | 2,742,823 | 66.6% | ~1,334,257 | 95.0% | 2.06× |
| `aten::_native_batch_norm_legit_no_training` | 985,505 | 23.9% | **0 (eliminated)** | — | ∞ |
| `aten::addmm` (8 nodes, Linear head) | 203,824 | 4.95% | ~70,400 | 5.0% | 2.90× |
| `aten::convolution` (Triton bias fused) | 68,384 | 1.66% | (absorbed into conv group) | — | — |
| **Total attributed** | **4,119,109** | **100%** | **~1,404,657** | **100%** | **2.93×** |

---

## Per-Pass Attribution

| Pass | Status | Contribution to Speedup |
|---|---|---|
| OPT-3 — BN fold | APPLIED (3 BN modules folded; 3× INFO log confirmed) | −985,505 ns (−23.9% of baseline); largest single contributor |
| OPT-1 — channels_last | APPLIED | `convertTensor_kernel` fully eliminated; enabled cuDNN NHWC-native BF16 kernel selection |
| OPT-2 — BF16 | APPLIED | `addmm` 2.90× (TC routing to BF16 HMMA path); conv TC% increased (60.9%→76.3% for 64→128 shape) |
| OPT-4 — max-autotune + TF32 | APPLIED | cuDNN algorithm selection for BF16 NHWC shapes; marginal contribution vs. dtype/layout changes |

---

## Hardware Counter Evidence

### OPT-3 — BN Fold
- **Baseline:** `aten::_native_batch_norm_legit_no_training` — 70 kernels, 985,505 ns, dominant kernel `triton_poi_fused__native_batch_norm_legit_no_training_relu_4`
- **Optimized:** Operator class entirely absent. Fused Triton kernels in optimized profile are `triton_poi_fused_convolution_relu_*` and `triton_poi_fused_convolution_max_pool2d_with_indices_relu_*` — BN is gone; ReLU/pool fused directly with conv bias correction.

### OPT-1 — channels_last
- **Baseline:** `convertTensor_kernel` appears dozens of times (k_00006, k_00007, k_00012, k_00013, … throughout warm-up and measure phases). Architecture notes confirm NCHW→NHWC coercion on every forward pass.
- **Optimized:** Zero `convertTensor_kernel` occurrences across the entire trace. `nhwcAddPaddingKernel` present (NHWC-native padding helper — not layout coercion).

### OPT-2 — BF16
| Operator | Baseline kernel | Optimized kernel | Baseline TC% | Optimized TC% |
|---|---|---|---|---|
| `addmm` (Linear head) | `ampere_sgemm_32x128_tn` | `Kernel2` (cuBLAS BF16) | 0.0% | 2.56% |
| Conv 64→128 | `Kernel` (FP32, 238 regs) | `Kernel` (BF16 NHWC, 232 regs) | 60.9% | 76.3% |
| Conv 128→256 | `Kernel` (FP32, 238 regs) | `Kernel` (BF16 NHWC, 246 regs) | 72.4% | 37.3% |

Note: 128→256 shape has slightly more registers in optimized (246 vs 238) — max-autotune selected a different BF16 tiling for this shape. Despite lower TC%, wall time improved 1.47× due to BF16 HMMA throughput advantage.

---

## Per-Shape Convolution Detail

| Conv Shape | Baseline regs | Optimized regs | Baseline occ% | Optimized occ% | Speedup |
|---|---|---|---|---|---|
| 3→64 (64×64 stem, ×6) | 80 | 230 | 32.5% | 25.3% | ~1.01× |
| 64→128 (64×64, ×3) | 238 | 232 | 24.2% | 12.5% | ~1.97× |
| 128→256 (32×32, ×4) | 238 | 246 | 20.6% | 11.4% | ~1.47× |
| Dedup group (all fused, ×50 kernels) | 238 | 27 (Triton avg) | 24.2% | 70.0% | ~2.0× |

---

## Residual Opportunities

| Rank | Operator | Time (ns) | % of Optimized | New Bottleneck | Proposed Fix |
|---|---|---|---|---|---|
| 1 | Conv dedup group (50 kernels, Triton-dominated) | 573,057 | 40.8% | memory_bound (DRAM=36.96%) | epilogue fusion to reduce Triton launches |
| 2 | Conv 128→256 (×4) | ~334,400 | 23.8% | register_pressure (246 regs) | `cudnn.benchmark=True` for per-shape algorithm re-selection |
| 3 | Conv 64→128 (×3) | ~188,250 | 13.4% | wave_starvation (occ=12.5%) | larger batch size amortizes launch overhead |
| 4 | Conv stem 3→64 (×6) | ~238,080 | 16.9% | wave_starvation (nhwcAddPaddingKernel overhead) | grouped conv or depthwise factorization |
| 5 | `addmm` (×8) | ~70,400 | 5.0% | wave_starvation (waves_per_sm=0.009) | B≥64 or fuse with softmax epilogue |

**Projected ceiling if residual proposals applied:** ~3.39× vs. baseline (from current 2.93×)
