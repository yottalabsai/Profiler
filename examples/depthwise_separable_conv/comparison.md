# Before/After Comparison — DepthwiseSepConv on NVIDIA A100-SXM4-80GB

**Overall: 3,650,578 ns → 2,257,934 ns = 1.62x faster (38.1% latency reduction)**

> All durations from ncu replay (2–5× wall-clock). Relative comparisons are valid.

---

## Summary Table

| Group | Baseline (ns) | Optimized (ns) | Speedup | Bottleneck Resolved |
|---|---|---|---|---|
| BN + activation (6 layers) | 1,674,636 | 875,132 | **1.91x** | YES — BN eliminated; epilogue-fused bias+ReLU6 |
| pw 128→256 (5 instances) | 381,211 | 229,117 | **1.66x** | PARTIAL — register count unchanged (238→250); layout speedup |
| dw 128×3×3 (5 instances) | 202,335 | 123,456 | **1.64x** | PARTIAL — channels_last NHWC path active |
| pw 64→128 (5 instances) | 165,756 | 113,822 | **1.46x** | PARTIAL — BF16 reduced registers (238→48), occ 12%→52% |
| cudnn_conv aggregate (30k) | 988,276 | 691,642 | **1.43x** | PARTIAL — layout + dtype improvement |
| dw 64×3×3 (5 instances) | 100,828 | 73,535 | **1.37x** | PARTIAL |
| pw 32→64 (5 instances) | 78,080 | 73,951 | 1.06x | MINIMAL — small tensors, BF16 benefit limited |
| dw 32×3×3 (5 instances) | 59,456 | 77,279 | **0.77x ⚠** | REGRESSION — BF16 increased registers 40→72, occ 65%→15% |

---

## Pass Attribution

### OPT-001: BN Fold — APPLIED ✓ (primary speedup driver)
- `aten::_native_batch_norm_legit_no_training` (60 kernels, 1,576,717 ns) → **eliminated**
- After fold, Inductor emitted `triton_poi_fused_convolution_hardtanh_{0-3}` epilogue kernels
- These 60 epilogue kernels (875,132 ns total) are 1.91x faster than the old BN+activation path
- **Kernel evidence:** `triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_*` absent in optimized profile

### OPT-002: channels_last — APPLIED ✓ (moderate contribution)
- Depthwise convolutions confirmed using `conv2d_c1_k1_nhwc_kernel` (NHWC path) in both profiles
- Pointwise convolutions: `Kernel2` still present in optimized (cuDNN still selecting this kernel for some sizes)
- Contributed to 1.37–1.66x speedup on depthwise and mid-size pointwise layers

### OPT-003: BF16 dtype promotion — APPLIED, MIXED RESULTS ⚠
- **64→128 pw convolutions:** regs 238→48, occupancy 12%→52% ✓ (as predicted)
- **128→256 pw convolutions:** regs still 250 (BF16 NHWC path uses different kernel; register count not reduced as expected)
- **32-ch depthwise (REGRESSION):** regs 40→72, occupancy 65%→15% — BF16 selected a higher-register kernel for these small depthwise sizes
- **Mitigation:** Run `torch.compile(mode='max-autotune')` to let Inductor find the best BF16 tile for each shape

### OPT-004: epilogue_fusion — APPLIED ✓
- Confirmed by `triton_poi_fused_convolution_hardtanh_*` kernels in optimized profile
- Conv + bias-add + ReLU6 fused into single Triton elementwise kernel per layer
- **Kernel evidence:** 6 distinct fused kernel variants × 10 iterations = 60 kernels (matches exactly)

---

## Hardware Counter Evidence

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| BN kernel launches | 60 | 0 | **−60 (eliminated)** |
| Total kernel launches | 130 | 120 | −10 |
| pw 128→256 occ % | 12.4% | 11.8% | flat (regs still ~250) |
| pw 64→128 occ % | 12%–31% | 52% | **+40pp** ✓ |
| dw 128 regs/thread | 40 | 72 | ↑ (BF16 regression) |
| Epilogue fusion | No | Yes (`_hardtanh_*`) | **confirmed** |

---

## Residual Opportunities

### 1. 128→256 pointwise convolutions (still register-pressure, ~10% of total)
- `regs_per_thread = 250`, `occ = 11.8%` — identical to baseline
- BF16 NHWC selected a high-register cuDNN kernel variant for this shape
- **Fix:** `torch.compile(mode='max-autotune')` to trigger cuDNN autotuning for BF16 NHWC 128×256 tiles

### 2. 32-ch depthwise regression (−23%, ~3.4% of total)
- BF16 increased registers from 40 to 72 on 32-channel depthwise convolutions
- **Fix:** Apply BF16 selectively — keep FP32 for small depthwise layers, or constrain to `mode='max-autotune'`

### 3. bias+ReLU6 epilogue kernels now largest operator (38.8% of optimized total, 875K ns)
- These Triton elementwise kernels fuse bias-add + hardtanh but still write/read from DRAM
- To fully fuse with the preceding conv, the depthwise conv would need to be Triton-compiled (not cuDNN)
- **Fix (low confidence):** Custom Triton depthwise kernel that writes directly to registers without intermediate materialization

---

## Reproduction Commands

```bash
# Capture baseline
python nvidia/scripts/run_workload.py \
    --workload examples/depthwise_separable_conv/depthwise_separable_conv.py \
    --compile-backend inductor --warmup-iters 5 --measure-iters 10

# Capture optimized
python nvidia/scripts/run_workload.py \
    --workload examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py \
    --compile-backend depthwise_sep_conv_opt --warmup-iters 5 --measure-iters 10
```
