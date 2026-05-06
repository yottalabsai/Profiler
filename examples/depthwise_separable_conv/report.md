# GPU Optimization Report — DepthwiseSepConv

**Result: 3,650,578 ns → 2,257,934 ns = 1.62x faster (38.1% latency reduction)**

> All duration values from ncu replay (2–5× wall-clock). Comparisons are relative only.

---

## Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (108 SMs) |
| Architecture | Ampere (sm80) |
| PyTorch | 2.11.0+cu130 |
| Compile Mode | inductor |
| Batch Size | 16 |
| Input Shape | [16, 32, 56, 56] |
| Measurement | 5 warmup + 10 measure iterations (ncu replay) |

---

## Baseline Operator Summary

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck |
|---|---|---|---|---|
| aten::_native_batch_norm_legit_no_training | 43.2% | 1,576,717 | 60 | memory_bound |
| aten::cudnn_convolution (aggregate, 30 pw/dw) | 27.1% | 988,276 | 30 | register_pressure |
| aten::convolution (bias+act, 6 layers) | 2.7% | 97,919 | 10 | memory_bound |
| aten::cudnn_convolution 128→256 (×5) | 10.5% | 381,211 | 5 | memory_bound (l1=0%) |
| aten::cudnn_convolution dw-128 (×5) | 5.6% | 202,335 | 5 | well_optimized |
| aten::cudnn_convolution pw-64→128 (×5) | 4.5% | 165,756 | 5 | memory_bound |
| aten::cudnn_convolution dw-64 (×5) | 2.8% | 100,828 | 5 | well_optimized |
| aten::cudnn_convolution pw-32→64 (×5) | 2.1% | 78,080 | 5 | tensor_core_idle |
| aten::cudnn_convolution dw-32 (×5) | 1.6% | 59,456 | 5 | well_optimized |

**Optimization ceiling:** Top 3 operator groups account for 73.0% of attributed time.

---

## Reading the Metrics

**DRAM throughput > 60% + SM < 30%:** HBM bandwidth is the bottleneck. The operator is reading/writing more data than the GPU can compute on, causing it to sit idle waiting for memory.

**registers_per_thread > 128:** Each warp occupies so many registers that the SM can only schedule a fraction of its maximum warps. This suppresses occupancy and prevents the GPU from hiding memory latency with other work. At 238 registers, the A100 can only fit 1 block per SM instead of the theoretical 4+.

**tensor_core_active_pct = 0.0 (not null):** A GEMM kernel ran but Tensor Cores were completely idle. This means the FP32 SIMT path was selected instead of the BF16/FP16 HMMA path. Casting to BF16 is the fix; it routes cuDNN to `sm80_xmma_gemm_bf16bf16` kernels with 16× higher peak throughput.

**ncu replay note:** All duration values are from ncu's application replay mode and are 2–5× longer than real wall-clock execution. They are valid for relative comparison within and across profiles captured with the same settings.

---

## Optimizations Applied

| ID | Type | Target | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-001 | bn_fold | BatchNorm (6 layers) | dram=60.5%, 60 kernel launches, inference mode | high | **APPLIED** |
| OPT-002 | memory_layout | All conv layers | NCHW→NHWC, im2col elimination | high | **APPLIED** |
| OPT-003 | dtype_promotion | All conv layers | regs=238, tc=20–39%, occ=12–31% | high | **APPLIED (mixed)** |
| OPT-004 | algorithm_selection | Conv epilogue | BN fold exposes standalone ReLU6 | medium | **APPLIED** |

---

## Results: Before vs. After (Batch = 16, both profiles)

| Operator Group | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| BN + activation (6 layers) | 1,674,636 | 875,132 | **1.91x** |
| pw 128→256 (×5) | 381,211 | 229,117 | **1.66x** |
| dw 128×3×3 (×5) | 202,335 | 123,456 | **1.64x** |
| pw 64→128 (×5) | 165,756 | 113,822 | **1.46x** |
| cudnn_conv aggregate (30k) | 988,276 | 691,642 | **1.43x** |
| dw 64×3×3 (×5) | 100,828 | 73,535 | **1.37x** |
| pw 32→64 (×5) | 78,080 | 73,951 | 1.06x |
| dw 32×3×3 (×5) | 59,456 | 77,279 | 0.77x ⚠ |
| **Total** | **3,650,578** | **2,257,934** | **1.62x** |

---

## What Drove Each Speedup

**BN fold (OPT-001) — dominant driver, +1.91x on activation path:**
In inference mode, BatchNorm parameters (`running_mean`, `running_var`, `weight`, `bias`) are constants. Absorbing them into the preceding Conv2d weights eliminates the BN computation entirely. The 60 kernel launches of `triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_*` (1,576,717 ns) are absent from the optimized profile. The remaining 875,132 ns attributed to `aten::convolution` are the Triton-fused `convolution_hardtanh_*` epilogue kernels — bias-add + ReLU6 fused by Inductor after BN fold exposes the standalone activation (OPT-004 effect). Evidence: zero BN kernels in `profile_optimized.json`; `triton_poi_fused_convolution_hardtanh_{0-3}` kernels present.

**channels_last (OPT-002) — moderate contribution across all conv layers:**
Applying `torch.channels_last` memory format eliminated the im2col overhead for pointwise convolutions and ensured cuDNN selected its NHWC-native paths. Depthwise convolutions were already using `conv2d_c1_k1_nhwc_kernel` in the baseline; channels_last provided layout consistency across the full graph. Contributed meaningfully to the 1.37–1.66x speedups on depthwise and mid-size pointwise layers.

**BF16 dtype promotion (OPT-003) — mixed results:**
- **64→128 pointwise (confirmed):** Registers dropped from 238 to 48 threads, occupancy rose from ~12% to ~52%. The BF16 NHWC cuDNN path selected a more register-efficient kernel for this tile size.
- **128→256 pointwise (incomplete):** Registers remain at ~250 threads and occupancy at ~12%. The BF16 NHWC kernel for this larger tile size did not reduce register pressure as predicted. `torch.compile(mode='max-autotune')` is needed to benchmark alternative tile configurations.
- **32-ch depthwise (regression):** Registers increased from 40 to 72; occupancy fell from 65% to 15%. The BF16 kernel selected by cuDNN for 32-channel depthwise convolutions at this spatial size uses more registers than the FP32 equivalent. The 32-ch depthwise layers (0.77x) partially offset gains elsewhere.

**Epilogue fusion (OPT-004) — confirmed via kernel naming:**
With BN eliminated, Inductor fused the bias-add + ReLU6 activation into the conv output path, emitting four variants of `triton_poi_fused_convolution_hardtanh_*` (one per channel width: 32, 64, 128, 128-wide). The 60 fused activation kernels total 875K ns, 1.91x faster than the old 1.67M ns BN+activation path.

---

## Remaining Opportunities

| ID | Type | Target | Reason / Next Step | Est. Gain |
|---|---|---|---|---|
| — | max-autotune | pw 128→256 | `mode='max-autotune'` to find lower-register BF16 tile | ~1.5x on these ops (~10% total) |
| — | selective dtype | dw-32 layers | Apply BF16 only to layers with C ≥ 64; keep FP32 for small depthwise | +3% (undo regression) |
| — | Triton depthwise | dw-128 layers | Custom Triton kernel fusing conv+bias+ReLU6 into one pass; eliminates `triton_poi_fused_convolution_hardtanh_3` round-trip | ~2% |

The largest remaining opportunity is the 128→256 pointwise convolution group (10.1% of optimized total) which is still register-pressure bound. Running with `mode='max-autotune'` allows Inductor/cuDNN to benchmark the BF16 NHWC GEMM across multiple tile sizes and select the one with lowest register pressure for the specific A100 hardware.

---

## Reproducing This Analysis

```bash
# Stage 0: Capture baseline
/capture examples/depthwise_separable_conv/depthwise_separable_conv.py \
    --ncu-sudo=true --ncu-path=/opt/nvidia/nsight-compute/2025.1.1/ncu

# Stage 1: Analyze bottlenecks
/analyze examples/depthwise_separable_conv/profile.json

# Stage 2: Generate optimization proposals
/propose examples/depthwise_separable_conv/triage.json

# Stage 3: Generate optimized backend
/backend examples/depthwise_separable_conv/depthwise_separable_conv.py \
         examples/depthwise_separable_conv/optimizations.json

# Stage 4: Validate backend
/validate examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py

# Stage 5: Re-profile optimized workload
/capture examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py \
    --compile-backend=depthwise_sep_conv_opt --profile-name=optimized \
    --ncu-sudo=true --ncu-path=/opt/nvidia/nsight-compute/2025.1.1/ncu

# Stage 6: Compare
/compare examples/depthwise_separable_conv/profile.json \
         examples/depthwise_separable_conv/profile_optimized.json
```
