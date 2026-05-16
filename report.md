# ConvBlock Optimization Report

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition | PyTorch 2.11.0+cu128 | 2026-05-16

---

## Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell |
| SM Count | 188 |
| Peak BF16 | 1,457 TFLOPS |
| Peak FP32 SIMT | 91 TFLOPS |
| Peak HBM BW | 1.79 TB/s |
| PyTorch | 2.11.0+cu128 |
| Compile Mode | inductor (baseline) / conv_block_opt (optimized) |
| Batch Size | 16 |
| Input Shape | 16 × 3 × 64 × 64 |
| Measurement | ncu application-mode replay, 2 warmup + 2 measure iterations |

> **ncu replay timing note:** All durations come from ncu's kernel replay mode and are
> 2–5× longer than real wall-clock execution. Use them for relative comparison only.

---

## Operator Summary — Baseline

| Operator | Time | Duration (ns) | Kernels | Dominant Kernel | Bottleneck |
|---|---|---|---|---|---|
| layer::unique::prologue (outer NVTX, 1 iter) | 59.5% | 354,495 | 39 | convertTensor + BN Triton | Double-counts inner ops |
| aten::cudnn_convolution 64→128 (2 instances) | 28.6% | 170,464 | 6 | Kernel (sm80_xmma TF32) | compute-bound, Ampere HMMA |
| aten::cudnn_convolution 128→256 (2 instances) | 24.9% | 149,568 | 6 | Kernel (sm80_xmma TF32) | compute-bound, Ampere HMMA |
| aten::_native_batch_norm_legit_no_training | 6.3% | 37,904 | 14 | triton_poi_fused_BN_relu | memory-bound, DRAM=67% |
| aten::cudnn_convolution 3→64 (2 instances) | 4.9% | 29,120 | 6 | sm80_xmma_fprop + convertTensor | NCHW→NHWC coercion |
| aten::addmm 256→10 (3 instances) | 1.5% | 8,896 | 3 | gemmSN_TN_kernel | FP32 SIMT, TC%=0 |

**True per-iteration latency: ~363 µs** (prologue 354 µs + addmm 9 µs; prologue outer
NVTX range double-counts the inner operator sub-ranges — do not sum both).

Across 2 measurement iterations: 15 `convertTensor_kernel` launches per pass (NCHW↔NHWC
coercion), 14 BatchNorm kernel launches, and all convolutions using the Ampere-generation
`sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f3` or unnamed `Kernel` path
rather than Blackwell-native BF16 WGMMA.

---

## Reading the Metrics

**tensor_core_active_pct:** Fraction of SM cycles where the tensor core pipeline was
active. `= 0.0` on a GEMM kernel means FP32 SIMT path — no Tensor Core hardware used.
On Blackwell, this counter is still available (unlike `warp_cycles_per_instruction`
which was removed).

**eligible_cycles_pct:** Blackwell's replacement for `warp_cycles_per_instruction`.
Fraction of SM cycles where warps were eligible to issue. Below 20% indicates scheduler
stalls; above 80% is ideal.

**convertTensor_kernel:** cuDNN format coercion between NCHW (PyTorch default) and
NHWC (cuDNN internal). Eliminated by using `memory_format=torch.channels_last`.

**sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32:** An Ampere-era TF32
convolution kernel running on a Blackwell GPU. Not wrong, but leaves significant
Blackwell WGMMA throughput on the table.

**Waves/SM formula:** `ceil(grid_x × grid_y × grid_z / 188)`. Values below 0.5 indicate
most SMs are idle.

---

## Optimizations Applied

| ID | Type | Target | Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | channels_last | All aten::cudnn_convolution | 30 convertTensor_kernel, 111 µs total | high | **APPLIED** |
| OPT-2 | BF16 dtype promotion | Conv 64→128, 128→256, addmm | sm80_xmma TF32 on Blackwell; gemmSN_TN TC%=0 | high | **APPLIED** |
| OPT-3 | BN fold (module-level) | aten::_native_batch_norm (14 kernels, 67,904 ns) | DRAM%=67.2%, 0 tensor core activity | high | **APPLIED** |

All three optimizations are applied in `get_model_and_input()` before Dynamo traces the model.
OPT-3 uses `torch.nn.utils.fusion.fuse_conv_bn_eval()` to fold BN weights into Conv2d;
the resulting FX graph contains no BN nodes. The backend (`conv_block_opt`) takes the
flat compile path (no duplicate partitions detected) and delegates directly to `compile_fx`.

---

## Results: Before vs. After (Batch = 16, both profiles)

| Operation | Baseline (ns/iter) | Optimized (ns/iter) | Speedup | Notes |
|---|---|---|---|---|
| Conv 3→64 (16×3×64×64) | 14,560 | 24,112 | 0.60x ⚠ | Regression — see below |
| Conv 64→128 (16×64×64×64) | 85,312 | 32,768 | **2.60x** | BF16 WGMMA; TC 65%→71% |
| Conv 128→256 (16×128×32×32) | 74,848 | 41,248 | **1.81x** | BF16 WGMMA; TC 60%→79% |
| BatchNorm (all 3 stages) | 33,952 | 0 | **eliminated** | 14 kernels → 0 |
| addmm 256→10 | 2,976 | 2,400 | 1.24x | gemmSN_TN → Kernel2; TC 0%→8% |
| Remaining (convertTensor, pool, post-conv) | ~151,000 | 19,088 | **7.9x** | 15 convertTensor + BN Triton → 5 BF16 Triton |
| **Total per forward pass** | **~363,000** | **~119,040** | **3.05x** | |

**Kernel launches per forward pass: 47 → 9 (81% reduction)**

> Optimized totals are averaged over the two captured iterations. Baseline per-iteration
> total is derived from the prologue outer NVTX range (354,495 ns) plus addmm (8,896 ns),
> consistent with the analysis in `optimizations.json`.

### Conv 3→64 regression

The first convolution (3 input channels) regressed from 14,560 to 24,192 ns per iteration.
cuDNN selected `convolve_common_engine_float_NHWC` (TC%=0) for the NHWC BF16 path — a
non-WGMMA fallback. With only 3 input channels, the GEMM tile dimensions are too small
for Blackwell WGMMA minimum alignment (64/64/16). This regression (9,552 ns/iter) is
outweighed by the savings elsewhere. No fix was proposed because the tile-size constraint
is fundamental to this layer's shape.

### Metric inconsistency on op_id=26

The second instance of the 64→128 convolution (`op_id=26`, 2nd measurement iteration)
shows `sm=0.11%, TC=8.15%` vs op_id=9 (`sm=61%, TC=71%`). The duration is consistent
(32,384 vs 32,768 ns), suggesting the metrics were collected for a neighboring small
kernel during the ncu replay invocation indexing. The op_id=9 metrics are used for
analysis above.

---

## What Drove Each Speedup

**OPT-1 — channels_last (+7.9x on post-conv overhead; eliminates convertTensor):**
All 15 `convertTensor_kernel` launches per forward pass are absent from the optimized
profile. In the baseline, cuDNN received NCHW tensors and had to insert NCHW→NHWC
conversion before each convolution and NHWC→NCHW after. With `channels_last`, tensors
arrive in NHWC natively, and cuDNN selects NHWC-native kernel paths
(`convolve_common_engine_float_NHWC`, or the `Kernel` BF16 path for larger stages).
The 15 convertTensor launches (estimated 22+ µs/iter for the named ops alone) are
entirely eliminated.

**OPT-2 — BF16 dtype promotion (+2.60x on Conv 64→128, +1.81x on Conv 128→256):**
Casting model weights and inputs to BF16 caused cuDNN to select Blackwell-native BF16
WGMMA kernels (labeled `Kernel` in ncu) for the 64→128 and 128→256 convolution stages.
Tensor core activity rose from 65% to 71% for 64→128 and from 60% to 79% for 128→256.
The post-conv Triton kernels (bias-add, ReLU, pool) also benefit from BF16: operating
on half-width data roughly halves their DRAM traffic, contributing to the 7.9x
improvement in the "remaining" bucket.

**OPT-3 — BN fold (+elimination of 14 kernel launches, 33,952 ns/iter):**
`fuse_conv_bn_eval()` folded the BatchNorm scale/shift into each Conv2d weight and
bias before Dynamo tracing. All 14 `triton_poi_fused__native_batch_norm_legit_no_training_*`
kernels are absent from the optimized profile. These kernels were purely memory-bound
(DRAM%=67.2%) — each read the full post-conv feature map, scaled, shifted, and wrote
it back with zero data reuse. With the fold, normalization is absorbed into the
convolution arithmetic at zero marginal cost.

---

## Operator Summary — Optimized

| Operator | Duration (ns/iter) | Kernels/iter | Dominant Kernel | TC% |
|---|---|---|---|---|
| Conv 3→64 (NHWC, 3-ch input) | 24,112 | 1 | convolve_common_engine_float_NHWC | 0.0 |
| Conv 64→128 (BF16 WGMMA) | 32,576 | 1 | Kernel (BF16) | 71.1 |
| Conv 128→256 (BF16 WGMMA) | 40,848 | 1 | Kernel (BF16) | 79.4 |
| BatchNorm | 0 | 0 | — | — |
| Post-conv Triton (bias+ReLU+pool) | 19,088 | 5 | triton_poi/red/per_fused | 0.0 |
| addmm 256→10 | 2,416 | 1 | Kernel2 (BF16) | 8.3 |

The bottleneck has shifted: the 3→64 conv (now 20% of runtime) is the new largest
single operator. Its `convolve_common_engine_float_NHWC` kernel does not engage Tensor
Cores due to the 3-channel input size. The two larger convolutions (64→128 and 128→256)
are now well-served by BF16 WGMMA with 71–79% tensor core activity.

---

## Remaining Opportunities

| Opportunity | Target | Potential Gain | Prerequisite |
|---|---|---|---|
| Pad input channels 3→4 or 3→8 | Conv 3→64 (TC%=0, 20% of runtime) | ~1.1–1.2x on this op | Modify model architecture or use `F.pad` |
| Custom Triton conv kernel for 3-channel NHWC BF16 | Conv 3→64 | ~2x on this op | Custom kernel development |
| `torch.compile(mode='max-autotune')` | All ops | 5–15% additional tuning | Replace custom backend with max-autotune |

These were not part of the original proposals and require either model changes or
custom kernel work. Given the 3→64 conv is now the dominant operator at 20% of total
runtime, padding its input to 4 channels (and zero-masking the added channel) is the
most accessible next step.

---

## Reproducing This Analysis

```bash
# 1. Capture baseline profile
/capture examples/conv_block/conv_block.py

# 2. Generate optimization proposals
/propose profile.json

# 3. Generate optimized backend
/backend examples/conv_block/conv_block.py optimizations.json

# 4. Validate backend
/validate examples/conv_block/conv_block_optimized.py

# 5. Profile optimized workload
/capture examples/conv_block/conv_block_optimized.py \
    --profile-name=optimized \
    --compile-backend=conv_block_opt

# 6. Generate this report
/report profile.json profile_optimized.json
```

**Direct profiling commands:**
```bash
# Baseline
PYTHONPATH=/home/ubuntu/Profiler nsys profile --trace=cuda,nvtx \
    --output=profiler_output/conv_block --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block.py \
        --warmup-iters 2 --measure-iters 2 \
        --output-prefix profiler_output/conv_block \
        --inductor-debug-dir profiler_output/conv_block_inductor_debug

# Optimized
PYTHONPATH=/home/ubuntu/Profiler nsys profile --trace=cuda,nvtx \
    --output=profiler_output/conv_block_opt --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block_optimized.py \
        --compile-backend conv_block_opt \
        --warmup-iters 2 --measure-iters 2 \
        --output-prefix profiler_output/conv_block_opt \
        --inductor-debug-dir profiler_output/conv_block_opt_inductor_debug
```
