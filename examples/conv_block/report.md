# conv_block Optimization Report

**Model:** VGG-style ConvBlock (3× Conv2d→BN→ReLU + pooling + Linear head)  
**GPU:** NVIDIA A100-SXM4-80GB  
**Profiling tools:** nsys 2024.6.2, ncu 2025.1.1  
**Batch size:** 16 | **Compile mode:** inductor (baseline) / conv_block_opt (optimized)

---

## Executive Summary

Applying four evidence-backed transformations to the baseline FP32 ConvBlock model yields a **2.93× end-to-end speedup** (4,119,109 ns → 1,404,657 ns attributed execution time):

| Transformation | Baseline Bottleneck Addressed | Time Saved |
|---|---|---|
| BatchNorm fold into Conv weights | `_native_batch_norm` (23.9% of baseline) | −985,505 ns |
| `channels_last` memory format | `convertTensor_kernel` layout coercion on every conv pass | layout overhead eliminated |
| BF16 precision | FP32 SIMT path on `addmm` (TC=0%), register pressure on conv (238 regs/thread) | −203,000+ ns |
| `max-autotune` + TF32 | Sub-optimal cuDNN/cuBLAS algorithm selection | kernel selection improved |

All four passes applied without error. 7/7 validation tests passed.

---

## Hardware Context

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| SM count | 108 |
| CUDA compute capability | 8.0 (Ampere) |
| Theoretical FP32 throughput | 19.5 TFLOPS |
| Theoretical BF16 throughput | 312 TFLOPS (Tensor Core) |
| Memory bandwidth | 2 TB/s (HBM2e) |
| nsys | 2024.6.2.225 |
| ncu | 2025.1.1.0 |
| PyTorch | 2.x (inductor backend) |

---

## Baseline Bottleneck Analysis

Total attributed time: **4,119,109 ns** across 28 operators, 273 kernels (0% unattributed).

### Bottleneck 1: register_pressure + layout_overhead — 66.6% of time

Every `aten::cudnn_convolution` operator dispatched cuDNN's implicit-GEMM NHWC path with **238 registers per thread**. On A100, each SM has 65,536 registers; at 238 regs/thread × 32 threads/warp = 7,616 regs/warp, a maximum of 8 warps per SM are schedulable — **24.2% occupancy** against a hardware maximum of 100%.

Simultaneously, `convertTensor_kernel` appeared dozens of times throughout the trace (k_00006, k_00007, k_00012, …), confirming cuDNN was coercing NCHW→NHWC on every forward pass because the model was not initialized with `memory_format=torch.channels_last`.

```
Metric (per conv node, 14 nodes)   Baseline
─────────────────────────────────────────────
registers_per_thread               238
achieved_occupancy                 24.2%
tensor_core_active_pct             57.4% (FP32 TC, not HMMA)
convertTensor_kernel               present (every pass)
secondary_issue                    layout_overhead
```

### Bottleneck 2: BN+ReLU — 23.9% of time

`aten::_native_batch_norm_legit_no_training` — the `_no_training` suffix confirms `model.eval()`. Inductor had already fused BN+ReLU into a single Triton kernel (`triton_poi_fused__native_batch_norm_legit_no_training_relu_4`) achieving 65.9% occupancy and 75.9 waves/SM. However, this entire operator class can be **eliminated at model-construction time** by folding the closed-form BN statistics into the preceding Conv2d weights — no runtime cost at all.

### Bottleneck 3: tensor_core_idle on addmm — 4.95% of time

The Linear classifier head (`aten::addmm`, 8 nodes, shape 16×256→10) dispatched `ampere_sgemm_32x128_tn` — the FP32 SIMT GEMM — with `tensor_core_active_pct=0.0`. Switching to BF16 routes this to `Kernel2` (cuBLAS BF16) and enables Tensor Core execution.

---

## Optimizations Applied

### OPT-3 — BatchNorm Fold (high confidence)

**Mechanism:** Closed-form fold of each BN's running statistics and affine parameters into the preceding Conv2d weight and bias before `torch.compile()` is called:

```
scale   = gamma / sqrt(running_var + eps)
W_new   = W_conv × scale.view(-1, 1, 1, 1)
b_new   = (b_conv - running_mean) × scale + beta
```

The BN module is replaced with `nn.Identity` so TorchDynamo never generates `aten::_native_batch_norm_legit_no_training` nodes. Arithmetic is performed in FP32 to preserve BN statistic precision before any dtype cast.

**Result:** 985,505 ns eliminated — the single largest contributor.

### OPT-1 — channels_last Memory Format (high confidence)

**Mechanism:** `model.to(memory_format=torch.channels_last)` and `x.to(memory_format=torch.channels_last)` applied before `torch.compile()`. This routes cuDNN to its NHWC-native kernel path, eliminating the per-pass `convertTensor_kernel` coercion.

**Result:** Zero `convertTensor_kernel` launches in the optimized profile. Layout overhead class fully resolved.

### OPT-2 — BF16 Precision (high confidence)

**Mechanism:** `model.to(torch.bfloat16)` and `x.to(torch.bfloat16)` applied after `channels_last` to preserve the layout flag. cuDNN then selects BF16 HMMA kernels (theoretical 16× throughput advantage over FP32 scalar on A100 Tensor Cores).

**Result:**

| Shape | Baseline TC% | Optimized TC% | Speedup |
|---|---|---|---|
| Conv 64→128 | 60.9% (FP32) | 76.3% (BF16 HMMA) | ~1.97× |
| Conv 128→256 | 72.4% (FP32) | 37.3% (BF16)* | ~1.47× |
| addmm Linear head | 0.0% (SIMT) | 2.56% (BF16 cuBLAS) | 2.90× |

*max-autotune selected a different BF16 tiling for 128→256 (246 regs vs 238); lower TC% but higher BF16 HMMA instruction throughput still yields a net improvement.

### OPT-4 — max-autotune + TF32 (medium confidence)

**Mechanism:** `torch.compile(mode='max-autotune')` runs Inductor's cuDNN/cuBLAS autotuning heuristics for each operator shape. `torch.backends.cuda.matmul.allow_tf32 = True` enables TF32 for any residual FP32 paths.

**Result:** Kernel algorithm selection improved for BF16 NHWC shapes. Marginal contribution relative to OPT-1/OPT-2 on this workload.

---

## Measured Results

### Total Attributed Time

| | Baseline | Optimized | Speedup |
|---|---|---|---|
| Total attributed time | 4,119,109 ns | ~1,404,657 ns | **2.93×** |
| Kernel launches | 273 | 214 | −22% |
| Unattributed kernels | 0% | 0% | — |

### Per-Operator Breakdown

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| `aten::cudnn_convolution` (all) | 2,742,823 | ~1,334,257 | 2.06× |
| `aten::_native_batch_norm` | 985,505 | **0 (eliminated)** | ∞ |
| `aten::addmm` (×8) | 203,824 | ~70,400 | 2.90× |
| `aten::convolution` (bias, Triton) | 68,384 | (absorbed into conv group) | — |

---

## Residual Opportunities

| Priority | Operator | Time (ns) | % Optimized | Bottleneck | Proposed Fix |
|---|---|---|---|---|---|
| 1 | Conv dedup group (Triton-dominated, 50 kernels) | 573,057 | 40.8% | memory_bound (DRAM=36.96%) | epilogue fusion to reduce Triton pointwise launches |
| 2 | Conv 128→256 (×4) | ~334,400 | 23.8% | register_pressure residual (246 regs) | `torch.backends.cudnn.benchmark=True` for per-shape re-selection |
| 3 | Conv stem 3→64 (×6) | ~238,080 | 16.9% | wave_starvation | `nhwcAddPaddingKernel` overhead; consider grouped factorization |
| 4 | Conv 64→128 (×3) | ~188,250 | 13.4% | wave_starvation (occ=12.5%) | larger batch size to amortize |
| 5 | `addmm` (×8) | ~70,400 | 5.0% | wave_starvation (waves_per_sm=0.009) | B≥64 or fuse with softmax epilogue |

**Projected ceiling (residual proposals applied):** ~3.39× vs. baseline.

---

## Validation

```
Step 1 (Syntax):       PASS
Step 2 (Import):       PASS
Step 3 (Registration): PASS  — conv_block_opt in torch._dynamo.list_backends()
Step 4 (Test Suite):   PASS  — 7/7 tests
Step 5 (Smoke Test):   PASS  — compiled forward pass, exit 0
```

Passes applied (confirmed via INFO logs):
- OPT-3: `fold_bn_into_conv` called 3× (BN(64)→Conv(3→64), BN(128)→Conv(64→128), BN(256)→Conv(128→256))
- OPT-1: model + input tensor converted to channels_last before compile
- OPT-2: model params and input cast to bfloat16
- OPT-4: TF32 flags set; backend delegated to Inductor flat compile path

---

## Reproduction Commands

```bash
# Baseline profile
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block.py \
    --output-prefix examples/conv_block/profiler_output/conv_block \
    --warmup-iters 5 --measure-iters 10 --correlation-pass

PYTHONPATH=/root/Profiler nsys profile \
    --trace=cuda,nvtx \
    --output=examples/conv_block/profiler_output/conv_block \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block.py \
        --output-prefix examples/conv_block/profiler_output/conv_block \
        --warmup-iters 5 --measure-iters 10

# Optimized profile
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block_optimized.py \
    --compile-backend conv_block_opt \
    --output-prefix examples/conv_block/profiler_output/conv_block_optimized \
    --warmup-iters 5 --measure-iters 10 --correlation-pass

# Validation
PYTHONPATH=/root/Profiler python3 -m pytest \
    examples/conv_block/test_conv_block_optimized.py -v
```
