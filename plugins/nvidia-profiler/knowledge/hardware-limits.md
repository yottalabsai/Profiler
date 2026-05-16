# GPU Hardware Limits Reference

This table is used by `/propose` to compute Waves/SM, ridge points, and architecture-specific bottleneck thresholds. Look up `capture_metadata.device_name` against `device_name_pattern` (case-insensitive substring match).

## Roofline Table

| GPU | device_name_pattern | Architecture | SM Count | Peak BF16 TFLOPS | Peak FP16 TFLOPS | Peak FP32 TFLOPS | Peak HBM BW (TB/s) | Ridge Point BF16 (FLOP/byte) | Preferred Tile Size |
|---|---|---|---|---|---|---|---|---|---|
| A100 SXM5 | `a100` | Ampere | 108 | 312 | 312 | 19.5 | 2.0 | 156 | 64 |
| A100 PCIe | `a100.*pcie` | Ampere | 108 | 312 | 312 | 19.5 | 1.555 | 201 | 64 |
| A10G | `a10g` or `a10` | Ampere | 80 | 31.2 | 31.2 | 31.2 | 0.600 | 52 | 32 |
| H100 SXM5 | `h100.*sxm` | Hopper | 132 | 989 | 1979 | 67 | 3.35 | 295 | 64 |
| H100 PCIe | `h100.*pcie` | Hopper | 114 | 756 | 1513 | 51 | 2.0 | 378 | 64 |
| H200 SXM5 | `h200` | Hopper | 132 | 989 | 1979 | 67 | 4.8 | 206 | 64 |
| RTX PRO 6000 Blackwell | `rtx.*6000` or `blackwell` | Blackwell | 188 | 1457 | 1457 | 91 | 1.79 | 814 | 64 |
| B100 | `b100` | Blackwell | 160 | 3500 | 7000 | 60 | 8.0 | 438 | 64 |
| B200 | `b200` | Blackwell | 160 | 4500 | 9000 | 60 | 8.0 | 563 | 64 |
| RTX 4090 | `rtx 4090` or `4090` | Ada Lovelace | 128 | 165 | 165 | 82.6 | 1.008 | 164 | 32 |
| RTX 3090 | `rtx 3090` | Ampere | 82 | 35.6 | 35.6 | 35.6 | 0.936 | 38 | 32 |

**Fallback (device_name null or unrecognized):** Use A100 SXM5 limits and flag the assumption in the report.

## Waves/SM Formula

```
waves_per_sm = ceil(grid_x * grid_y * grid_z / sm_count)
```

From a `KernelRecord.grid_dim = (grid_x, grid_y, grid_z)` and `sm_count` from this table.

Interpretation:
- `waves_per_sm < 0.5` → severe wave starvation; batch size or operator fusion is the fix
- `0.5 ≤ waves_per_sm < 2.0` → moderate starvation; padding may help
- `waves_per_sm ≥ 2.0` → sufficient occupancy to hide latency; look elsewhere for the bottleneck

## Tensor Core Tile Requirements

For Tensor Cores to fire, the GEMM tile dimensions must meet minimum alignment. If they don't, the kernel falls back to CUDA SIMT (HMMA disabled).

| Architecture | BF16 min M/N/K | FP16 min M/N/K |
|---|---|---|
| Ampere (A100) | 16 / 16 / 16 | 16 / 16 / 16 |
| Hopper (H100/H200) | 64 / 64 / 16 (WGMMA) | 64 / 64 / 16 |
| Blackwell | 64 / 64 / 16 (WGMMA) | 64 / 64 / 16 |

If `block_dim.x * block_dim.y < tile_min`, Tensor Cores cannot engage regardless of dtype.

## FP8 Support and Tile Requirements

FP8 (`torch.float8_e4m3fn`, `torch.float8_e5m2`) is Hopper/Blackwell only. On Ampere, FP8 ops silently fall back to FP16 or raise a runtime error depending on PyTorch version. **Never propose FP8 for `device_name` matching `a100`, `a10`, `3090`, or `4090`.**

| GPU | FP8 Hardware Support | Peak FP8 TFLOPS (e4m3) | FP8 Min Tile M/N/K | Notes |
|---|---|---|---|---|
| Ampere (A100, A10G) | No | N/A | N/A | Falls back to FP16 silently |
| Hopper (H100 SXM5) | Yes | 1979 | 64 / 64 / 32 | Requires `torch >= 2.1`, `cuBLAS >= 12.4` |
| Hopper (H200) | Yes | 1979 | 64 / 64 / 32 | Same as H100 FP8 path |
| Blackwell (B100) | Yes | 7000 | 64 / 64 / 32 | Use `torch.float8_e4m3fn` |
| Blackwell (B200) | Yes | 9000 | 64 / 64 / 32 | Use `torch.float8_e4m3fn` |
| Ada (RTX 4090) | Software emulated | N/A | N/A | No dedicated FP8 Tensor Core path; avoid |

When proposing FP8 for eligible GPUs, add to notes: `"Requires torch >= 2.1 and cuBLAS >= 12.4. Verify: torch.cuda.get_device_capability() >= (9, 0)."`

## L2 Cache Sizes

L2 size determines whether intermediate tensors from adjacent fused operators fit in cache between kernel launches. If the intermediate exceeds L2, fusion saves no memory traffic.

| GPU | L2 Cache | Key Implication |
|---|---|---|
| A100 SXM5 | 40 MB | ~10M elements (BF16); elementwise fusion with GEMM output is effective |
| H100 SXM5 | 50 MB | ~12.5M elements; activation fusion with transformer FFN output is effective |
| H200 | 50 MB | Same as H100 |
| B100 / B200 | 192 MB | Most transformer layer intermediates fit; prefer aggressive fusion |
| RTX 4090 | 72 MB | Larger than A100; lower HBM BW limits fusion ROI despite large cache |
| A10G | 24 MB | Smaller; be conservative with fusion proposals |
| RTX 3090 | 6 MB | Very small; HBM bandwidth is the primary limit, not cache |

## Architecture-Specific Counter Availability

| Counter | Ampere | Hopper | Blackwell |
|---|---|---|---|
| `tensor_core_active_pct` (`smsp__pipe_tensor_cycles_active...`) | ✓ | ✓ | ✓ |
| `warp_cycles_per_instruction` (`smsp__warp_cycles_per_issued...`) | ✓ | ✓ | **Removed** — use `eligible_cycles_pct` instead |
| `dram_bytes_read/written` | ✓ | ✓ | May require fallback metric name |
| `achieved_occupancy` | ✓ | ✓ | ✓ |

**Blackwell note:** `warp_cycles_per_instruction == null` is expected and correct on Blackwell GPUs, not a profiler bug. Use `eligible_cycles_pct < 20` as the equivalent latency-bound indicator.

## Roofline Interpretation

An operator is **compute-bound** if its arithmetic intensity exceeds the ridge point:
```
arithmetic_intensity = (total_flops) / (dram_bytes_read + dram_bytes_written)
```

- Arithmetic intensity **above** ridge point → compute-bound (maximize FLOP throughput: dtype, tile size, fusion)
- Arithmetic intensity **below** ridge point → memory-bound (maximize data reuse: cache tiling, operator fusion, layout)

`profile.json` does not include `total_flops` directly. Use `sm_throughput_pct` and `memory_throughput_pct` as proxy indicators.

## cuBLAS Kernel Name Reference

Key cuBLAS kernel names and what they indicate:

| Kernel Name | Meaning | Fix |
|---|---|---|
| `gemmSN_NN_*` | FP32 SIMT, non-transposed inputs | Cast to BF16/FP16 → routes to Tensor Cores |
| `gemmSN_TN_*` | FP32 SIMT, transposed A (weight stored row-major) | Cast dtype + consider pre-transposing weight |
| `sm90_xmma_gemm_bf16bf16_*` | BF16 WGMMA (Hopper/Blackwell Tensor Core) | Optimal — no action needed |
| `sm80_xmma_gemm_f16f16_*` | FP16 HMMA (Ampere Tensor Core) | Optimal for FP16 — no action needed |
| `convertTensor_kernel` | Memory layout coercion (NCHW ↔ NHWC) | Apply `memory_format=torch.channels_last` at model creation |
| `triton__*_fused_*` | Fused Triton kernel from Inductor | Attribution via NVTX or torch.profiler correlation only; appears in `unattributed_kernels` if neither matched |
| `volta_fp16_s884*` | Legacy FP16 Tensor Core (Volta/Turing) | Use `torch.compile` with `mode='max-autotune'` |
| `cudnn_infer_ampere_*` | cuDNN Ampere inference conv (FP16/INT8) | Optimal — no action needed |
| `sm80_xmma_implicit_gemm_*` | Ampere implicit GEMM for convolution | Optimal — no action needed |
| `sm90_xmma_implicit_gemm_*` | Hopper implicit GEMM for convolution | Optimal — no action needed |
| `wgrad_*` / `dgrad_*` | Weight/data gradient kernels (backward pass) | Only in training profiles; skip in inference analysis |

## Triton Kernel Naming Conventions (Inductor-Generated)

Inductor-generated Triton kernels follow a consistent naming pattern used by nsys and ncu. Recognizing these names is essential for attribution.

| Prefix | Meaning | Attribution |
|---|---|---|
| `triton_poi_fused_*` | Pointwise (elementwise) fused kernel | NVTX range if `emit_nvtx` was active; else `unattributed_kernels` |
| `triton_tem_fused_*` | Template-based (reduction, softmax, layernorm) | Same; often spans multiple `aten::` operators |
| `triton_red_fused_*` | Reduction-specific fused kernel | Same |
| `triton_per_fused_*` | Persistent kernel (e.g. persistent reduction) | Same |
| `triton__*` (generic) | Any Inductor Triton kernel not matching above | Treat as `unattributed_kernels` unless `corr.json` links it |
| `_triton_gemm_*` | Triton GEMM (rare; cuBLAS preferred for large GEMMs) | Attribution via NVTX only |

**High unattributed Triton kernel count?** Re-run manifest build with `--inductor-fusion-dir <inductor_debug_dir>` to upgrade these to MEDIUM-confidence attribution.
