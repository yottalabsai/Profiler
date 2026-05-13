# TransformerStack Optimization Comparison

## Summary

| Metric | Baseline (FP32) | Optimized (BF16) | Change |
|--------|-----------------|------------------|--------|
| Wall-clock latency | 4.00 ms/iter | 3.13 ms/iter | **1.28x faster** |
| Latency reduction | — | — | **22%** |
| tensor_core_active_pct (GEMM) | ~1% | ~36% | +35pp |
| Unattributed kernels | 0% | ~53% | Higher (no inductor fusion map) |

## Hardware Evidence (ncu)

**Baseline**: All matrix multiply kernels run on the FP32 SIMT path.
- `tensor_core_active_pct ≈ 1%` — tensor cores idle
- `sm_throughput_pct ≈ 32%` — compute under-utilized

**Optimized**: BF16 dtype routes matmuls through the tensor core path.
- `tensor_core_active_pct ≈ 36%` — tensor cores active
- This confirms OPT-1 (BF16 dtype promotion) is working

## Pass Attribution

| Pass | Applied | Impact |
|------|---------|--------|
| OPT-1 BF16 dtype promotion | ✓ APPLIED | 22% latency reduction, tensor cores from 1%→36% |
| OPT-2 QKV fusion | NOT_APPLIED (graceful) | — |
| OPT-3 SDPA replacement | NOT_APPLIED (graceful) | — |

## Residual Opportunities

1. **QKV fusion** (OPT-2): Q/K/V projections still run as 3 separate mm calls. Fixing the FX pass to handle Dynamo's pre-partition IR (weights as placeholders, not get_attr nodes) would eliminate 16 kernel launches per forward pass.

2. **SDPA** (OPT-3): Manual attention (matmul→softmax→matmul) could be replaced with `F.scaled_dot_product_attention`. The pattern detection needs to target pre-Dynamo Python-level ops (`torch.matmul`, `torch.softmax`) correctly.

3. **tensor_core_utilization ceiling**: Even at BF16, only 36% tensor core activity. Larger batch sizes or longer sequence lengths would improve occupancy and push toward the ~80% achievable range for A100 GEMMs of this shape.
