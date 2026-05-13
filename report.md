# TransformerStack Optimization Report

**Model**: TransformerStack (8-layer GPT-2-style transformer)  
**Hardware**: NVIDIA A100-SXM4-80GB  
**Date**: 2026-05-13  

---

## Hardware Context

- GPU: NVIDIA A100-SXM4-80GB (108 SMs, 312 TFLOPS BF16, 2 TB/s HBM2e)
- Ridge point: ~156 TFLOPS / 2 TB/s = ~78 FLOPs/byte
- Peak BF16 tensor core throughput: 312 TFLOPS

---

## Bottleneck Analysis

The profiler identified **tensor_core_idle** as the dominant bottleneck across all 8 transformer layers:

| Metric | Baseline Value | Interpretation |
|--------|---------------|----------------|
| `tensor_core_active_pct` | ~1% | Tensor cores nearly idle — FP32 SIMT path |
| `sm_throughput_pct` | ~32% | SM under-utilized |
| `achieved_occupancy` | ~32% | Low wave count |

**Root cause**: Model initialized in FP32 (default). All matmul operations in attention (Q×K^T, A×V) and FFN (fc_up, fc_down) use the FP32 CUDA SIMT path, bypassing tensor cores which require BF16/FP16/TF32.

---

## Transformations Applied

| ID | Type | Status | Evidence |
|----|------|--------|---------|
| OPT-1 | BF16 dtype promotion | **APPLIED** | `get_model_and_input()` casts model + input to BF16 |
| OPT-2 | QKV weight fusion | NOT_APPLIED | Pattern targets pre-Inductor Dynamo IR placeholders |
| OPT-3 | SDPA replacement | NOT_APPLIED | Softmax/matmul pattern not matched in partitioned IR |

---

## Measured Results

```
Baseline  (FP32):   4.00 ms/iter   — tensor_core_active_pct ≈ 1%
Optimized (BF16):   3.13 ms/iter   — tensor_core_active_pct ≈ 36%

Overall speedup:    1.28x  (22% latency reduction)
```

OPT-1 activated tensor cores: `tensor_core_active_pct` rose from ~1% to ~36%.

---

## Reproduction Commands

```bash
# Baseline profile
PYTHONPATH=/root/Profiler /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile \
    --trace=cuda,nvtx --output=profiler_output/transformer_stack/transformer_stack \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/transformer_stack/transformer_stack.py \
        --warmup-iters 2 --measure-iters 2 \
        --output-prefix profiler_output/transformer_stack/transformer_stack

# Optimized profile
PYTHONPATH=/root/Profiler /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile \
    --trace=cuda,nvtx --output=profiler_output/transformer_stack_opt/transformer_stack_opt \
    python3 nvidia/scripts/run_workload.py \
        --workload transformer_stack_optimized.py \
        --compile-backend transformer_stack_opt \
        --warmup-iters 2 --measure-iters 2
```

---

## Known Caveats

1. **ncu replay timing**: All `total_duration_ns` values in `profile.json` and `profile_optimized.json` are 2–5× real execution time (ncu application-mode replay). Use wall-clock benchmarks for absolute latency comparisons.

2. **CUPTI conflict with nsys**: `torch.profiler` correlation pass runs inside nsys; CUPTI conflict produces a 0-entry `.corr.json`. Attribution falls back to NVTX-only (MEDIUM confidence). The Inductor fusion map provides MEDIUM-confidence attribution for Triton fused kernels.

3. **Partition-level attribution (baseline)**: The dedup+inductor backend wraps each partition with a single NVTX range (`layer::unique::modules_0`), not individual aten:: ranges. All kernels inside the partition are attributed to that single operator name. Fine-grained per-operator analysis requires running without dedup or using the standard inductor backend.

4. **OPT-2/OPT-3 graceful degradation**: QKV fusion and SDPA replacement passes did not match patterns in the Dynamo pre-partition FX IR (weights are lifted to placeholder nodes, not get_attr). The backend degrades gracefully — OPT-1 still applies.

---

## Future Work

1. Fix OPT-2 (QKV fusion): target `F.linear` calls in Dynamo IR; resolve weights from placeholder inputs via `_capture_partition_inputs` rather than `named_parameters()`.
2. Fix OPT-3 (SDPA): target `torch.matmul` / `torch.softmax` (Python builtins in pre-partition IR) correctly; use `n.target is torch.matmul` identity checks.
3. Consider running with `--compile-backend=inductor` (flat, no dedup) for aten::-level attribution to get per-operator bottleneck analysis.
