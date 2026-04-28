---
name: compare
description: Compare a baseline profile.json against profile_optimized.json. Normalizes for batch-size differences, matches operators across profiles, attributes speedups to specific transformations using hardware counter evidence, and identifies residual optimization opportunities.
---

# /compare — Before/After Performance Comparison

Measures the actual hardware impact of your optimizations by comparing two `profile.json` files. Produces a normalized speedup table with causal attribution — each speedup is traced back to the specific transformation that caused it.

## Usage

```
/compare profile.json profile_optimized.json
/compare profile.json profile_optimized.json --validation-log=validation_report.json
```

## What It Does

1. **Batch normalization** — adjusts for batch-size differences between captures
2. **Operator matching** — aligns operators across profiles by name (not ID)
3. **Speedup attribution** — traces each speedup to a specific transformation using hardware counter evidence
4. **Residual analysis** — identifies the new top bottlenecks after optimization

## Batch Size Normalization

The most important step before any comparison. If you optimized with batch padding (B=16 → B=64), raw durations are not directly comparable.

```
normalized_optimized_ns = raw_optimized_ns × (baseline_batch / optimized_batch)
```

Batch size is extracted from `capture_metadata.model_name` (convention: `ModelName-B{N}`) or inferred from GEMM kernel grid dimensions.

**Without normalization:** A padded-batch workload appears 4× slower on paper but may actually be faster per sample. Always normalize.

## Operator Matching Across Profiles

Match by `operator_name`. When fusion reduces operator count:
- Baseline: 3 × `aten::linear` → Optimized: 1 merged entry = QKV fusion confirmed
- Baseline: `aten::mm` + `aten::softmax` chain → Optimized: single FlashAttention kernel = SDPA confirmed

**Kernel name evidence for transformations:**

| What you see in optimized profile | Transformation confirmed |
|---|---|
| `gemmSN_NN` / `gemmSN_TN` replaced by `sm90_xmma_gemm_bf16bf16_*` | BF16 dtype promotion → Tensor Cores active |
| `convertTensor_kernel` absent | channels_last applied |
| 3 GEMM kernels collapsed to 1 | QKV weight fusion |
| `aten::softmax` + 2× GEMM replaced by `flash_fwd*` or SDPA kernel | FlashAttention replacement |
| `BatchNorm` kernel absent | BN fold into Conv2d |

## Attribution Rules

A speedup is attributed to a transformation ONLY when:
1. The validation log shows the pass was APPLIED (INFO log, not WARNING)
2. The corresponding hardware metric changed in the expected direction
3. The operator containing those kernels shows speedup

If the validation log shows `WARNING: Pattern not found` for a pass, that pass did NOT contribute to any speedup — even if the operator shows improvement (Inductor's own optimization may be responsible).

## Residual Opportunity Detection

After measuring the optimized profile, re-rank operators by duration and classify their new bottlenecks. The optimization typically reveals second-order bottlenecks:

- Example: After eliminating `convertTensor_kernel` (layout overhead), `aten::cudnn_convolution` may now be `wave_starvation` (exposed because the layout bottleneck was masking it)
- Cross-reference with `optimizations.json`: which medium/low confidence proposals were not applied? These are the next optimization candidates.

## Output

```
Before vs. After Comparison: {ModelName} on {GPU}

Batch: baseline B={N}, optimized B={M}  →  normalized by {ratio}x

Operator Comparison (normalized):
| Operator           | Baseline (ns) | Optimized (ns) | Speedup | Bottleneck Resolved |
|--------------------|---------------|----------------|---------|---------------------|
| aten::conv2d       | 1,740,320     |    892,000     |  1.95x  | YES (channels_last) |
| aten::batch_norm   |   330,944     |    315,000     |  1.05x  | PARTIAL             |

Hardware Evidence:
  conv2d: tensor_core_active_pct 0.0% → 41.2%  (Tensor Cores now active)
  conv2d: convertTensor_kernel launches 60 → 0  (layout overhead eliminated)

Pass Attribution:
  pass_channels_last:  APPLIED → convertTensor_kernel eliminated
  pass_bf16_dtype:     APPLIED → Tensor Core path activated
  pass_fuse_qkv:       NOT APPLIED (pattern not found — no attribution)

Overall: {baseline_total}ns → {opt_total}ns = {speedup}x ({pct}% faster per sample)

Residual Opportunities:
  1. aten::batch_norm: wave_starvation (waves_per_sm=0.4) — consider batch padding (medium confidence)
  2. aten::conv2d: still register_pressure (registers_per_thread=148) — consider max-autotune (low confidence)
```

## Notes on ncu Replay Timing

All duration values in both `profile.json` and `profile_optimized.json` come from ncu replay, which runs kernels in isolation and is 2–5× slower than real execution. The absolute values are not wall-clock times.

However, **relative comparisons are valid** because both profiles are captured under the same ncu conditions. A 2× speedup in ncu-replayed durations corresponds to a real 2× speedup (within ~10% measurement noise).
