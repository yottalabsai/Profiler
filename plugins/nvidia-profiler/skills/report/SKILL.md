---
name: report
description: Generate a human-readable report.md summarizing the complete optimization lifecycle — from baseline hardware metrics to implemented transformations to measured speedups. Suitable for sharing with teams, publishing as documentation, or including in PR descriptions.
---

# /report — Optimization Report Generation

Generates `report.md` documenting the full optimization lifecycle. The report is technically precise and evidence-based — each claim is backed by hardware counter data.

## Usage

```
/report
/report --audience=team          # include hardware counter explainers (default)
/report --audience=executive     # leads with speedup numbers, minimal counter detail
/report --output=custom_name.md
```

## Inputs (reads from current directory)

- `profile.json` — baseline hardware metrics
- `profile_optimized.json` — optimized hardware metrics (if available)
- `triage.json` — bottleneck classification from `/analyze`
- `optimizations.json` — proposed optimizations with evidence
- `validation_report.json` — which passes applied (from `/validate`)

Missing files are noted as "not yet available" and those sections are skipped.

## Report Structure

Mirror `examples/transformer_block/report.md` as the gold standard. Include these sections:

### 1. Hardware Context

```markdown
## Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA H100 SXM5 (132 SMs) |
| Architecture | Hopper |
| PyTorch | 2.3.0 |
| Compile Mode | inductor |
| Batch Size | 16 |
| Measurement | 10 iterations (ncu replay — relative timing only) |
```

### 2. Operator Summary

Time-budget table sorted by wall-time percentage:
```markdown
## Operator Summary (Baseline)

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| aten::linear | 42.3% | 5,234,567 | 30 | tensor_core_idle |
| aten::layer_norm | 18.1% | 2,241,200 | 20 | well_optimized |
```

### 3. Reading the Metrics

Include a brief explainer for non-GPU-expert readers:

```markdown
## Reading the Metrics

**Waves/SM:** Number of waves of warps dispatched per SM. A wave is one full occupancy
cycle. `waves_per_sm < 0.5` means most SMs are idle most of the time — not enough
parallel work. Formula: `ceil(grid_x × grid_y × grid_z / sm_count)`.

**Achieved Occupancy:** Fraction of theoretical maximum warps active simultaneously.
Low occupancy (< 20%) means insufficient parallelism to hide memory latency.

**tensor_core_active_pct = 0.0 (not null):** The GEMM kernel ran but did not use
Tensor Core hardware. This always means the kernel took the FP32 SIMT path (no
dtype cast to BF16/FP16). Fixing this is the highest-ROI optimization available.

**tensor_core_active_pct = null:** Normal for non-GEMM kernels (elementwise, reductions)
and for Blackwell GPUs where this counter was removed. Not a problem.

**ncu replay timing note:** All durations come from ncu's kernel replay mode and are
2–5× longer than real execution. Use for relative comparison only.
```

### 4. Optimizations Applied

```markdown
## Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype_promotion | aten::linear (all) | tensor_core_active_pct = 0.0 | high | APPLIED |
| OPT-2 | memory_layout | aten::conv2d | convertTensor_kernel ×60 | high | APPLIED |
| OPT-3 | qkv_fusion | aten::linear (0,1,2) | waves_per_sm = 0.3 | medium | NOT APPLIED |
```

Only include optimizations confirmed as APPLIED in the validation log. Mark others as NOT APPLIED with the reason.

### 5. Before/After Results

The normalized comparison table from `/compare`. If `profile_optimized.json` is not available, this section reads "Profiling in progress — run `/capture workload_optimized.py --profile-name=optimized` and then `/compare`."

```markdown
## Results: Before vs. After (Normalized to B=16)

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| aten::conv2d | 1,740,320 | 892,000 | **1.95x** |
| aten::batch_norm | 330,944 | 315,000 | 1.05x |
| **Total** | **2,122,456** | **1,232,000** | **1.72x** |
```

### 6. What Drove Each Speedup

Causal attribution section — one paragraph per applied optimization:

```markdown
## What Drove Each Speedup

**channels_last layout (OPT-2, +0.95x speedup on aten::conv2d):**
Converting the model to `channels_last` memory format eliminated 60 `convertTensor_kernel`
launches per forward pass. cuDNN previously had to coerce from NCHW to NHWC for each
convolution call; with NHWC-native tensors, no coercion is needed. Evidence: the
`convertTensor_kernel` entries are entirely absent from the optimized profile.
```

### 7. Remaining Opportunities

```markdown
## Remaining Opportunities

These optimizations from the proposal were not applied in this iteration:

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-3 | qkv_fusion | aten::linear | Pattern not found in Inductor graph | ~1.3x on linear ops |

Estimated additional gain if OPT-3 is applied: ~15% total throughput improvement.
```

### 8. Reproduction Commands

```markdown
## Reproducing This Analysis

```bash
# 1. Capture baseline profile
/capture examples/conv_block/conv_block.py

# 2. Analyze bottlenecks
/analyze profile.json

# 3. Generate optimization proposals
/propose profile.json

# 4. Generate optimized backend
/backend conv_block.py optimizations.json

# 5. Validate backend
/validate conv_block_optimized.py

# 6. Profile optimized workload
/capture conv_block_optimized.py --profile-name=optimized --compile-backend=conv_block_opt

# 7. Compare results
/compare profile.json profile_optimized.json
```
```

## Anti-Patterns to Avoid

These make reports inaccurate and misleading:

- **Do NOT** credit a transformation that logged `WARNING: Pattern not found` in the validation output — even if the operator shows speedup (Inductor may have optimized it independently)
- **Do NOT** compare raw durations across different batch sizes — always normalize first
- **Do NOT** omit the batch size in the results table — "1.95x speedup" is meaningless without knowing if batch changed
- **Do NOT** claim `tensor_core_active_pct = null` is a bottleneck — it's expected for non-GEMM kernels and on Blackwell
- **Do NOT** state absolute latency values as wall-clock times — ncu replay values are 2–5× longer than real execution; always add the caveat
- **Do NOT** describe future work without clearly labeling it as "not yet implemented" in the optimization table
