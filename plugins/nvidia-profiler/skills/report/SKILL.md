---
name: report
description: Generate a human-readable report.md summarizing the complete optimization lifecycle — from baseline hardware metrics to implemented transformations to measured speedups. Suitable for sharing with teams, publishing as documentation, or including in PR descriptions.
---

# /report — Optimization Report Generation

Generates `report.md` documenting the full optimization lifecycle. The report is technically precise and evidence-based — each claim is backed by hardware counter data.

## Usage

```
/report
```

## Inputs (reads from current directory)

- `profile.json` — baseline hardware metrics (required)
- `optimizations.json` — proposed optimizations with evidence (required)
- `profile_optimized.json` — optimized hardware metrics (optional; Section 6 skipped if absent)
- `profiler_output/validation_report.json` — pass application status from `/validate` (optional; Section 4 statuses marked UNKNOWN if absent)
- `profiler_output/implementation_notes.md` — backend design rationale from `/backend` (required; stop if missing)

Missing optional files are noted as "not yet available" and those sections are skipped. If `profiler_output/implementation_notes.md` is missing, stop and tell the user to run `/backend` first.

## Report Structure

Include these sections:

### 1. Hardware Context

Open the report with a one-sentence headline: "This optimization achieved X× total speedup on [model] (B=[N], [GPU])." This ensures the document is scannable without requiring hardware expertise.

Key-value table. Fields to include: GPU model (with SM count), architecture family (Ampere / Hopper / Blackwell), PyTorch version, compile mode, batch size, iteration count with the note "(ncu replay — relative timing only)".

### 2. Operator Summary

Table columns: `Operator`, `Time (%)`, `Duration (ns)`, `Kernels`, `Bottleneck Class`. Sort by `Time (%)` descending.

On Blackwell (B100/B200), `tensor_core_active_pct` is unavailable; Bottleneck Class is derived from memory throughput % and achieved occupancy instead.

### 3. Reading the Metrics

Explain only metrics that appear in this workload's profile and drive the identified
bottlenecks. Give thresholds and interpretations that make the value actionable —
not textbook definitions.

One non-obvious rule: **tensor_core_active_pct = 0.0 (not null)** means the GEMM ran
on the FP32 SIMT path with Tensor Cores completely idle — the highest-ROI optimization
signal available. A null value is expected for non-GEMM kernels and on architectures
where the counter was removed; do not flag it as a problem.

### 4. Optimizations Applied

Read `validation_report.json → passes[].status`. Table columns: `ID`, `Type`, `Target Operators`, `Hardware Evidence`, `Confidence` (all from `optimizations.json`); `Status` from `validation_report.json → passes[].status`.

If `validation_report.json` is absent, mark all statuses as UNKNOWN and add: "Run `/validate` to determine which passes were applied."

### 5. Implementation Notes

Read `implementation_notes.md` and paste its full raw text directly under this heading — every line, as-is, with no wrapper prose, no introduction sentence, no italicized preamble. The file's own content is the section body. Do not add any text before or after it.

### 6. Before/After Results

If `profile_optimized.json` is not available, this section reads "Profiling in progress — run `/capture workload_optimized.py --profile-name=optimized` to generate it."

When both profiles are present:

**Before comparison**, verify both profiles share the same batch size (`capture_metadata.batch_size`). If they differ, skip this section and add: "Batch size mismatch between baseline and optimized captures — comparison skipped. Re-capture with matching batch sizes."

**Cross-session check.** Compute a cross-session flag from the two profiles' `capture_metadata`: it is true when EITHER `device_name` differs, OR the gap between the two `capture_timestamp_utc` values exceeds **6 hours**. **If the flag is false, add nothing and proceed normally.** If it is true, do NOT skip the section — the numbers are still meaningful, just less certain — and prepend this caveat block to Section 6:

> ⚠️ **Cross-session capture.** Baseline and optimized profiles were captured {Xh apart | on different GPUs (A vs B)}. GPU clock state (boost/thermal) can differ between sessions, so small speedups may reflect clock variation rather than the optimization. For a clean comparison, re-capture both profiles back-to-back in one session with locked clocks (`nvidia-smi -lgc`).

If either `capture_timestamp_utc` is missing or unparseable, fall back to the `device_name` check alone and note that the timestamp gap was unavailable.

**Step A — Operator matching**

Match operators across profiles by `operator_name` (NOT `operator_id` — it changes between captures). When N baseline entries collapse to 1 optimized entry due to fusion, sum the baseline durations and compare the sum to the single optimized entry. Report a single row labeled e.g. `aten::linear (fused ×3)` with combined baseline, optimized, and speedup.

**Step B — Speedup attribution**

Read `validation_report.json → passes[].status`. A speedup is attributed to a transformation only when all three hold:
1. `status == APPLIED` in `validation_report.json`
2. The corresponding hardware metric changed in the expected direction
3. The operator containing those kernels shows speedup

If a pass has `status == NOT_APPLIED` or `status == FAILED`, that pass did NOT contribute to any speedup — even if the operator shows improvement (Inductor's own optimization may be responsible).

If `validation_report.json` is absent, mark all attribution as "Validation results unavailable — speedup source unconfirmed."

**Step C — Residual opportunity detection**

After measuring the optimized profile, re-rank operators by `total_duration_ns` and classify their new bottlenecks (the optimization typically exposes second-order bottlenecks). Cross-reference with unapplied proposals in `optimizations.json` and estimate residual gain using their `estimated_impact` discounted by confidence.

Table columns: `Operator`, `Baseline (ns)`, `Optimized (ns)`, `Speedup`. Include a **Total** row.

### 7. What Drove Each Speedup

One paragraph per applied optimization. Header: `**{description} ({OPT-N}, +{X}x on {operator}):**` followed by one sentence explaining the mechanism and one sentence citing the hardware evidence (which counter changed, or which kernel appeared/disappeared).

### 8. Remaining Opportunities

If all proposed optimizations were applied: "All proposed optimizations were applied. No further FX-level gains identified in this profile."

Otherwise, table columns: `ID`, `Type`, `Target`, `Reason Not Applied` (from `validation_report.json → passes[].detail`), `Projected Gain` (from `optimizations.json → estimated_impact`). End with one sentence estimating total additional gain if all remaining passes were applied.

## Anti-Patterns to Avoid

These make reports inaccurate and misleading:

- **Do NOT** credit a transformation that has `status != APPLIED` in `validation_report.json` — even if the operator shows speedup (Inductor may have optimized it independently)
- **Do NOT** omit the batch size in the results table — "1.95x speedup" is meaningless without it
- **Do NOT** claim `tensor_core_active_pct = null` is a bottleneck — it's expected for non-GEMM kernels and on Blackwell
- **Do NOT** state absolute latency values as wall-clock times — ncu replay values are 2–5× longer than real execution; always add the caveat
- **Do NOT** describe future work without clearly labeling it as "not yet implemented" in the optimization table
- **Do NOT** report a small speedup as a definitive win while the cross-session caveat is active — it may be clock variation, not the optimization

## Post-Generation

`profiler_output/implementation_notes.md` is retained after report generation — do not delete it.

## Reference Output

See `examples/conv_block/report.md` for a complete reference output. Use it only as a formatting fallback — the structured instructions above take precedence.
