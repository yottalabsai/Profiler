---
name: optimize
description: End-to-end GPU optimization workflow. Pass workload.py and the plugin runs all stages automatically — nsys+ncu capture, bottleneck analysis, optimization proposals, FX backend generation, validation, re-profiling, comparison, and report. Can also resume from any checkpoint or run individual stages.
---

# /optimize — End-to-End Optimization Workflow

The single command that drives the complete GPU optimization pipeline. Pass your `workload.py` and get back a production-ready optimized backend with a measured speedup report.

## Usage

```bash
# Full pipeline from scratch (most common)
/optimize workload.py

# Full pipeline with explicit options
/optimize workload.py --ncu-sudo=true

# Resume from a specific stage (skips completed stages)
/optimize workload.py --resume --from=propose
/optimize workload.py --resume --from=backend
/optimize workload.py --resume --from=validate

# Single stage only
/optimize --stage=capture workload.py
/optimize --stage=propose profile.json
/optimize --stage=report

# Batch: multiple workloads
/optimize workload_a.py workload_b.py

# Eager mode (no torch.compile)
/optimize workload.py --compile-backend=none
```

## Pre-Flight Check

Before starting any stage, run `/preflight` to validate the full environment. If any required check fails, report the failure and fix to the user before proceeding. The preflight output is authoritative — do not attempt to work around failures by adjusting commands.

```
/preflight
```

## The 7-Stage Pipeline

### Stage 0: Capture (→ profile.json)
Runs nsys+ncu pipeline on `workload.py`. Auto-detects executables, sudo requirement, and PYTHONPATH. Produces `profile.json` with per-operator hardware metrics.

Two-phase capture to avoid CUPTI conflict (nsys and torch.profiler cannot share CUPTI simultaneously):
- Phase A (no nsys): `run_workload.py --correlation-pass` → `.corr.json` + `.part.json`
- Phase B (under nsys): `run_workload.py` (no `--correlation-pass`) → `.nsys-rep`

Layer deduplication is unconditional when using the built-in backend: the FX graph is always split by detected layer structure; only unique representatives are compiled; structural duplicates share the same compiled callable. `profiler_output/{stem}.part.json` is written in Phase A; pass it to ncu replay via `--partition-map` to skip replaying duplicate-partition kernels.

```
Delegates to: capture-agent
Output: profile.json, profiler_output/{stem}.part.json (when built-in backend used)
Skip if: profile.json exists and --resume is set
```

### Stage 1: Propose (→ optimizations.json)
Reads `profile.json` directly. Derives time budget, edge case flags, and architecture context, then generates ranked, evidence-backed FX graph transformation proposals using open-ended reasoning from raw hardware counters.

```
Delegates to: optimization-strategist
Output: optimizations.json
Skip if: optimizations.json exists and --resume is set
```

### Stage 2: Backend (→ workload_optimized.py)
Generates a production-ready custom `torch.compile()` backend implementing the proposed transformations. Runs syntax check before completing.

```
Delegates to: backend-engineer
Output: {workload}_optimized.py, test_{workload}_optimized.py, OPTIMIZED_WORKLOAD.md
Skip if: {workload}_optimized.py exists and --resume is set
```

### Stage 3: Validate (→ validation_report.json)
Runs the 5-step validation sequence (syntax, import, registration, pytest, smoke test). Reports which FX passes applied vs. degraded gracefully.

```
Delegates to: validation-agent
Output: validation_report.json
Blocks: Stage 4 if any validation step fails
```

### Stage 4: Re-Capture (→ profile_optimized.json)
Runs the same nsys+ncu pipeline on `workload_optimized.py`. Uses the same `--warmup-iters` and `--measure-iters` as Stage 0 to ensure comparable kernel counts.

```
Delegates to: capture-agent (with --profile-name=optimized)
Output: profile_optimized.json
```

### Stage 4.5: Cleanup intermediary files
After both `profile.json` and `profile_optimized.json` are verified, remove Inductor debug directories from `profiler_output/`. These directories (named `{stem}_inductor_debug/`) contain Triton-compiled artifacts used only for kernel attribution during the pipeline; they are not needed afterward and can be large.

```bash
rm -rf profiler_output/*_inductor_debug/
```

Do not remove other files in `profiler_output/` (`.nsys-rep`, `.ncu-rep`, `.corr.json`, `.part.json`) — they are useful for debugging and re-running individual stages without a full re-capture.

```
Output: (none — cleanup only)
Skip if: neither inductor_debug directory exists
```

### Stage 5: Report (→ report.md)
Generates a human-readable summary of the full optimization lifecycle — hardware context, bottleneck classification, transformations applied, measured speedup with hardware counter evidence, residual opportunities, and reproduction commands.

```
Output: report.md
```

## Checkpoint Behavior

Before each stage, the agent checks whether the output artifact from the previous stage exists:

| Artifact | Triggers |
|---|---|
| `profile.json` | Skips Stage 0 when `--resume` is set |
| `optimizations.json` | Skips Stage 1 |
| `{workload}_optimized.py` | Skips Stage 2 |
| `validation_report.json` | Skips Stage 3 |
| `profile_optimized.json` | Skips Stage 4 (re-capture) |

Without `--resume`, all stages run even if artifacts exist (fresh run).

## Edge Case Handling

| Condition | Action |
|---|---|
| `unattributed_kernels > 10%` | Warn at Stage 1; reduce confidence on all proposals |
| `compile_mode == "eager"` | Skip FX pass generation at Stage 2; propose `torch.compile` migration instead |
| `device_name == null` | Use Ampere A100 limits as fallback; flag in Stage 6 report |
| High duration variance across same operator's `call_index` values | Flag dynamic shapes; disable batch-padding optimization |
| Stage 3 validation fails | Block Stage 4 (do not waste ncu replay time on broken code) |
| `ERR_NVGPUCTRPERM` in Stage 0 or 4 | Halt and provide exact remediation: `--ncu-sudo=true` or elevated prompt |

## MCP Tool Usage by Stage

| Stage | MCP Tool | Purpose |
|---|---|---|
| Stage 1 | `context7` | Fetch PyTorch FX API docs before writing `fx_steps[]` |
| Stage 1 | `exa-search` | Find similar optimization patterns for medium/low confidence transforms |
| Stage 1 | `sequential-thinking` | Multi-operator dependency analysis when > 5 operators above 5% threshold |
| Stage 1 | `memory` | Cache profile analysis; retrieve for similar models seen before |

## Configuration Options

| Option | Default | Applies To |
|---|---|---|
| `--compile-backend` | *(none)* | Stages 0, 4 — named `@register_backend` backend for optimized workloads. Omit for baseline (uses built-in dedup+inductor backend). Required at Stage 5 when the optimized workload has complex FX passes (SDPA, BN fold, pre-transposed weights). |
| `--warmup-iters` | `2` | Stages 0, 4 (must match) |
| `--measure-iters` | `2` | Stages 0, 4 (must match) |
| `--ncu-sudo` | `auto` | Stages 0, 4 |
| `--ncu-path` | `auto` | Stages 0, 4 |
| `--nsys-path` | `auto` | Stages 0, 4 |
| `--confidence-threshold` | `medium` | Stage 2 (only implement opts at or above this level) |
| `--max-optimizations` | `10` | Stage 1 |
| `--skip-validation` | `false` | Skips Stage 3 (not recommended) |
| `--resume` | `false` | Skip stages with existing artifacts |
| `--from` | `capture` | Which stage to resume from |
| `--audience` | `team` | Report audience for Stage 5 |

## Progress Output

The agent prints progress at each stage boundary:

```
[optimize] Stage 0/5: Capturing baseline profile...
[optimize]   → nsys: /opt/nvidia/.../nsys (v2025.3)
[optimize]   → ncu:  /opt/nvidia/.../ncu (v2025.4.1, sudo=true)
[optimize]   ✓ profile.json: 12 operators, 162 kernels

[optimize] Stage 1/5: Generating optimization proposals...
[optimize]   ✓ optimizations.json: 3 proposals (2 high, 1 medium)

[optimize] Stage 2/5: Generating custom backend...
[optimize]   ✓ conv_block_optimized.py: syntax OK

[optimize] Stage 3/5: Validating backend...
[optimize]   ✓ All 5 validation steps passed

[optimize] Stage 4/5: Capturing optimized profile...
[optimize]   ✓ profile_optimized.json: 10 operators, 102 kernels

[optimize] Stage 5/5: Generating report...
[optimize]   ✓ report.md written (overall speedup: 1.87x, 102 → 54 kernel launches)

[optimize] Complete. See report.md for full results.
```
