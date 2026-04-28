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
/optimize workload.py --compile-backend=inductor --ncu-sudo=true

# Resume from a specific stage (skips completed stages)
/optimize workload.py --resume --from=analyze
/optimize workload.py --resume --from=backend
/optimize workload.py --resume --from=validate

# Single stage only
/optimize --stage=capture workload.py
/optimize --stage=analyze profile.json
/optimize --stage=compare profile.json profile_optimized.json

# Batch: multiple workloads
/optimize workload_a.py workload_b.py

# Eager mode (no torch.compile)
/optimize workload.py --compile-backend=none
```

## The 8-Stage Pipeline

### Stage 0: Capture (→ profile.json)
Runs nsys+ncu pipeline on `workload.py`. Auto-detects executables, sudo requirement, and PYTHONPATH. Produces `profile.json` with per-operator hardware metrics.

```
Delegates to: capture-agent
Output: profile.json
Skip if: profile.json exists and --resume is set
```

### Stage 1: Analyze (→ triage.json)
Classifies each operator by bottleneck type. Computes wall-time percentages. Flags attribution edge cases.

```
Delegates to: profile-analyzer
Output: triage.json
Skip if: triage.json exists and --resume is set
```

### Stage 2: Propose (→ optimizations.json)
Generates ranked, evidence-backed FX graph transformation proposals. Uses `context7` for current PyTorch API docs and `sequential-thinking` when > 5 operators are above 5% threshold.

```
Delegates to: optimization-strategist
Output: optimizations.json
Skip if: optimizations.json exists and --resume is set
```

### Stage 3: Backend (→ workload_optimized.py)
Generates a production-ready custom `torch.compile()` backend implementing the proposed transformations. Runs syntax check before completing.

```
Delegates to: backend-engineer
Output: {workload}_optimized.py, test_{workload}_optimized.py, OPTIMIZED_WORKLOAD.md
Skip if: {workload}_optimized.py exists and --resume is set
```

### Stage 4: Validate (→ validation_report.json)
Runs the 5-step validation sequence (syntax, import, registration, pytest, smoke test). Reports which FX passes applied vs. degraded gracefully.

```
Delegates to: validation-agent
Output: validation_report.json
Blocks: Stage 5 if any validation step fails
```

### Stage 5: Re-Capture (→ profile_optimized.json)
Runs the same nsys+ncu pipeline on `workload_optimized.py`. Uses the same `--warmup-iters` and `--measure-iters` as Stage 0 to ensure comparable kernel counts.

```
Delegates to: capture-agent (with --profile-name=optimized)
Output: profile_optimized.json
```

### Stage 6: Compare (→ comparison table)
Normalizes for batch-size differences and computes per-operator speedups with hardware counter evidence. Reports residual opportunities.

```
Delegates to: comparison-agent
Output: printed comparison table + comparison.md
```

### Stage 7: Report (→ report.md)
Generates a human-readable summary of the full optimization lifecycle — hardware context, bottleneck classification, transformations applied, measured results, and reproduction commands.

```
Output: report.md
```

## Checkpoint Behavior

Before each stage, the agent checks whether the output artifact from the previous stage exists:

| Artifact | Triggers |
|---|---|
| `profile.json` | Skips Stage 0 when `--resume` is set |
| `triage.json` | Skips Stage 1 |
| `optimizations.json` | Skips Stage 2 |
| `{workload}_optimized.py` | Skips Stage 3 |
| `validation_report.json` | Skips Stage 4 |
| `profile_optimized.json` | Skips Stage 5 |

Without `--resume`, all stages run even if artifacts exist (fresh run).

## Edge Case Handling

| Condition | Action |
|---|---|
| `unattributed_kernels > 10%` | Warn at Stage 1; reduce confidence on all Stage 2 proposals |
| `compile_mode == "eager"` | Skip FX pass generation at Stage 3; propose `torch.compile` migration instead |
| `device_name == null` | Use Ampere A100 limits as fallback; flag in Stage 7 report |
| High duration variance across same operator's `call_index` values | Flag dynamic shapes; disable batch-padding optimization |
| Stage 4 validation fails | Block Stage 5 (do not waste ncu replay time on broken code) |
| `ERR_NVGPUCTRPERM` in Stage 0 or 5 | Halt and provide exact remediation: `--ncu-sudo=true` or elevated prompt |

## MCP Tool Usage by Stage

| Stage | MCP Tool | Purpose |
|---|---|---|
| Stage 2 | `context7` | Fetch PyTorch FX API docs before writing `fx_steps[]` |
| Stage 2 | `exa-search` | Find similar optimization patterns for medium/low confidence transforms |
| Stage 2 | `sequential-thinking` | Multi-operator dependency analysis when > 5 operators above 5% threshold |
| Stage 1 & 2 | `memory` | Cache profile analysis; retrieve for similar models seen before |

## Configuration Options

| Option | Default | Applies To |
|---|---|---|
| `--compile-backend` | `inductor` | Stages 0, 3, 5 |
| `--warmup-iters` | `5` | Stages 0, 5 (must match) |
| `--measure-iters` | `10` | Stages 0, 5 (must match) |
| `--ncu-sudo` | `auto` | Stages 0, 5 |
| `--ncu-path` | `auto` | Stages 0, 5 |
| `--nsys-path` | `auto` | Stages 0, 5 |
| `--confidence-threshold` | `medium` | Stage 3 (only implement opts at or above this level) |
| `--max-optimizations` | `10` | Stage 2 |
| `--skip-validation` | `false` | Skips Stage 4 (not recommended) |
| `--resume` | `false` | Skip stages with existing artifacts |
| `--from` | `capture` | Which stage to resume from |
| `--audience` | `team` | Report audience for Stage 7 |

## Progress Output

The agent prints progress at each stage boundary:

```
[optimize] Stage 0/7: Capturing baseline profile...
[optimize]   → nsys: /opt/nvidia/.../nsys (v2025.3)
[optimize]   → ncu:  /opt/nvidia/.../ncu (v2025.4.1, sudo=true)
[optimize]   ✓ profile.json: 12 operators, 162 kernels

[optimize] Stage 1/7: Analyzing bottlenecks...
[optimize]   ✓ triage.json: top bottleneck = tensor_core_idle on aten::linear (42.3%)

[optimize] Stage 2/7: Generating optimization proposals...
[optimize]   ✓ optimizations.json: 3 proposals (2 high, 1 medium)

[optimize] Stage 3/7: Generating custom backend...
[optimize]   ✓ conv_block_optimized.py: syntax OK

[optimize] Stage 4/7: Validating backend...
[optimize]   ✓ All 5 validation steps passed

[optimize] Stage 5/7: Capturing optimized profile...
[optimize]   ✓ profile_optimized.json: 10 operators, 102 kernels

[optimize] Stage 6/7: Comparing results...
[optimize]   ✓ Overall speedup: 1.87x (102 → 54 kernel launches)

[optimize] Stage 7/7: Generating report...
[optimize]   ✓ report.md written

[optimize] Complete. See report.md for full results.
```
