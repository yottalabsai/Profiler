---
name: optimize
description: End-to-end GPU optimization workflow. Pass workload.py and the plugin runs all stages automatically — nsys+ncu capture, bottleneck analysis, optimization proposals, FX backend generation, validation, re-profiling, comparison, and report. Can also resume from any checkpoint or run individual stages.
---

# /optimize — End-to-End Optimization Workflow

The single command that drives the complete GPU optimization pipeline. Pass your `workload.py` and get back a production-ready optimized backend with a measured speedup report.

## Usage

```bash
# Full pipeline
/optimize workload.py

# Start from a specific stage
/optimize workload.py --from=validate
```

## Pre-Flight Check

Before starting any stage, run `/preflight` to validate the full environment. If any required check fails, report the failure to the user and fix it before proceeding. The preflight output is authoritative — do not attempt to work around failures by adjusting commands.

```
/preflight
```

## Pre-Capture: torch.compile Check

Before Stage 0, read `workload.py` and scan for `torch.compile`. If not found, halt immediately:

> "workload.py does not appear to use `torch.compile`. FX graph passes require compiled execution. Add `model = torch.compile(model, ...)` to `get_model_and_input()`, then re-run `/optimize`."

Do not proceed to capture — an eager-mode profile cannot feed FX pass generation, and the nsys+ncu capture (~30 min) would be wasted.

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
Skip if: --from is set to a later stage
```

### Stage 1: Propose (→ optimizations.json)
Reads `profile.json` directly. Derives time budget, edge case flags, and architecture context, then generates ranked, evidence-backed FX graph transformation proposals using open-ended reasoning from raw hardware counters.

```
Delegates to: optimization-strategist
Output: optimizations.json
Skip if: --from is set to a later stage
```

### Stage 2: Backend (→ workload_optimized.py)
Generates a production-ready custom `torch.compile()` backend implementing the proposed transformations. Runs syntax check before completing.

```
Delegates to: backend-engineer
Output: {workload}_optimized.py, test_{workload}_optimized.py, implementation_notes.md
Skip if: --from is set to a later stage
```

### Stage 3: Validate
Runs the 4-step validation sequence (syntax, import, registration, pytest). The test suite includes a compiled smoke test. Reports which FX passes applied vs. degraded gracefully.

```
Delegates to: validation-agent
Blocks: Stage 4 if any validation step fails
```

### Stage 4: Re-Capture (→ profile_optimized.json)
Runs the same nsys+ncu pipeline on `workload_optimized.py`. Before invoking the capture-agent, scan `workload_optimized.py` for the `@register_backend` decorator to extract the backend name, then pass it as `--compile-backend` so the capture-agent uses the registered backend instead of the built-in dedup backend.

```
Delegates to: capture-agent (with --profile-name=optimized --compile-backend=<auto-detected name>)
Output: profile_optimized.json
```

### Stage 4.5: Cleanup intermediary files
After both `profile.json` and `profile_optimized.json` are verified, remove Inductor debug directories from `profiler_output/`. These directories (named `{stem}_inductor_debug/`) contain Triton-compiled artifacts used only for kernel attribution during the pipeline; they are not needed afterward and can be large. Do not remove other files in `profiler_output/` — they are useful for re-running individual stages without a full re-capture.

### Stage 5: Report (→ report.md)
Generates a human-readable summary of the full optimization lifecycle — hardware context, bottleneck classification, transformations applied, measured speedup with hardware counter evidence, residual opportunities, and reproduction commands.

```
Delegates to: report skill
Output: report.md
```

## Checkpoint Behavior

`--from=<stage>` skips all stages before the specified stage. All stages from `<stage>` onward run unconditionally.

## Edge Case Handling

| Condition | Action |
|---|---|
| `compile_mode == "eager"` in `profile.json` | Skip FX pass generation at Stage 2; warn user (pre-capture check should have caught this) |
| High duration variance across same operator's `call_index` values | Flag dynamic shapes; disable batch-padding optimization |
| Stage 3 validation fails | Block Stage 4; tell user to fix the backend and re-run with `--from=validate` |
| `ERR_NVGPUCTRPERM` in Stage 0 or 4 | Halt and provide exact remediation: run ncu with sudo or from an elevated terminal |

## Configuration Options

| Option | Default | Description |
|---|---|---|
| `--from` | `capture` | Start pipeline from a specific stage (`capture`, `propose`, `backend`, `validate`, `report`) |

## Progress Output

Print a short status line at each stage boundary (start and completion). Exact format is flexible — the key requirement is that the user can see which stage is running and what artifact was produced.
