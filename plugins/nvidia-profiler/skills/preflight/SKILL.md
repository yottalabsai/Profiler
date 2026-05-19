---
name: preflight
description: Validate the full profiler environment — Python packages, CUDA, nsys, ncu — before running the pipeline. Prints every failure with an exact fix command. Run this before /capture or /optimize on any new machine.
---

# /preflight — Environment Validation

Runs `nvidia/scripts/preflight.py` and reports the status of every dependency the pipeline needs. Call this before starting any profiling work on a new machine or after changing your Python environment.

## Usage

```
/preflight                       # full check (Python + CUDA + nsys + ncu)
/preflight --json /tmp/env.json  # same check + write detected env facts to JSON file
```

## What It Checks

| Check | Required | What fails if missing |
|---|---|---|
| Python ≥ 3.10 | yes | entire pipeline |
| packaging importable | yes | version checks inside preflight crash |
| torch importable | yes | run_workload.py crashes at import |
| CUDA available | yes | no GPU profiling possible |
| torch ≥ 2.0 | yes | torch.compile / FX graph features missing |
| torch._inductor importable | yes | dedup backend crashes mid-run |
| pydantic ≥ 2.0 | yes | schema validation crashes |
| nvidia.operator_profiler importable | yes | PYTHONPATH wrong or package not installed |
| operator_profiler.fx sub-package | yes | dedup backend import fails at runtime |
| operator_profiler.capture sub-package | yes | correlation pass import fails |
| nsys | yes | Stage 0a capture impossible |
| ncu | yes | Stage 0d kernel replay impossible |

## Instructions for Claude

1. Find the project root (directory containing `nvidia/operator_profiler/`).

2. Run the preflight script:
```bash
python3 {project_root}/nvidia/scripts/preflight.py
```

To also capture detected environment facts for use in pipeline commands:
```bash
python3 {project_root}/nvidia/scripts/preflight.py --json /tmp/preflight_env.json
```

3. Parse output: each line is prefixed `OK` or `FAIL`.
   - All `OK` → environment is ready
   - Any `FAIL` → report the check name, error, and fix command to the user; do not proceed with the pipeline until fixed

4. When `--json PATH` is used and all checks pass, the JSON file contains:
```json
{
  "project_root": "/abs/path/to/repo",
  "nsys_path":    "/opt/nvidia/nsight-systems/.../bin/nsys",
  "ncu_path":     "/opt/nvidia/nsight-compute/.../ncu",
  "sudo_required": true,
  "pythonpath":   "/abs/path/to/repo:/home/user/.local/lib/python3.x/site-packages:..."
}
```
The `pythonpath` field is the validated `sys.path` from the preflight run — use it verbatim as `PYTHONPATH=` for all pipeline subprocesses. The JSON file is only written on success (exit code 0); on failure no file is written.

5. If `nvidia.operator_profiler importable` fails, the most common fixes are:
   - Package not installed: `pip install -e .` from project root
   - PYTHONPATH missing: `export PYTHONPATH={project_root}`
   - Wrong project root detected — verify `nvidia/operator_profiler/` exists under it

6. If `CUDA available` fails, preflight cannot help — this is a driver/hardware issue. Confirm GPU is present with `nvidia-smi`.

## When to Use

- **Before first `/capture` or `/optimize` on a new machine** — run preflight
- **After `pip install` changes** — run to verify packages and tooling
- **After a failed pipeline run with import errors** — run to identify the broken dependency
- **In CI** — run `python3 nvidia/scripts/preflight.py` as a setup validation step
