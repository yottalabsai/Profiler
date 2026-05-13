---
name: preflight
description: Validate the full profiler environment — Python packages, CUDA, nsys, ncu — before running the pipeline. Prints every failure with an exact fix command. Run this before /capture or /optimize on any new machine.
---

# /preflight — Environment Validation

Runs `nvidia/scripts/preflight.py` and reports the status of every dependency the pipeline needs. Call this before starting any profiling work on a new machine or after changing your Python environment.

## Usage

```
/preflight                  # full check (Python + CUDA + nsys + ncu)
/preflight --no-tools       # Python + CUDA only (skip nsys/ncu, faster)
```

## What It Checks

| Check | Required | What fails if missing |
|---|---|---|
| Python ≥ 3.10 | yes | entire pipeline |
| torch importable | yes | run_workload.py crashes at import |
| CUDA available | yes | no GPU profiling possible |
| torch ≥ 2.0 | yes | torch.compile / FX graph features missing |
| torch._inductor importable | yes | dedup backend crashes mid-run |
| pydantic ≥ 2.0 | yes | schema validation crashes |
| nvidia.operator_profiler importable | yes | PYTHONPATH wrong or package not installed |
| operator_profiler.fx sub-package | yes | dedup backend import fails at runtime |
| operator_profiler.capture sub-package | yes | correlation pass import fails |
| rich | optional | no formatted output, everything else works |
| nsys | yes (full) | Stage 0a capture impossible |
| ncu | yes (full) | Stage 0d kernel replay impossible |

## Instructions for Claude

1. Find the project root (directory containing `nvidia/operator_profiler/`).

2. Run the preflight script:
```bash
python3 {project_root}/nvidia/scripts/preflight.py
```
For Python-only check (no nsys/ncu):
```bash
python3 {project_root}/nvidia/scripts/preflight.py --no-tools
```

3. Parse output: each line is prefixed `OK`, `WARN`, or `FAIL`.
   - All `OK` → environment is ready
   - Any `FAIL` → report the check name, error, and fix command to the user; do not proceed with the pipeline until fixed
   - `WARN` on `rich` only → safe to continue, suggest `pip install 'rich>=13.0'`

4. If `nvidia.operator_profiler importable` fails, the most common fixes are:
   - Package not installed: `pip install -e .` from project root
   - PYTHONPATH missing: `export PYTHONPATH={project_root}`
   - Wrong project root detected — verify `nvidia/operator_profiler/` exists under it

5. If `CUDA available` fails, preflight cannot help — this is a driver/hardware issue. Confirm GPU is present with `nvidia-smi`.

## Fix Commands by Check

| Check | Fix |
|---|---|
| torch importable | `pip install torch` — see https://pytorch.org for CUDA variant |
| CUDA available | Install CUDA-enabled torch wheel; verify with `nvidia-smi` |
| pydantic ≥ 2.0 | `pip install 'pydantic>=2.0'` |
| nvidia.operator_profiler | `pip install -e .` from project root, or `export PYTHONPATH={project_root}` |
| rich | `pip install 'rich>=13.0'` |
| nsys | `sudo apt install nsight-systems` or download from developer.nvidia.com |
| ncu | `sudo apt install nsight-compute` or download from developer.nvidia.com |

## When to Use

- **Before first `/capture` or `/optimize` on a new machine** — run full preflight
- **After `pip install` changes** — run `--no-tools` to verify packages
- **After a failed pipeline run with import errors** — run full preflight to identify the broken dependency
- **In CI** — run `python3 nvidia/scripts/preflight.py` as a setup validation step
