---
name: validate
description: Systematically validate a generated workload_optimized.py before profiling. Runs 5 steps in order: syntax check, import check, backend registration, pytest suite, compiled smoke test. Reports which FX passes applied vs. degraded gracefully. Saves wasted ncu replay time on broken code.
---

# /validate

## Usage

```
/validate workload_optimized.py
/validate workload_optimized.py --backend-name=my_model_opt
/validate workload_optimized.py --numerical
```

## Flags

| Flag | Default | Agent instruction |
|---|---|---|
| `--backend-name=NAME` | auto-detected | "Use NAME as the backend name for registration and smoke test checks." |
| `--numerical` | false | "After Step 5, run the numerical correctness check with BF16 tolerances (atol=1e-2, rtol=1e-2)." |

## Execution

Delegates to: validation-agent

Translate any flags present into their agent instructions above and include them in the human-turn prompt alongside the file path.
