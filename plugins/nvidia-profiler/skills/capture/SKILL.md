---
name: capture
description: Run the full nsys+ncu profiling pipeline on a workload.py file and produce profile.json. Handles executable detection, sudo permissions, PYTHONPATH propagation, and --script-args ordering automatically.
---

# /capture

## Usage

```
/capture workload.py
/capture workload.py --ncu-sudo=true
/capture workload.py --profile-name=optimized
/capture workload.py --warmup-iters=5 --measure-iters=5
/capture workload_optimized.py --compile-backend=my_model_opt
```

## Workload Interface

Your `workload.py` must expose:

```python
def get_model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    model = YourModel().to("cuda").eval()
    x = torch.randn(..., device="cuda")
    return model, x
```

The model and input must be on CUDA. The pipeline reads no other function from your file.

## Flags

| Flag | Default | Agent instruction |
|---|---|---|
| `--ncu-sudo=true/false` | auto | "Force sudo on/off for ncu; skip auto-detection." |
| `--profile-name=NAME` | baseline | "Write output to profile_{NAME}.json instead of profile.json." |
| `--warmup-iters=N` | 2 | "Use N warmup iterations in all pipeline stages." |
| `--measure-iters=N` | 2 | "Use N measurement iterations in all pipeline stages." |
| `--compile-backend=NAME` | none | "Use the named @register_backend instead of the built-in dedup backend. Required for optimized workloads with complex FX passes." |

## Execution

Delegates to: capture-agent

Translate any flags present into their agent instructions above and include them in the human-turn prompt alongside the workload path.
