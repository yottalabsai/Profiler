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
| `--compile-backend=NAME` | none | "Use the named @register_backend instead of the built-in dedup backend. Required for optimized workloads with complex FX passes." |
| `--no-lock-clocks` | clocks locked | "Disable GPU clock locking during capture (passes `--no-lock-clocks` to `run_workload.py` in Stage 0a)." |

## GPU clock locking

The nsys capture phase produces the per-operator **durations** in `profile.json`, and it
runs the workload at the GPU's dynamic boost clock by default — which floats with
temperature/power and differs between two captures. To make baseline-vs-optimized
comparisons fair, `run_workload.py` **locks the GPU clocks for the captured iterations**
(default-on) and resets them afterward. The lock lives in `run_workload.py`, so it applies on
every capture route — the agent's raw `nsys profile` command, `operator-profiler profile`, and
the batch runners — and the context manager guarantees a reset even on exception, timeout, or
Ctrl-C.

By default the target is **probe-and-lock**: a brief synthetic load measures the clock the GPU
actually *sustains* (the power/thermal cap sits below the max boost clock, so locking to max
would leave the clock floating), and the lock is set at/below that value so it is genuinely
held. The probed result is **cached per GPU** in `profiler_output/.gpu_clock_lock.json`, so the
**baseline capture probes once and the optimized capture reuses the identical clock** — both
lock to exactly the same frequency, making their durations directly comparable.

This is best-effort: clock setting uses `sudo -n` (non-interactive), so on a machine without
clock-set permission the capture logs a WARNING and proceeds at dynamic clocks rather than
prompting or failing. Pass `--no-lock-clocks` to disable; `--lock-clocks-freq=max` forces the
old highest-supported behavior; `--lock-clocks-freq=<gr>,<mem>` sets an explicit target and
bypasses the cache (all are `run_workload.py` flags, supplied in Stage 0a). ncu's replay phase
is unaffected — it already self-locks to base clocks and the durations do not come from it.

## Execution

Delegates to: capture-agent

Translate any flags present into their agent instructions above and include them in the human-turn prompt alongside the workload path.

## Output

Produces `profile.json` (or `profile_optimized.json`) at the workload's parent directory.
