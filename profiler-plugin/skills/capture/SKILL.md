---
name: capture
description: Run the full nsys+ncu profiling pipeline on a workload.py file and produce profile.json. Handles executable detection, sudo permissions, PYTHONPATH propagation, and --script-args ordering automatically.
---

# /capture — GPU Profiling Pipeline

Runs the complete profiling pipeline on your `workload.py` and produces `profile.json` with per-operator hardware metrics. You only need to provide the workload file path.

## Usage

```
/capture workload.py
/capture workload.py --compile-backend=none          # eager mode (no torch.compile)
/capture workload.py --ncu-sudo=true                 # force sudo for ncu
/capture workload.py --profile-name=optimized        # produces profile_optimized.json
/capture workload.py --warmup-iters=10 --measure-iters=20
```

## Workload Interface

Your `workload.py` MUST expose:
```python
def get_model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    """Return (uncompiled model on CUDA, input tensor on CUDA)."""
    model = YourModel().to("cuda").eval()
    x = torch.randn(..., device="cuda")
    return model, x
```

The pipeline does not read any other function from your file. The model must be on CUDA, not CPU.

## Pipeline Stages

### Stage 0a: nsys Capture
Runs your workload under NVIDIA Nsight Systems with CUDA + NVTX tracing.

```bash
nsys profile --trace=cuda,nvtx --output=profiler_output/{stem} --force-overwrite=true \
    python nvidia/scripts/run_workload.py \
        --workload workload.py \
        --compile-backend inductor \
        --warmup-iters 5 \
        --measure-iters 10 \
        --correlation-pass
```

Output: `profiler_output/{stem}.nsys-rep`

### Stage 0b: SQLite Export
Exports the binary nsys report to SQLite for programmatic parsing.

```bash
nsys export --type=sqlite --output=profiler_output/{stem}.sqlite profiler_output/{stem}.nsys-rep
```

### Stage 0c: Manifest Build
Parses the SQLite database and joins CUDA kernel launches to NVTX operator ranges.

```bash
PYTHONPATH=/project/root python -m nvidia.operator_profiler profile \
    --workload workload.py \
    --nsys-rep profiler_output/{stem}.nsys-rep \
    --output profiler_output/{stem}.manifest.json
```

Output: `profiler_output/{stem}.manifest.json` (the `MappingManifest`)

### Stage 0d: Kernel Replay with ncu
Replays each unique kernel under Nsight Compute to collect 20 hardware performance counters.

```bash
[sudo -E] ncu --target-processes all \
    python -m nvidia.operator_profiler map \
        --manifest profiler_output/{stem}.manifest.json \
        --output profile.json \
        --ncu-sudo true \
        --ncu-env PYTHONPATH=/project/root \
        --warmup-iters 5 \
        --measure-iters 10 \
        --script-args --workload workload.py --compile-backend inductor
```

Output: `profile.json`

## Automatic System Detection

The capture-agent detects your system configuration automatically:

| Configuration | How Detected |
|---|---|
| nsys executable | `nsys --version`; fallback: scan `/opt/nvidia/nsight-systems/*/bin/nsys` |
| ncu executable | `ncu --version`; fallback: scan `/opt/nvidia/nsight-compute/*/ncu` |
| sudo requirement | Attempt `ncu --version`; if `ERR_NVGPUCTRPERM` → needs sudo |
| PYTHONPATH | `python -c "import sys; print(':'.join(sys.path))"` + project root prepended |
| Project root | Search upward from workload for directory containing `nvidia/operator_profiler/` |

## The --script-args Ordering Rule

`operator-profiler map` uses `--script-args` as a remainder argument. It MUST be the last flag. The capture-agent enforces this automatically, but if you write the command manually:

```bash
# CORRECT — all map flags before --script-args
operator-profiler map manifest.json \
    --ncu-sudo true \
    --ncu-env PYTHONPATH=/repo \
    --script-args --workload workload.py --compile-backend inductor

# WRONG — ncu-env after script-args gets passed to the workload script
operator-profiler map manifest.json \
    --script-args --workload workload.py \
    --ncu-env PYTHONPATH=/repo    # ← silently ignored as map flag
```

## Warmup/Measure Iteration Matching

The `--warmup-iters` and `--measure-iters` in Stage 0a (nsys) MUST match Stage 0d (ncu replay). Mismatching causes kernel count mismatches — ncu collects metrics for the wrong kernel invocations relative to the nsys attribution.

```
Stage 0a: --warmup-iters 5 --measure-iters 10
Stage 0d: --warmup-iters 5 --measure-iters 10   ← must be identical
```

## Permission Issues

| Error | Platform | Solution |
|---|---|---|
| `ERR_NVGPUCTRPERM` | Linux | Re-run with `--ncu-sudo=true` or as root |
| `ERR_NVGPUCTRPERM` | Windows | Restart terminal as Administrator; DevMode GPU profiling must be enabled |
| `Permission denied: /dev/nvidiactl` | Linux | Add user to `video` group: `sudo usermod -aG video $USER` (requires logout) |

## Profiling Optimized Workloads

To profile your optimized workload for comparison:
```
/capture workload_optimized.py --profile-name=optimized --compile-backend={your_backend_name}
```

This produces `profile_optimized.json`. Use `/compare profile.json profile_optimized.json` to measure speedup.

## Success Criteria

The capture is successful when `profile.json`:
1. Exists and is valid JSON
2. Has non-empty `operators[]` array
3. Has `operators[*].aggregated != null` (metrics were collected)
4. Has `unattributed_kernels` count < 50% of total kernel count

If `unattributed_kernels` is high (> 20%), check:
- Was `emit_nvtx()` / `--correlation-pass` active during capture?
- Did torch.compile complete before the measurement window?
- Is the model using `cudagraphs` mode? (Graph replay kernels have different attribution)

## Configuration Options

| Option | Default | Description |
|---|---|---|
| `--compile-backend` | `inductor` | Compile backend: `inductor`, `none` (eager), `cudagraphs`, or any registered custom backend name |
| `--warmup-iters` | `5` | Warmup iterations before NVTX capture window |
| `--measure-iters` | `10` | Measurement iterations within NVTX capture window |
| `--ncu-sudo` | `auto` | `auto` detects, `true` forces sudo, `false` skips |
| `--ncu-path` | `auto` | Explicit path to ncu executable |
| `--nsys-path` | `auto` | Explicit path to nsys executable |
| `--output-dir` | `profiler_output/` | Directory for intermediate files (nsys-rep, sqlite, manifest) |
| `--profile-name` | `baseline` | `baseline` → `profile.json`, `optimized` → `profile_optimized.json` |
