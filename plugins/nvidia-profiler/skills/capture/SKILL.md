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

### Stage 0a-pre: Correlation Pass (standalone, before nsys)
Runs the torch.profiler correlation pass as a plain Python invocation — **not inside nsys**. nsys and torch.profiler both use CUPTI and cannot run simultaneously; running `--correlation-pass` inside nsys produces a 0-entry sidecar and inflates the nsys kernel count, causing a mismatch with ncu.

Also passes `--inductor-debug-dir` so Inductor writes its compiled `.py` artifacts to a known location during compilation. These are used in Stage 0c to attribute Triton fused kernels.

```bash
PYTHONPATH={project_root} python nvidia/scripts/run_workload.py \
    --workload workload.py \
    --compile-backend inductor \
    --warmup-iters 5 \
    --measure-iters 10 \
    --correlation-pass \
    --output-prefix profiler_output/{stem} \
    --inductor-debug-dir profiler_output/{stem}_inductor_debug
```

Output: `profiler_output/{stem}.corr.json`, Inductor compiled artifacts in `profiler_output/{stem}_inductor_debug/`

### Stage 0a: nsys Capture (WITHOUT --correlation-pass)
Runs your workload under NVIDIA Nsight Systems with CUDA + NVTX tracing. The correlation pass already ran in Stage 0a-pre; do not include `--correlation-pass` here. Pass the same `--inductor-debug-dir` so Inductor reuses the cached compilation and produces the same kernel names in the trace.

```bash
PYTHONPATH={project_root} nsys profile --trace=cuda,nvtx --output=profiler_output/{stem} --force-overwrite=true \
    python nvidia/scripts/run_workload.py \
        --workload workload.py \
        --compile-backend inductor \
        --warmup-iters 5 \
        --measure-iters 10 \
        --inductor-debug-dir profiler_output/{stem}_inductor_debug
```

Output: `profiler_output/{stem}.nsys-rep`

### Stage 0b: SQLite Export
Exports the binary nsys report to SQLite for programmatic parsing.

```bash
nsys export --type=sqlite --output=profiler_output/{stem}.sqlite profiler_output/{stem}.nsys-rep
```

### Stage 0c: Manifest Build
Parses the SQLite database and joins CUDA kernel launches to NVTX operator ranges. Auto-detects the GPU device name, loads the correlation map from Stage 0a-pre for HIGH-confidence attribution, and applies the Inductor fusion map to attribute Triton fused kernels that bypass torch.profiler correlation.

```bash
PYTHONPATH={project_root}:{full_sys_path} operator-profiler manifest \
    --nsys-rep profiler_output/{stem}.nsys-rep \
    --output profiler_output/{stem}.manifest.json \
    --model-name {model_name} \
    --compile-backend {compile_backend} \
    --corr-json profiler_output/{stem}.corr.json \
    --inductor-fusion-dir profiler_output/{stem}_inductor_debug
```

`--inductor-fusion-dir` parses the Inductor-compiled `.py` files (hash-named, not `output_code.py`) to extract `{kernel_name → [aten::ops]}` mappings. Upgrades all UNATTRIBUTED Triton fused kernels to MEDIUM confidence. Omit when `--compile-backend=none`.

Output: `profiler_output/{stem}.manifest.json` (the `MappingManifest`, with `device_name` populated)

### Stage 0d: Kernel Replay with ncu
Replays each unique kernel under Nsight Compute to collect 20 hardware performance counters. Always pass `--ncu-output-dir profiler_output/ncu_reps/` so `.ncu-rep` files are written to a persistent location, not `/tmp/`.

```bash
PYTHONPATH={project_root} operator-profiler map \
    profiler_output/{stem}.manifest.json \
    --script {project_root}/nvidia/scripts/run_workload.py \
    --ncu-executable {ncu_path} \
    --ncu-sudo \
    --ncu-env "PYTHONPATH={project_root}:{full_sys_path}" \
    --ncu-output-dir profiler_output/ncu_reps/ \
    --model-name {model_name} \
    --output profile.json \
    --script-args --workload workload.py --compile-backend inductor --warmup-iters 5 --measure-iters 10 --inductor-debug-dir profiler_output/{stem}_inductor_debug
```

`--inductor-debug-dir` in `--script-args` ensures the ncu replay reuses the same cached Inductor compilation (same `TORCHINDUCTOR_CACHE_DIR`). Without it, Inductor may recompile with differently-ordered kernels, silently corrupting the kernel→invocation mapping.

Output: `profile.json`, `.ncu-rep` files in `profiler_output/ncu_reps/`

## Automatic System Detection

The capture-agent detects your system configuration automatically:

| Configuration | How Detected |
|---|---|
| nsys executable | `nsys --version`; fallback: scan `/opt/nvidia/nsight-systems/*/bin/nsys` |
| ncu executable | `ncu --version`; fallback: scan `/opt/nvidia/nsight-compute/*/ncu` |
| sudo requirement | Read `/proc/driver/nvidia/params`; if `RmProfilingAdminOnly: 1` → needs sudo |
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
4. Has `unattributed_kernels` count < 60% of total kernel count

**Expected unattributed rates:** With the Inductor fusion map enabled, unattributed rates should be near 0% for Inductor-compiled models. Triton fused kernels (which bypass torch.profiler correlation) are attributed via ground-truth `# Original ATen: [...]` comments in the Inductor-compiled `.py` files.

If `unattributed_kernels` exceeds 10%, check:
- Was `--inductor-debug-dir` passed to all three stages (0a-pre, 0a, 0d `--script-args`)? (required for Triton kernel attribution)
- Was `--inductor-fusion-dir` passed to `operator-profiler manifest`?
- Was `--correlation-pass` used during capture? (provides HIGH confidence for cuBLAS kernels)
- Were NVTX ranges emitted? (`--trace=cuda,nvtx` must be set in nsys flags)
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
| `--inductor-debug-dir` | `auto` | Directory for Inductor compiled artifacts. Auto-set to `profiler_output/{stem}_inductor_debug/` when `compile_backend=inductor`. Passed to all three stages (0a-pre, 0a, 0d) for consistent compilation caching and Triton kernel attribution. |
