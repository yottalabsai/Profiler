# Operator Profiler — Developer Notes

## Session setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

`torch` must be installed separately with the correct CUDA variant for the system's GPU driver.
See https://pytorch.org for the appropriate install command.

Run the preflight check to validate the environment before any pipeline work:

```bash
python3 nvidia/scripts/preflight.py
```

The package is importable as `nvidia.operator_profiler` after `pip install -e .` or with `PYTHONPATH=$(pwd)` set.

---

## Repository layout

```
nvidia/
  operator_profiler/      # main package (installed by pyproject.toml)
    aggregator/           # metric aggregation, profile assembly
    capture/              # nsys runner, torch.profiler correlator, inductor extractor
    cli/                  # operator-profiler CLI (subcommands: profile, map, manifest)
    fx/                   # UniqueSubgraphRegistry, FxPassRunner
    mapper/               # ManifestBuilder, AttributionEngine, KernelProfileOrchestrator
    schema/               # Pydantic models: manifest, metrics, profile
    utils/
  scripts/                # run_workload.py, preflight.py, verify_ncu_pipeline.py
  tests/
    unit/
    integration/          # require CUDA + nsys/ncu; excluded by default (-m 'not integration')
examples/                 # per-model workload and generated optimized scripts
```

---

## GPU profiling tools

### nsys (Nsight Systems)
- Captures CUDA kernel launches and NVTX ranges → `.nsys-rep` file
- Run as a normal user — no elevated privileges needed
- SQLite export is handled internally; do not run `nsys export` manually

### ncu (Nsight Compute)
- Collects hardware performance counters for every kernel in one workload sweep
- **Requires sudo** on this system — GPU performance counters are restricted to root (`ERR_NVGPUCTRPERM`)
- Uses **application-mode replay** (`--replay-mode application`): ncu replays the entire workload once per counter group (~4–8 passes), collecting counters for all kernels in a single invocation. This replaced the old per-kernel-name subprocess loop.
- **NVTX range mode was intentionally abandoned**: PyTorch 2.x statically links NVTX v3 inside `libtorch_cuda.so`, so ncu's injection mechanism never sees any NVTX events — range mode silently captures nothing.

When constructing `KernelProfileConfig` for the orchestrator:

```python
from nvidia.operator_profiler.mapper.kernel_profiler import KernelProfileConfig

config = KernelProfileConfig(
    replay_script="nvidia/scripts/run_workload.py",
    replay_script_args=["--workload", "my_workload.py"],
    ncu_executable="/opt/nvidia/nsight-compute/2025.4.1/ncu",
    ncu_sudo=True,
    ncu_extra_env={"PYTHONPATH": "/root/Profiler"},
)
```

`ncu_extra_env` is required because `sudoers env_reset` drops `PYTHONPATH`. The runner uses `sudo env KEY=VAL ... ncu` to force variables through regardless of the sudoers `env_keep` list. The lower-level `NcuKernelProfileConfig` (internal to `ncu_runner.py`) uses different field names (`use_sudo`, `extra_env`) — always use `KernelProfileConfig` at the call site.

---

## Pipeline overview

nsys and torch.profiler both consume CUPTI and cannot run simultaneously. The capture is split into two explicit phases.

**Phase 1 — correlation pass** (standalone, before nsys):

```bash
python3 nvidia/scripts/run_workload.py --workload <script.py> \
    --output-prefix <prefix> --inductor-debug-dir <dir> \
    --correlation-pass
```

Compiles the model, warms up, runs torch.profiler, writes:
- `<prefix>.corr.json` — HIGH-confidence kernel→op attribution map
- `<prefix>.part.json` — partition equivalence map (from layer deduplication)

Then exits without NVTX capture.

**Phase 2 — NVTX capture** (under nsys):

```bash
nsys profile --trace=cuda,nvtx --output=<prefix> \
    python3 nvidia/scripts/run_workload.py --workload <script.py> \
        --output-prefix <prefix> --inductor-debug-dir <dir>
```

Reuses the cached Inductor compilation from Phase 1, warms up, then runs `--measure-iters` iterations under `emit_nvtx`.

**Remaining pipeline stages** (all run post-capture):

3. **ManifestBuilder** (`nvidia/operator_profiler/mapper/manifest_builder.py`) — exports `.nsys-rep` to SQLite internally, builds a `MappingManifest` with kernel attribution and (when dedup is active) layer partition tags per kernel.
4. **AttributionEngine** (`nvidia/operator_profiler/mapper/attribution_engine.py`) — converts the manifest into `OperatorRecord` list + `unattributed_kernels`. Skips pre-NVTX warm-up kernels.
5. **KernelProfileOrchestrator** (`nvidia/operator_profiler/mapper/kernel_profiler.py`) — runs one application-mode ncu invocation, fans out `(kernel_name, invocation_index) → metrics` to manifest entries. When `partition_equivalence_map` is set, skips duplicate-partition kernels and propagates their metrics from unique representatives.
6. **build_profile** (`nvidia/operator_profiler/aggregator/profile_builder.py`) — assembles `OperatorAttributedProfile` with aggregated metrics → `profile.json`.

**Critical invariant:** `--warmup-iters` and `--measure-iters` must be identical across Phase 1, Phase 2, and the ncu replay. Kernels are matched by `(kernel_name, invocation_index)` in launch order — never by timestamp. Any iteration-count mismatch shifts the index and corrupts attribution.

---

## Attribution tiers (three, in priority order)

1. **torch.profiler correlation** (`high` confidence) — kernel matched via `--correlation-pass` CUPTI External-id join. Loaded by `ManifestBuilder` from `<prefix>.corr.json` via `correlation_map`. Most reliable; use when attribution quality matters.

2. **NVTX enclosure** (`medium` confidence) — kernel's CPU launch timestamp falls within an `aten::` NVTX range pushed by `emit_nvtx`. Uses the CPU launch timestamp (when `cuLaunchKernel` fired), not the GPU start timestamp, because GPU execution is asynchronous.

3. **Inductor fusion enrichment** (`medium` confidence) — optional post-attribution pass. When `--inductor-debug-dir` is set, `parse_inductor_debug_dir()` reads hash-named `.py` files written by `torch._inductor.config.debug = True` and maps Triton kernel names to their fused aten ops. Upgrades `UNATTRIBUTED` Triton kernels to `INDUCTOR_FUSION`; adds `fused_ops` to already-attributed kernels without changing their confidence.

Kernels that match none of the three tiers land in `unattributed_kernels[]`. Expect 20–40% unattributed for Inductor-compiled models without `--correlation-pass`.

---

## Layer deduplication

For models with repeated structure (e.g. transformer blocks), the pipeline detects structurally identical FX subgraphs, profiles only one representative per equivalence class, and propagates metrics to all duplicates.

**How it works:**

- `run_workload.py` runs layer deduplication unconditionally. `UniqueSubgraphRegistry` splits the FX graph by detected layer structure; unique representatives are compiled with Inductor and duplicates share the compiled callable.
- Each partition's forward pass is wrapped with an NVTX range: `layer::unique::<label>` or `layer::duplicate::<label>`.
- The partition equivalence map (`duplicate_label → unique_label`) is written to `<prefix>.part.json` during the correlation pass.
- `ManifestBuilder._tag_layer_partitions()` reads these NVTX ranges from the nsys trace and tags each kernel with `layer_partition` and `is_unique_partition`. This is a no-op if the trace has no `layer::` ranges.
- Pass `partition_equivalence_map` (loaded from `.part.json`) to `KernelProfileConfig` so the orchestrator skips duplicate-partition kernels during ncu replay and propagates metrics from unique representatives by positional index.

**Failure mode:** If `.part.json` is not passed to the orchestrator, dedup silently degrades — all kernels are profiled and duplicate partitions get independent ncu metrics (equivalent anyway), but profiling takes longer. No incorrect results.

**Why this matters:** GPT-2 (12 identical transformer blocks) goes from ~30–45 min ncu replay time to ~3–5 min.

---

## Running tests

```bash
# Unit tests only (no GPU required)
pytest nvidia/tests/unit/

# Integration tests (require CUDA + nsys/ncu on PATH)
pytest -m integration nvidia/tests/integration/
```
