# Operator Profiler — Claude Code Notes

## Session setup

At the start of every session, install required dependencies:

```bash
pip install -r requirements.txt
```

`torch` must be installed separately with the correct CUDA variant for the system's GPU driver.
See https://pytorch.org for the appropriate install command.

---

## GPU profiling tools

### nsys (Nsight Systems)
- Captures CUDA kernel launches and NVTX ranges → `.nsys-rep` file
- Export to SQLite for parsing: `nsys export --type=sqlite --output=<path>.sqlite <path>.nsys-rep`
- Run as a normal user — no elevated privileges needed

### ncu (Nsight Compute)
- Collects hardware performance counters per kernel invocation
- **Requires sudo** on this system — GPU performance counters are restricted to root (`ERR_NVGPUCTRPERM`)
- Always use `ncu_sudo=True` in `KernelProfileConfig`, and pass the full path:
  ```python
  KernelProfileConfig(
      ncu_executable="/opt/nvidia/nsight-compute/2025.4.1/ncu",
      ncu_sudo=True,
      ncu_extra_env={"PYTHONPATH": "/home/ubuntu/Operator-Profiler"},
      ...
  )
  ```
- `ncu_extra_env` is needed because `sudo` drops `PYTHONPATH`

---

## Pipeline overview

1. **nsys capture** — run workload under `nsys profile --trace=cuda,nvtx` via `scripts/run_workload.py`
2. **ManifestBuilder** — parse SQLite export → `MappingManifest` (kernel → NVTX attribution)
3. **AttributionEngine** — convert manifest to `OperatorRecord` list
4. **KernelProfileOrchestrator** — run `ncu --kernel-name <name>` for each unique kernel; match back by invocation order (never by timestamp)
5. **build_profile** — assemble `OperatorAttributedProfile` with aggregated metrics → `profile.json`

Use consistent `--warmup-iters` and `--measure-iters` between the nsys capture and the ncu replay script to avoid invocation count mismatches.

---

## Attribution

Two tiers (provenance sidecar was removed — it did not work reliably; name heuristic tier was also removed):
1. **torch.profiler correlation** (`high` confidence) — kernel matched via `--correlation-pass` CUPTI data
2. **NVTX enclosure** (`medium` confidence) — kernel falls within an `aten::` NVTX range pushed by `emit_nvtx`

Kernels that cannot be matched by either tier are stored in `unattributed_kernels[]`. With the name heuristic removed, Triton fused kernels that previously received `low` confidence attribution now appear in `unattributed_kernels`. Expect higher unattributed rates (~20–40%) for Inductor-compiled models without `--correlation-pass`.

