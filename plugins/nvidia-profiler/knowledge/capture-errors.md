# Capture Pipeline Error Reference

Error lookup table for the nsys+ncu pipeline. Each row maps a stderr pattern to its cause and remediation.

| Error in stderr | Meaning | Action |
|---|---|---|
| `ERR_NVGPUCTRPERM` | GPU counter access denied | Re-run with `--ncu-sudo`; on Windows, restart terminal with admin privileges |
| `ModuleNotFoundError: nvidia` | PYTHONPATH missing in nsys subprocess | Fix PYTHONPATH — run Step 6 import validation before retrying |
| `ModuleNotFoundError: torch` | User-local site-packages missing from ncu env | Add `~/.local/lib/python3.x/site-packages` to `--ncu-env PYTHONPATH=` |
| `no such table: NVTX_EVENTS` | nsys was run on raw workload (not run_workload.py) | Re-run Stage 0a with `run_workload.py`, not the raw workload file |
| `Kernel count mismatch` | warmup/measure-iters differ between Stage 0a and Stage 0d | Re-run both stages with matching `--warmup-iters` and `--measure-iters` |
| `metrics.raw is empty` | `--script-args` was not last flag | Rebuild command with `--script-args` strictly at the end |
| Many `metrics.raw` dicts empty despite matching kernel names | `TORCHINDUCTOR_CACHE_DIR` not forwarded to ncu (pre-fix) — ncu recompiled to different cache, kernel names shifted | Should not occur with current orchestrator; if seen, verify `_ncu_env()` is being called in both `_profile_one` and `_profile_all` |
| `ncu: command not found` | ncu not in PATH | Use full path: `/opt/nvidia/nsight-compute/*/ncu` |
| `operator-profiler: command not found` | Package not installed | Run `pip install .` from project root (requires `pyproject.toml` with `where = ["nvidia"]`) |
| Duplicate partition metrics identical to zero | `--partition-map` not passed but `.part.json` exists | Pass `--partition-map {output_dir}/{workload_stem}.part.json` to `operator-profiler map` |
