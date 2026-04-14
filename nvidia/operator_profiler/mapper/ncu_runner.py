"""
ncu subprocess wrapper.

Provides two entry points:
  1. run_kernel_profile()  — collect metrics for one kernel name across the workload
  2. import_ncu_report()   — export an existing .ncu-rep to CSV

Kernel-name based profiling
----------------------------
Instead of NVTX range filtering (which requires a shared libnvToolsExt.so that
recent PyTorch builds no longer expose), we filter by kernel name using ncu's
--kernel-name flag.  This works regardless of how the workload links NVTX and
handles both eager and compiled (Triton) workloads:

  - Eager mode:   kernel names are stable cuBLAS/cuDNN identifiers.
  - Compiled mode: Inductor-generated Triton kernels have unique names per fused
                   operation (e.g. triton_per_fused_addmm_relu_0), so one kernel
                   name == one fused unit — no grouping needed.

Edge case #8: ncu timestamps are NEVER used for attribution ordering;
only metric values are extracted from ncu output.
"""
from __future__ import annotations

import logging
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from nvidia.operator_profiler.schema.metrics import AGGREGATE_NCU_METRICS
from nvidia.operator_profiler.utils.subprocess_utils import run_subprocess

log = logging.getLogger(__name__)


@dataclass
class NcuKernelProfileConfig:
    """Configuration for a single ncu kernel-name-filtered profile run."""
    script: str | Path                 # Python script to replay
    script_args: list[str] = field(default_factory=list)
    kernel_name_filter: str = ""       # Exact name or regex passed to --kernel-name
    # ncu_metric_set takes precedence over metrics when non-empty.
    # Leave empty (default) to use AGGREGATE_NCU_METRICS via --metrics.
    # Pass a named set ("full", "default", "roofline", "basic") to override.
    ncu_metric_set: str = ""
    metrics: list[str] = field(default_factory=lambda: list(AGGREGATE_NCU_METRICS))
    output_path: str | Path = ""       # .ncu-rep output path
    ncu_executable: str = "ncu"
    extra_ncu_args: list[str] = field(default_factory=list)
    # Prefix the ncu command with "sudo -E" to gain GPU counter access when
    # the system restricts profiling to root (ERR_NVGPUCTRPERM).
    use_sudo: bool = False
    # Extra environment variables forwarded to the ncu subprocess.
    extra_env: dict[str, str] = field(default_factory=dict)


def run_kernel_profile(config: NcuKernelProfileConfig) -> Path:
    """
    Run ncu with --kernel-name <filter> --replay-mode kernel.

    Profiles every invocation of matching kernels across the full workload
    execution.  Results are written to a .ncu-rep file which is then imported
    via import_ncu_report().

    Returns the path to the .ncu-rep output file.
    """
    output_path = Path(config.output_path)

    script_cmd: list[str] = []
    script_path = Path(config.script)
    if script_path.suffix == ".py":
        script_cmd = [sys.executable, str(script_path)]
    else:
        script_cmd = [str(script_path)]

    if config.ncu_metric_set:
        metric_args = ["--set", config.ncu_metric_set]
    else:
        metric_args = ["--metrics", ",".join(config.metrics)]

    ncu_cmd = [
        config.ncu_executable,
        "--replay-mode", "kernel",
        *metric_args,
        "--export", str(output_path),
        "--force-overwrite",
        *config.extra_ncu_args,
    ]
    if config.kernel_name_filter:
        ncu_cmd += ["--kernel-name", config.kernel_name_filter]
    ncu_cmd += [*script_cmd, *config.script_args]

    # Prepend sudo -E to preserve the environment when root access is needed
    # for GPU performance counters (ERR_NVGPUCTRPERM).
    cmd = (["sudo", "-E"] + ncu_cmd) if config.use_sudo else ncu_cmd

    log.info(
        "Running ncu kernel profile for '%s': %s",
        config.kernel_name_filter or "(all kernels)",
        shlex.join(cmd),
    )
    run_subprocess(
        cmd,
        description=f"ncu kernel profile '{config.kernel_name_filter}'",
        extra_env=config.extra_env or None,
    )
    return output_path


def import_ncu_report(ncu_rep_path: str | Path, ncu_executable: str = "ncu") -> str:
    """
    Run `ncu --import <file> --csv` and return the raw CSV text.

    This is the only way to read ncu metric values — we never parse the
    binary .ncu-rep format directly.
    """
    ncu_rep_path = Path(ncu_rep_path)
    cmd = [ncu_executable, "--import", str(ncu_rep_path), "--csv"]
    log.info("Importing ncu report: %s", ncu_rep_path)
    result = run_subprocess(cmd, description="ncu --import --csv", capture_output=True)
    return result.stdout
