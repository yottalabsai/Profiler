"""
ncu subprocess wrapper.

Provides two entry points:
  1. run_kernel_profile()  — collect metrics via --replay-mode application
                             (full workload, all kernels in one pass); optionally
                             filtered to one kernel name via --kernel-name
  2. import_ncu_report()   — export an existing .ncu-rep to CSV

Application-mode profiling
---------------------------
ncu replays the entire workload once per counter group (typically 4–8 passes),
collecting hardware counters for all kernels in each pass.  This is far more
efficient than kernel-mode replay for multi-layer models, because the number
of ncu subprocess calls is bounded by the counter-group count rather than the
unique kernel count.

--kernel-name filtering is still supported (and used by _profile_one() for
targeted single-kernel re-profiling), but the primary path passes no filter.

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
    """Configuration for a single ncu profile run."""
    script: str | Path                 # Python script to replay
    script_args: list[str] = field(default_factory=list)
    kernel_name_filter: str | None = None  # Exact name or regex for --kernel-name; None = profile all kernels
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
    Run ncu with --replay-mode application [--kernel-name <filter>].

    Replays the full workload once per counter group, collecting hardware
    counters for all kernels (or only matching kernels when kernel_name_filter
    is set).  Results are written to a .ncu-rep file imported via
    import_ncu_report().

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
        "--replay-mode", "application",
        *metric_args,
        "--export", str(output_path),
        "--force-overwrite",
        *config.extra_ncu_args,
    ]
    if config.kernel_name_filter:
        ncu_cmd += ["--kernel-name", config.kernel_name_filter]
    ncu_cmd += [*script_cmd, *config.script_args]

    # Prepend sudo + explicit env injection when root access is needed.
    # We can't rely on sudo -E alone because sudoers env_reset strips vars
    # like PYTHONPATH even with -E. Using `sudo env KEY=VAL ...` forces them
    # through regardless of the sudoers env_keep list.
    if config.use_sudo:
        env_pairs = [f"{k}={v}" for k, v in (config.extra_env or {}).items()]
        cmd = ["sudo", "env"] + env_pairs + ncu_cmd
    else:
        cmd = ncu_cmd

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
