"""
ncu subprocess wrapper.

Provides two entry points:
  1. run_range_replay()   — collect metrics for one NVTX range
  2. import_ncu_report()  — export an existing .ncu-rep to CSV

Edge case #8: ncu timestamps are NEVER used for attribution ordering;
only metric values are extracted from ncu output.
"""
from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from operator_profiler.schema.metrics import DEFAULT_NCU_METRICS
from operator_profiler.utils.subprocess_utils import run_subprocess

log = logging.getLogger(__name__)


@dataclass
class NcuRangeReplayConfig:
    """Configuration for a single ncu range-replay run."""
    script: str | Path            # Python script to replay
    script_args: list[str] = field(default_factory=list)
    nvtx_include: str = ""        # NVTX range text to filter (glob)
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    replay_mode: str = "range"    # "range" or "kernel"
    output_path: str | Path = ""  # .ncu-rep output path
    ncu_executable: str = "ncu"
    extra_ncu_args: list[str] = field(default_factory=list)


def run_range_replay(config: NcuRangeReplayConfig) -> Path:
    """
    Run ncu with --nvtx --nvtx-include <range> --replay-mode range.

    Returns the path to the .ncu-rep output file.

    --replay-mode range preserves cache state across the fused kernel sequence,
    which is critical for fused kernel metrics (see architecture plan §4).
    """
    output_path = Path(config.output_path)

    metrics_arg = ",".join(config.metrics)
    cmd = [
        config.ncu_executable,
        "--nvtx",
        "--nvtx-include", config.nvtx_include,
        "--replay-mode", config.replay_mode,
        "--metrics", metrics_arg,
        "--export", str(output_path),
        "--force-overwrite",
        *config.extra_ncu_args,
        str(config.script),
        *config.script_args,
    ]
    log.info(
        "Running ncu range replay for range '%s': %s",
        config.nvtx_include,
        shlex.join(cmd),
    )
    run_subprocess(cmd, description=f"ncu range replay '{config.nvtx_include}'")
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
