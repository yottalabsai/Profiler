"""
nsys subprocess wrapper.

Runs the user's script under nsys capture with CUDA + NVTX tracing enabled.
"""
from __future__ import annotations

import logging
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from operator_profiler.utils.subprocess_utils import run_subprocess

log = logging.getLogger(__name__)


@dataclass
class NsysRunConfig:
    """Configuration for an nsys profile run."""
    script: str | Path
    script_args: list[str] = field(default_factory=list)
    output_path: str | Path = "profile"   # .nsys-rep extension appended by nsys
    trace: list[str] = field(default_factory=lambda: ["cuda", "nvtx"])
    nsys_executable: str = "nsys"
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_nsys_args: list[str] = field(default_factory=list)


def run_nsys_profile(config: NsysRunConfig) -> Path:
    """
    Run `nsys profile --trace=cuda,nvtx python <script>` and return the
    path to the resulting .nsys-rep file.
    """
    output_path = Path(config.output_path)
    trace_str = ",".join(config.trace)

    # Build environment string for INDUCTOR_PROVENANCE and INDUCTOR_COMPILE_THREADS
    env_prefix: list[str] = []
    for k, v in config.extra_env.items():
        env_prefix.append(f"{k}={v}")

    cmd = [
        config.nsys_executable,
        "profile",
        f"--trace={trace_str}",
        "--output", str(output_path),
        "--force-overwrite", "true",
        *config.extra_nsys_args,
        sys.executable, str(config.script),
        *config.script_args,
    ]
    log.info("Running nsys: %s", shlex.join(cmd))
    run_subprocess(cmd, description="nsys profile", extra_env=config.extra_env)

    # nsys appends .nsys-rep automatically
    rep_path = output_path.with_suffix(".nsys-rep")
    if not rep_path.exists():
        rep_path = Path(str(output_path) + ".nsys-rep")
    return rep_path
