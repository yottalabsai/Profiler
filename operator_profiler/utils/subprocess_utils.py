"""
Safe subprocess runner with timeout and environment injection.

Used by nsys_runner.py, ncu_runner.py — any module that shells out to
external tools.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 3600  # 1 hour — profiling runs can be long


class SubprocessError(RuntimeError):
    """Raised when a managed subprocess exits with a non-zero return code."""


def run_subprocess(
    cmd: list[str],
    description: str = "",
    extra_env: dict[str, str] | None = None,
    capture_output: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run *cmd* as a subprocess.

    Parameters
    ----------
    cmd:
        Command and arguments (list form — never a shell string).
    description:
        Human-readable name for logging.
    extra_env:
        Additional environment variables merged into the current process env.
    capture_output:
        If True, stdout/stderr are captured and returned in the result.
        If False, they inherit from the parent process (visible in terminal).
    timeout:
        Seconds before subprocess is killed (default: 1 hour).
    cwd:
        Working directory for the subprocess.

    Returns
    -------
    subprocess.CompletedProcess — check .stdout / .stderr when capture_output=True.

    Raises
    ------
    SubprocessError if the process exits with a non-zero return code.
    FileNotFoundError if the executable is not found on PATH.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    label = description or cmd[0]
    log.debug("Running %s: %s", label, " ".join(cmd))

    kwargs: dict[str, Any] = {
        "env": env,
        "timeout": timeout,
        "cwd": cwd,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True

    try:
        result = subprocess.run(cmd, check=False, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Executable not found: '{cmd[0]}'. "
            f"Is it installed and on PATH?  (command: {' '.join(cmd)})"
        )
    except subprocess.TimeoutExpired:
        raise SubprocessError(
            f"{label} timed out after {timeout}s. "
            f"Increase timeout or check for a hanging process."
        )

    if result.returncode != 0:
        stderr_snippet = ""
        if capture_output and result.stderr:
            stderr_snippet = f"\nstderr:\n{result.stderr[:2000]}"
        raise SubprocessError(
            f"{label} exited with code {result.returncode}.{stderr_snippet}"
        )

    return result
