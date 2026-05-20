"""
preflight.py — environment sanity check before any pipeline stage.

Validates Python packages, CUDA availability, and GPU tooling so that
missing dependencies surface immediately with actionable fix commands
rather than as cryptic ImportErrors mid-run.

Standalone usage:
    python3 nvidia/scripts/preflight.py

Programmatic usage (from run_workload.py or other entry points):
    from nvidia.scripts.preflight import check_all
    check_all()        # exits with code 1 on any required failure
"""
from __future__ import annotations

import glob
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


# ── colour helpers (no deps required) ────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if sys.stdout.isatty() else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


# ── individual checks ─────────────────────────────────────────────────────────

class _Check:
    def __init__(
        self,
        name: str,
        fn: Callable[[], str | None],
        fix: str,
        required: bool = True,
    ):
        self.name = name
        self.fn = fn
        self.fix = fix
        self.required = required
        self.detail: str | None = None
        self.passed: bool = False

    def run(self) -> bool:
        try:
            detail = self.fn()
            self.detail = detail
            self.passed = True
        except Exception as exc:
            self.detail = str(exc)
            self.passed = False
        return self.passed


def _check_python_version() -> str:
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} found; 3.10+ required")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def _check_torch() -> str:
    import torch  # noqa: F401
    import torch as _t
    return f"torch {_t.__version__}"


def _check_cuda() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() returned False — "
            "install the CUDA-enabled torch wheel for your driver"
        )
    return f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}"


def _check_torch_version() -> str:
    import torch
    from packaging.version import Version  # type: ignore[import-untyped]
    v = torch.__version__.split("+")[0]  # strip +cu121 suffix
    if Version(v) < Version("2.0"):
        raise RuntimeError(f"torch {v} found; ≥2.0 required")
    return f"torch {torch.__version__} ≥ 2.0 OK"


def _check_pydantic() -> str:
    import pydantic
    from packaging.version import Version  # type: ignore[import-untyped]
    if Version(pydantic.VERSION) < Version("2.0"):
        raise RuntimeError(f"pydantic {pydantic.VERSION} found; ≥2.0 required")
    return f"pydantic {pydantic.VERSION}"


def _check_packaging() -> str:
    import packaging  # noqa: F401
    return "packaging available"


def _check_operator_profiler() -> str:
    import importlib
    spec = importlib.util.find_spec("nvidia.operator_profiler")
    if spec is None:
        raise RuntimeError(
            "nvidia.operator_profiler not importable — "
            "run: pip install -e . (from repo root), or set PYTHONPATH=<repo root>"
        )
    from nvidia.operator_profiler import __version__
    return f"operator_profiler {__version__}"


def _check_operator_profiler_fx() -> str:
    from nvidia.operator_profiler.fx import UniqueSubgraphRegistry  # noqa: F401
    return "fx sub-package OK"


def _check_operator_profiler_capture() -> str:
    from nvidia.operator_profiler.capture.torch_profiler_correlator import build_correlation_map  # noqa: F401
    return "capture sub-package OK"


def _check_inductor() -> str:
    from torch._inductor.compile_fx import compile_fx  # noqa: F401
    return "torch._inductor.compile_fx importable"


def _find_executable(name: str, glob_patterns: list[str]) -> str:
    """Return path to executable or raise."""
    path = shutil.which(name)
    if path:
        return path
    for pattern in glob_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # newest version last
    raise RuntimeError(
        f"'{name}' not found on PATH or in standard install locations "
        f"({', '.join(glob_patterns)})"
    )


def _check_nsys() -> str:
    path = _find_executable("nsys", [
        "/opt/nvidia/nsight-systems/*/bin/nsys",
        "/opt/nvidia/nsight-systems-cli/*/bin/nsys",
        "/usr/local/bin/nsys",
    ])
    result = subprocess.run(
        [path, "--version"], capture_output=True, text=True, timeout=10
    )
    version_line = (result.stdout or result.stderr).strip().splitlines()[0]
    return f"{path} — {version_line}"


def _check_ncu() -> str:
    path = _find_executable(
        "ncu",
        ["/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda/bin/ncu"],
    )
    result = subprocess.run(
        [path, "--version"], capture_output=True, text=True, timeout=10
    )
    version_line = (result.stdout or result.stderr).strip().splitlines()[0]
    return f"{path} — {version_line}"


def _detect_sudo_required() -> bool:
    """Return True if the NVIDIA kernel module requires root for performance counters."""
    try:
        content = Path("/proc/driver/nvidia/params").read_text()
        for line in content.splitlines():
            if line.startswith("RmProfilingAdminOnly:"):
                return line.split(":", 1)[1].strip() == "1"
    except OSError:
        pass
    return False  # Windows or driver not loaded → assume no sudo needed


# ── check registry ────────────────────────────────────────────────────────────

def _build_checks() -> list[_Check]:
    return [
        _Check(
            "Python ≥ 3.10",
            _check_python_version,
            fix="Install Python 3.10 or later.",
        ),
        _Check(
            "packaging",
            _check_packaging,
            fix="pip install packaging",
        ),
        _Check(
            "torch importable",
            _check_torch,
            fix="pip install torch  # or see https://pytorch.org for CUDA-enabled wheel",
        ),
        _Check(
            "CUDA available",
            _check_cuda,
            fix=(
                "Install the CUDA-enabled torch wheel matching your driver:\n"
                "    see https://pytorch.org — select your CUDA version\n"
                "  If torch is installed, verify: python3 -c \"import torch; print(torch.cuda.is_available())\""
            ),
        ),
        _Check(
            "torch ≥ 2.0",
            _check_torch_version,
            fix="pip install 'torch>=2.0'  # or see https://pytorch.org",
        ),
        _Check(
            "torch._inductor importable",
            _check_inductor,
            fix="torch._inductor ships with torch≥2.0 — reinstall torch if this fails.",
        ),
        _Check(
            "pydantic ≥ 2.0",
            _check_pydantic,
            fix="pip install 'pydantic>=2.0'",
        ),
        _Check(
            "nvidia.operator_profiler importable",
            _check_operator_profiler,
            fix=(
                "From the repo root run one of:\n"
                "    pip install -e .\n"
                "    export PYTHONPATH=$(pwd)"
            ),
        ),
        _Check(
            "operator_profiler.fx sub-package",
            _check_operator_profiler_fx,
            fix="pip install -e .  (from repo root)",
        ),
        _Check(
            "operator_profiler.capture sub-package",
            _check_operator_profiler_capture,
            fix="pip install -e .  (from repo root)",
        ),
        _Check(
            "nsys",
            _check_nsys,
            fix=(
                "Install NVIDIA Nsight Systems:\n"
                "    sudo apt install nsight-systems   # or download from developer.nvidia.com"
            ),
        ),
        _Check(
            "ncu",
            _check_ncu,
            fix=(
                "Install NVIDIA Nsight Compute:\n"
                "    sudo apt install nsight-compute   # or download from developer.nvidia.com"
            ),
        ),
    ]


# ── runner ────────────────────────────────────────────────────────────────────

def check_all(label: str = "preflight") -> dict:
    """
    Run all environment checks.

    Args:
        label: Prefix for log lines (e.g. 'run_workload').

    Returns a dict of detected environment facts on success.
    Exits with code 1 if any required check fails.
    """
    checks = _build_checks()
    print(f"[{label}] Environment check ({len(checks)} checks)...", flush=True)

    failures: list[_Check] = []

    for chk in checks:
        chk.run()
        if chk.passed:
            print(f"[{label}]   {_green('OK')}  {chk.name}: {chk.detail}", flush=True)
        else:
            print(f"[{label}]   {_red('FAIL')} {chk.name}: {chk.detail}", flush=True)
            failures.append(chk)

    if failures:
        print(f"\n[{label}] {_red(_bold(f'{len(failures)} required check(s) failed:'))}", flush=True)
        for chk in failures:
            print(f"\n[{label}]   {_bold(chk.name)}", flush=True)
            print(f"[{label}]   Error: {chk.detail}", flush=True)
            print(f"[{label}]   Fix:   {chk.fix}", flush=True)
        print(f"\n[{label}] Aborting — fix the above before running the pipeline.", flush=True)
        sys.exit(1)

    print(f"[{label}] All required checks passed.", flush=True)

    nsys_chk = next(c for c in checks if c.name == "nsys")
    ncu_chk = next(c for c in checks if c.name == "ncu")
    return {
        "project_root": str(Path(__file__).resolve().parent.parent.parent),
        "nsys_path": nsys_chk.detail.split(" — ")[0] if nsys_chk.detail else "",
        "ncu_path": ncu_chk.detail.split(" — ")[0] if ncu_chk.detail else "",
        "sudo_required": _detect_sudo_required(),
        "pythonpath": ":".join(p for p in sys.path if p),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Validate the profiler environment.")
    parser.add_argument(
        "--json", metavar="PATH",
        help="Write detected environment facts as JSON to PATH after all checks pass.",
    )
    args = parser.parse_args()

    env = check_all(label="preflight")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env, indent=2))
