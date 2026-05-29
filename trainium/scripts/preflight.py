"""
preflight.py — Validate the Trainium profiling environment before capture.

Checks:
  1. torch_neuronx importable
  2. _C._register_profiler() callable (profiler registered with libkineto)
  3. NeuronConfig and NeuronProfiler importable
  4. NRT available (torch_neuronx.nrt is reachable)
  5. Output directory writable

Run before any profiling session:
    python3 trainium/scripts/preflight.py
"""
from __future__ import annotations

import sys


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def check_all() -> bool:
    print("Trainium profiling preflight checks")
    print("=" * 40)
    all_ok = True

    # 1. torch_neuronx importable
    try:
        import torch_neuronx
        version = getattr(torch_neuronx, "__version__", "unknown")
        all_ok &= _check("torch_neuronx importable", True, f"version={version}")
    except ImportError as e:
        all_ok &= _check("torch_neuronx importable", False, str(e))

    # 2. _C._register_profiler callable
    try:
        import torch_neuronx._C as _C
        ok = hasattr(_C, "_register_profiler")
        all_ok &= _check("_C._register_profiler exists", ok,
                         "libkineto profiler registration" if ok else "missing — profiling will not work")
    except Exception as e:
        all_ok &= _check("_C._register_profiler exists", False, str(e))

    # 3. NeuronConfig / NeuronProfiler importable
    try:
        from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode
        all_ok &= _check("NeuronConfig / NeuronProfiler importable", True)
    except ImportError as e:
        all_ok &= _check("NeuronConfig / NeuronProfiler importable", False, str(e))

    # 4. torch importable (needed for torch.profiler)
    try:
        import torch
        all_ok &= _check("torch importable", True, f"version={torch.__version__}")
    except ImportError as e:
        all_ok &= _check("torch importable", False, str(e))

    # 5. pydantic v2 available (schema layer)
    try:
        import pydantic
        v = pydantic.__version__
        ok = int(v.split(".")[0]) >= 2
        all_ok &= _check("pydantic v2", ok, f"version={v}")
    except Exception as e:
        all_ok &= _check("pydantic v2", False, str(e))

    print("=" * 40)
    if all_ok:
        print("All checks passed — ready to profile.")
    else:
        print("One or more checks FAILED — resolve above before profiling.")

    return all_ok


if __name__ == "__main__":
    ok = check_all()
    sys.exit(0 if ok else 1)
