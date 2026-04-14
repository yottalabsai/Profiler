"""
run_workload.py — Generic nsys capture runner.

Loads a user workload, compiles and warms up the model, then runs it
under emit_nvtx so the nsys trace carries aten:: NVTX ranges alongside
CUDA kernel launches.

Run under nsys:
    nsys profile --trace=cuda,nvtx --output=<prefix> \\
        python scripts/run_workload.py --workload <script.py>

Workload interface
------------------
The workload script must expose:

    def get_model_and_input() -> tuple[model, input_tensor]:
        ...

The returned model should be a plain nn.Module on CUDA. Compilation and
warmup are handled by this tool via --compile-backend and --warmup-iters.

Example:
    nsys profile --trace=cuda,nvtx --output=profile \\
        python scripts/run_workload.py --workload scripts/inductor_workload.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.autograd.profiler as autograd_profiler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_workload(script_path: str):
    path = Path(script_path).resolve()
    if not path.exists():
        print(f"[run_workload] ERROR: workload script not found: {path}", flush=True)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model_and_input"):
        print(
            f"[run_workload] ERROR: {path.name} does not expose "
            "get_model_and_input(). Add:\n\n"
            "    def get_model_and_input():\n"
            "        # return (model, input_tensor)\n",
            flush=True,
        )
        sys.exit(1)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", required=True,
        help="Path to a workload script that exposes get_model_and_input().",
    )
    parser.add_argument(
        "--compile-backend", default="inductor",
        help="torch.compile backend to use (default: inductor). Pass 'none' to skip compilation.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5,
        help="Number of warmup iterations before NVTX capture (default: 5).",
    )
    parser.add_argument(
        "--measure-iters", type=int, default=10,
        help="Number of capture iterations under emit_nvtx (default: 10).",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"[run_workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    workload = _load_workload(args.workload)
    print(f"[run_workload] Loading workload: {args.workload}", flush=True)
    model, x = workload.get_model_and_input()

    if args.compile_backend != "none":
        print(f"[run_workload] Compiling with backend='{args.compile_backend}'...", flush=True)
        model = torch.compile(model, backend=args.compile_backend)

    print(f"[run_workload] Warmup ({args.warmup_iters} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            model(x)
    torch.cuda.synchronize()

    print(f"[run_workload] Capture ({args.measure_iters} iters with emit_nvtx)...", flush=True)
    with torch.no_grad():
        with autograd_profiler.emit_nvtx(record_shapes=True):
            for _ in range(args.measure_iters):
                _ = model(x)
    torch.cuda.synchronize()

    print("[run_workload] Done.", flush=True)


if __name__ == "__main__":
    main()
