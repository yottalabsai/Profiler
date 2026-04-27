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
import json
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
    parser.add_argument(
        "--correlation-pass", action="store_true", default=False,
        help=(
            "Run a torch.profiler pass after warmup to build a HIGH-confidence "
            "kernel→op attribution map.  Writes <output-prefix>.corr.json alongside "
            "the nsys report.  n_iters matches --measure-iters."
        ),
    )
    parser.add_argument(
        "--output-prefix", default="profile",
        help="Prefix used to name sidecar files (default: 'profile').",
    )
    parser.add_argument(
        "--inductor-debug-dir", default=None,
        help=(
            "Directory where Inductor debug artifacts (output_code.py) will be "
            "written during torch.compile().  Enables torch._inductor.config.debug "
            "and redirects TORCHINDUCTOR_CACHE_DIR to this path so the artifacts "
            "are in a predictable location.  Pass this path to "
            "parse_inductor_debug_dir() after the nsys run to build a fusion map."
        ),
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"[run_workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    workload = _load_workload(args.workload)
    print(f"[run_workload] Loading workload: {args.workload}", flush=True)
    model, x = workload.get_model_and_input()

    if args.compile_backend != "none":
        if args.inductor_debug_dir:
            import torch._inductor.config as _ind_cfg
            debug_dir = Path(args.inductor_debug_dir).resolve()
            debug_dir.mkdir(parents=True, exist_ok=True)
            _ind_cfg.trace.enabled = True
            _ind_cfg.trace.debug_dir = str(debug_dir)
            print(f"[run_workload] Inductor debug artifacts → {debug_dir}", flush=True)
        print(f"[run_workload] Compiling with backend='{args.compile_backend}'...", flush=True)
        model = torch.compile(model, backend=args.compile_backend)

    print(f"[run_workload] Warmup ({args.warmup_iters} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            model(x)
    torch.cuda.synchronize()

    if args.correlation_pass:
        from nvidia.operator_profiler.capture.torch_profiler_correlator import build_correlation_map
        print(
            f"[run_workload] torch.profiler correlation pass ({args.measure_iters} iters)...",
            flush=True,
        )
        corr_map = build_correlation_map(lambda: model(x), n_iters=args.measure_iters)
        corr_path = Path(args.output_prefix + ".corr.json")
        corr_path.write_text(json.dumps({
            "schema_version": "1.0",
            "entries": [
                {"kernel_name": k[0], "invocation": k[1], "op_name": v}
                for k, v in corr_map.items()
            ],
        }))
        print(
            f"[run_workload] Correlation map → {corr_path} ({len(corr_map)} entries)",
            flush=True,
        )
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
