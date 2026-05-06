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

Layer deduplication (--layer-deduplicate)
-----------------------------------------
When --layer-deduplicate is set, the script uses a custom torch.compile backend
(_make_dedup_backend) that receives the dynamo-traced FX graph and:
  1. Builds a UniqueSubgraphRegistry — splits the graph by layer and groups
     structurally identical partitions by graph signature.
  2. Optionally applies replace_pattern FX passes loaded from --fx-pass-module
     to unique subgraphs only, propagating edits to all structural duplicates.
  3. Compiles each unique-representative partition with inductor (Triton kernels).
     Duplicate partitions share the same compiled callable as their representative.
  4. Wraps each partition's compiled forward with an NVTX range:
       layer::unique::LABEL   — for unique-representative partitions
       layer::duplicate::LABEL — for structural duplicates
  5. Writes a .part.json sidecar mapping duplicate partition labels to their
     unique representative (used by KernelProfileConfig.partition_equivalence_map).

The NVTX partition ranges sit outside the aten:: ranges from emit_nvtx and are
transparent to ManifestBuilder's operator attribution logic.  ManifestBuilder
reads them via _tag_layer_partitions() and sets KernelManifestEntry.layer_partition
and .is_unique_partition.  KernelProfileOrchestrator then only sends
unique-partition kernels to ncu and propagates metrics to duplicates.

Pass file interface (--fx-pass-module):
    def get_passes() -> list[tuple[callable, callable]]:
        # each tuple is (pattern_fn, replacement_fn) for replace_pattern
        return [(pattern, replacement)]

Example:
    nsys profile --trace=cuda,nvtx --output=profile \\
        python scripts/run_workload.py --workload scripts/inductor_workload.py \\
            --layer-deduplicate --fx-pass-module my_passes.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.autograd.profiler as autograd_profiler
import torch.fx as fx

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root, contains nvidia/
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


def _load_fx_passes(script_path: str) -> list[tuple[Callable, Callable]]:
    """Load (pattern, replacement) pairs from a pass file exposing get_passes()."""
    path = Path(script_path).resolve()
    if not path.exists():
        print(f"[run_workload] ERROR: fx-pass-module not found: {path}", flush=True)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_fx_passes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_passes"):
        print(
            f"[run_workload] ERROR: {path.name} does not expose get_passes(). Add:\n\n"
            "    def get_passes():\n"
            "        return [(pattern_fn, replacement_fn), ...]\n",
            flush=True,
        )
        sys.exit(1)
    passes = module.get_passes()
    print(f"[run_workload] Loaded {len(passes)} FX pass(es) from {path.name}", flush=True)
    return passes


def _wrap_nvtx(submod: fx.GraphModule, nvtx_text: str) -> None:
    """Replace submod.forward with a closure that pushes/pops an NVTX range."""
    original_forward = submod.forward

    def _wrapped(*args, **kwargs):
        torch.cuda.nvtx.range_push(nvtx_text)
        try:
            return original_forward(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    submod.forward = _wrapped


def _capture_partition_inputs(registry, example_inputs: list) -> dict:
    """
    Run one interpreted forward pass through registry.split, recording the
    input tensors received by each partition submodule.

    These per-partition inputs are passed to compile_fx so inductor can compile
    each unique subgraph with concrete example shapes.
    """
    captured: dict = {}
    saved: dict = {}

    for name, submod in registry.split.named_children():
        if isinstance(submod, fx.GraphModule):
            saved[name] = submod.forward

            def _make_capturing(n: str, orig: Callable) -> Callable:
                def _fwd(*args, **kwargs):
                    captured[n] = list(args)
                    return orig(*args, **kwargs)
                return _fwd

            submod.forward = _make_capturing(name, submod.forward)

    with torch.no_grad():
        registry.split(*example_inputs)

    for name, submod in registry.split.named_children():
        if isinstance(submod, fx.GraphModule):
            submod.forward = saved[name]

    return captured


def _make_dedup_backend(
    passes: list[tuple[Callable, Callable]],
    output_prefix: str,
) -> tuple[Callable, dict]:
    """
    Return (backend, state) for integrating layer deduplication with inductor.

    backend — pass to torch.compile(model, backend=backend); invoked once by
              dynamo with the traced FX graph.
    state   — populated by the backend with state['run_fn']: a zero-argument
              callable that runs registry.split() directly (bypassing dynamo).

    Callers should use state['run_fn'] for warmup and nsys capture rather than
    re-calling the compiled model, to avoid dynamo re-tracing under emit_nvtx.
    """
    from torch._inductor.compile_fx import compile_fx
    from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

    _state: dict = {}

    def _backend(gm: fx.GraphModule, example_inputs: list) -> Callable:
        # Step 1: Build unique subgraph registry
        registry = UniqueSubgraphRegistry(gm)
        n_total = sum(
            1 for _, submod in registry.split.named_children()
            if isinstance(submod, fx.GraphModule)
        )
        n_unique = len(registry.unique_reps)
        print(
            f"[run_workload] Registry: {n_total} partition(s), "
            f"{n_unique} unique signature(s)",
            flush=True,
        )

        # Step 2: Apply FX passes to unique reps (propagated to duplicates)
        if passes:
            runner = FxPassRunner(registry)
            for i, (pattern, replacement) in enumerate(passes):
                n = runner.apply_pass(pattern, replacement)
                print(f"[run_workload] FX pass {i + 1}: {n} replacement(s)", flush=True)

        # Step 3: Capture per-partition example inputs, then compile each unique
        # representative with inductor. Duplicate partitions share the same
        # compiled callable as their representative.
        print("[run_workload] Capturing per-partition inputs...", flush=True)
        partition_inputs = _capture_partition_inputs(registry, example_inputs)

        print(
            f"[run_workload] Compiling {n_unique} unique partition(s) with inductor...",
            flush=True,
        )
        for rep_name, rep_mod in registry.unique_reps:
            compiled_fn = compile_fx(rep_mod, partition_inputs.get(rep_name, []))
            rep_mod.forward = compiled_fn
            for _dup_name, dup_mod in registry.duplicates_of(rep_name):
                dup_mod.forward = compiled_fn
        print("[run_workload] Inductor compilation done.", flush=True)

        # Step 4: Wrap each partition's compiled forward with an NVTX range
        for name, submod in registry.split.named_children():
            if isinstance(submod, fx.GraphModule):
                label  = registry.partition_label(name)
                prefix = "unique" if registry.is_unique_rep(name) else "duplicate"
                _wrap_nvtx(submod, f"layer::{prefix}::{label}")

        # Step 5: Write partition equivalence map sidecar
        equiv_map = registry.build_partition_equivalence_map()
        part_path = Path(output_prefix + ".part.json")
        part_path.write_text(json.dumps(equiv_map, indent=2))
        print(
            f"[run_workload] Partition equivalence map → {part_path} "
            f"({len(equiv_map)} duplicate(s))",
            flush=True,
        )

        # _state['run_fn'] is zero-arg for direct warmup/capture calls (no dynamo).
        # The returned callable accepts *args so dynamo can invoke it normally.
        _state['run_fn'] = lambda: registry.split(*example_inputs)
        return lambda *args: registry.split(*args)

    return _backend, _state


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
            "Directory where Inductor debug artifacts will be written during "
            "torch.compile().  Enables torch._inductor.config.debug and sets "
            "TORCHINDUCTOR_CACHE_DIR to this path so hash-named compiled .py "
            "files land in a predictable location.  Pass this path to "
            "parse_inductor_debug_dir() after the nsys run to build a fusion map."
        ),
    )
    parser.add_argument(
        "--layer-deduplicate", action="store_true", default=False,
        help=(
            "Capture the FX graph, split by layer, and run only unique-representative "
            "partitions under full nsys annotation.  Duplicate partitions receive "
            "layer::duplicate::LABEL NVTX ranges so ManifestBuilder can tag them.  "
            "Writes a .part.json equivalence-map sidecar for use with "
            "KernelProfileConfig.partition_equivalence_map.  Works for any model "
            "architecture without requiring --fx-pass-module."
        ),
    )
    parser.add_argument(
        "--fx-pass-module", default=None,
        help=(
            "Path to a Python file exposing get_passes() -> list[(pattern, replacement)]. "
            "Applies replace_pattern passes to unique subgraphs only and propagates "
            "edits to all structural duplicates before nsys capture.  "
            "Implies --layer-deduplicate."
        ),
    )
    args = parser.parse_args()

    # --fx-pass-module implies --layer-deduplicate
    if args.fx_pass_module and not args.layer_deduplicate:
        args.layer_deduplicate = True

    assert torch.cuda.is_available(), "CUDA required"
    print(f"[run_workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    workload = _load_workload(args.workload)
    print(f"[run_workload] Loading workload: {args.workload}", flush=True)
    original_model, x = workload.get_model_and_input()

    # ------------------------------------------------------------------
    # Layer-deduplication path
    # ------------------------------------------------------------------
    if args.layer_deduplicate:
        passes = _load_fx_passes(args.fx_pass_module) if args.fx_pass_module else []
        dedup_backend, _state = _make_dedup_backend(passes, args.output_prefix)

        # One torch.compile call triggers the backend: captures the FX graph,
        # compiles unique partitions with inductor, wraps NVTX, writes .part.json.
        print("[run_workload] Compiling with dedup+inductor backend...", flush=True)
        _compiled = torch.compile(original_model, backend=dedup_backend)
        with torch.no_grad():
            _compiled(x)

        # Use registry.split directly for warmup and capture so that emit_nvtx
        # does not trigger dynamo re-tracing (which would recompile partitions
        # and fire Triton JIT inside the unique-partition NVTX window).
        run_fn = _state['run_fn']

        print(f"[run_workload] Warmup ({args.warmup_iters} iters)...", flush=True)
        with torch.no_grad():
            for _ in range(args.warmup_iters):
                run_fn()
        torch.cuda.synchronize()

        if args.correlation_pass:
            from nvidia.operator_profiler.capture.torch_profiler_correlator import build_correlation_map
            print(
                f"[run_workload] torch.profiler correlation pass ({args.measure_iters} iters)...",
                flush=True,
            )
            corr_map = build_correlation_map(run_fn, n_iters=args.measure_iters)
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

        print(
            f"[run_workload] Capture ({args.measure_iters} iters with NVTX)...",
            flush=True,
        )
        with torch.no_grad():
            with autograd_profiler.emit_nvtx(record_shapes=True):
                for _ in range(args.measure_iters):
                    run_fn()
        torch.cuda.synchronize()
        print("[run_workload] Done.", flush=True)
        return

    # ------------------------------------------------------------------
    # Standard path (no deduplication)
    # ------------------------------------------------------------------
    model = original_model

    if args.compile_backend != "none":
        if args.inductor_debug_dir:
            import os
            import torch._inductor.config as _ind_cfg
            debug_dir = Path(args.inductor_debug_dir).resolve()
            debug_dir.mkdir(parents=True, exist_ok=True)
            _ind_cfg.debug = True
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(debug_dir)
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
