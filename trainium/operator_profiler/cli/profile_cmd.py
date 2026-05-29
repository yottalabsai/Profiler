"""
trainium-profiler profile <script> [args]

End-to-end command: runs NeuronConfig capture, builds manifest, runs attribution,
aggregates metrics, and writes profile.json.

Unlike the NVIDIA pipeline (which has two separate commands — `profile` then `map`),
Trainium collapses both into one step because NRT captures attribution + hardware
metrics in a single execution.

Usage:
    python -m trainium.operator_profiler profile model.py \\
        --model-name MyModel --output profile.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "profile",
        help="Run NeuronConfig capture and produce an operator-attributed profile.",
    )
    p.add_argument("script", help="Python script to profile")
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    p.add_argument("--model-name", default="model")
    p.add_argument("--output", default="profile.json", help="Output profile JSON path")
    p.add_argument(
        "--profile-output-dir",
        default="/tmp/trainium_profile",
        help="Directory for NRT trace files (ntrace.pb, .ntff, trace.json)",
    )
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--measure-iters", type=int, default=1)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["DEVICE", "RUNTIME"],
        choices=["DEVICE", "RUNTIME", "CPU_UTIL", "HOST_MEMORY"],
        help="NRT ProfileMode(s) to enable",
    )
    p.add_argument("--neuroncore-indices", nargs="+", type=int, default=None)
    p.add_argument("--device-name", default=None)
    p.set_defaults(func=_run)


def _run(args) -> None:
    import importlib.util

    from trainium.operator_profiler.capture.neuron_capture import run_capture
    from trainium.operator_profiler.mapper.attribution_engine import AttributionEngine
    from trainium.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from trainium.operator_profiler.aggregator.profile_builder import build_profile
    from trainium.operator_profiler.schema.manifest import CaptureManifestMetadata

    # Load workload script and get model + input
    spec = importlib.util.spec_from_file_location("_workload", args.script)
    workload_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload_mod)

    model, inputs = workload_mod.get_model_and_input()

    def workload_fn() -> None:
        model(inputs)

    profile_output_dir = Path(args.profile_output_dir)

    # Capture
    capture_result = run_capture(
        workload_fn=workload_fn,
        profile_output_dir=profile_output_dir,
        model_name=args.model_name,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        modes=args.modes,
        neuroncore_indices=args.neuroncore_indices,
    )

    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unknown"

    metadata = CaptureManifestMetadata(
        model_name=args.model_name,
        torch_version=torch_version,
        compile_mode="neuron",
        nrt_session_dir=str(capture_result.nrt_session_dir),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device_name=args.device_name or capture_result.device_name,
    )

    # Build manifest
    builder = ManifestBuilder(
        trace_json_path=capture_result.trace_json_path,
        nrt_session_dir=capture_result.nrt_session_dir,
        metadata=metadata,
    )
    manifest = builder.build()

    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest → %s", manifest_path)

    # Attribution
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    # Assemble profile
    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=args.model_name,
        torch_version=torch_version,
        neuron_sdk_version=capture_result.neuron_sdk_version,
        device_name=args.device_name or capture_result.device_name,
        nrt_session_dir=str(capture_result.nrt_session_dir),
    )

    output_path = Path(args.output)
    output_path.write_text(profile.model_dump_json(indent=2))
    log.info("Profile → %s", output_path)
    print(f"Profile written to: {output_path}")
