"""
run_workload.py — Trainium capture runner.

Single-phase workflow (no nsys, no correlation-pass split)
----------------------------------------------------------
NRT captures both the operator timeline and hardware metrics in one execution
via nrt_inspect_begin_with_options() / nrt_inspect_stop().  Run this script
directly (no wrapper tool needed):

    python3 trainium/scripts/run_workload.py \\
        --workload <script.py> \\
        --output-prefix <prefix> \\
        --profile-output-dir /tmp/traces

Outputs:
    <prefix>.manifest.json    — MappingManifest (operator→device event map)
    <prefix>.profile.json     — OperatorAttributedProfile (final output)

NRT session directory (containing ntrace.pb, .ntff, trace.json) is written to
--profile-output-dir and the path is embedded in profile.json capture_metadata.

Workload interface
------------------
The workload script must expose:

    def get_model_and_input() -> tuple[nn.Module, torch.Tensor]:
        ...

The returned model should be on XLA (Neuron) device.  Compilation and warm-up
are handled by this script via --warmup-iters.

Example:
    python3 trainium/scripts/run_workload.py \\
        --workload trainium/scripts/workload.py \\
        --output-prefix runs/gpt2 \\
        --model-name gpt2
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trainium profiling capture runner")
    p.add_argument("--workload", required=True, help="Workload script (must expose get_model_and_input)")
    p.add_argument("--output-prefix", default="profile", help="Output path prefix (no extension)")
    p.add_argument("--model-name", default="model")
    p.add_argument("--profile-output-dir", default="/tmp/trainium_profile")
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--measure-iters", type=int, default=1)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["DEVICE", "RUNTIME"],
        choices=["DEVICE", "RUNTIME", "CPU_UTIL", "HOST_MEMORY"],
    )
    p.add_argument("--neuroncore-indices", nargs="+", type=int, default=None)
    p.add_argument("--device-name", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from trainium.scripts.preflight import check_all
    check_all()

    from trainium.operator_profiler.capture.neuron_capture import run_capture
    from trainium.operator_profiler.mapper.attribution_engine import AttributionEngine
    from trainium.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from trainium.operator_profiler.aggregator.profile_builder import build_profile
    from trainium.operator_profiler.schema.manifest import CaptureManifestMetadata

    # Load workload
    spec = importlib.util.spec_from_file_location("_workload", args.workload)
    workload_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload_mod)
    model, inputs = workload_mod.get_model_and_input()

    def workload_fn() -> None:
        model(inputs)

    output_prefix = Path(args.output_prefix)
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

    manifest_path = output_prefix.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest → %s", manifest_path)

    # Attribution + profile assembly
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

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

    profile_path = output_prefix.with_suffix(".profile.json")
    profile_path.write_text(profile.model_dump_json(indent=2))
    log.info("Profile → %s", profile_path)
    print(f"Profile written to: {profile_path}")


if __name__ == "__main__":
    main()
