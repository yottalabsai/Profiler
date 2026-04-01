"""
operator-profiler map <manifest.json>

Runs ncu range replays for each NVTX range in the manifest and emits the
Operator-Attributed Profile JSON.

Usage:
    operator-profiler map profile.manifest.json --script model.py --output profile.json
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "map",
        help="Run ncu range replays and produce the operator-attributed profile.",
    )
    p.add_argument("manifest", help="Path to mapping_manifest.json")
    p.add_argument("--script", required=True, help="Replay script (same as capture)")
    p.add_argument("--script-args", nargs="*", default=[])
    p.add_argument("--output", default="profile.json", help="Output profile JSON path")
    p.add_argument("--ncu-executable", default="ncu")
    p.add_argument("--model-name", default="model")
    p.add_argument("--torch-version", default=None)
    p.add_argument("--device-name", default=None)
    p.set_defaults(func=_run)


def _run(args) -> None:
    from operator_profiler.schema.manifest import MappingManifest
    from operator_profiler.mapper.attribution_engine import AttributionEngine
    from operator_profiler.mapper.range_replay import RangeReplayConfig, RangeReplayOrchestrator
    from operator_profiler.aggregator.profile_builder import build_profile

    manifest_path = Path(args.manifest)
    manifest = MappingManifest.model_validate_json(manifest_path.read_text())

    # Run attribution
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    # Run range replays
    torch_version = args.torch_version
    if torch_version is None:
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "unknown"

    replay_config = RangeReplayConfig(
        replay_script=args.script,
        replay_script_args=args.script_args or [],
        ncu_executable=args.ncu_executable,
        expected_input_shapes=manifest.capture_metadata.input_shapes,
    )
    orch = RangeReplayOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    # Assemble profile
    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=args.model_name,
        torch_version=torch_version,
        device_name=args.device_name,
        ncu_report_path=str(ncu_output_dir),
    )

    output_path = Path(args.output)
    output_path.write_text(profile.model_dump_json(indent=2))
    log.info("Operator-attributed profile → %s", output_path)
    print(f"Profile written to: {output_path}")
