"""
operator-profiler map <manifest.json>

Runs ncu range replays for each NVTX range in the manifest and emits the
Operator-Attributed Profile JSON.

Usage:
    operator-profiler map profile.manifest.json --script model.py --output profile.json
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--output", default="profile.json", help="Output profile JSON path")
    p.add_argument("--ncu-executable", default="ncu")
    p.add_argument(
        "--ncu-sudo",
        action="store_true",
        default=False,
        help="Prefix ncu with 'sudo -E' (required when GPU perf counters are restricted to root)",
    )
    p.add_argument(
        "--ncu-env",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Extra environment variable forwarded to ncu (e.g. PYTHONPATH=/my/repo). Repeatable.",
    )
    p.add_argument("--model-name", default="model")
    p.add_argument("--torch-version", default=None)
    p.add_argument("--device-name", default=None)
    p.add_argument(
        "--ncu-output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory where per-kernel .ncu-rep files are written. "
            "Defaults to a temp directory under /tmp/ if not set. "
            "Set this to a path inside profiler_output/ so .ncu-rep files "
            "survive the run and can be inspected or re-imported later."
        ),
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    from nvidia.operator_profiler.schema.manifest import MappingManifest
    from nvidia.operator_profiler.mapper.attribution_engine import AttributionEngine
    from nvidia.operator_profiler.mapper.kernel_profiler import KernelProfileConfig, KernelProfileOrchestrator
    from nvidia.operator_profiler.aggregator.profile_builder import build_profile

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

    ncu_extra_env: dict[str, str] = {}
    for pair in args.ncu_env:
        key, _, value = pair.partition("=")
        ncu_extra_env[key] = value

    ncu_output_dir = args.ncu_output_dir
    if ncu_output_dir:
        import os
        os.makedirs(ncu_output_dir, exist_ok=True)

    replay_config = KernelProfileConfig(
        replay_script=args.script,
        replay_script_args=args.script_args or [],
        output_dir=ncu_output_dir or "",
        ncu_executable=args.ncu_executable,
        ncu_sudo=args.ncu_sudo,
        ncu_extra_env=ncu_extra_env,
        expected_input_shapes=manifest.capture_metadata.input_shapes,
    )
    orch = KernelProfileOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    # Resolve device_name: explicit flag wins; fall back to what was captured at
    # manifest-build time (operator-profiler manifest records torch.cuda.get_device_name).
    device_name = args.device_name or manifest.capture_metadata.device_name

    # Assemble profile
    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=args.model_name,
        torch_version=torch_version,
        device_name=device_name,
        ncu_report_path=str(ncu_output_dir),
    )

    output_path = Path(args.output)
    output_path.write_text(profile.model_dump_json(indent=2))
    log.info("Operator-attributed profile → %s", output_path)
    print(f"Profile written to: {output_path}")
