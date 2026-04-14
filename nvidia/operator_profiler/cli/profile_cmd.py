"""
operator-profiler profile <script> [args]

End-to-end command: runs nsys capture, then builds the mapping manifest.
Does NOT run ncu replay (use `map` for that).

Usage:
    operator-profiler profile model.py --model-name MyModel --output profile.nsys-rep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "profile",
        help="Run nsys + inductor-provenance capture on a Python script.",
    )
    p.add_argument("script", help="Python script to profile")
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    p.add_argument("--model-name", default="model", help="Human-readable model name")
    p.add_argument("--output", default="profile", help="Output path prefix (no extension)")
    p.add_argument("--compile-mode", choices=["eager", "inductor", "cudagraphs"], default="eager")
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--nsys-executable", default="nsys")
    p.set_defaults(func=_run)


def _run(args) -> None:
    from nvidia.operator_profiler.capture.nsys_runner import NsysRunConfig, run_nsys_profile
    from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

    output_prefix = Path(args.output)

    extra_env: dict[str, str] = {}

    nsys_config = NsysRunConfig(
        script=args.script,
        script_args=args.script_args,
        output_path=str(output_prefix),
        nsys_executable=args.nsys_executable,
        extra_env=extra_env,
    )
    rep_path = run_nsys_profile(nsys_config)
    log.info("nsys report: %s", rep_path)

    # Build manifest
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unknown"

    metadata = CaptureManifestMetadata(
        model_name=args.model_name,
        torch_version=torch_version,
        compile_mode=args.compile_mode,
        nsys_report_path=str(rep_path),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    builder = ManifestBuilder(
        nsys_rep_path=rep_path,
        metadata=metadata,
    )
    manifest = builder.build()

    manifest_path = output_prefix.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Mapping manifest → %s", manifest_path)
    print(f"Manifest written to: {manifest_path}")
