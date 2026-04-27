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
    p.add_argument(
        "--correlation-pass",
        action="store_true",
        default=False,
        help=(
            "Run the script once outside nsys before capture to build a HIGH-confidence "
            "CUPTI correlation map.  The script must support --correlation-pass and "
            "--output-prefix (e.g. run_workload.py)."
        ),
    )
    p.add_argument(
        "--inductor-debug-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing Inductor trace artifacts (output_code.py files) "
            "written when TORCH_COMPILE_DEBUG=1 is set during the capture run. "
            "When provided, parse_inductor_debug_dir() is called after nsys capture "
            "and the resulting fusion map is passed to ManifestBuilder so that "
            "unattributed fused Triton kernels receive INDUCTOR_FUSION attribution."
        ),
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    from nvidia.operator_profiler.capture.nsys_runner import NsysRunConfig, run_nsys_profile
    from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata
    from nvidia.operator_profiler.utils.subprocess_utils import run_subprocess

    output_prefix = Path(args.output)

    # ── Optional: correlation pass (must run outside nsys — nsys holds CUPTI) ──
    if getattr(args, "correlation_pass", False):
        log.info("Running correlation pass (outside nsys) ...")
        corr_cmd = [
            sys.executable, args.script,
            "--correlation-pass",
            "--output-prefix", str(output_prefix),
            *[a for a in args.script_args if a not in ("--correlation-pass",)],
        ]
        run_subprocess(corr_cmd, description="correlation pass")
        log.info("Correlation pass complete.")

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

    # Load correlation sidecar if present (written by run_workload --correlation-pass)
    correlation_map: dict[tuple[str, int], str] | None = None
    corr_sidecar = output_prefix.with_suffix(".corr.json")
    if corr_sidecar.exists():
        data = json.loads(corr_sidecar.read_text())
        correlation_map = {
            (e["kernel_name"], e["invocation"]): e["op_name"]
            for e in data.get("entries", [])
        }
        log.info(
            "Loaded correlation map: %d entries from %s",
            len(correlation_map),
            corr_sidecar,
        )

    # Load inductor fusion map if the user captured Inductor debug artifacts
    inductor_fusion_map: dict[str, list[str]] | None = None
    if args.inductor_debug_dir:
        from nvidia.operator_profiler.capture.inductor_fusion_extractor import (
            parse_inductor_debug_dir,
        )
        inductor_fusion_map = parse_inductor_debug_dir(args.inductor_debug_dir)
        log.info(
            "Inductor fusion map: %d kernel→ops entries from %s",
            len(inductor_fusion_map),
            args.inductor_debug_dir,
        )

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
        correlation_map=correlation_map,
        inductor_fusion_map=inductor_fusion_map,
    )
    manifest = builder.build()

    manifest_path = output_prefix.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Mapping manifest → %s", manifest_path)
    print(f"Manifest written to: {manifest_path}")
