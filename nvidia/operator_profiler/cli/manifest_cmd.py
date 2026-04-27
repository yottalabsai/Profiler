"""
operator-profiler manifest --nsys-rep <path.nsys-rep> --output <manifest.json>

Builds the MappingManifest from an existing nsys capture without re-running the
workload.  Replaces the inline ManifestBuilder python3 -c invocation in Stage 0c.

Usage:
    operator-profiler manifest \\
        --nsys-rep profiler_output/workload.nsys-rep \\
        --output   profiler_output/workload.manifest.json \\
        --model-name MyModel \\
        --compile-backend inductor \\
        --corr-json profiler_output/workload.corr.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "manifest",
        help="Build a MappingManifest from an existing nsys capture.",
    )
    p.add_argument("--nsys-rep", required=True, metavar="PATH",
                   help="Path to the .nsys-rep file produced by nsys profile.")
    p.add_argument("--output", required=True, metavar="PATH",
                   help="Output path for the manifest JSON.")
    p.add_argument("--model-name", default="model",
                   help="Human-readable model name stored in manifest metadata.")
    p.add_argument("--compile-backend", default="inductor",
                   help="Compile backend used during capture (e.g. inductor, none).")
    p.add_argument("--corr-json", default=None, metavar="PATH",
                   help="Optional .corr.json from --correlation-pass; enables HIGH-confidence attribution.")
    p.add_argument("--nsys-executable", default="nsys",
                   help="Path to nsys executable (used for SQLite export if not already cached).")
    p.set_defaults(func=_run)


def _run(args) -> None:
    import torch
    from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

    nsys_rep = Path(args.nsys_rep)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect device name
    device_name: str | None = None
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    # Load correlation map if provided
    correlation_map: dict[tuple[str, int], str] | None = None
    if args.corr_json:
        corr_path = Path(args.corr_json)
        if corr_path.exists():
            raw = json.loads(corr_path.read_text())
            correlation_map = {
                (entry["kernel_name"], entry["invocation"]): entry["op_name"]
                for entry in raw.get("entries", [])
            }
            log.info("Loaded correlation map: %d entries from %s", len(correlation_map), corr_path)
        else:
            log.warning("--corr-json path does not exist: %s (skipping)", corr_path)

    meta = CaptureManifestMetadata(
        model_name=args.model_name,
        torch_version=torch.__version__,
        compile_mode=args.compile_backend,
        nsys_report_path=str(nsys_rep.resolve()),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device_name=device_name,
    )

    builder = ManifestBuilder(
        nsys_rep_path=nsys_rep,
        metadata=meta,
        sqlite_cache_dir=nsys_rep.parent,
        correlation_map=correlation_map,
        nsys_executable=args.nsys_executable,
    )
    manifest = builder.build()

    output_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest written → %s (%d kernels)", output_path, len(manifest.kernels))
    print(f"Manifest written to: {output_path} ({len(manifest.kernels)} kernels, device: {device_name})")
