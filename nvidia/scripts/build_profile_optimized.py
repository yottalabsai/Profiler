"""
build_profile_optimized.py — Phase D: assemble profile_optimized.json from
already-collected nsys-rep + ncu-rep artifacts.

This script skips the nsys and ncu capture phases (both already done) and
only runs the post-capture pipeline:
  1. Load correlation map from <prefix>.corr.json for HIGH-confidence attribution
  2. ManifestBuilder on <nsys-rep>  →  MappingManifest
  3. AttributionEngine              →  OperatorRecords + unattributed
  4. import_ncu_report + parse_ncu_csv_by_id  →  (kernel_name, id) → KernelMetrics
  5. Fan metrics out to OperatorRecords via KernelProfileOrchestrator-style merge
  6. build_profile()                →  profile_optimized.json

Usage:
    PYTHONPATH=/home/ubuntu/Profiler python3 nvidia/scripts/build_profile_optimized.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT))

from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
from nvidia.operator_profiler.mapper.attribution_engine import AttributionEngine
from nvidia.operator_profiler.mapper.ncu_runner import import_ncu_report
from nvidia.operator_profiler.mapper.ncu_parser import parse_ncu_csv_by_id
from nvidia.operator_profiler.aggregator.profile_builder import build_profile
from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
EXAMPLE_DIR      = ROOT / "examples" / "gpt2"
PROFILER_OUT     = EXAMPLE_DIR / "profiler_output"
NCU_EXECUTABLE   = "/opt/nvidia/nsight-compute/2025.4.1/ncu"

NSYS_REP         = PROFILER_OUT / "gpt2_optimized.nsys-rep"
CORR_JSON        = PROFILER_OUT / "gpt2_optimized.corr.json"
NCU_REP          = PROFILER_OUT / "ncu_reps" / "all_kernels.ncu-rep"
SQLITE_CACHE_DIR = PROFILER_OUT          # reuse the existing gpt2.sqlite if nsys export re-runs
PROFILE_OUT      = EXAMPLE_DIR / "profile_optimized.json"

MODEL_NAME       = "GPT2-Optimized"


def main() -> None:
    import torch
    torch_version = torch.__version__

    # ── Step 1: Load correlation map ─────────────────────────────────────────
    correlation_map: dict[tuple[str, int], str] = {}
    if CORR_JSON.exists():
        data = json.loads(CORR_JSON.read_text())
        for entry in data.get("entries", []):
            correlation_map[(entry["kernel_name"], entry["invocation"])] = entry["op_name"]
        log.info("Loaded correlation map: %d entries from %s", len(correlation_map), CORR_JSON)
    else:
        log.warning("No corr.json found at %s — falling back to NVTX attribution only", CORR_JSON)

    # ── Step 2: ManifestBuilder ───────────────────────────────────────────────
    log.info("Building manifest from %s", NSYS_REP)
    metadata = CaptureManifestMetadata(
        model_name=MODEL_NAME,
        torch_version=torch_version,
        compile_mode="inductor",
        nsys_report_path=str(NSYS_REP),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    builder = ManifestBuilder(
        nsys_rep_path=NSYS_REP,
        metadata=metadata,
        sqlite_cache_dir=SQLITE_CACHE_DIR,
        correlation_map=correlation_map,
        nsys_executable="nsys",
    )
    manifest = builder.build()
    log.info("Manifest: %d kernels, %d warnings", len(manifest.kernels), len(manifest.warnings))

    # ── Step 3: AttributionEngine ─────────────────────────────────────────────
    log.info("Running attribution engine")
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()
    log.info(
        "Attribution: %d operator records, %d unattributed kernels",
        len(operator_records), len(unattributed),
    )

    # ── Step 4: Import ncu metrics from the already-collected .ncu-rep ───────
    log.info("Importing ncu metrics from %s", NCU_REP)
    csv_text = import_ncu_report(NCU_REP, NCU_EXECUTABLE)
    metrics_map = parse_ncu_csv_by_id(csv_text)
    log.info("ncu metrics: %d (kernel_name, id) entries", len(metrics_map))

    # ── Step 5: Fan metrics out to OperatorRecord kernels ────────────────────
    # Build a per-kernel_name invocation-ordered list from the manifest (same
    # strategy as KernelProfileOrchestrator._build_replay_targets + _merge_metrics).
    log.info("Merging ncu metrics into operator records")

    # kernel_id → KernelMetrics mapping
    kernel_metrics: dict[str, object] = {}

    # Collect unique kernel names in manifest launch order
    seen: dict[str, list[str]] = defaultdict(list)  # kernel_name → [kernel_id, ...]
    for entry in manifest.kernels:
        seen[entry.kernel_name].append(entry.kernel_id)

    matched = 0
    missing = 0
    for kernel_name, kernel_ids in seen.items():
        # Filter metrics_map to rows whose full ncu name contains the manifest name
        # (nsys stores short names; ncu CSV stores full mangled names).
        name_rows = {
            kid: m for (kname, kid), m in metrics_map.items()
            if kernel_name in kname
        }
        ordered = [
            name_rows[k]
            for k in sorted(name_rows, key=lambda x: int(x) if x.isdigit() else 0)
        ]
        for i, kid in enumerate(kernel_ids):
            if i < len(ordered):
                kernel_metrics[kid] = ordered[i]
                matched += 1
            else:
                missing += 1

    log.info(
        "Metric fan-out: %d matched, %d missing (no ncu row for launch index)",
        matched, missing,
    )

    # Write metrics into KernelRecord objects in-place
    for op_record in operator_records:
        for kernel in op_record.kernels:
            if kernel.kernel_id in kernel_metrics:
                kernel.metrics = kernel_metrics[kernel.kernel_id]

    # ── Step 6: build_profile() ───────────────────────────────────────────────
    log.info("Building profile")
    import torch.version
    cuda_version = torch.version.cuda

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=MODEL_NAME,
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_name=None,
        ncu_report_path=str(NCU_REP),
    )

    PROFILE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_OUT.write_text(profile.model_dump_json(indent=2))
    log.info("Profile written to: %s", PROFILE_OUT)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_duration_ns = sum(
        sum(k.duration_ns for k in op.kernels)
        for op in operator_records
    )
    total_duration_ms = total_duration_ns / 1e6

    # Top 5 operators by total kernel duration
    op_durations = [
        (op.operator_name, sum(k.duration_ns for k in op.kernels))
        for op in operator_records
    ]
    op_durations.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("Profile built successfully")
    print("=" * 60)
    print(f"  Output:              {PROFILE_OUT}")
    print(f"  Operator records:    {len(operator_records)}")
    print(f"  Unattributed kernels:{len(unattributed)}")
    print(f"  Total kernel time:   {total_duration_ms:.2f} ms")
    print(f"  Warnings:            {len(manifest.warnings)}")
    print()
    print("Top 5 operators by kernel time:")
    for name, dur_ns in op_durations[:5]:
        print(f"  {name:50s}  {dur_ns/1e6:8.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
