"""
Profile builder — assembles the final OperatorAttributedProfile from
operator records and unattributed kernels.

Called after:
  1. ManifestBuilder.build()         → MappingManifest
  2. AttributionEngine.run()         → (operator_records, unattributed_kernels)
  3. KernelProfileOrchestrator.run() → metrics populated in operator_records
  4. build_aggregated_metrics()      → AggregatedMetrics per operator
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from operator_profiler.aggregator.metric_aggregator import build_aggregated_metrics
from operator_profiler.schema.manifest import MappingManifest
from operator_profiler.schema.profile import (
    CaptureMetadata,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)

log = logging.getLogger(__name__)


def build_profile(
    manifest: MappingManifest,
    operator_records: list[OperatorRecord],
    unattributed_kernels: list[KernelRecord],
    model_name: str,
    torch_version: str,
    cuda_version: str | None = None,
    device_name: str | None = None,
    ncu_report_path: str | None = None,
    extra_warnings: list[str] | None = None,
) -> OperatorAttributedProfile:
    """
    Assemble the top-level OperatorAttributedProfile.

    Fills in AggregatedMetrics for each OperatorRecord and collects all
    warnings from the manifest.
    """
    # Aggregate metrics for each operator
    for op in operator_records:
        if op.kernels:
            op.aggregated = build_aggregated_metrics(op.kernels)

    meta = manifest.capture_metadata
    capture_metadata = CaptureMetadata(
        model_name=model_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        compile_mode=meta.compile_mode,  # type: ignore[arg-type]
        nsys_report_path=meta.nsys_report_path,
        ncu_report_path=ncu_report_path,
        capture_timestamp_utc=(
            meta.capture_timestamp_utc or datetime.now(timezone.utc).isoformat()
        ),
        device_name=device_name,
    )

    warnings = list(manifest.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)

    log.info(
        "Built profile: %d operators, %d unattributed kernels, %d warnings",
        len(operator_records),
        len(unattributed_kernels),
        len(warnings),
    )

    return OperatorAttributedProfile(
        capture_metadata=capture_metadata,
        operators=operator_records,
        unattributed_kernels=unattributed_kernels,
        warnings=warnings,
    )
