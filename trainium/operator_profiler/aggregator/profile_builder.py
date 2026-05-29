"""
Profile builder — assembles the final OperatorAttributedProfile.

Adapted from nvidia/operator_profiler/aggregator/profile_builder.py.

Changes:
  - cuda_version / nsys_report_path / ncu_report_path replaced with
    neuron_sdk_version / nrt_session_dir in CaptureMetadata.
  - No ncu replay step — metrics are already populated from ntrace_parser
    before this function is called.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from trainium.operator_profiler.aggregator.metric_aggregator import build_aggregated_metrics
from trainium.operator_profiler.schema.manifest import MappingManifest
from trainium.operator_profiler.schema.profile import (
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
    neuron_sdk_version: str | None = None,
    device_name: str | None = None,
    nrt_session_dir: str | None = None,
    extra_warnings: list[str] | None = None,
) -> OperatorAttributedProfile:
    """
    Assemble the top-level OperatorAttributedProfile.

    Fills in AggregatedMetrics for each OperatorRecord and collects all
    warnings from the manifest.
    """
    for op in operator_records:
        if op.kernels:
            op.aggregated = build_aggregated_metrics(op.kernels)

    meta = manifest.capture_metadata
    capture_metadata = CaptureMetadata(
        model_name=model_name,
        torch_version=torch_version,
        neuron_sdk_version=neuron_sdk_version,
        compile_mode=meta.compile_mode,
        nrt_session_dir=nrt_session_dir or meta.nrt_session_dir,
        capture_timestamp_utc=(
            meta.capture_timestamp_utc or datetime.now(timezone.utc).isoformat()
        ),
        device_name=device_name or meta.device_name,
        neuroncore_count=meta.neuroncore_count,
    )

    warnings = list(manifest.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)

    log.info(
        "Built profile: %d operators, %d unattributed, %d warnings",
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
