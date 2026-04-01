"""
Profile builder — assembles the final OperatorAttributedProfile from
operator records and unattributed kernels.

Called after:
  1. ManifestBuilder.build()         → MappingManifest
  2. AttributionEngine.run()         → (operator_records, unattributed_kernels)
  3. RangeReplayOrchestrator.run()   → metrics populated in operator_records
  4. build_aggregated_metrics()      → AggregatedMetrics per operator

Optional DiagnosisAgent post-pass
----------------------------------
Pass ``diagnosis_agent=DiagnosisAgent(...)`` to replace the heuristic
bottleneck_classification with LLM-reasoned labels.  The agent runs after
all AggregatedMetrics are built so it can compare each operator against the
model-wide metric distribution.  Falls back to the heuristic value on any
API error — the profile is always returned.
"""
from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from operator_profiler.aggregator.metric_aggregator import build_aggregated_metrics
from operator_profiler.schema.manifest import MappingManifest
from operator_profiler.schema.profile import (
    CaptureMetadata,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)

if TYPE_CHECKING:
    from operator_profiler.agents.diagnosis import DiagnosisAgent, ModelStats

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
    diagnosis_agent: "DiagnosisAgent | None" = None,
) -> OperatorAttributedProfile:
    """
    Assemble the top-level OperatorAttributedProfile.

    Fills in AggregatedMetrics for each OperatorRecord and collects all
    warnings from the manifest.

    Parameters
    ----------
    diagnosis_agent:
        Optional DiagnosisAgent.  When provided, a second pass runs after
        all operators have AggregatedMetrics filled, replacing the heuristic
        bottleneck_classification with an LLM-reasoned label.
    """
    # Aggregate metrics for each operator
    for op in operator_records:
        if op.kernels:
            op.aggregated = build_aggregated_metrics(op.kernels)

    # Optional DiagnosisAgent post-pass — re-classifies each operator using
    # model-wide metric distributions for relative comparison.
    if diagnosis_agent is not None:
        stats = _compute_model_stats(operator_records, device_name)
        for op in operator_records:
            if op.aggregated is None:
                continue
            result = diagnosis_agent.diagnose(op, stats)
            op.aggregated.bottleneck_classification = result.classification
            log.debug(
                "DiagnosisAgent: %s → %s (%s)",
                op.operator_id, result.classification,
                result.reasoning[:80],
            )

    meta = manifest.capture_metadata
    capture_metadata = CaptureMetadata(
        model_name=model_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        compile_mode=meta.compile_mode,  # type: ignore[arg-type]
        nsys_report_path=meta.nsys_report_path,
        ncu_report_path=ncu_report_path,
        provenance_log_path=meta.provenance_log_path,
        capture_timestamp_utc=meta.capture_timestamp_utc
        or datetime.now(timezone.utc).isoformat(),
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_model_stats(
    operator_records: list[OperatorRecord],
    device_name: str | None,
) -> "ModelStats":
    """
    Compute model-wide metric distributions for DiagnosisAgent context.

    The agent needs these to classify bottlenecks *relatively* — an operator
    at 40% occupancy is alarming if the model median is 75%, but unremarkable
    if the median is 35%.
    """
    from operator_profiler.agents.diagnosis import ModelStats
    from operator_profiler.aggregator.roofline import KNOWN_GPU_SPECS

    # Fuzzy GPU spec lookup for ridge point
    ridge_point: float | None = None
    resolved_name = device_name or ""
    for key, specs in KNOWN_GPU_SPECS.items():
        if key.lower() in resolved_name.lower() or resolved_name.lower() in key.lower():
            ridge_point = specs["peak_compute_gflops"] / specs["peak_bandwidth_gbs"]
            break

    # Collect cross-operator metric vectors
    ais: list[float] = []
    occs: list[float] = []
    tc_pcts: list[float] = []

    for op in operator_records:
        for k in op.kernels:
            if k.metrics.arithmetic_intensity is not None:
                ais.append(k.metrics.arithmetic_intensity)
        if op.aggregated is not None:
            if op.aggregated.mean_achieved_occupancy is not None:
                occs.append(op.aggregated.mean_achieved_occupancy)
            if op.aggregated.mean_tensor_core_active_pct is not None:
                tc_pcts.append(op.aggregated.mean_tensor_core_active_pct)

    def _percentile(vals: list[float], p: int) -> float | None:
        if not vals:
            return None
        sv = sorted(vals)
        idx = max(0, min(len(sv) - 1, int(p / 100 * len(sv))))
        return sv[idx]

    return ModelStats(
        median_ai=statistics.median(ais) if ais else None,
        p25_ai=_percentile(ais, 25),
        p75_ai=_percentile(ais, 75),
        median_occupancy=statistics.median(occs) if occs else None,
        median_tensor_core_pct=statistics.median(tc_pcts) if tc_pcts else None,
        ridge_point=ridge_point,
        device_name=device_name,
    )
