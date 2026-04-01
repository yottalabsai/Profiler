"""
Metric aggregator — combines KernelMetrics from multiple kernels into
AggregatedMetrics.

Aggregation strategy:
  Additive quantities (sum):
    dram_bytes_read/written, executed_instructions, issued_instructions

  Duration-weighted mean (rate/utilization metrics):
    achieved_occupancy, tensor_core_active_pct
    Weighting by kernel duration ensures a 10µs kernel contributes 10x more
    than a 1µs kernel — critical for heterogeneous 1:many operator mappings.

  Identity (dominant kernel):
    dominant_kernel_id — kernel with highest duration_ns
"""
from __future__ import annotations

import statistics
from typing import Sequence

from operator_profiler.schema.metrics import (
    AggregationOp, NCU_NAME_TO_POLICY, get_raw_value,
)
from operator_profiler.schema.profile import AggregatedMetrics, KernelMetrics, KernelRecord


def aggregate_fused_metrics(
    kernel_metrics: Sequence[KernelMetrics],
) -> KernelMetrics:
    """
    Merge a sequence of KernelMetrics into one using the canonical policy table.

    Used when a single NVTX range covers multiple ncu kernel rows (fused kernels).
    Unknown keys (not in NCU_NAME_TO_POLICY) take the last observed value.
    """
    if not kernel_metrics:
        return KernelMetrics()
    if len(kernel_metrics) == 1:
        return kernel_metrics[0]

    all_keys = {k for m in kernel_metrics for k in m.raw}
    merged_raw: dict[str, float | int | str] = {}

    for key in all_keys:
        vals = [m.raw[key] for m in kernel_metrics if key in m.raw]
        policy = NCU_NAME_TO_POLICY.get(key)
        numeric = [v for v in vals if isinstance(v, (int, float))]
        if not numeric or policy is None:
            merged_raw[key] = vals[-1]
            continue
        if policy.aggregation == AggregationOp.SUM:
            merged_raw[key] = sum(numeric)
        elif policy.aggregation == AggregationOp.MEAN:
            merged_raw[key] = statistics.mean(numeric)
        elif policy.aggregation == AggregationOp.MAX:
            merged_raw[key] = max(numeric)
        elif policy.aggregation == AggregationOp.MIN:
            merged_raw[key] = min(numeric)

    return KernelMetrics(raw=merged_raw)


def build_aggregated_metrics(kernels: list[KernelRecord]) -> AggregatedMetrics:
    """
    Build an AggregatedMetrics summary for an OperatorRecord from its kernels.

    Rate metrics (occupancy, tensor core %) use duration-weighted means so that
    long-running kernels contribute proportionally to the aggregate.
    """
    if not kernels:
        return AggregatedMetrics(total_duration_ns=0, kernel_count=0)

    total_ns = sum(k.duration_ns for k in kernels)
    dominant = max(kernels, key=lambda k: k.duration_ns)

    # --- Additive quantities ---
    dram_read_vals = [v for k in kernels if (v := get_raw_value(k.metrics.raw, "dram_bytes_read")) is not None]
    dram_read = int(sum(dram_read_vals)) if dram_read_vals else None
    dram_write_vals = [v for k in kernels if (v := get_raw_value(k.metrics.raw, "dram_bytes_written")) is not None]
    dram_write = int(sum(dram_write_vals)) if dram_write_vals else None
    total_executed = int(sum(
        v for k in kernels
        if isinstance(v := k.metrics.raw.get("Executed Instructions"), (int, float))
    ))
    total_issued = int(sum(
        v for k in kernels
        if isinstance(v := k.metrics.raw.get("Issued Instructions"), (int, float))
    ))

    # --- Duration-weighted means ---
    mean_occ = _duration_weighted_mean(kernels, "achieved_occupancy")
    mean_tensor = _duration_weighted_mean(kernels, "tensor_core_active_pct")

    return AggregatedMetrics(
        total_duration_ns=total_ns,
        kernel_count=len(kernels),
        dominant_kernel_id=dominant.kernel_id,
        total_dram_bytes_read=dram_read,   # None if hardware doesn't expose counter
        total_dram_bytes_written=dram_write,
        total_executed_instructions=total_executed,
        total_issued_instructions=total_issued,
        mean_achieved_occupancy=mean_occ,
        mean_tensor_core_active_pct=mean_tensor,
    )


def _duration_weighted_mean(kernels: list[KernelRecord], profile_field: str) -> float | None:
    """
    Compute a duration-weighted mean of a metric across kernels.

    Kernels without the metric are excluded from both numerator and denominator,
    so a missing metric on a short kernel doesn't suppress the aggregate.
    """
    weighted_sum = 0.0
    total_weight = 0
    for k in kernels:
        v = get_raw_value(k.metrics.raw, profile_field)
        if v is not None:
            weighted_sum += v * k.duration_ns
            total_weight += k.duration_ns
    return weighted_sum / total_weight if total_weight > 0 else None
