"""
Metric aggregator — combines KernelMetrics from multiple kernels into
AggregatedMetrics using the canonical aggregation policy table.

Policy table (from architecture plan §4):
  dram__bytes_read.sum          → sum   (additive bytes)
  dram__bytes_written.sum       → sum   (additive bytes)
  sm__active_cycles_elapsed.sum → sum   (total GPU work cycles)
  sm__throughput.avg.*          → mean  (rate metric)
  l1tex__t_hit_rate.pct         → mean  (ratio)
  sm__warps_active.avg.*        → mean  (ratio — achieved occupancy)
  smsp__pipe_tensor_cycles.*    → mean  (ratio)
  gpu__time_duration.sum        → sum   (wall time)
  smsp__sass_thread_inst.*      → sum   (instruction counts)
"""
from __future__ import annotations

import statistics
from typing import Sequence

from operator_profiler.schema.metrics import AggregationOp, METRIC_POLICIES
from operator_profiler.schema.profile import AggregatedMetrics, KernelMetrics


def aggregate_fused_metrics(
    kernel_metrics: Sequence[KernelMetrics],
) -> KernelMetrics:
    """
    Aggregate a sequence of KernelMetrics into a single KernelMetrics using
    the canonical policy table.

    Used when one NVTX range contains multiple ncu kernel rows (fused kernels).
    """
    if not kernel_metrics:
        return KernelMetrics()
    if len(kernel_metrics) == 1:
        return kernel_metrics[0]

    # Collect all values for named fields
    dram_read_vals = _collect_int_field(kernel_metrics, "dram_bytes_read")
    dram_write_vals = _collect_int_field(kernel_metrics, "dram_bytes_written")
    sm_cycles_vals = _collect_float_field(kernel_metrics, "sm_active_cycles")
    l1_hit_vals = _collect_float_field(kernel_metrics, "l1_hit_rate")
    occupancy_vals = _collect_float_field(kernel_metrics, "achieved_occupancy")
    tensor_vals = _collect_float_field(kernel_metrics, "tensor_core_active_pct")
    ai_vals = _collect_float_field(kernel_metrics, "arithmetic_intensity")
    gflops_vals = _collect_float_field(kernel_metrics, "achieved_gflops")

    # Aggregate raw dict using per-metric policies
    merged_raw: dict[str, float | int | str] = {}
    for policy in METRIC_POLICIES:
        raw_vals = [
            m.raw[policy.ncu_name]
            for m in kernel_metrics
            if policy.ncu_name in m.raw
        ]
        if not raw_vals:
            continue
        numeric = [v for v in raw_vals if isinstance(v, (int, float))]
        if not numeric:
            merged_raw[policy.ncu_name] = raw_vals[-1]
            continue
        if policy.aggregation == AggregationOp.SUM:
            merged_raw[policy.ncu_name] = sum(numeric)
        elif policy.aggregation == AggregationOp.MEAN:
            merged_raw[policy.ncu_name] = statistics.mean(numeric)
        elif policy.aggregation == AggregationOp.MAX:
            merged_raw[policy.ncu_name] = max(numeric)
        elif policy.aggregation == AggregationOp.MIN:
            merged_raw[policy.ncu_name] = min(numeric)

    return KernelMetrics(
        sm_active_cycles=sum(sm_cycles_vals) if sm_cycles_vals else None,
        dram_bytes_read=sum(dram_read_vals) if dram_read_vals else None,
        dram_bytes_written=sum(dram_write_vals) if dram_write_vals else None,
        l1_hit_rate=statistics.mean(l1_hit_vals) if l1_hit_vals else None,
        achieved_occupancy=statistics.mean(occupancy_vals) if occupancy_vals else None,
        tensor_core_active_pct=statistics.mean(tensor_vals) if tensor_vals else None,
        arithmetic_intensity=statistics.mean(ai_vals) if ai_vals else None,
        achieved_gflops=sum(gflops_vals) if gflops_vals else None,
        raw=merged_raw,
    )


def build_aggregated_metrics(kernels_metrics: Sequence[KernelMetrics], total_duration_ns: int) -> AggregatedMetrics:
    """
    Build an AggregatedMetrics summary for an OperatorRecord from its kernels.
    """
    count = len(kernels_metrics)
    if count == 0:
        return AggregatedMetrics(total_duration_ns=total_duration_ns, kernel_count=0)

    dram_read = sum(m.dram_bytes_read for m in kernels_metrics if m.dram_bytes_read is not None)
    dram_write = sum(m.dram_bytes_written for m in kernels_metrics if m.dram_bytes_written is not None)

    occ_vals = [m.achieved_occupancy for m in kernels_metrics if m.achieved_occupancy is not None]
    tensor_vals = [m.tensor_core_active_pct for m in kernels_metrics if m.tensor_core_active_pct is not None]

    mean_occ = statistics.mean(occ_vals) if occ_vals else None
    mean_tensor = statistics.mean(tensor_vals) if tensor_vals else None

    bottleneck = _classify_bottleneck(kernels_metrics, total_duration_ns)

    return AggregatedMetrics(
        total_duration_ns=total_duration_ns,
        kernel_count=count,
        total_dram_bytes_read=dram_read,
        total_dram_bytes_written=dram_write,
        mean_achieved_occupancy=mean_occ,
        mean_tensor_core_active_pct=mean_tensor,
        bottleneck_classification=bottleneck,
    )


def _classify_bottleneck(
    kernels_metrics: Sequence[KernelMetrics], total_duration_ns: int
) -> str:
    """
    Simple bottleneck classification from arithmetic intensity + occupancy.

    This is a heuristic — the roofline module provides a more accurate analysis.
    """
    ai_vals = [m.arithmetic_intensity for m in kernels_metrics if m.arithmetic_intensity is not None]
    occ_vals = [m.achieved_occupancy for m in kernels_metrics if m.achieved_occupancy is not None]

    if not ai_vals or not occ_vals:
        return "unknown"

    mean_ai = statistics.mean(ai_vals)
    mean_occ = statistics.mean(occ_vals)

    # Roofline ridge point is typically ~10 FLOP/byte for modern GPUs
    if mean_ai > 10.0 and mean_occ > 50.0:
        return "compute_bound"
    if mean_ai < 5.0:
        return "memory_bound"
    if total_duration_ns < 10_000:  # < 10 µs — likely latency bound
        return "latency_bound"
    return "unknown"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _collect_int_field(metrics: Sequence[KernelMetrics], field: str) -> list[int]:
    return [getattr(m, field) for m in metrics if getattr(m, field) is not None]


def _collect_float_field(metrics: Sequence[KernelMetrics], field: str) -> list[float]:
    return [getattr(m, field) for m in metrics if getattr(m, field) is not None]
