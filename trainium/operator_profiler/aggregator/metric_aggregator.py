"""
Metric aggregator — combines KernelMetrics from multiple NeuronCore operations
into AggregatedMetrics for an OperatorRecord.

Adapted from nvidia/operator_profiler/aggregator/metric_aggregator.py.
Aggregation strategy is identical; only the field names change:
  - dram_bytes_read/written  → dma_bytes_read/written
  - sm_throughput_pct        → tensor_engine_utilization_pct
  - achieved_occupancy       → (kept as alias if NRT exposes it)
  - l1_hit_rate              → sbuf_utilization_pct
  - l2_hit_rate              → hbuf_utilization_pct
  - warp_cycles_per_instruction / eligible_cycles_pct / ipc_active →
      operations_per_cycle / stall_cycles_pct (NRT equivalents)

TODO(blocker: ntrace.pb schema): The _WMEAN_FIELDS / _SUM_FIELDS tuples below
use the logical profile_field names from schema/metrics.py MetricPolicy.
Once the real NRT counter names are confirmed and METRIC_POLICIES is updated,
this file requires no further changes — get_raw_value() handles the name
translation automatically.
"""
from __future__ import annotations

import statistics
from typing import Sequence

from trainium.operator_profiler.schema.metrics import (
    AggregationOp,
    NRT_NAME_TO_POLICY,
    get_raw_value,
)
from trainium.operator_profiler.schema.profile import (
    AggregatedMetrics,
    KernelMetrics,
    KernelRecord,
)

# Duration-weighted mean: rate/utilisation metrics where a longer-running
# operation should contribute proportionally more to the aggregate.
_WMEAN_FIELDS: tuple[str, ...] = (
    "tensor_engine_utilization_pct",
    "vector_engine_utilization_pct",
    "memory_utilization_pct",
    "ddr_throughput_pct",
    "stall_cycles_pct",
    "operations_per_cycle",
    "sbuf_utilization_pct",
    "hbuf_utilization_pct",
    # NVIDIA-compatible alias fields (populated if NRT exposes them)
    "achieved_occupancy",
    "sm_throughput_pct",
    "l1_hit_rate",
    "l2_hit_rate",
)

# Additive: absolute quantities that sum across operations
_SUM_FIELDS: tuple[str, ...] = (
    "dma_bytes_read",
    "dma_bytes_written",
    "stall_cycles",
    "execution_cycles",
)

# Max: per-operation constants where the worst case limits the operator
_MAX_FIELDS: tuple[str, ...] = ()


def aggregate_fused_metrics(
    kernel_metrics: Sequence[KernelMetrics],
) -> KernelMetrics:
    """
    Merge a sequence of KernelMetrics into one using the canonical policy table.

    Used when a single aten:: op spans multiple NeuronCore operations.
    """
    if not kernel_metrics:
        return KernelMetrics()
    if len(kernel_metrics) == 1:
        return kernel_metrics[0]

    all_keys = {k for m in kernel_metrics for k in m.raw}
    merged_raw: dict[str, float | int | str] = {}

    for key in all_keys:
        vals = [m.raw[key] for m in kernel_metrics if key in m.raw]
        policy = NRT_NAME_TO_POLICY.get(key)
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

    Rate metrics use duration-weighted means so that longer-running operations
    contribute proportionally to the aggregate.
    """
    if not kernels:
        return AggregatedMetrics(total_duration_ns=0, kernel_count=0)

    total_ns = 0
    dominant = kernels[0]
    sum_acc: dict[str, float] = {}
    wmean_acc: dict[str, list[float]] = {}
    max_acc: dict[str, float] = {}

    for k in kernels:
        dur = k.duration_ns
        total_ns += dur
        if dur > dominant.duration_ns:
            dominant = k

        raw = k.metrics.raw

        for field in _SUM_FIELDS:
            v = get_raw_value(raw, field)
            if v is not None:
                sum_acc[field] = sum_acc.get(field, 0.0) + v

        for field in _WMEAN_FIELDS:
            v = get_raw_value(raw, field)
            if v is not None:
                acc = wmean_acc.get(field)
                if acc is None:
                    wmean_acc[field] = [v * dur, dur]
                else:
                    acc[0] += v * dur
                    acc[1] += dur

        for field in _MAX_FIELDS:
            v = get_raw_value(raw, field)
            if v is not None and (field not in max_acc or v > max_acc[field]):
                max_acc[field] = v

    def _s(field: str, *, as_int: bool = False) -> "int | float | None":
        val = sum_acc.get(field)
        return (int(val) if as_int else val) if val is not None else None

    def _w(field: str) -> float | None:
        acc = wmean_acc.get(field)
        return acc[0] / acc[1] if acc else None

    total_dma_read  = _s("dma_bytes_read", as_int=True)
    total_dma_write = _s("dma_bytes_written", as_int=True)

    return AggregatedMetrics(
        total_duration_ns=total_ns,
        kernel_count=len(kernels),
        dominant_kernel_id=dominant.kernel_id,
        # DDR bandwidth
        total_dma_bytes_read=total_dma_read,
        total_dma_bytes_written=total_dma_write,
        # NVIDIA-compatible aliases so comparison tools work across backends
        total_dram_bytes_read=total_dma_read,
        total_dram_bytes_written=total_dma_write,
        # Memory throughput
        memory_utilization_pct=_w("memory_utilization_pct"),
        ddr_throughput_pct=_w("ddr_throughput_pct"),
        # Compute utilization
        tensor_engine_utilization_pct=_w("tensor_engine_utilization_pct"),
        vector_engine_utilization_pct=_w("vector_engine_utilization_pct"),
        sm_throughput_pct=_w("tensor_engine_utilization_pct"),  # alias
        # Stall / latency
        stall_cycles_pct=_w("stall_cycles_pct"),
        # Instruction throughput
        total_execution_cycles=_s("execution_cycles", as_int=True),
        operations_per_cycle=_w("operations_per_cycle"),
        # On-chip memory (scratchpad hit rates)
        l1_hit_rate=_w("sbuf_utilization_pct"),    # alias
        l2_hit_rate=_w("hbuf_utilization_pct"),    # alias
        # If NRT exposes occupancy-equivalent
        achieved_occupancy=_w("achieved_occupancy"),
    )
