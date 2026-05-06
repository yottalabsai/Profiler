"""
Metric aggregator — combines KernelMetrics from multiple kernels into
AggregatedMetrics.

Aggregation strategy:
  Additive quantities (sum):
    dram_bytes_read/written, executed_instructions, issued_instructions,
    local_memory_spills

  Duration-weighted mean (rate/utilization metrics):
    achieved_occupancy, tensor_core_active_pct, sm_throughput_pct,
    l1_hit_rate, l2_hit_rate, memory_throughput_pct, dram_throughput_pct,
    mem_busy_pct, warp_cycles_per_instruction, eligible_cycles_pct,
    ipc_active, avg_threads_per_warp
    Weighting by kernel duration ensures a 10µs kernel contributes 10x more
    than a 1µs kernel — critical for heterogeneous 1:many operator mappings.

  Max across kernels (per-kernel constants that limit occupancy):
    registers_per_thread, dynamic_smem_per_block

  Identity (dominant kernel):
    dominant_kernel_id — kernel with highest duration_ns
"""
from __future__ import annotations

import statistics
from typing import Sequence

# Field-name constants for the single-pass aggregation loop
_WMEAN_FIELDS: tuple[str, ...] = (
    "achieved_occupancy", "tensor_core_active_pct", "sm_throughput_pct",
    "l1_hit_rate", "l2_hit_rate", "memory_throughput_pct",
    "dram_throughput_pct", "mem_busy_pct", "warp_cycles_per_instruction",
    "eligible_cycles_pct", "ipc_active", "avg_threads_per_warp",
)
_SUM_FIELDS: tuple[str, ...] = (
    "dram_bytes_read", "dram_bytes_written", "local_memory_spills",
)
_MAX_FIELDS: tuple[str, ...] = ("registers_per_thread", "dynamic_smem_per_block")

from nvidia.operator_profiler.schema.metrics import (
    AggregationOp, NCU_NAME_TO_POLICY, get_raw_value,
)
from nvidia.operator_profiler.schema.profile import AggregatedMetrics, KernelMetrics, KernelRecord


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

    Rate metrics use duration-weighted means so that long-running kernels
    contribute proportionally to the aggregate.
    """
    if not kernels:
        return AggregatedMetrics(total_duration_ns=0, kernel_count=0)

    total_ns = 0
    dominant = kernels[0]
    # sum_acc: field -> running total; key absent means no kernel had the metric
    sum_acc: dict[str, float] = {}
    # wmean_acc: field -> [weighted_sum, total_weight]
    wmean_acc: dict[str, list[float]] = {}
    # max_acc: field -> current max; key absent means no kernel had the metric
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

        for field in ("executed_instructions", "issued_instructions"):
            v = get_raw_value(raw, field)
            if v is not None:
                sum_acc[field] = sum_acc.get(field, 0.0) + v

    def _s(field: str, *, as_int: bool = False) -> "int | float | None":
        val = sum_acc.get(field)
        return (int(val) if as_int else val) if val is not None else None

    def _w(field: str) -> float | None:
        acc = wmean_acc.get(field)
        return acc[0] / acc[1] if acc else None

    def _m(field: str) -> float | None:
        return max_acc.get(field)

    return AggregatedMetrics(
        total_duration_ns=total_ns,
        kernel_count=len(kernels),
        dominant_kernel_id=dominant.kernel_id,
        # Memory bandwidth
        total_dram_bytes_read=_s("dram_bytes_read", as_int=True),
        total_dram_bytes_written=_s("dram_bytes_written", as_int=True),
        # Memory throughput
        memory_throughput_pct=_w("memory_throughput_pct"),
        dram_throughput_pct=_w("dram_throughput_pct"),
        mem_busy_pct=_w("mem_busy_pct"),
        # Cache efficiency
        l1_hit_rate=_w("l1_hit_rate"),
        l2_hit_rate=_w("l2_hit_rate"),
        # Compute utilization
        sm_throughput_pct=_w("sm_throughput_pct"),
        tensor_core_active_pct=_w("tensor_core_active_pct"),
        # Occupancy & latency
        achieved_occupancy=_w("achieved_occupancy"),
        warp_cycles_per_instruction=_w("warp_cycles_per_instruction"),
        eligible_cycles_pct=_w("eligible_cycles_pct"),
        # Instruction throughput
        total_executed_instructions=int(sum_acc.get("executed_instructions", 0.0)),
        total_issued_instructions=int(sum_acc.get("issued_instructions", 0.0)),
        ipc_active=_w("ipc_active"),
        # Thread utilization
        avg_threads_per_warp=_w("avg_threads_per_warp"),
        # Register / shared memory pressure
        registers_per_thread=_m("registers_per_thread"),
        local_memory_spills=_s("local_memory_spills", as_int=True),
        dynamic_smem_per_block=_m("dynamic_smem_per_block"),
    )
