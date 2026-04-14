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

    total_ns = sum(k.duration_ns for k in kernels)
    dominant = max(kernels, key=lambda k: k.duration_ns)

    # --- Additive quantities ---
    dram_read = _sum_kernels(kernels, "dram_bytes_read", as_int=True)
    dram_write = _sum_kernels(kernels, "dram_bytes_written", as_int=True)
    total_executed = int(sum(
        v for k in kernels
        if (v := get_raw_value(k.metrics.raw, "executed_instructions")) is not None
    ))
    total_issued = int(sum(
        v for k in kernels
        if (v := get_raw_value(k.metrics.raw, "issued_instructions")) is not None
    ))
    local_spills = _sum_kernels(kernels, "local_memory_spills", as_int=True)

    # --- Duration-weighted means (rate / utilization metrics) ---
    mean_occ = _duration_weighted_mean(kernels, "achieved_occupancy")
    mean_tensor = _duration_weighted_mean(kernels, "tensor_core_active_pct")
    mean_sm_thru = _duration_weighted_mean(kernels, "sm_throughput_pct")
    mean_l1 = _duration_weighted_mean(kernels, "l1_hit_rate")
    mean_l2 = _duration_weighted_mean(kernels, "l2_hit_rate")
    mean_mem_thru = _duration_weighted_mean(kernels, "memory_throughput_pct")
    mean_dram_thru = _duration_weighted_mean(kernels, "dram_throughput_pct")
    mean_mem_busy = _duration_weighted_mean(kernels, "mem_busy_pct")
    mean_warp_cycles = _duration_weighted_mean(kernels, "warp_cycles_per_instruction")
    mean_eligible = _duration_weighted_mean(kernels, "eligible_cycles_pct")
    mean_ipc = _duration_weighted_mean(kernels, "ipc_active")
    mean_threads = _duration_weighted_mean(kernels, "avg_threads_per_warp")

    # --- Max across kernels (per-kernel constants) ---
    max_regs = _max_kernels(kernels, "registers_per_thread")
    max_smem = _max_kernels(kernels, "dynamic_smem_per_block")

    return AggregatedMetrics(
        total_duration_ns=total_ns,
        kernel_count=len(kernels),
        dominant_kernel_id=dominant.kernel_id,
        # Memory bandwidth
        total_dram_bytes_read=dram_read,
        total_dram_bytes_written=dram_write,
        # Memory throughput
        memory_throughput_pct=mean_mem_thru,
        dram_throughput_pct=mean_dram_thru,
        mem_busy_pct=mean_mem_busy,
        # Cache efficiency
        l1_hit_rate=mean_l1,
        l2_hit_rate=mean_l2,
        # Compute utilization
        sm_throughput_pct=mean_sm_thru,
        tensor_core_active_pct=mean_tensor,
        # Occupancy & latency
        achieved_occupancy=mean_occ,
        warp_cycles_per_instruction=mean_warp_cycles,
        eligible_cycles_pct=mean_eligible,
        # Instruction throughput
        total_executed_instructions=total_executed,
        total_issued_instructions=total_issued,
        ipc_active=mean_ipc,
        # Thread utilization
        avg_threads_per_warp=mean_threads,
        # Register / shared memory pressure
        registers_per_thread=max_regs,
        local_memory_spills=local_spills,
        dynamic_smem_per_block=max_smem,
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


def _sum_kernels(
    kernels: list[KernelRecord], profile_field: str, *, as_int: bool = False
) -> "int | float | None":
    """Sum a metric across kernels. Returns None if no kernel has the metric."""
    vals = [v for k in kernels if (v := get_raw_value(k.metrics.raw, profile_field)) is not None]
    if not vals:
        return None
    total = sum(vals)
    return int(total) if as_int else total


def _max_kernels(kernels: list[KernelRecord], profile_field: str) -> float | None:
    """Max of a metric across kernels. Returns None if no kernel has the metric."""
    vals = [v for k in kernels if (v := get_raw_value(k.metrics.raw, profile_field)) is not None]
    return max(vals) if vals else None
