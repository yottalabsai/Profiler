"""
Metrics schema helpers — aggregation policies and metric name constants.

This module is the single source of truth for ncu metric names used across
ncu_parser.py and metric_aggregator.py.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class AggregationOp(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class MetricPolicy(NamedTuple):
    ncu_name: str          # exact ncu CSV column name
    profile_field: str     # field name in KernelMetrics
    aggregation: AggregationOp
    description: str


# ---------------------------------------------------------------------------
# Canonical metric policies — used by metric_aggregator.py
#
# These 20 counters cover every major bottleneck axis for PyTorch FX graph
# operator-level optimization: memory bandwidth, cache efficiency, compute
# utilization, occupancy, latency, instruction throughput, and register
# pressure.  They replace --set full (~90 metrics) to keep profile.json small
# enough for LLM context.
# ---------------------------------------------------------------------------
METRIC_POLICIES: list[MetricPolicy] = [
    # --- Time ---
    MetricPolicy(
        "gpu__time_duration.sum",
        "gpu_time_duration_ns",
        AggregationOp.SUM,
        "GPU wall time (sequential assumption — use sum)",
    ),
    # --- Memory bandwidth ---
    MetricPolicy(
        "dram__bytes_read.sum",
        "dram_bytes_read",
        AggregationOp.SUM,
        "Total DRAM bytes read",
    ),
    MetricPolicy(
        "dram__bytes_written.sum",
        "dram_bytes_written",
        AggregationOp.SUM,
        "Total DRAM bytes written",
    ),
    # --- Memory subsystem throughput ---
    MetricPolicy(
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "memory_throughput_pct",
        AggregationOp.MEAN,
        "Overall memory subsystem utilization % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram_throughput_pct",
        AggregationOp.MEAN,
        "DRAM bandwidth utilization % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "mem_busy_pct",
        AggregationOp.MEAN,
        "Memory pipeline busy % of peak (rate — use mean)",
    ),
    # --- Cache efficiency ---
    MetricPolicy(
        "l1tex__t_hit_rate.pct",
        "l1_hit_rate",
        AggregationOp.MEAN,
        "L1 cache hit rate (ratio — use mean)",
    ),
    MetricPolicy(
        "lts__t_hit_rate.pct",
        "l2_hit_rate",
        AggregationOp.MEAN,
        "L2 cache hit rate (ratio — use mean)",
    ),
    # --- Compute utilization ---
    MetricPolicy(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm_throughput_pct",
        AggregationOp.MEAN,
        "SM throughput % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "tensor_core_active_pct",
        AggregationOp.MEAN,
        "Tensor core active % (ratio — use mean)",
    ),
    # --- SM cycles ---
    MetricPolicy(
        "sm__active_cycles_elapsed.sum",
        "sm_active_cycles",
        AggregationOp.SUM,
        "SM active cycles (total across kernels)",
    ),
    # --- Occupancy ---
    MetricPolicy(
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "achieved_occupancy",
        AggregationOp.MEAN,
        "Achieved occupancy (ratio — use mean)",
    ),
    # --- Latency / warp scheduling ---
    MetricPolicy(
        "smsp__warp_cycles_per_issued_instruction.ratio",
        "warp_cycles_per_instruction",
        AggregationOp.MEAN,
        "Avg cycles per issued instruction — high values indicate latency-bound (use mean)",
    ),
    MetricPolicy(
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "eligible_cycles_pct",
        AggregationOp.MEAN,
        "% of cycles with at least one eligible warp — low values = latency-bound (use mean)",
    ),
    # --- Instruction throughput ---
    MetricPolicy(
        "smsp__inst_executed.sum",
        "executed_instructions",
        AggregationOp.SUM,
        "Total executed instructions",
    ),
    MetricPolicy(
        "smsp__inst_executed.avg.per_cycle_active",
        "ipc_active",
        AggregationOp.MEAN,
        "Instructions per active SM cycle — instruction throughput efficiency (use mean)",
    ),
    # --- Thread utilization ---
    MetricPolicy(
        "smsp__average_threads_executed_per_instruction.ratio",
        "avg_threads_per_warp",
        AggregationOp.MEAN,
        "Avg active threads per warp — values below 32 indicate control-flow divergence (use mean)",
    ),
    # --- Register / shared memory pressure ---
    MetricPolicy(
        "launch__registers_per_thread",
        "registers_per_thread",
        AggregationOp.MAX,
        "Registers per thread — high values limit occupancy (max across kernels)",
    ),
    MetricPolicy(
        "l1tex__data_pipe_lsu_wavefronts_mem_local.sum",
        "local_memory_spills",
        AggregationOp.SUM,
        "Register spills to local (DRAM-backed) memory — nonzero is costly",
    ),
    MetricPolicy(
        "launch__shared_mem_per_block_dynamic",
        "dynamic_smem_per_block",
        AggregationOp.MAX,
        "Dynamic shared memory per block (bytes) — high values limit occupancy (max across kernels)",
    ),
]

# Map ncu column name → MetricPolicy for fast lookup
NCU_NAME_TO_POLICY: dict[str, MetricPolicy] = {p.ncu_name: p for p in METRIC_POLICIES}

# Human-readable aliases emitted by `ncu --set` (default/full) instead of raw
# counter names.  These are NOT added to METRIC_POLICIES (and thus not to
# AGGREGATE_NCU_METRICS) since they cannot be passed to `--metrics`.
# They allow get_raw_value() to find metrics in profile.json files that were
# captured with --set full.
_HUMAN_READABLE_ALIASES: list[MetricPolicy] = [
    # --- Existing aliases ---
    MetricPolicy(
        "Achieved Occupancy",
        "achieved_occupancy",
        AggregationOp.MEAN,
        "Achieved occupancy (--set alias)",
    ),
    MetricPolicy(
        "SM Active Cycles",
        "sm_active_cycles",
        AggregationOp.MEAN,
        "SM active cycles (--set alias)",
    ),
    MetricPolicy(
        "L1/TEX Hit Rate",
        "l1_hit_rate",
        AggregationOp.MEAN,
        "L1 cache hit rate (--set alias)",
    ),
    MetricPolicy(
        "Compute (SM) Throughput",
        "sm_throughput_pct",
        AggregationOp.MEAN,
        "SM throughput % of peak (--set alias)",
    ),
    # --- New aliases for --set full backward compatibility ---
    MetricPolicy(
        "Executed Instructions",
        "executed_instructions",
        AggregationOp.SUM,
        "Executed instructions (--set alias)",
    ),
    MetricPolicy(
        "Issued Instructions",
        "issued_instructions",
        AggregationOp.SUM,
        "Issued instructions (--set alias; backward compat for total_issued_instructions)",
    ),
    MetricPolicy(
        "L2 Hit Rate",
        "l2_hit_rate",
        AggregationOp.MEAN,
        "L2 cache hit rate (--set alias)",
    ),
    MetricPolicy(
        "Warp Cycles Per Issued Instruction",
        "warp_cycles_per_instruction",
        AggregationOp.MEAN,
        "Warp cycles per instruction (--set alias)",
    ),
    MetricPolicy(
        "Memory Throughput",
        "memory_throughput_pct",
        AggregationOp.MEAN,
        "Overall memory throughput % (--set alias)",
    ),
    MetricPolicy(
        "DRAM Throughput",
        "dram_throughput_pct",
        AggregationOp.MEAN,
        "DRAM throughput % (--set alias)",
    ),
    MetricPolicy(
        "Mem Busy",
        "mem_busy_pct",
        AggregationOp.MEAN,
        "Memory pipeline busy % (--set alias)",
    ),
    MetricPolicy(
        "One or More Eligible",
        "eligible_cycles_pct",
        AggregationOp.MEAN,
        "Cycles with at least one eligible warp % (--set alias)",
    ),
    MetricPolicy(
        "Executed Ipc Active",
        "ipc_active",
        AggregationOp.MEAN,
        "IPC when SM active (--set alias)",
    ),
    MetricPolicy(
        "Avg. Active Threads Per Warp",
        "avg_threads_per_warp",
        AggregationOp.MEAN,
        "Avg active threads per warp (--set alias)",
    ),
    MetricPolicy(
        "Registers Per Thread",
        "registers_per_thread",
        AggregationOp.MAX,
        "Registers per thread (--set alias)",
    ),
    MetricPolicy(
        "Local Memory Spilling Requests",
        "local_memory_spills",
        AggregationOp.SUM,
        "Register spills to local memory (--set alias)",
    ),
    MetricPolicy(
        "Dynamic Shared Memory Per Block",
        "dynamic_smem_per_block",
        AggregationOp.MAX,
        "Dynamic shared memory per block (--set alias)",
    ),
]
NCU_NAME_TO_POLICY.update({p.ncu_name: p for p in _HUMAN_READABLE_ALIASES})

# Reverse lookup: logical profile_field → all NCU names (raw counters + aliases)
# Used by get_raw_value() to find a metric regardless of which name NCU used.
PROFILE_FIELD_TO_NCU_NAMES: dict[str, list[str]] = {}
for _p in NCU_NAME_TO_POLICY.values():
    PROFILE_FIELD_TO_NCU_NAMES.setdefault(_p.profile_field, []).append(_p.ncu_name)


def get_raw_value(
    raw: dict[str, "float | int | str"], profile_field: str
) -> "float | None":
    """
    Look up a metric from a KernelMetrics.raw dict by logical name.

    Checks all NCU name aliases (raw counter names and human-readable names
    from --set mode) so callers don't need to know which variant NCU used.
    Returns the first numeric match, or None if absent or non-numeric.
    """
    for ncu_name in PROFILE_FIELD_TO_NCU_NAMES.get(profile_field, []):
        val = raw.get(ncu_name)
        if isinstance(val, (int, float)):
            return float(val)
    return None


# Explicit metric list passed to `ncu --metrics` when ncu_metric_set is empty.
# Derived from METRIC_POLICIES — covers all 20 counters needed for bottleneck
# classification at the operator level.  Replaces --set full (~90 metrics) to
# keep KernelMetrics.raw small enough for LLM context.
#
# NOTE: smsp__sass_thread_inst_executed.sum is NOT supported in
# --replay-mode range (ncu restriction); omitted intentionally.
AGGREGATE_NCU_METRICS: list[str] = [p.ncu_name for p in METRIC_POLICIES]
