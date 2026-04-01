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
# ---------------------------------------------------------------------------
METRIC_POLICIES: list[MetricPolicy] = [
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
    MetricPolicy(
        "sm__active_cycles_elapsed.sum",
        "sm_active_cycles",
        AggregationOp.SUM,
        "SM active cycles (total across kernels)",
    ),
    MetricPolicy(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm_throughput_pct",
        AggregationOp.MEAN,
        "SM throughput % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "l1tex__t_hit_rate.pct",
        "l1_hit_rate",
        AggregationOp.MEAN,
        "L1 cache hit rate (ratio — use mean)",
    ),
    MetricPolicy(
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "achieved_occupancy",
        AggregationOp.MEAN,
        "Achieved occupancy (ratio — use mean)",
    ),
    MetricPolicy(
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "tensor_core_active_pct",
        AggregationOp.MEAN,
        "Tensor core active % (ratio — use mean)",
    ),
    MetricPolicy(
        "gpu__time_duration.sum",
        "gpu_time_duration_ns",
        AggregationOp.SUM,
        "GPU wall time (sequential assumption — use sum)",
    ),
    # NOTE: smsp__sass_thread_inst_executed.sum is NOT supported in
    # --replay-mode range (ncu restriction); omitted from DEFAULT_NCU_METRICS.
    # Collect it separately with --replay-mode kernel if needed.
]

# Map ncu column name → MetricPolicy for fast lookup
NCU_NAME_TO_POLICY: dict[str, MetricPolicy] = {p.ncu_name: p for p in METRIC_POLICIES}

# Human-readable aliases emitted by `ncu --set` (default/full) instead of raw
# counter names.  These are NOT added to METRIC_POLICIES (and thus not to
# DEFAULT_NCU_METRICS) since they cannot be passed to `--metrics`.
_HUMAN_READABLE_ALIASES: list[MetricPolicy] = [
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


# Minimal set of metrics for a standard ncu run
DEFAULT_NCU_METRICS: list[str] = [p.ncu_name for p in METRIC_POLICIES]
