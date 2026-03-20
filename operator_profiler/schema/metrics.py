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
    MetricPolicy(
        "smsp__sass_thread_inst_executed.sum",
        "thread_inst_executed",
        AggregationOp.SUM,
        "Thread instructions executed (additive)",
    ),
]

# Map ncu column name → MetricPolicy for fast lookup
NCU_NAME_TO_POLICY: dict[str, MetricPolicy] = {p.ncu_name: p for p in METRIC_POLICIES}

# Minimal set of metrics for a standard ncu run
DEFAULT_NCU_METRICS: list[str] = [p.ncu_name for p in METRIC_POLICIES]
