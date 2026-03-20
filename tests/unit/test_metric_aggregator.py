"""
Unit tests for the metric aggregator.

Verifies:
  - SUM policy for additive metrics (dram bytes, instruction counts)
  - MEAN policy for ratio/rate metrics (occupancy, hit rates)
  - Single-kernel passthrough
  - Empty input returns a blank KernelMetrics
  - build_aggregated_metrics populates AggregatedMetrics correctly
  - Bottleneck classification heuristic
"""
import pytest

from operator_profiler.aggregator.metric_aggregator import (
    aggregate_fused_metrics,
    build_aggregated_metrics,
)
from operator_profiler.schema.profile import KernelMetrics


def km(**kwargs) -> KernelMetrics:
    return KernelMetrics(**kwargs)


class TestAggregateFusedMetrics:
    def test_empty_returns_blank(self):
        result = aggregate_fused_metrics([])
        assert result.dram_bytes_read is None
        assert result.achieved_occupancy is None

    def test_single_passthrough(self):
        m = km(dram_bytes_read=1024, achieved_occupancy=50.0)
        result = aggregate_fused_metrics([m])
        assert result.dram_bytes_read == 1024
        assert result.achieved_occupancy == pytest.approx(50.0)

    def test_dram_bytes_summed(self):
        """dram_bytes_read uses SUM policy — additive."""
        metrics = [km(dram_bytes_read=1000), km(dram_bytes_read=2000), km(dram_bytes_read=3000)]
        result = aggregate_fused_metrics(metrics)
        assert result.dram_bytes_read == 6000

    def test_dram_bytes_written_summed(self):
        metrics = [km(dram_bytes_written=500), km(dram_bytes_written=700)]
        result = aggregate_fused_metrics(metrics)
        assert result.dram_bytes_written == 1200

    def test_achieved_occupancy_meaned(self):
        """achieved_occupancy uses MEAN policy — ratio metric."""
        metrics = [km(achieved_occupancy=40.0), km(achieved_occupancy=60.0)]
        result = aggregate_fused_metrics(metrics)
        assert result.achieved_occupancy == pytest.approx(50.0)

    def test_l1_hit_rate_meaned(self):
        metrics = [km(l1_hit_rate=70.0), km(l1_hit_rate=90.0)]
        result = aggregate_fused_metrics(metrics)
        assert result.l1_hit_rate == pytest.approx(80.0)

    def test_tensor_core_pct_meaned(self):
        metrics = [km(tensor_core_active_pct=30.0), km(tensor_core_active_pct=50.0)]
        result = aggregate_fused_metrics(metrics)
        assert result.tensor_core_active_pct == pytest.approx(40.0)

    def test_raw_dict_aggregated(self):
        """Raw dict entries follow their policy from METRIC_POLICIES."""
        m1 = km(raw={"dram__bytes_read.sum": 1000, "l1tex__t_hit_rate.pct": 80.0})
        m2 = km(raw={"dram__bytes_read.sum": 2000, "l1tex__t_hit_rate.pct": 60.0})
        result = aggregate_fused_metrics([m1, m2])
        assert result.raw["dram__bytes_read.sum"] == 3000         # sum
        assert result.raw["l1tex__t_hit_rate.pct"] == pytest.approx(70.0)  # mean

    def test_none_values_skipped(self):
        """None values must not affect aggregation of other kernels."""
        metrics = [
            km(dram_bytes_read=None),
            km(dram_bytes_read=1000),
        ]
        result = aggregate_fused_metrics(metrics)
        assert result.dram_bytes_read == 1000

    def test_all_none_fields_remain_none(self):
        metrics = [km(), km(), km()]
        result = aggregate_fused_metrics(metrics)
        assert result.dram_bytes_read is None
        assert result.achieved_occupancy is None


class TestBuildAggregatedMetrics:
    def test_basic_aggregation(self):
        metrics = [
            km(dram_bytes_read=1000, dram_bytes_written=500, achieved_occupancy=60.0),
            km(dram_bytes_read=2000, dram_bytes_written=800, achieved_occupancy=80.0),
        ]
        agg = build_aggregated_metrics(metrics, total_duration_ns=500)
        assert agg.kernel_count == 2
        assert agg.total_duration_ns == 500
        assert agg.total_dram_bytes_read == 3000
        assert agg.total_dram_bytes_written == 1300
        assert agg.mean_achieved_occupancy == pytest.approx(70.0)

    def test_empty_metrics(self):
        agg = build_aggregated_metrics([], total_duration_ns=0)
        assert agg.kernel_count == 0
        assert agg.total_duration_ns == 0

    def test_bottleneck_compute_bound(self):
        metrics = [km(arithmetic_intensity=15.0, achieved_occupancy=65.0)]
        agg = build_aggregated_metrics(metrics, total_duration_ns=50_000)
        assert agg.bottleneck_classification == "compute_bound"

    def test_bottleneck_memory_bound(self):
        metrics = [km(arithmetic_intensity=2.0, achieved_occupancy=30.0)]
        agg = build_aggregated_metrics(metrics, total_duration_ns=50_000)
        assert agg.bottleneck_classification == "memory_bound"

    def test_bottleneck_latency_bound(self):
        metrics = [km(arithmetic_intensity=7.0, achieved_occupancy=30.0)]
        agg = build_aggregated_metrics(metrics, total_duration_ns=5_000)
        assert agg.bottleneck_classification == "latency_bound"

    def test_bottleneck_unknown_without_ai(self):
        metrics = [km(achieved_occupancy=50.0)]  # no arithmetic_intensity
        agg = build_aggregated_metrics(metrics, total_duration_ns=50_000)
        assert agg.bottleneck_classification == "unknown"
