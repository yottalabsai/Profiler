"""
Unit tests for the metric aggregator.

Verifies:
  - SUM policy for additive metrics (dram bytes) via raw NCU names
  - MEAN policy for ratio/rate metrics (occupancy, hit rates)
  - Single-kernel passthrough
  - Empty input returns a blank KernelMetrics
  - build_aggregated_metrics populates AggregatedMetrics correctly from KernelRecords
"""
import pytest

from operator_profiler.aggregator.metric_aggregator import (
    aggregate_fused_metrics,
    build_aggregated_metrics,
)
from operator_profiler.schema.metrics import PROFILE_FIELD_TO_NCU_NAMES, get_raw_value
from operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    KernelMetrics,
    KernelRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def km(**kwargs) -> KernelMetrics:
    """
    Build a KernelMetrics using logical field names (dram_bytes_read, achieved_occupancy,
    etc.) or raw NCU names via raw={...}.

    Logical names are translated to the first canonical NCU name from PROFILE_FIELD_TO_NCU_NAMES.
    None values are treated as absent (not inserted into raw).
    """
    raw: dict[str, float | int | str] = {}
    for field, value in kwargs.items():
        if field == "raw":
            raw.update(value)
        elif field in PROFILE_FIELD_TO_NCU_NAMES and value is not None:
            raw[PROFILE_FIELD_TO_NCU_NAMES[field][0]] = value
        # None or unknown fields: skip (absent metric)
    return KernelMetrics(raw=raw)


def kr(duration_ns: int = 100, **metrics_kwargs) -> KernelRecord:
    """Build a minimal KernelRecord with given duration and KernelMetrics."""
    return KernelRecord(
        kernel_id="k_00000",
        kernel_name="test_kernel",
        stream_id=0,
        device_id=0,
        start_ns=0,
        end_ns=duration_ns,
        duration_ns=duration_ns,
        metrics=km(**metrics_kwargs),
        attribution_method=AttributionMethod.UNATTRIBUTED,
        confidence=Confidence.UNATTRIBUTED,
    )


# ---------------------------------------------------------------------------
# aggregate_fused_metrics
# ---------------------------------------------------------------------------

class TestAggregateFusedMetrics:
    def test_empty_returns_blank(self):
        result = aggregate_fused_metrics([])
        assert get_raw_value(result.raw, "dram_bytes_read") is None
        assert get_raw_value(result.raw, "achieved_occupancy") is None

    def test_single_passthrough(self):
        m = km(dram_bytes_read=1024, achieved_occupancy=50.0)
        result = aggregate_fused_metrics([m])
        assert get_raw_value(result.raw, "dram_bytes_read") == pytest.approx(1024)
        assert get_raw_value(result.raw, "achieved_occupancy") == pytest.approx(50.0)

    def test_dram_bytes_summed(self):
        """dram_bytes_read uses SUM policy — additive."""
        metrics = [km(dram_bytes_read=1000), km(dram_bytes_read=2000), km(dram_bytes_read=3000)]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "dram_bytes_read") == pytest.approx(6000)

    def test_dram_bytes_written_summed(self):
        metrics = [km(dram_bytes_written=500), km(dram_bytes_written=700)]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "dram_bytes_written") == pytest.approx(1200)

    def test_achieved_occupancy_meaned(self):
        """achieved_occupancy uses MEAN policy — ratio metric."""
        metrics = [km(achieved_occupancy=40.0), km(achieved_occupancy=60.0)]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "achieved_occupancy") == pytest.approx(50.0)

    def test_l1_hit_rate_meaned(self):
        metrics = [km(l1_hit_rate=70.0), km(l1_hit_rate=90.0)]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "l1_hit_rate") == pytest.approx(80.0)

    def test_tensor_core_pct_meaned(self):
        metrics = [km(tensor_core_active_pct=30.0), km(tensor_core_active_pct=50.0)]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "tensor_core_active_pct") == pytest.approx(40.0)

    def test_raw_dict_aggregated(self):
        """Raw dict entries follow their policy from METRIC_POLICIES."""
        m1 = km(raw={"dram__bytes_read.sum": 1000, "l1tex__t_hit_rate.pct": 80.0})
        m2 = km(raw={"dram__bytes_read.sum": 2000, "l1tex__t_hit_rate.pct": 60.0})
        result = aggregate_fused_metrics([m1, m2])
        assert result.raw["dram__bytes_read.sum"] == 3000         # sum
        assert result.raw["l1tex__t_hit_rate.pct"] == pytest.approx(70.0)  # mean

    def test_none_values_skipped(self):
        """None values (absent metrics) must not affect aggregation of other kernels."""
        metrics = [
            km(dram_bytes_read=None),   # absent → empty raw
            km(dram_bytes_read=1000),
        ]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "dram_bytes_read") == pytest.approx(1000)

    def test_all_none_fields_remain_none(self):
        metrics = [km(), km(), km()]
        result = aggregate_fused_metrics(metrics)
        assert get_raw_value(result.raw, "dram_bytes_read") is None
        assert get_raw_value(result.raw, "achieved_occupancy") is None


# ---------------------------------------------------------------------------
# build_aggregated_metrics (operates on list[KernelRecord])
# ---------------------------------------------------------------------------

class TestBuildAggregatedMetrics:
    def test_basic_aggregation(self):
        kernels = [
            kr(duration_ns=200, dram_bytes_read=1000, dram_bytes_written=500, achieved_occupancy=60.0),
            kr(duration_ns=300, dram_bytes_read=2000, dram_bytes_written=800, achieved_occupancy=80.0),
        ]
        agg = build_aggregated_metrics(kernels)
        assert agg.kernel_count == 2
        assert agg.total_duration_ns == 500
        assert agg.total_dram_bytes_read == 3000
        assert agg.total_dram_bytes_written == 1300
        # Duration-weighted mean: (60*200 + 80*300) / 500 = 36000/500 = 72.0
        assert agg.mean_achieved_occupancy == pytest.approx(72.0)

    def test_empty_metrics(self):
        agg = build_aggregated_metrics([])
        assert agg.kernel_count == 0
        assert agg.total_duration_ns == 0

    def test_single_kernel(self):
        kernels = [kr(duration_ns=100, dram_bytes_read=512, achieved_occupancy=45.0)]
        agg = build_aggregated_metrics(kernels)
        assert agg.kernel_count == 1
        assert agg.total_duration_ns == 100
        assert agg.total_dram_bytes_read == 512
        assert agg.mean_achieved_occupancy == pytest.approx(45.0)

    def test_missing_metrics_are_none(self):
        """Kernels with no metrics should yield None for rate fields."""
        kernels = [kr(duration_ns=100), kr(duration_ns=200)]
        agg = build_aggregated_metrics(kernels)
        assert agg.total_dram_bytes_read is None
        assert agg.mean_achieved_occupancy is None

    def test_bottleneck_classification_not_set_by_aggregator(self):
        """bottleneck_classification is set by DiagnosisAgent, not the aggregator."""
        kernels = [kr(duration_ns=100, achieved_occupancy=65.0)]
        agg = build_aggregated_metrics(kernels)
        assert agg.bottleneck_classification is None

    def test_dominant_kernel_id(self):
        kernels = [
            kr(duration_ns=50),
            kr(duration_ns=200),
            kr(duration_ns=100),
        ]
        # Give distinct IDs
        kernels[0].kernel_id = "k_00000"
        kernels[1].kernel_id = "k_00001"
        kernels[2].kernel_id = "k_00002"
        agg = build_aggregated_metrics(kernels)
        assert agg.dominant_kernel_id == "k_00001"
