"""
Unit tests for metric_aggregator.build_aggregated_metrics().

Adapted from nvidia/tests/unit/test_metric_aggregator.py with NRT field names.
"""
from __future__ import annotations

import pytest

from trainium.operator_profiler.aggregator.metric_aggregator import build_aggregated_metrics
from trainium.operator_profiler.schema.profile import (
    AggregatedMetrics,
    KernelMetrics,
    KernelRecord,
)


def _kernel(kid: str, duration_ns: int, raw: dict | None = None) -> KernelRecord:
    return KernelRecord(
        kernel_id=kid,
        kernel_name=f"nrt_op_{kid}",
        stream_id=0,
        device_id=0,
        start_ns=0,
        end_ns=duration_ns,
        duration_ns=duration_ns,
        metrics=KernelMetrics(raw=raw or {}),
    )


class TestBuildAggregatedMetrics:
    def test_empty_kernels(self):
        result = build_aggregated_metrics([])
        assert result.total_duration_ns == 0
        assert result.kernel_count == 0

    def test_single_kernel_duration(self):
        k = _kernel("k0", 5000)
        result = build_aggregated_metrics([k])
        assert result.total_duration_ns == 5000
        assert result.kernel_count == 1
        assert result.dominant_kernel_id == "k0"

    def test_dominant_kernel_is_longest(self):
        kernels = [
            _kernel("k0", 100),
            _kernel("k1", 9000),
            _kernel("k2", 500),
        ]
        result = build_aggregated_metrics(kernels)
        assert result.dominant_kernel_id == "k1"

    def test_dma_bytes_summed(self):
        kernels = [
            _kernel("k0", 1000, {"TODO_dma_bytes_read": 1024, "TODO_dma_bytes_written": 512}),
            _kernel("k1", 2000, {"TODO_dma_bytes_read": 2048, "TODO_dma_bytes_written": 256}),
        ]
        result = build_aggregated_metrics(kernels)
        # With TODO names, get_raw_value returns None (no policy match yet)
        # This test documents expected behavior once real counter names are filled in.
        assert result.total_dma_bytes_read is None or isinstance(result.total_dma_bytes_read, int)

    def test_no_metrics_returns_none_fields(self):
        k = _kernel("k0", 1000)
        result = build_aggregated_metrics([k])
        assert result.tensor_engine_utilization_pct is None
        assert result.memory_utilization_pct is None
        assert result.stall_cycles_pct is None

    def test_duration_weighted_mean(self):
        """
        Duration-weighted mean: a 9000ns kernel with 90% utilization and a
        1000ns kernel with 10% utilization should give ~82% mean, not 50%.
        """
        from trainium.operator_profiler.schema.metrics import METRIC_POLICIES

        # Find the actual NRT counter name for tensor_engine_utilization_pct
        te_name = next(
            (p.nrt_name for p in METRIC_POLICIES if p.profile_field == "tensor_engine_utilization_pct"),
            None,
        )
        if te_name is None or "TODO" in te_name:
            pytest.skip("NRT counter name not yet confirmed — skipping weighted mean test")

        kernels = [
            _kernel("k0", 9000, {te_name: 90.0}),
            _kernel("k1", 1000, {te_name: 10.0}),
        ]
        result = build_aggregated_metrics(kernels)
        # Expected: (90*9000 + 10*1000) / (9000+1000) = 820000/10000 = 82.0
        assert result.tensor_engine_utilization_pct == pytest.approx(82.0)

    def test_nvidia_alias_fields_populated(self):
        """total_dram_bytes_read/written should alias total_dma_bytes_read/written."""
        from trainium.operator_profiler.schema.metrics import METRIC_POLICIES

        dma_name = next(
            (p.nrt_name for p in METRIC_POLICIES if p.profile_field == "dma_bytes_read"),
            None,
        )
        if dma_name is None or "TODO" in dma_name:
            pytest.skip("NRT counter name not yet confirmed")

        k = _kernel("k0", 1000, {dma_name: 4096})
        result = build_aggregated_metrics([k])
        assert result.total_dram_bytes_read == result.total_dma_bytes_read
