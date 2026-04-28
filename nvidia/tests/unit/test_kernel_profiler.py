"""
Unit tests for KernelProfileOrchestrator.

Tests _build_replay_targets(), _merge_metrics(), _apply_metrics_to_records(),
and the run() integration path. No GPU or subprocess required — _profile_all()
is mocked to return a pre-built metrics_map.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nvidia.operator_profiler.mapper.kernel_profiler import (
    KernelProfileConfig,
    KernelProfileOrchestrator,
    KernelReplayTarget,
)
from nvidia.operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    KernelAttribution,
    KernelManifestEntry,
    MappingManifest,
)
from nvidia.operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    KernelMetrics,
    KernelRecord,
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _manifest_entry(kernel_id: str, kernel_name: str, start_ns: int = 0) -> KernelManifestEntry:
    return KernelManifestEntry(
        kernel_id=kernel_id,
        kernel_name=kernel_name,
        stream_id=7,
        device_id=0,
        start_ns=start_ns,
        end_ns=start_ns + 1000,
        duration_ns=1000,
        attribution=KernelAttribution(
            method=AttributionMethod.NVTX,
            confidence=Confidence.MEDIUM,
        ),
    )


def _kernel_record(kernel_id: str, kernel_name: str) -> KernelRecord:
    return KernelRecord(
        kernel_id=kernel_id,
        kernel_name=kernel_name,
        stream_id=7,
        device_id=0,
        start_ns=0,
        end_ns=1000,
        duration_ns=1000,
    )


def _make_manifest(*entries: KernelManifestEntry) -> MappingManifest:
    meta = CaptureManifestMetadata(
        model_name="test",
        torch_version="2.0",
        compile_mode="eager",
        capture_timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    return MappingManifest(capture_metadata=meta, kernels=list(entries))


def _metrics(cycles: float = 1000.0) -> KernelMetrics:
    return KernelMetrics(raw={"sm__cycles_active.sum": cycles})


def _make_orch(manifest: MappingManifest, operator_records: list[OperatorRecord]) -> KernelProfileOrchestrator:
    config = KernelProfileConfig(replay_script="/tmp/fake_replay.py")
    return KernelProfileOrchestrator(manifest, operator_records, config)


# ---------------------------------------------------------------------------
# _build_replay_targets()
# ---------------------------------------------------------------------------

class TestBuildReplayTargets:
    def test_single_kernel_one_invocation(self):
        manifest = _make_manifest(_manifest_entry("k_00000", "gemm_kernel"))
        orch = _make_orch(manifest, [])
        targets = orch._build_replay_targets()
        assert len(targets) == 1
        assert targets[0].kernel_name == "gemm_kernel"
        assert targets[0].kernel_ids == ["k_00000"]

    def test_two_kernels_one_invocation_each(self):
        manifest = _make_manifest(
            _manifest_entry("k_00000", "kernel_A"),
            _manifest_entry("k_00001", "kernel_B"),
        )
        orch = _make_orch(manifest, [])
        targets = orch._build_replay_targets()
        names = {t.kernel_name for t in targets}
        assert names == {"kernel_A", "kernel_B"}

    def test_same_kernel_multiple_invocations_preserves_launch_order(self):
        """K1 at positions 0 and 2; K2 at position 1 — kernel_ids must stay in order."""
        manifest = _make_manifest(
            _manifest_entry("k_00000", "K1", start_ns=0),
            _manifest_entry("k_00001", "K2", start_ns=1000),
            _manifest_entry("k_00002", "K1", start_ns=2000),
        )
        orch = _make_orch(manifest, [])
        targets = orch._build_replay_targets()
        k1 = next(t for t in targets if t.kernel_name == "K1")
        assert k1.kernel_ids == ["k_00000", "k_00002"], "Launch order must be preserved"

    def test_three_unique_kernels(self):
        manifest = _make_manifest(
            _manifest_entry("k_0", "A"),
            _manifest_entry("k_1", "B"),
            _manifest_entry("k_2", "C"),
            _manifest_entry("k_3", "A"),
        )
        orch = _make_orch(manifest, [])
        targets = orch._build_replay_targets()
        assert len(targets) == 3


# ---------------------------------------------------------------------------
# _merge_metrics()
# ---------------------------------------------------------------------------

class TestMergeMetrics:
    def _run_merge(self, target: KernelReplayTarget, metrics_map) -> dict:
        manifest = _make_manifest()
        orch = _make_orch(manifest, [])
        orch._merge_metrics(target, metrics_map)
        return orch._kernel_metrics

    def test_perfect_match_two_invocations(self):
        target = KernelReplayTarget(
            kernel_name="gemv2T_kernel",
            kernel_ids=["k_00000", "k_00001"],
        )
        m0 = _metrics(cycles=100.0)
        m1 = _metrics(cycles=200.0)
        metrics_map = {
            ("void gemv2T_kernel<int>(float*)", "0"): m0,
            ("void gemv2T_kernel<int>(float*)", "1"): m1,
        }
        result = self._run_merge(target, metrics_map)
        assert result["k_00000"].raw["sm__cycles_active.sum"] == 100.0
        assert result["k_00001"].raw["sm__cycles_active.sum"] == 200.0

    def test_substring_match_short_vs_mangled_name(self):
        """Short manifest name must match against full mangled ncu name."""
        target = KernelReplayTarget(
            kernel_name="gemv2T_kernel",
            kernel_ids=["k_00000"],
        )
        metrics_map = {
            ("void gemv2T_kernel<int, float, 128>(float*, int)", "0"): _metrics(999.0),
        }
        result = self._run_merge(target, metrics_map)
        assert "k_00000" in result
        assert result["k_00000"].raw["sm__cycles_active.sum"] == 999.0

    def test_no_false_match_for_unrelated_kernel(self):
        """A kernel whose name does not contain the short name must not match."""
        target = KernelReplayTarget(
            kernel_name="softmax_kernel",
            kernel_ids=["k_00000"],
        )
        metrics_map = {
            ("void gemm_kernel<float>(float*, float*)", "0"): _metrics(42.0),
        }
        result = self._run_merge(target, metrics_map)
        assert "k_00000" not in result  # no match → no entry

    def test_invocation_count_mismatch_logs_warning(self, caplog):
        """Manifest has 3 entries, ncu returned only 2 rows → log warning, no crash."""
        import logging
        target = KernelReplayTarget(
            kernel_name="kernel_X",
            kernel_ids=["k_0", "k_1", "k_2"],
        )
        metrics_map = {
            ("kernel_X", "0"): _metrics(10.0),
            ("kernel_X", "1"): _metrics(20.0),
        }
        with caplog.at_level(logging.WARNING, logger="nvidia.operator_profiler.mapper.kernel_profiler"):
            result = self._run_merge(target, metrics_map)

        assert "k_0" in result
        assert "k_1" in result
        assert "k_2" not in result  # third entry has no matching row
        assert any("k_2" in r.message or "invocation" in r.message.lower() for r in caplog.records)

    def test_id_sort_order_is_numeric(self):
        """IDs must be sorted numerically ("9" < "10"), not lexicographically."""
        target = KernelReplayTarget(
            kernel_name="kk",
            kernel_ids=[f"k_{i:05d}" for i in range(12)],
        )
        metrics_map = {("kk", str(i)): _metrics(float(i * 10)) for i in range(12)}
        result = self._run_merge(target, metrics_map)
        # k_00000 → ncu id "0" (cycles 0), k_00009 → id "9" (cycles 90), k_00010 → id "10" (cycles 100)
        assert result["k_00000"].raw["sm__cycles_active.sum"] == 0.0
        assert result["k_00009"].raw["sm__cycles_active.sum"] == 90.0
        assert result["k_00010"].raw["sm__cycles_active.sum"] == 100.0


# ---------------------------------------------------------------------------
# _apply_metrics_to_records()
# ---------------------------------------------------------------------------

class TestApplyMetricsToRecords:
    def test_metrics_written_to_matching_kernel_records(self):
        op = OperatorRecord(
            operator_id="aten::linear_0",
            operator_name="aten::linear",
            call_index=0,
            kernels=[
                _kernel_record("k_00000", "gemm"),
                _kernel_record("k_00001", "bias_add"),
            ],
        )
        manifest = _make_manifest()
        orch = _make_orch(manifest, [op])
        orch._kernel_metrics = {
            "k_00000": _metrics(cycles=500.0),
            "k_00001": _metrics(cycles=600.0),
        }
        orch._apply_metrics_to_records()

        assert op.kernels[0].metrics.raw["sm__cycles_active.sum"] == 500.0
        assert op.kernels[1].metrics.raw["sm__cycles_active.sum"] == 600.0

    def test_kernels_without_metrics_entry_remain_empty(self):
        op = OperatorRecord(
            operator_id="op_0",
            operator_name="aten::relu",
            call_index=0,
            kernels=[_kernel_record("k_00000", "relu_kernel")],
        )
        manifest = _make_manifest()
        orch = _make_orch(manifest, [op])
        orch._kernel_metrics = {}  # no metrics at all
        orch._apply_metrics_to_records()

        assert op.kernels[0].metrics.raw == {}

    def test_metrics_not_overwritten_for_unmatched_id(self):
        op = OperatorRecord(
            operator_id="op_0",
            operator_name="aten::mm",
            call_index=0,
            kernels=[
                _kernel_record("k_00000", "gemm"),
                _kernel_record("k_99999", "gemm"),  # id not in metrics map
            ],
        )
        manifest = _make_manifest()
        orch = _make_orch(manifest, [op])
        orch._kernel_metrics = {"k_00000": _metrics(111.0)}
        orch._apply_metrics_to_records()

        assert op.kernels[0].metrics.raw["sm__cycles_active.sum"] == 111.0
        assert op.kernels[1].metrics.raw == {}


# ---------------------------------------------------------------------------
# run() integration (mocked _profile_all)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    def test_run_calls_profile_all_once_and_populates_records(self, tmp_path):
        """run() must call _profile_all exactly once (not per-kernel) and populate metrics."""
        manifest = _make_manifest(
            _manifest_entry("k_00000", "kernel_A"),
            _manifest_entry("k_00001", "kernel_A"),
            _manifest_entry("k_00002", "kernel_B"),
        )
        op_a = OperatorRecord(
            operator_id="op_0", operator_name="aten::mm", call_index=0,
            kernels=[
                _kernel_record("k_00000", "kernel_A"),
                _kernel_record("k_00001", "kernel_A"),
            ],
        )
        op_b = OperatorRecord(
            operator_id="op_1", operator_name="aten::relu", call_index=1,
            kernels=[_kernel_record("k_00002", "kernel_B")],
        )

        fake_metrics_map = {
            ("kernel_A", "0"): _metrics(100.0),
            ("kernel_A", "1"): _metrics(200.0),
            ("kernel_B", "0"): _metrics(300.0),
        }

        config = KernelProfileConfig(
            replay_script="/tmp/fake.py",
            output_dir=str(tmp_path),
        )
        orch = KernelProfileOrchestrator(manifest, [op_a, op_b], config)

        profile_all_calls = []

        def fake_profile_all(output_dir):
            profile_all_calls.append(output_dir)
            return fake_metrics_map

        orch._profile_all = fake_profile_all
        orch.run()

        assert len(profile_all_calls) == 1, "_profile_all must be called exactly once"
        assert op_a.kernels[0].metrics.raw["sm__cycles_active.sum"] == 100.0
        assert op_a.kernels[1].metrics.raw["sm__cycles_active.sum"] == 200.0
        assert op_b.kernels[0].metrics.raw["sm__cycles_active.sum"] == 300.0

    def test_run_handles_empty_metrics_map_gracefully(self, tmp_path):
        """If ncu produces no output, run() should complete without raising."""
        manifest = _make_manifest(_manifest_entry("k_00000", "some_kernel"))
        op = OperatorRecord(
            operator_id="op_0", operator_name="aten::mm", call_index=0,
            kernels=[_kernel_record("k_00000", "some_kernel")],
        )
        config = KernelProfileConfig(
            replay_script="/tmp/fake.py",
            output_dir=str(tmp_path),
        )
        orch = KernelProfileOrchestrator(manifest, [op], config)
        orch._profile_all = lambda output_dir: {}
        orch.run()  # must not raise
        assert op.kernels[0].metrics.raw == {}
