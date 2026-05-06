"""
Unit tests for layer-partition tagging and metric propagation.

Tests the new functionality added to the deduplication pipeline:
  - ManifestBuilder._tag_layer_partitions()  — NVTX prefix → layer_partition + is_unique_partition
  - KernelProfileOrchestrator._propagate_partition_metrics() — unique→duplicate metric copy
  - KernelProfileOrchestrator._build_replay_targets()        — skip non-unique partitions

No GPU required — all tests use mocked nsys I/O and in-memory manifests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
from nvidia.operator_profiler.mapper.nsys_export import KernelRow, NvtxRow
from nvidia.operator_profiler.mapper.kernel_profiler import (
    KernelProfileConfig,
    KernelProfileOrchestrator,
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
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_metadata() -> CaptureManifestMetadata:
    return CaptureManifestMetadata(
        model_name="TestModel",
        torch_version="2.3.0",
        compile_mode="inductor",
        capture_timestamp_utc="2026-01-01T00:00:00+00:00",
    )


def _make_kernel_row(**kw) -> KernelRow:
    defaults = dict(
        correlation_id=1,
        kernel_name="relu_kernel",
        start_ns=2000,
        end_ns=2500,
        stream_id=7,
        device_id=0,
        grid_x=32, grid_y=1, grid_z=1,
        block_x=128, block_y=1, block_z=1,
        host_tid=7,
        cpu_launch_start_ns=1100,   # falls inside the NVTX range below
    )
    defaults.update(kw)
    return KernelRow(**defaults)


def _make_nvtx_row(**kw) -> NvtxRow:
    defaults = dict(
        text="layer::unique::modules_0",
        start_ns=1000,
        end_ns=3000,
        nesting_level=1,
        domain="default",
        stream_id=7,
        device_id=0,
    )
    defaults.update(kw)
    return NvtxRow(**defaults)


def _unattributed() -> KernelAttribution:
    return KernelAttribution(
        method=AttributionMethod.UNATTRIBUTED,
        confidence=Confidence.UNATTRIBUTED,
    )


def _make_entry(kernel_id: str, **kw) -> KernelManifestEntry:
    defaults = dict(
        kernel_id=kernel_id,
        kernel_name="some_kernel",
        stream_id=7,
        device_id=0,
        start_ns=2000,
        end_ns=2500,
        duration_ns=500,
        attribution=_unattributed(),
    )
    defaults.update(kw)
    return KernelManifestEntry(**defaults)


def _build_manifest_with(kernel_rows, nvtx_rows) -> "MappingManifest":
    """Run ManifestBuilder with mocked nsys I/O."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        builder = ManifestBuilder(
            nsys_rep_path=tmp_path / "fake.nsys-rep",
            metadata=_make_metadata(),
        )
        with (
            patch("nvidia.operator_profiler.mapper.manifest_builder.export_to_sqlite",
                  return_value=tmp_path / "fake.sqlite"),
            patch("nvidia.operator_profiler.mapper.manifest_builder.query_kernels",
                  return_value=kernel_rows),
            patch("nvidia.operator_profiler.mapper.manifest_builder.query_nvtx_events",
                  return_value=nvtx_rows),
        ):
            return builder.build()


# ---------------------------------------------------------------------------
# _tag_layer_partitions
# ---------------------------------------------------------------------------

class TestTagLayerPartitions:

    def test_unique_prefix_sets_is_unique_true(self):
        """A layer::unique:: NVTX range enclosing the kernel sets is_unique_partition=True."""
        kr = _make_kernel_row(cpu_launch_start_ns=1100)
        nvtx = _make_nvtx_row(text="layer::unique::modules_0", start_ns=1000, end_ns=3000)
        manifest = _build_manifest_with([kr], [nvtx])

        assert len(manifest.kernels) == 1
        entry = manifest.kernels[0]
        assert entry.layer_partition == "modules_0"
        assert entry.is_unique_partition is True

    def test_duplicate_prefix_sets_is_unique_false(self):
        """A layer::duplicate:: NVTX range enclosing the kernel sets is_unique_partition=False."""
        kr = _make_kernel_row(cpu_launch_start_ns=1100)
        nvtx = _make_nvtx_row(text="layer::duplicate::modules_1", start_ns=1000, end_ns=3000)
        manifest = _build_manifest_with([kr], [nvtx])

        entry = manifest.kernels[0]
        assert entry.layer_partition == "modules_1"
        assert entry.is_unique_partition is False

    def test_no_layer_ranges_leaves_fields_at_defaults(self):
        """Non-layer NVTX ranges (aten::) do not set layer_partition or is_unique_partition."""
        kr = _make_kernel_row(cpu_launch_start_ns=1100)
        nvtx = _make_nvtx_row(text="aten::relu", start_ns=1000, end_ns=3000)
        manifest = _build_manifest_with([kr], [nvtx])

        entry = manifest.kernels[0]
        assert entry.layer_partition is None
        assert entry.is_unique_partition is False

    def test_kernel_outside_layer_range_untagged(self):
        """A kernel whose CPU launch timestamp falls outside the layer range is not tagged."""
        kr = _make_kernel_row(cpu_launch_start_ns=500)   # before range start 1000
        nvtx = _make_nvtx_row(text="layer::unique::modules_0", start_ns=1000, end_ns=3000)
        manifest = _build_manifest_with([kr], [nvtx])

        entry = manifest.kernels[0]
        assert entry.layer_partition is None

    def test_label_extracted_correctly(self):
        """The partition label is the text after the prefix, not including the prefix itself."""
        kr = _make_kernel_row(cpu_launch_start_ns=1100)
        nvtx = _make_nvtx_row(text="layer::unique::transformer_layer_7", start_ns=1000, end_ns=3000)
        manifest = _build_manifest_with([kr], [nvtx])

        assert manifest.kernels[0].layer_partition == "transformer_layer_7"

    def test_multiple_kernels_tagged_independently(self):
        """Two kernels in different partitions are tagged independently."""
        kr_a = _make_kernel_row(kernel_name="k_a", cpu_launch_start_ns=1100)
        kr_b = _make_kernel_row(kernel_name="k_b", cpu_launch_start_ns=4100)

        nvtx_a = _make_nvtx_row(text="layer::unique::modules_0",    start_ns=1000, end_ns=3000)
        nvtx_b = _make_nvtx_row(text="layer::duplicate::modules_1", start_ns=4000, end_ns=6000)

        manifest = _build_manifest_with([kr_a, kr_b], [nvtx_a, nvtx_b])

        assert len(manifest.kernels) == 2
        a, b = manifest.kernels
        assert a.layer_partition == "modules_0"
        assert a.is_unique_partition is True
        assert b.layer_partition == "modules_1"
        assert b.is_unique_partition is False


# ---------------------------------------------------------------------------
# KernelProfileOrchestrator._build_replay_targets
# ---------------------------------------------------------------------------

class TestBuildReplayTargets:

    def _make_orchestrator(self, entries, equiv_map) -> KernelProfileOrchestrator:
        manifest = MappingManifest(
            capture_metadata=_make_metadata(),
            kernels=entries,
        )
        config = KernelProfileConfig(
            replay_script="dummy.py",
            partition_equivalence_map=equiv_map,
        )
        return KernelProfileOrchestrator(manifest, [], config)

    def test_skips_duplicate_partition_kernels(self):
        """When equiv_map is set, _build_replay_targets excludes is_unique_partition=False entries."""
        unique = _make_entry("k_00000", kernel_name="relu", layer_partition="modules_0",
                             is_unique_partition=True)
        dup1   = _make_entry("k_00001", kernel_name="relu", layer_partition="modules_1",
                             is_unique_partition=False)
        dup2   = _make_entry("k_00002", kernel_name="relu", layer_partition="modules_2",
                             is_unique_partition=False)

        orch = self._make_orchestrator(
            [unique, dup1, dup2],
            {"modules_1": "modules_0", "modules_2": "modules_0"},
        )
        targets = orch._build_replay_targets()

        all_target_kids = [kid for t in targets for kid in t.kernel_ids]
        assert "k_00000" in all_target_kids
        assert "k_00001" not in all_target_kids
        assert "k_00002" not in all_target_kids

    def test_no_skipping_when_equiv_map_empty(self):
        """Without a partition_equivalence_map, all kernels are included as replay targets."""
        unique = _make_entry("k_00000", kernel_name="relu", layer_partition="modules_0",
                             is_unique_partition=True)
        dup    = _make_entry("k_00001", kernel_name="relu", layer_partition="modules_1",
                             is_unique_partition=False)

        orch = self._make_orchestrator([unique, dup], {})
        targets = orch._build_replay_targets()

        all_target_kids = [kid for t in targets for kid in t.kernel_ids]
        assert "k_00000" in all_target_kids
        assert "k_00001" in all_target_kids


# ---------------------------------------------------------------------------
# KernelProfileOrchestrator._propagate_partition_metrics
# ---------------------------------------------------------------------------

class TestPropagatePartitionMetrics:

    def _make_orchestrator_with_metrics(
        self,
        entries: list[KernelManifestEntry],
        prepopulated: dict[str, KernelMetrics],
        equiv_map: dict[str, str],
    ) -> KernelProfileOrchestrator:
        manifest = MappingManifest(
            capture_metadata=_make_metadata(),
            kernels=entries,
        )
        config = KernelProfileConfig(
            replay_script="dummy.py",
            partition_equivalence_map=equiv_map,
        )
        orch = KernelProfileOrchestrator(manifest, [], config)
        orch._kernel_metrics.update(prepopulated)
        return orch

    def test_metrics_propagated_from_unique_to_duplicate(self):
        """Duplicate partition kernels inherit metrics from the unique rep by position."""
        metrics_0 = KernelMetrics(raw={"achieved_occupancy": 0.75})
        metrics_1 = KernelMetrics(raw={"achieved_occupancy": 0.80})

        unique_k0 = _make_entry("k_00000", layer_partition="modules_0", is_unique_partition=True)
        unique_k1 = _make_entry("k_00001", layer_partition="modules_0", is_unique_partition=True)
        dup_k0    = _make_entry("k_00002", layer_partition="modules_1", is_unique_partition=False)
        dup_k1    = _make_entry("k_00003", layer_partition="modules_1", is_unique_partition=False)

        orch = self._make_orchestrator_with_metrics(
            [unique_k0, unique_k1, dup_k0, dup_k1],
            {"k_00000": metrics_0, "k_00001": metrics_1},
            {"modules_1": "modules_0"},
        )
        orch._propagate_partition_metrics()

        assert orch._kernel_metrics["k_00002"] is metrics_0
        assert orch._kernel_metrics["k_00003"] is metrics_1

    def test_noop_when_equiv_map_empty(self):
        """With an empty partition_equivalence_map, _propagate_partition_metrics does nothing."""
        metrics_0 = KernelMetrics(raw={"achieved_occupancy": 0.75})
        unique_k  = _make_entry("k_00000", layer_partition="modules_0", is_unique_partition=True)
        dup_k     = _make_entry("k_00001", layer_partition="modules_1", is_unique_partition=False)

        orch = self._make_orchestrator_with_metrics(
            [unique_k, dup_k],
            {"k_00000": metrics_0},
            {},   # empty → disabled
        )
        before_count = len(orch._kernel_metrics)
        orch._propagate_partition_metrics()
        assert len(orch._kernel_metrics) == before_count
        assert "k_00001" not in orch._kernel_metrics

    def test_two_duplicates_both_get_metrics(self):
        """Two duplicate partitions with the same unique rep both receive propagated metrics."""
        m = KernelMetrics(raw={"achieved_occupancy": 0.9})
        unique_k  = _make_entry("k_00000", layer_partition="modules_0", is_unique_partition=True)
        dup1_k    = _make_entry("k_00001", layer_partition="modules_1", is_unique_partition=False)
        dup2_k    = _make_entry("k_00002", layer_partition="modules_2", is_unique_partition=False)

        orch = self._make_orchestrator_with_metrics(
            [unique_k, dup1_k, dup2_k],
            {"k_00000": m},
            {"modules_1": "modules_0", "modules_2": "modules_0"},
        )
        orch._propagate_partition_metrics()

        assert orch._kernel_metrics.get("k_00001") is m
        assert orch._kernel_metrics.get("k_00002") is m
