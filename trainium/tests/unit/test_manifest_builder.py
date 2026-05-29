"""
Unit tests for ManifestBuilder.

Uses synthetic inputs (no trace.json or ntrace.pb files on disk) by constructing
the intermediate data structures directly and testing the manifest assembly logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainium.operator_profiler.capture.trace_correlator import NrtDeviceEvent
from trainium.operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    MappingManifest,
)
from trainium.operator_profiler.schema.profile import AttributionMethod, Confidence


def _make_metadata() -> CaptureManifestMetadata:
    return CaptureManifestMetadata(
        model_name="test",
        torch_version="2.1.0",
        compile_mode="neuron",
        capture_timestamp_utc="2024-01-01T00:00:00+00:00",
    )


def _make_device_event(name: str, corr_id: int, start_ns: int = 0, dur_ns: int = 1000, nc: int = 0) -> NrtDeviceEvent:
    return NrtDeviceEvent(
        event_name=name,
        correlation_id=corr_id,
        start_ns=start_ns,
        end_ns=start_ns + dur_ns,
        duration_ns=dur_ns,
        neuroncore_id=nc,
        device_id=0,
    )


class TestManifestBuilderLogic:
    """Tests the manifest assembly logic without file I/O by subclassing."""

    def _build_from_data(
        self,
        corr_id_to_op: dict[int, str],
        device_events: list[NrtDeviceEvent],
    ) -> MappingManifest:
        """
        Drive ManifestBuilder._build_manifest_from_data() — a helper that
        accepts pre-parsed inputs so tests don't need real trace files.
        """
        from trainium.operator_profiler.schema.manifest import (
            CaptureManifestMetadata,
            KernelAttribution,
            KernelManifestEntry,
            MappingManifest,
        )
        from trainium.operator_profiler.schema.profile import AttributionMethod, Confidence, NrtEventInfo
        from trainium.operator_profiler.mapper.manifest_builder import ManifestBuilder

        meta = _make_metadata()

        # Monkey-patch build_attribution_maps to return our synthetic data
        import trainium.operator_profiler.mapper.manifest_builder as mb_module
        import trainium.operator_profiler.capture.ntrace_parser as nt_module
        original_ba = mb_module.build_attribution_maps
        original_nt = mb_module.parse_ntrace

        mb_module.build_attribution_maps = lambda _p: (corr_id_to_op, device_events)
        mb_module.parse_ntrace = lambda _d: {}

        try:
            builder = ManifestBuilder(
                trace_json_path=Path("/dev/null"),
                nrt_session_dir=Path("/tmp"),
                metadata=meta,
            )
            return builder.build()
        finally:
            mb_module.build_attribution_maps = original_ba
            mb_module.parse_ntrace = original_nt

    def test_attributed_entries(self):
        events = [
            _make_device_event("nrt_op_linear", corr_id=10, start_ns=1000, dur_ns=5000),
            _make_device_event("nrt_op_relu",   corr_id=11, start_ns=7000, dur_ns=1000),
        ]
        corr_id_to_op = {10: "aten::linear", 11: "aten::relu"}

        manifest = self._build_from_data(corr_id_to_op, events)

        assert len(manifest.kernels) == 2
        k0 = manifest.kernels[0]
        assert k0.attribution.method == AttributionMethod.NRT_CORRELATION
        assert k0.attribution.confidence == Confidence.HIGH
        assert k0.attribution.source_operators == ["aten::linear"]
        assert k0.kineto_correlation_id == 10

    def test_unattributed_entry(self):
        events = [_make_device_event("nrt_op_internal", corr_id=99)]
        manifest = self._build_from_data({}, events)

        assert manifest.kernels[0].attribution.method == AttributionMethod.UNATTRIBUTED
        assert manifest.kernels[0].attribution.confidence == Confidence.UNATTRIBUTED

    def test_warmup_events_flagged_in_warnings(self):
        # Two warm-up events (no corr_id match) followed by two attributed events
        events = [
            _make_device_event("warmup_op_0", corr_id=1, start_ns=0),
            _make_device_event("warmup_op_1", corr_id=2, start_ns=1000),
            _make_device_event("nrt_op_linear", corr_id=10, start_ns=2000),
            _make_device_event("nrt_op_relu",   corr_id=11, start_ns=3000),
        ]
        corr_id_to_op = {10: "aten::linear", 11: "aten::relu"}

        manifest = self._build_from_data(corr_id_to_op, events)

        warmup_warnings = [w for w in manifest.warnings if "warm-up" in w]
        assert len(warmup_warnings) == 2

    def test_kernel_ids_are_sequential(self):
        events = [_make_device_event(f"op_{i}", corr_id=i) for i in range(5)]
        manifest = self._build_from_data({}, events)

        ids = [k.kernel_id for k in manifest.kernels]
        assert ids == ["k_00000", "k_00001", "k_00002", "k_00003", "k_00004"]

    def test_neuroncore_id_stored_as_stream_id(self):
        events = [_make_device_event("op", corr_id=5, nc=3)]
        manifest = self._build_from_data({5: "aten::add"}, events)
        assert manifest.kernels[0].stream_id == 3
