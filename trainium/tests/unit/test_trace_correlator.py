"""
Unit tests for trace_correlator.build_attribution_maps().

Tests use synthetic trace.json files constructed in-memory (no Trainium required).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trainium.operator_profiler.capture.trace_correlator import (
    NrtDeviceEvent,
    build_attribution_maps,
)


def _write_trace(events: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump({"traceEvents": events}, tmp)
    tmp.close()
    return Path(tmp.name)


def _cpu_op(name: str, ext_id: int, ts: float = 0, dur: float = 100) -> dict:
    return {"cat": "cpu_op", "name": name, "ts": ts, "dur": dur, "args": {"External id": ext_id}}


def _driver_event(name: str, ext_id: int, ts: float, dur: float, nc_id: int = 0) -> dict:
    return {
        "cat": "privateuse1_driver",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {"External id": ext_id, "device": nc_id},
    }


class TestBuildAttributionMaps:
    def test_basic_attribution(self, tmp_path):
        trace = _write_trace([
            _cpu_op("aten::linear", 10, ts=0),
            _cpu_op("aten::relu", 11, ts=200),
            _driver_event("nrt_op_linear", 10, ts=50, dur=100, nc_id=0),
            _driver_event("nrt_op_relu",   11, ts=160, dur=40, nc_id=0),
        ])
        corr_id_to_op, device_events = build_attribution_maps(trace)

        assert corr_id_to_op[10] == "aten::linear"
        assert corr_id_to_op[11] == "aten::relu"
        assert len(device_events) == 2
        assert device_events[0].event_name == "nrt_op_linear"
        assert device_events[0].correlation_id == 10
        assert device_events[0].neuroncore_id == 0

    def test_timestamps_converted_to_ns(self, tmp_path):
        trace = _write_trace([
            _cpu_op("aten::mm", 5, ts=0),
            _driver_event("nrt_op_mm", 5, ts=1.5, dur=2.0),  # 1.5 µs = 1500 ns
        ])
        _, device_events = build_attribution_maps(trace)
        assert device_events[0].start_ns == 1500
        assert device_events[0].duration_ns == 2000
        assert device_events[0].end_ns == 3500

    def test_unattributed_driver_event(self, tmp_path):
        trace = _write_trace([
            # driver event with no matching cpu_op
            _driver_event("nrt_op_internal", 99, ts=0, dur=10),
        ])
        corr_id_to_op, device_events = build_attribution_maps(trace)
        assert 99 not in corr_id_to_op
        assert len(device_events) == 1
        assert device_events[0].correlation_id == 99

    def test_non_kernel_op_namespaces_excluded(self, tmp_path):
        trace = _write_trace([
            _cpu_op("prims::add", 20),    # rejected: prims:: namespace
            _cpu_op("torch::foo", 21),    # rejected: torch:: namespace
            _cpu_op("aten::add", 22),     # accepted
            _driver_event("nrt_op_add", 22, ts=0, dur=10),
        ])
        corr_id_to_op, _ = build_attribution_maps(trace)
        assert 20 not in corr_id_to_op
        assert 21 not in corr_id_to_op
        assert corr_id_to_op[22] == "aten::add"

    def test_events_sorted_by_start_time(self, tmp_path):
        trace = _write_trace([
            _cpu_op("aten::linear", 1),
            _cpu_op("aten::relu", 2),
            _driver_event("nrt_op_relu",   2, ts=200, dur=10),
            _driver_event("nrt_op_linear", 1, ts=100, dur=50),  # earlier
        ])
        _, device_events = build_attribution_maps(trace)
        # Should be sorted by ts: linear (100µs) before relu (200µs)
        assert device_events[0].event_name == "nrt_op_linear"
        assert device_events[1].event_name == "nrt_op_relu"

    def test_empty_trace(self, tmp_path):
        trace = _write_trace([])
        corr_id_to_op, device_events = build_attribution_maps(trace)
        assert corr_id_to_op == {}
        assert device_events == []

    def test_driver_event_without_external_id_skipped(self, tmp_path):
        trace = _write_trace([
            {"cat": "privateuse1_driver", "name": "nrt_metadata", "ts": 0, "dur": 1, "args": {}},
        ])
        _, device_events = build_attribution_maps(trace)
        assert device_events == []
