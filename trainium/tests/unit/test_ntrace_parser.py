"""
Unit tests for ntrace_parser.

Currently a stub — all tests that require real ntrace.pb content are marked
xfail(strict=False) until the ntrace.pb schema is confirmed and the parser
is implemented.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from trainium.operator_profiler.capture.ntrace_parser import parse, _find_ntrace


class TestFindNtrace:
    def test_returns_none_for_empty_dir(self, tmp_path):
        result = _find_ntrace(tmp_path)
        assert result is None

    def test_finds_ntrace_in_direct_child(self, tmp_path):
        ntrace = tmp_path / "ntrace.pb"
        ntrace.write_bytes(b"")
        result = _find_ntrace(tmp_path)
        assert result == ntrace

    def test_finds_ntrace_one_level_deeper(self, tmp_path):
        session = tmp_path / "inst_pid_12345" / "1700000000"
        session.mkdir(parents=True)
        ntrace = session / "ntrace.pb"
        ntrace.write_bytes(b"")
        # Pass the parent (instance dir) to simulate NRT session layout
        result = _find_ntrace(tmp_path / "inst_pid_12345")
        assert result == ntrace


class TestParse:
    def test_missing_ntrace_returns_empty_dict(self, tmp_path):
        result = parse(tmp_path)
        assert result == {}

    def test_empty_ntrace_pb_returns_empty_dict(self, tmp_path):
        (tmp_path / "ntrace.pb").write_bytes(b"")
        result = parse(tmp_path)
        # Stub returns {} and logs a warning — no exception
        assert result == {}

    @pytest.mark.xfail(
        reason="ntrace_parser._parse_ntrace_pb not yet implemented (schema unknown)",
        strict=False,
    )
    def test_real_ntrace_returns_metrics_by_corr_id(self, tmp_path):
        # This test documents the expected interface once the parser is implemented.
        # A real ntrace.pb should return a dict keyed by Kineto correlation IDs
        # with metric dicts as values.
        ntrace = tmp_path / "ntrace.pb"
        ntrace.write_bytes(b"<real ntrace.pb content here>")

        result = parse(tmp_path)

        assert isinstance(result, dict)
        for corr_id, metrics in result.items():
            assert isinstance(corr_id, int)
            assert isinstance(metrics, dict)
            # At minimum expect a duration metric
            assert any("duration" in k or "cycles" in k for k in metrics)
