"""
Unit tests for ncu_parser.parse_ncu_csv_by_id().

Application-mode ncu produces a single CSV covering all kernels; each kernel
appears with per-name invocation IDs 0, 1, 2, ... These tests exercise the
parser against that multi-kernel, multi-invocation output shape.
"""
from __future__ import annotations

import textwrap

import pytest

from nvidia.operator_profiler.mapper.ncu_parser import parse_ncu_csv_by_id, parse_ncu_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(*rows: tuple) -> str:
    """Build a minimal ncu CSV string from (kernel_name, id, metric, value) rows."""
    header = '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",' \
             '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
    lines = [header]
    for kernel_name, kid, metric, value in rows:
        lines.append(
            f'"{kid}","1","python","host","{kernel_name}","1000us",'
            f'"1","7","Section","  {metric}","","  {value}"'
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_kernel_single_invocation():
    csv = _make_csv(("gemm_kernel", "0", "sm__active_cycles.sum", "12345"))
    result = parse_ncu_csv_by_id(csv)
    assert ("gemm_kernel", "0") in result
    m = result[("gemm_kernel", "0")]
    assert m.raw.get("sm__active_cycles.sum") == 12345


def test_multi_kernel_multi_invocation():
    """Application-mode output: two kernels, 2 invocations each → 4 dict keys."""
    csv = _make_csv(
        ("kernel_A", "0", "sm__active_cycles.sum", "100"),
        ("kernel_A", "0", "dram__bytes_read.sum", "200"),
        ("kernel_A", "1", "sm__active_cycles.sum", "110"),
        ("kernel_A", "1", "dram__bytes_read.sum", "210"),
        ("kernel_B", "0", "sm__active_cycles.sum", "300"),
        ("kernel_B", "1", "sm__active_cycles.sum", "310"),
    )
    result = parse_ncu_csv_by_id(csv)
    assert len(result) == 4
    assert ("kernel_A", "0") in result
    assert ("kernel_A", "1") in result
    assert ("kernel_B", "0") in result
    assert ("kernel_B", "1") in result

    assert result[("kernel_A", "0")].raw["sm__active_cycles.sum"] == 100
    assert result[("kernel_A", "1")].raw["sm__active_cycles.sum"] == 110
    assert result[("kernel_B", "1")].raw["sm__active_cycles.sum"] == 310


def test_non_numeric_values_dropped():
    """'n/a' metric values must be silently dropped — not stored in .raw."""
    csv = _make_csv(
        ("kernel_X", "0", "sm__active_cycles.sum", "5000"),
        ("kernel_X", "0", "some_unavailable_metric", "n/a"),
    )
    result = parse_ncu_csv_by_id(csv)
    m = result[("kernel_X", "0")]
    assert "sm__active_cycles.sum" in m.raw
    assert "some_unavailable_metric" not in m.raw


def test_comma_separated_large_numbers():
    """ncu formats large integers with commas; these must parse as plain ints."""
    csv = _make_csv(("k", "0", "dram__bytes_read.sum", "1,234,567"))
    result = parse_ncu_csv_by_id(csv)
    assert result[("k", "0")].raw["dram__bytes_read.sum"] == 1234567


def test_float_metric_values():
    csv = _make_csv(("k", "0", "sm__warps_active.avg.pct_of_peak_sustained_active", "63.5"))
    result = parse_ncu_csv_by_id(csv)
    assert abs(result[("k", "0")].raw["sm__warps_active.avg.pct_of_peak_sustained_active"] - 63.5) < 1e-9


def test_empty_csv_returns_empty_dict():
    result = parse_ncu_csv_by_id("")
    assert result == {}


def test_missing_required_columns_returns_empty():
    """CSV that has no Metric Name column should return {} without raising."""
    bad_csv = '"ID","Kernel Name","Metric Value"\n"0","k","42"\n'
    result = parse_ncu_csv_by_id(bad_csv)
    assert result == {}


def test_parse_ncu_csv_last_id_wins():
    """parse_ncu_csv (non-by-id) returns one entry per kernel name; last ID wins."""
    csv = _make_csv(
        ("gemm", "0", "sm__active_cycles.sum", "100"),
        ("gemm", "1", "sm__active_cycles.sum", "999"),
    )
    result = parse_ncu_csv(csv)
    assert "gemm" in result
    # Sorted by (name, id) so "1" > "0" → last = id "1"
    assert result["gemm"].raw["sm__active_cycles.sum"] == 999


def test_full_mangled_kernel_name_preserved():
    """Full C++ mangled names (with template args) are stored as-is in the key."""
    mangled = "void gemv2T_kernel_val<int, float, 128>(float*, int)"
    csv = _make_csv((mangled, "0", "sm__active_cycles.sum", "77"))
    result = parse_ncu_csv_by_id(csv)
    assert (mangled, "0") in result
