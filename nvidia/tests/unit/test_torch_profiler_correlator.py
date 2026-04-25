"""
Unit tests for torch_profiler_correlator._parse_chrome_trace.

Uses synthetic Chrome trace JSON — no GPU hardware required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from nvidia.operator_profiler.capture.torch_profiler_correlator import _parse_chrome_trace


def _write_trace(events: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump({"traceEvents": events}, tmp)
    tmp.close()
    return Path(tmp.name)


def _cpu_op(tid: int, name: str, ts: int, dur: int, ext_id: int = 0) -> dict:
    return {
        "ph": "X", "cat": "cpu_op", "name": name,
        "tid": tid, "ts": ts, "dur": dur,
        "args": {"External id": ext_id},
    }


def _gpu_kernel(name: str, ext_id: int, ts: float = 0.0) -> dict:
    """GPU kernel event — cat='kernel', linked to cpu_op via matching External id."""
    return {
        "ph": "X", "cat": "kernel", "name": name,
        "tid": 7, "ts": ts, "dur": 10.0,
        "args": {"External id": ext_id},
    }


class TestParseChromeTrace:
    def test_basic_attribution(self):
        """Kernel External id matches cpu_op External id → attributed."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm", ts=100, dur=200, ext_id=7),
            _gpu_kernel("triton_mm_0", ext_id=7, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert result == {("triton_mm_0", 0): "aten::mm"}

    def test_multiple_kernels_same_op(self):
        """Two launches with the same External id increment invocation index."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm", ts=100, dur=500, ext_id=7),
            _gpu_kernel("triton_mm_0", ext_id=7, ts=110.0),
            _gpu_kernel("triton_mm_0", ext_id=7, ts=200.0),
        ])
        result = _parse_chrome_trace(trace)
        assert result[("triton_mm_0", 0)] == "aten::mm"
        assert result[("triton_mm_0", 1)] == "aten::mm"

    def test_different_ops_different_kernels(self):
        """Each kernel gets the aten:: op from its own External id."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm",   ts=100, dur=100, ext_id=7),
            _cpu_op(1, "aten::relu", ts=300, dur=100, ext_id=9),
            _gpu_kernel("gemmSN_TN_kernel", ext_id=7, ts=110.0),
            _gpu_kernel("triton_relu_1",    ext_id=9, ts=310.0),
        ])
        result = _parse_chrome_trace(trace)
        assert result[("gemmSN_TN_kernel", 0)] == "aten::mm"
        assert result[("triton_relu_1", 0)] == "aten::relu"

    def test_non_aten_cpu_op_skipped(self):
        """Kernels whose cpu_op is a Triton name (not aten::) are not emitted."""
        trace = _write_trace([
            _cpu_op(1, "triton_per_fused_layer_norm_0", ts=100, dur=200, ext_id=6),
            _gpu_kernel("triton_per_fused_layer_norm_0", ext_id=6, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert ("triton_per_fused_layer_norm_0", 0) not in result

    def test_no_matching_ext_id(self):
        """Kernel with no matching cpu_op External id produces no entry."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm", ts=100, dur=200, ext_id=7),
            _gpu_kernel("triton_mm_0", ext_id=99, ts=110.0),  # ext_id 99 has no cpu_op
        ])
        result = _parse_chrome_trace(trace)
        assert ("triton_mm_0", 0) not in result

    def test_kernel_sort_order_determines_index(self):
        """Kernels are processed in GPU-timestamp order; earlier ts → lower index."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm",  ts=100, dur=500, ext_id=7),
            _cpu_op(1, "aten::addmm", ts=600, dur=100, ext_id=13),
            _gpu_kernel("gemmSN_TN_kernel", ext_id=7,  ts=110.0),
            _gpu_kernel("gemmSN_TN_kernel", ext_id=7,  ts=200.0),
            _gpu_kernel("gemmSN_TN_kernel", ext_id=13, ts=610.0),
        ])
        result = _parse_chrome_trace(trace)
        assert result[("gemmSN_TN_kernel", 0)] == "aten::mm"
        assert result[("gemmSN_TN_kernel", 1)] == "aten::mm"
        assert result[("gemmSN_TN_kernel", 2)] == "aten::addmm"

    def test_empty_trace(self):
        trace = _write_trace([])
        assert _parse_chrome_trace(trace) == {}

    def test_no_kernel_events(self):
        """cpu_op events with no corresponding kernel events → empty result."""
        trace = _write_trace([
            _cpu_op(1, "aten::mm", ts=100, dur=200, ext_id=7),
        ])
        assert _parse_chrome_trace(trace) == {}

    # --- quantized:: and torch.library custom op attribution ---

    def test_quantized_op_stored_in_correlation_map(self):
        """quantized:: cpu_op is now stored (was previously filtered out)."""
        trace = _write_trace([
            _cpu_op(1, "quantized::linear", ts=100, dur=200, ext_id=7),
            _gpu_kernel("quantized_linear_kernel", ext_id=7, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert ("quantized_linear_kernel", 0) in result
        assert result[("quantized_linear_kernel", 0)] == "quantized::linear"

    def test_custom_torch_library_op_stored(self):
        """Arbitrary torch.library namespace is stored in correlation map."""
        trace = _write_trace([
            _cpu_op(1, "flash_attn::fwd", ts=100, dur=200, ext_id=8),
            _gpu_kernel("flash_attn_fwd_kernel", ext_id=8, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert ("flash_attn_fwd_kernel", 0) in result
        assert result[("flash_attn_fwd_kernel", 0)] == "flash_attn::fwd"

    def test_prims_op_still_skipped(self):
        """prims:: ops do not dispatch kernels directly and are excluded."""
        trace = _write_trace([
            _cpu_op(1, "prims::mm", ts=100, dur=200, ext_id=9),
            _gpu_kernel("triton_mm_0", ext_id=9, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert ("triton_mm_0", 0) not in result

    def test_non_namespace_cpu_op_still_skipped(self):
        """Non-namespace cpu_op names (no '::') are still excluded."""
        trace = _write_trace([
            _cpu_op(1, "ProfilerStep#0", ts=100, dur=200, ext_id=10),
            _gpu_kernel("some_kernel", ext_id=10, ts=110.0),
        ])
        result = _parse_chrome_trace(trace)
        assert ("some_kernel", 0) not in result
