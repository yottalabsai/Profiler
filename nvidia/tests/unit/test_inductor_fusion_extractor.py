"""
Unit tests for parse_inductor_debug_dir — uses mock output_code.py files.

No GPU hardware or real Inductor compilation required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nvidia.operator_profiler.capture.inductor_fusion_extractor import (
    parse_inductor_debug_dir,
)


def _write_output_code(tmp_dir: Path, content: str, subdir: str = "abc123") -> Path:
    p = tmp_dir / subdir / "output_code.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class TestParseInductorDebugDir:
    def test_single_fused_kernel(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu, y], Original ATen: [aten.relu, aten.addmm]
triton_poi_fused_relu_addmm_0.run(in_ptr0, out_ptr0, 128, grid=grid(128))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert result["triton_poi_fused_relu_addmm_0"] == ["aten::relu", "aten::addmm"]

    def test_single_op_kernel(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
triton_poi_fused_relu_0.run(in_ptr0, out_ptr0, 16, grid=grid(16))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert result["triton_poi_fused_relu_0"] == ["aten::relu"]

    def test_overload_suffix_stripped(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add.Tensor]
triton_poi_fused_add_0.run(buf0, buf1, 64, grid=grid(64))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert result["triton_poi_fused_add_0"] == ["aten::add"]

    def test_prims_ops_filtered(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [mm], Original ATen: [prims.mm, aten.addmm]
triton_mm_0.run(buf0, buf1, out, 64, grid=grid(64))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert result["triton_mm_0"] == ["aten::addmm"]

    def test_all_prims_filtered_yields_no_entry(self, tmp_path):
        """Comment with only prims:: ops → no .run() match (pending_ops is None)."""
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [mm], Original ATen: [prims.mm]
triton_mm_0.run(buf0, buf1, out, 64, grid=grid(64))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert "triton_mm_0" not in result

    def test_extern_kernel_no_run_skipped(self, tmp_path):
        """extern_kernels calls don't use .run() — comment is discarded."""
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
extern_kernels.addmm(primals_1, buf0, primals_2, alpha=1, beta=1, out=buf1)
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert "extern_kernels" not in result
        assert len(result) == 0

    def test_multiple_kernels_in_one_file(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
triton_poi_fused_relu_0.run(buf0, out0, 16, grid=grid(16))
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
triton_red_fused_softmax_0.run(buf1, out1, 16, grid=grid(16))
""")
        result = parse_inductor_debug_dir(tmp_path)
        assert result["triton_poi_fused_relu_0"] == ["aten::relu"]
        assert result["triton_red_fused_softmax_0"] == ["aten::_softmax"]

    def test_multiple_output_code_files_merged(self, tmp_path):
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
triton_poi_fused_relu_0.run(buf0, out0, 16, grid=grid(16))
""", subdir="graph0")
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.addmm]
triton_mm_0.run(buf0, buf1, out, 64, grid=grid(64))
""", subdir="graph1")
        result = parse_inductor_debug_dir(tmp_path)
        assert "triton_poi_fused_relu_0" in result
        assert "triton_mm_0" in result

    def test_first_seen_wins_on_collision(self, tmp_path):
        """Same kernel in two files → first file's ops are kept."""
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
triton_poi_fused_relu_0.run(buf0, out0, 16, grid=grid(16))
""", subdir="aaa")
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
triton_poi_fused_relu_0.run(buf0, out0, 16, grid=grid(16))
""", subdir="zzz")
        result = parse_inductor_debug_dir(tmp_path)
        # Both subdirs present; whichever rglob finds first is kept
        assert "triton_poi_fused_relu_0" in result
        assert len(result["triton_poi_fused_relu_0"]) == 1

    def test_stream_setup_lines_between_comment_and_run(self, tmp_path):
        """Intermediate lines (stream setup) between comment and .run() reset pending."""
        _write_output_code(tmp_path, """\
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
stream0 = get_raw_stream(0)
triton_poi_fused_relu_0.run(buf0, out0, 16, grid=grid(16), stream=stream0)
""")
        result = parse_inductor_debug_dir(tmp_path)
        # stream0 = ... is non-comment, non-empty, non-.run() → pending reset
        assert "triton_poi_fused_relu_0" not in result

    def test_empty_dir(self, tmp_path):
        assert parse_inductor_debug_dir(tmp_path) == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = parse_inductor_debug_dir(tmp_path / "does_not_exist")
        assert result == {}
