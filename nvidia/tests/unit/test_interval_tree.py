"""
Unit tests for the per-stream NVTX interval tree.

Verifies correctness of enclosure queries, per-stream isolation (edge case #3),
and multi-range depth ordering (edge case #7).
"""
import pytest
from nvidia.operator_profiler.mapper.interval_tree import (
    NvtxIntervalForest,
    StreamIntervalTree,
    StreamKey,
)
from nvidia.operator_profiler.schema.profile import NvtxRangeInfo


def make_range(text: str, start: int, end: int, depth: int = 1) -> NvtxRangeInfo:
    return NvtxRangeInfo(text=text, depth=depth, start_ns=start, end_ns=end)


# ---------------------------------------------------------------------------
# StreamIntervalTree
# ---------------------------------------------------------------------------

class TestStreamIntervalTree:
    def test_single_range_enclosed(self):
        tree = StreamIntervalTree()
        tree.insert(make_range("aten::mm", 100, 200))
        result = tree.query_enclosing(150)
        assert len(result) == 1
        assert result[0].text == "aten::mm"

    def test_point_before_range(self):
        tree = StreamIntervalTree()
        tree.insert(make_range("aten::mm", 100, 200))
        assert tree.query_enclosing(50) == []

    def test_point_after_range(self):
        tree = StreamIntervalTree()
        tree.insert(make_range("aten::mm", 100, 200))
        assert tree.query_enclosing(250) == []

    def test_point_at_boundary(self):
        tree = StreamIntervalTree()
        tree.insert(make_range("aten::mm", 100, 200))
        # Inclusive on both ends
        assert len(tree.query_enclosing(100)) == 1
        assert len(tree.query_enclosing(200)) == 1

    def test_nested_ranges_all_returned(self):
        """All enclosing ranges should be returned, not just innermost."""
        tree = StreamIntervalTree()
        tree.insert(make_range("outer", 0, 1000, depth=1))
        tree.insert(make_range("middle", 100, 500, depth=2))
        tree.insert(make_range("inner", 200, 300, depth=3))
        result = tree.query_enclosing(250)
        names = [r.text for r in result]
        assert names == ["outer", "middle", "inner"]  # outermost first

    def test_innermost_enclosing(self):
        tree = StreamIntervalTree()
        tree.insert(make_range("outer", 0, 1000, depth=1))
        tree.insert(make_range("inner", 100, 200, depth=2))
        r = tree.innermost_enclosing(150)
        assert r is not None
        assert r.text == "inner"

    def test_non_overlapping_ranges(self):
        """Adjacent non-overlapping ranges — only one should match."""
        tree = StreamIntervalTree()
        tree.insert(make_range("first", 0, 100))
        tree.insert(make_range("second", 200, 300))
        assert len(tree.query_enclosing(50)) == 1
        assert tree.query_enclosing(50)[0].text == "first"
        assert len(tree.query_enclosing(250)) == 1
        assert tree.query_enclosing(250)[0].text == "second"

    def test_overlapping_same_depth(self):
        """Two overlapping ranges at same depth — both returned for mid-point."""
        tree = StreamIntervalTree()
        tree.insert(make_range("a", 0, 200, depth=1))
        tree.insert(make_range("b", 100, 300, depth=1))
        result = tree.query_enclosing(150)
        texts = {r.text for r in result}
        assert "a" in texts
        assert "b" in texts

    def test_no_ranges(self):
        tree = StreamIntervalTree()
        assert tree.query_enclosing(100) == []
        assert tree.innermost_enclosing(100) is None


# ---------------------------------------------------------------------------
# NvtxIntervalForest — per-stream isolation
# ---------------------------------------------------------------------------

class TestNvtxIntervalForest:
    def test_per_stream_isolation(self):
        """Kernels on stream 7 should NOT match NVTX ranges on stream 8."""
        forest = NvtxIntervalForest()
        forest.insert(7, 0, make_range("aten::mm", 100, 200))
        forest.insert(8, 0, make_range("aten::relu", 100, 200))

        result_s7 = forest.query_enclosing(7, 0, 150)
        assert len(result_s7) == 1
        assert result_s7[0].text == "aten::mm"

        result_s8 = forest.query_enclosing(8, 0, 150)
        assert len(result_s8) == 1
        assert result_s8[0].text == "aten::relu"

    def test_per_device_isolation(self):
        """Same stream_id on different devices should not mix."""
        forest = NvtxIntervalForest()
        forest.insert(7, 0, make_range("gpu0_op", 100, 200))
        forest.insert(7, 1, make_range("gpu1_op", 100, 200))

        r0 = forest.query_enclosing(7, 0, 150)
        r1 = forest.query_enclosing(7, 1, 150)
        assert r0[0].text == "gpu0_op"
        assert r1[0].text == "gpu1_op"

    def test_missing_stream_returns_empty(self):
        forest = NvtxIntervalForest()
        assert forest.query_enclosing(99, 0, 100) == []
        assert forest.innermost_enclosing(99, 0, 100) is None

    def test_stream_keys(self):
        forest = NvtxIntervalForest()
        forest.insert(7, 0, make_range("a", 0, 100))
        forest.insert(8, 0, make_range("b", 0, 100))
        keys = forest.stream_keys
        assert StreamKey(7, 0) in keys
        assert StreamKey(8, 0) in keys

    def test_multiple_ranges_same_stream_nested(self):
        """
        Multi-level NVTX nesting on one stream — verifies fused-kernel
        attribution stores all enclosing ranges (edge case #7).
        """
        forest = NvtxIntervalForest()
        forest.insert(7, 0, make_range("model_forward", 0, 10000, depth=1))
        forest.insert(7, 0, make_range("aten::linear", 100, 500, depth=2))
        forest.insert(7, 0, make_range("aten::mm", 150, 300, depth=3))

        # Point inside aten::mm should see all three enclosing ranges
        result = forest.query_enclosing(7, 0, 200)
        assert len(result) == 3
        assert result[0].text == "model_forward"
        assert result[-1].text == "aten::mm"
