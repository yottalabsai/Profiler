"""
Per-stream NVTX interval tree.

Each kernel is matched against NVTX ranges on the same (stream_id, device_id)
to find all enclosing ranges, from which the innermost (deepest depth) is
selected as the primary attribution range.

Design:
  - Uses a sorted list + bisect for O(log n) insertion + O(k log n) query,
    which is sufficient for typical profiling sizes (< 100k events).
  - Per-stream isolation prevents cross-stream false matches (edge case #3).
  - All enclosing ranges are returned (not just innermost) to support fused
    kernel attribution (edge case #7).

GPU timestamps from CUPTI are used; host-side timestamps are never used for
enclosure queries (edge case #5).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import NamedTuple

from operator_profiler.schema.profile import NvtxRangeInfo


class StreamKey(NamedTuple):
    stream_id: int
    device_id: int


@dataclass(order=True)
class _IntervalNode:
    """Internal node stored in the sorted list, ordered by start_ns."""
    start_ns: int
    end_ns: int
    depth: int
    range_info: NvtxRangeInfo = field(compare=False)


class StreamIntervalTree:
    """
    Interval tree for a single (stream_id, device_id) pair.

    Stores NVTX ranges and answers: "which ranges enclose timestamp T?"
    """

    def __init__(self) -> None:
        # Sorted by start_ns for bisect-based range queries
        self._nodes: list[_IntervalNode] = []
        self._sorted_starts: list[int] = []

    def insert(self, range_info: NvtxRangeInfo) -> None:
        node = _IntervalNode(
            start_ns=range_info.start_ns,
            end_ns=range_info.end_ns,
            depth=range_info.depth,
            range_info=range_info,
        )
        idx = bisect.bisect_left(self._sorted_starts, range_info.start_ns)
        self._nodes.insert(idx, node)
        self._sorted_starts.insert(idx, range_info.start_ns)

    def query_enclosing(self, point_ns: int) -> list[NvtxRangeInfo]:
        """
        Return all NVTX ranges that enclose *point_ns*, sorted by depth
        ascending (outermost first).

        A range encloses a point when: range.start_ns <= point_ns <= range.end_ns
        """
        # Candidates: all ranges that started at or before point_ns
        right_idx = bisect.bisect_right(self._sorted_starts, point_ns)
        candidates = self._nodes[:right_idx]

        enclosing = [
            node.range_info
            for node in candidates
            if node.end_ns >= point_ns
        ]
        # Sort outermost (lowest depth) first
        enclosing.sort(key=lambda r: r.depth)
        return enclosing

    def innermost_enclosing(self, point_ns: int) -> NvtxRangeInfo | None:
        """Return the deepest (highest depth) enclosing range, or None."""
        ranges = self.query_enclosing(point_ns)
        return ranges[-1] if ranges else None


class NvtxIntervalForest:
    """
    A collection of per-stream NVTX interval trees.

    Usage:
        forest = NvtxIntervalForest()
        for row in nvtx_rows:
            forest.insert(stream_id, device_id, range_info)

        enclosing = forest.query_enclosing(stream_id=7, device_id=0, point_ns=...)
    """

    def __init__(self) -> None:
        self._trees: dict[StreamKey, StreamIntervalTree] = {}

    def _get_or_create(self, stream_id: int, device_id: int) -> StreamIntervalTree:
        key = StreamKey(stream_id, device_id)
        if key not in self._trees:
            self._trees[key] = StreamIntervalTree()
        return self._trees[key]

    def insert(
        self, stream_id: int, device_id: int, range_info: NvtxRangeInfo
    ) -> None:
        self._get_or_create(stream_id, device_id).insert(range_info)

    def query_enclosing(
        self, stream_id: int, device_id: int, point_ns: int
    ) -> list[NvtxRangeInfo]:
        """
        Return all NVTX ranges on (stream_id, device_id) that enclose point_ns,
        sorted outermost first.  Returns [] if no stream tree exists.
        """
        key = StreamKey(stream_id, device_id)
        tree = self._trees.get(key)
        if tree is None:
            return []
        return tree.query_enclosing(point_ns)

    def innermost_enclosing(
        self, stream_id: int, device_id: int, point_ns: int
    ) -> NvtxRangeInfo | None:
        key = StreamKey(stream_id, device_id)
        tree = self._trees.get(key)
        if tree is None:
            return None
        return tree.innermost_enclosing(point_ns)

    @property
    def stream_keys(self) -> list[StreamKey]:
        return list(self._trees.keys())
