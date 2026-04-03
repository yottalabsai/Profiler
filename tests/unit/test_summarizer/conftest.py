"""
Shared fixtures for test_summarizer.

All fixtures are fully in-memory. No GPU, no API key, no file I/O unless
using tmp_path.
"""
from __future__ import annotations

import pytest

from operator_profiler.planner.loop import LoopResult
from operator_profiler.planner.schema import GraphPattern, MemoryEntry, OptMemoryStore
from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    KernelMetrics,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _make_metadata(model_name: str = "TestModel", device: str = "cuda:0") -> CaptureMetadata:
    return CaptureMetadata(
        model_name=model_name,
        torch_version="2.3.0",
        capture_timestamp_utc="2026-03-22T00:00:00+00:00",
        device_name=device,
    )


def _make_kernel(
    kernel_id: str,
    duration_ns: int = 1_000_000,
    dram_read: int | None = None,
    dram_write: int | None = None,
    occupancy: float | None = None,
    ai: float | None = None,
) -> KernelRecord:
    raw: dict = {}
    if dram_read is not None:
        raw["dram__bytes_read.sum"] = dram_read
    if dram_write is not None:
        raw["dram__bytes_written.sum"] = dram_write
    if occupancy is not None:
        raw["sm__warps_active.avg.pct_of_peak_sustained_active"] = occupancy
    if ai is not None:
        raw["arithmetic_intensity"] = ai
    return KernelRecord(
        kernel_id=kernel_id,
        kernel_name=f"kernel_{kernel_id}",
        demangled_name=f"demangled_{kernel_id}",
        stream_id=0,
        device_id=0,
        start_ns=0,
        end_ns=duration_ns,
        duration_ns=duration_ns,
        metrics=KernelMetrics(raw=raw),
    )


# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def before_profile() -> OperatorAttributedProfile:
    """2-op profile: linear (memory_bound, 3 ms) + relu (compute_bound, 1 ms)."""
    linear_kernel = _make_kernel("k_linear", duration_ns=3_000_000, dram_read=1024, occupancy=0.4, ai=2.0)
    relu_kernel = _make_kernel("k_relu", duration_ns=1_000_000, occupancy=0.8, ai=10.0)
    return OperatorAttributedProfile(
        capture_metadata=_make_metadata(),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                kernels=[linear_kernel],
                aggregated=AggregatedMetrics(
                    total_duration_ns=3_000_000,
                    kernel_count=1,
                    bottleneck_classification="memory_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::relu_0",
                operator_name="aten::relu",
                call_index=0,
                kernels=[relu_kernel],
                aggregated=AggregatedMetrics(
                    total_duration_ns=1_000_000,
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
        ],
    )


@pytest.fixture
def after_profile_exact() -> OperatorAttributedProfile:
    """2-op profile matching before keys exactly, each 2x faster."""
    return OperatorAttributedProfile(
        capture_metadata=_make_metadata(),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                aggregated=AggregatedMetrics(
                    total_duration_ns=1_500_000,
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::relu_0",
                operator_name="aten::relu",
                call_index=0,
                aggregated=AggregatedMetrics(
                    total_duration_ns=500_000,
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
        ],
    )


@pytest.fixture
def fuse_plan() -> RewritePlan:
    """A RewritePlan with one FuseOp merging linear_0 + relu_0."""
    return RewritePlan(
        description="Fuse linear+relu",
        ops=[
            FuseOp(
                op="fuse",
                id="fuse_linear_relu",
                nodes=["aten::linear_0", "aten::relu_0"],
                strategy="inductor_fuse",
            )
        ],
    )


@pytest.fixture
def after_profile_fused(fuse_plan) -> OperatorAttributedProfile:
    """1-op profile: linear_relu fused (compute_bound), total 2 ms (from 4 ms)."""
    return OperatorAttributedProfile(
        capture_metadata=_make_metadata(),
        operators=[
            OperatorRecord(
                operator_id="fused_linear_relu_0",
                operator_name="fused_linear_relu",
                call_index=0,
                is_fused=True,
                fused_with=["aten::linear_0", "aten::relu_0"],
                aggregated=AggregatedMetrics(
                    total_duration_ns=2_000_000,
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
        ],
    )


@pytest.fixture
def loop_result_with_history(fuse_plan) -> LoopResult:
    """LoopResult with 2 iterations and speedup 2.0."""
    return LoopResult(
        best_plan=fuse_plan,
        best_speedup=2.0,
        history=[
            {
                "iteration": 0,
                "bottleneck": "memory_bound",
                "worst_op_id": "aten::linear_0",
                "memory_hits": 0,
                "plans_tried": 3,
                "best_speedup_so_far": 1.8,
                "beam_scores": [1.8, 1.5, 1.2],
            },
            {
                "iteration": 1,
                "bottleneck": "memory_bound",
                "worst_op_id": "aten::linear_0",
                "memory_hits": 1,
                "plans_tried": 3,
                "best_speedup_so_far": 2.0,
                "beam_scores": [2.0],
            },
        ],
    )


@pytest.fixture
def memory_store_with_entries() -> OptMemoryStore:
    """OptMemoryStore with 3 entries of varying speedup."""
    def _entry(entry_id, ops, bottleneck, speedup, model=None):
        pat = GraphPattern(
            op_sequence=ops,
            pattern_hash="aabbcc",
            input_shapes={},
        )
        plan = RewritePlan(
            description=f"plan_{entry_id}",
            ops=[
                FuseOp(op="fuse", id=f"fuse_{entry_id}", nodes=ops[:2], strategy="inductor_fuse")
            ] if len(ops) >= 2 else [],
        )
        return MemoryEntry(
            entry_id=entry_id,
            graph_pattern=pat,
            bottleneck=bottleneck,
            rewrite_plan=plan,
            speedup=speedup,
            model_name=model,
            created_at="2026-03-22T00:00:00+00:00",
        )

    return OptMemoryStore(
        entries=[
            _entry("e1", ["aten::conv2d", "aten::relu"], "memory_bound", 1.8, "ResNet50"),
            _entry("e2", ["aten::linear", "aten::gelu"], "latency_bound", 1.3, "GPT2"),
            _entry("e3", ["aten::mm", "aten::add"], "compute_bound", 1.15),
        ]
    )
