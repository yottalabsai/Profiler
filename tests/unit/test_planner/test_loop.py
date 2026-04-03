"""
Unit tests for OptimizationLoop.

ThetaPlanner and profiler_fn are mocked — no GPU, no API key required.
HybridExecutor runs in skip_verification mode with simple FX graphs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.fx

from operator_profiler.planner.loop import (
    LoopConfig,
    LoopResult,
    OptimizationLoop,
    _total_duration,
    _worst_operator_id,
)
from operator_profiler.planner.memory import OptimizationMemory
from operator_profiler.planner.search import BeamSearch
from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.rewriter.executor import ExecutorConfig
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    OperatorAttributedProfile,
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    op_names: list[str] | None = None,
    bottleneck: str = "memory_bound",
    duration_ns: int = 1000,
) -> OperatorAttributedProfile:
    if op_names is None:
        op_names = ["aten::add"]
    operators = [
        OperatorRecord(
            operator_id=f"{name}_{i}",
            operator_name=name,
            call_index=i,
            aggregated=AggregatedMetrics(
                total_duration_ns=duration_ns,
                kernel_count=1,
                bottleneck_classification=bottleneck,
            ),
        )
        for i, name in enumerate(op_names)
    ]
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="M",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-21T00:00:00+00:00",
        ),
        operators=operators,
    )


def _make_gm() -> torch.fx.GraphModule:
    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x
    return torch.fx.symbolic_trace(M())


def _make_loop(
    tmp_path: Path,
    planner_plan_fn=None,
    profiler_return: OperatorAttributedProfile | None = None,
    n_iterations: int = 2,
    beam_width: int = 2,
) -> tuple[OptimizationLoop, OperatorAttributedProfile]:
    """
    Build an OptimizationLoop with mocked planner + profiler.
    """
    mock_planner = MagicMock()
    mock_planner.plan.side_effect = (
        planner_plan_fn if planner_plan_fn else lambda *a, **kw: RewritePlan()
    )
    # rank_candidates passes candidates through unchanged (no API call in tests)
    mock_planner.rank_candidates.side_effect = lambda profile, candidates, **kw: candidates

    profiler_fn = MagicMock()
    profiler_fn.return_value = (
        profiler_return if profiler_return is not None else _make_profile()
    )

    mem = OptimizationMemory(tmp_path / "opt.json")
    search = BeamSearch(width=beam_width, seed=0)
    cfg = LoopConfig(
        n_iterations=n_iterations,
        beam_width=beam_width,
        executor_config=ExecutorConfig(skip_verification=True),
    )
    loop = OptimizationLoop(
        planner=mock_planner,
        memory=mem,
        search=search,
        profiler_fn=profiler_fn,
        config=cfg,
    )
    initial_profile = _make_profile(duration_ns=1000)
    return loop, initial_profile


# ---------------------------------------------------------------------------
# _total_duration
# ---------------------------------------------------------------------------

def test_total_duration_sums_all_ops():
    profile = _make_profile(["aten::a", "aten::b"], duration_ns=500)
    assert _total_duration(profile) == 1000


def test_total_duration_skips_none_aggregated():
    profile = OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="M",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-21T00:00:00+00:00",
        ),
        operators=[
            OperatorRecord(operator_id="x", operator_name="aten::relu", call_index=0)
        ],
    )
    assert _total_duration(profile) == 0


# ---------------------------------------------------------------------------
# _worst_operator_id
# ---------------------------------------------------------------------------

def test_worst_operator_id_picks_slowest():
    profile = _make_profile(["aten::a", "aten::b"], duration_ns=100)
    profile.operators[1].aggregated.total_duration_ns = 9000
    assert _worst_operator_id(profile) == "aten::b_1"


# ---------------------------------------------------------------------------
# OptimizationLoop.run()
# ---------------------------------------------------------------------------

def test_loop_returns_loop_result(tmp_path):
    loop, initial_profile = _make_loop(tmp_path)
    gm = _make_gm()
    result = loop.run(gm, initial_profile, [torch.ones(1)])
    assert isinstance(result, LoopResult)
    assert result.best_speedup >= 1.0


def test_loop_history_length_matches_iterations(tmp_path):
    n = 3
    loop, initial_profile = _make_loop(tmp_path, n_iterations=n)
    result = loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert len(result.history) == n


def test_loop_history_contains_expected_keys(tmp_path):
    loop, initial_profile = _make_loop(tmp_path)
    result = loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    for entry in result.history:
        assert "iteration" in entry
        assert "bottleneck" in entry
        assert "memory_hits" in entry
        assert "plans_tried" in entry
        assert "best_speedup_so_far" in entry


def test_loop_curates_memory_on_speedup(tmp_path):
    """If profiler_fn returns a faster profile (lower duration), memory is populated."""
    # baseline: 1000 ns; new profile: 500 ns → speedup=2.0 > 1.05
    faster_profile = _make_profile(duration_ns=500)
    loop, initial_profile = _make_loop(tmp_path, profiler_return=faster_profile, n_iterations=1)
    loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert len(loop._memory) >= 1


def test_loop_no_curate_when_no_speedup(tmp_path):
    """If profiler_fn returns same duration, memory stays empty."""
    same_profile = _make_profile(duration_ns=1000)
    loop, initial_profile = _make_loop(tmp_path, profiler_return=same_profile, n_iterations=2)
    loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert len(loop._memory) == 0


def test_loop_best_plan_none_when_all_plans_empty(tmp_path):
    """All empty plans (no ops) → best_plan is None (no improvement)."""
    loop, initial_profile = _make_loop(
        tmp_path,
        planner_plan_fn=lambda *a, **kw: RewritePlan(),
        profiler_return=_make_profile(duration_ns=1000),
    )
    result = loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert result.best_plan is None


def test_loop_executor_failure_does_not_crash(tmp_path):
    """Plans that cause the executor to raise are skipped gracefully."""
    def bad_plan(*args, **kwargs):
        # Reference a node that doesn't exist → PreFlightError
        from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
        return RewritePlan(
            ops=[FuseOp(op="fuse", id="f0", nodes=["nonexistent_a", "nonexistent_b"])]
        )

    loop, initial_profile = _make_loop(
        tmp_path, planner_plan_fn=bad_plan, n_iterations=1
    )
    # Should not raise
    result = loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert isinstance(result, LoopResult)


def test_loop_planner_called_n_times_per_iteration(tmp_path):
    """ThetaPlanner.plan should be called beam_width times per iteration."""
    loop, initial_profile = _make_loop(tmp_path, n_iterations=1, beam_width=3)
    loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    # beam_width=3, 1 iteration → planner called 3 times
    assert loop._planner.plan.call_count == 3


def test_loop_calls_rank_candidates_each_iteration(tmp_path):
    """rank_candidates is called once per iteration to re-rank broad_search results."""
    n = 2
    loop, initial_profile = _make_loop(tmp_path, n_iterations=n)
    loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert loop._planner.rank_candidates.call_count == n


def test_loop_uses_broad_search_not_narrow_search(tmp_path):
    """Loop calls broad_search (no bottleneck filter) rather than the legacy search."""
    loop, initial_profile = _make_loop(tmp_path, n_iterations=1)
    # Patch both search methods so we can distinguish which was called
    from unittest.mock import patch
    with patch.object(loop._memory, "broad_search", wraps=loop._memory.broad_search) as mock_broad, \
         patch.object(loop._memory, "search", wraps=loop._memory.search) as mock_narrow:
        loop.run(_make_gm(), initial_profile, [torch.ones(1)])
    assert mock_broad.call_count >= 1
    assert mock_narrow.call_count == 0
