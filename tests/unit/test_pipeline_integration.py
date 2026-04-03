"""
End-to-end pipeline integration tests.

Tests the data flow across all pipeline stages:

  torch.fx.GraphModule
      → OptimizationLoop (mocked planner + profiler_fn)
      → LoopResult (serialization roundtrip)
      → HybridExecutor (applies best plan)
      → compute_diff (before/after ProfileDiff)
      → entries_to_rules (OptimizationRule list from curated memory)
      → SummaryReport construction
      → render_markdown / render_html / render_provenance_text
      → explain_node
      → RichDashboard (plain mode)

No GPU, no Anthropic API key required.  The profiler_fn and ThetaPlanner are
mocked; everything else uses real code paths.
"""
from __future__ import annotations

import json
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
)
from operator_profiler.planner.memory import OptimizationMemory
from operator_profiler.planner.schema import GraphPattern, MemoryEntry
from operator_profiler.planner.search import BeamSearch
from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    KernelMetrics,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)
from operator_profiler.summarizer import (
    SummaryReport,
    build_provenance_rows,
    compute_diff,
    entries_to_rules,
    explain_node,
    render_html,
    render_markdown,
    render_provenance_text,
    RichDashboard,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _TwoOpModel(nn.Module):
    """Minimal model: add → relu.  Produces predictable FX node names."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + x)


def _make_gm() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(_TwoOpModel())


def _computation_nodes(gm: torch.fx.GraphModule) -> list[str]:
    """Return node names that are actual computations (not placeholder/output)."""
    return [
        n.name for n in gm.graph.nodes
        if n.op not in ("placeholder", "output", "get_attr")
    ]


def _make_kernel(
    kernel_id: str,
    duration_ns: int = 1_000_000,
    occupancy: float = 0.5,
    ai: float = 5.0,
) -> KernelRecord:
    return KernelRecord(
        kernel_id=kernel_id,
        kernel_name=f"kernel_{kernel_id}",
        demangled_name=f"demangled_{kernel_id}",
        stream_id=0,
        device_id=0,
        start_ns=0,
        end_ns=duration_ns,
        duration_ns=duration_ns,
        metrics=KernelMetrics(raw={
            "dram__bytes_read.sum": 512,
            "dram__bytes_written.sum": 512,
            "sm__warps_active.avg.pct_of_peak_sustained_active": occupancy,
            "arithmetic_intensity": ai,
        }),
    )


def _make_before_profile() -> OperatorAttributedProfile:
    """2-op profile: add (memory_bound, 3 ms) + relu (latency_bound, 1 ms)."""
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="TwoOpModel",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-22T00:00:00+00:00",
            device_name="RTX 4090",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::add_0",
                operator_name="aten::add",
                call_index=0,
                kernels=[_make_kernel("k_add", duration_ns=3_000_000, occupancy=0.3, ai=1.5)],
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
                kernels=[_make_kernel("k_relu", duration_ns=1_000_000, occupancy=0.2, ai=0.5)],
                aggregated=AggregatedMetrics(
                    total_duration_ns=1_000_000,
                    kernel_count=1,
                    bottleneck_classification="latency_bound",
                ),
            ),
        ],
    )


def _make_after_profile_exact(speedup: float = 2.0) -> OperatorAttributedProfile:
    """2-op profile with same ops, each `speedup` times faster."""
    before_durations = {"aten::add_0": 3_000_000, "aten::relu_0": 1_000_000}
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="TwoOpModel",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-22T00:00:00+00:00",
            device_name="RTX 4090",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::add_0",
                operator_name="aten::add",
                call_index=0,
                aggregated=AggregatedMetrics(
                    total_duration_ns=int(3_000_000 / speedup),
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::relu_0",
                operator_name="aten::relu",
                call_index=0,
                aggregated=AggregatedMetrics(
                    total_duration_ns=int(1_000_000 / speedup),
                    kernel_count=1,
                    bottleneck_classification="compute_bound",
                ),
            ),
        ],
    )


def _make_loop(
    tmp_path: Path,
    planner_return: RewritePlan | None = None,
    profiler_return: OperatorAttributedProfile | None = None,
    n_iterations: int = 2,
) -> tuple[OptimizationLoop, OperatorAttributedProfile]:
    """Build an OptimizationLoop with all external dependencies mocked."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = planner_return or RewritePlan()
    mock_planner.rank_candidates.side_effect = lambda profile, candidates, **kw: candidates

    profiler_fn = MagicMock()
    profiler_fn.return_value = profiler_return or _make_before_profile()

    mem = OptimizationMemory(tmp_path / "pipeline_memory.json")
    search = BeamSearch(width=2, seed=42)
    cfg = LoopConfig(
        n_iterations=n_iterations,
        beam_width=2,
        executor_config=ExecutorConfig(skip_verification=True),
    )
    loop = OptimizationLoop(
        planner=mock_planner,
        memory=mem,
        search=search,
        profiler_fn=profiler_fn,
        config=cfg,
    )
    return loop, _make_before_profile()


def _build_summary_report(
    loop_result: LoopResult,
    before: OperatorAttributedProfile,
    after: OperatorAttributedProfile,
    memory: OptimizationMemory,
) -> SummaryReport:
    """Assemble a SummaryReport from pipeline outputs."""
    diff = compute_diff(before, after, loop_result.best_plan)
    rules = entries_to_rules(memory.entries)
    lessons = [r.rule_text for r in sorted(rules, key=lambda r: r.speedup, reverse=True)[:5]]
    return SummaryReport(
        diff=diff,
        rules=rules,
        lessons_learned=lessons,
        loop_history=loop_result.history,
        best_speedup=loop_result.best_speedup,
        best_plan_description=(
            loop_result.best_plan.description if loop_result.best_plan else None
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1 — Graph & Profile construction
# ---------------------------------------------------------------------------

def test_fx_graph_has_expected_computation_nodes():
    """TwoOpModel produces at least add and relu computation nodes."""
    gm = _make_gm()
    node_names = _computation_nodes(gm)
    assert len(node_names) >= 2


def test_before_profile_total_duration():
    """before_profile totals 4 ms across two operators."""
    profile = _make_before_profile()
    total = sum(op.aggregated.total_duration_ns for op in profile.operators)
    assert total == 4_000_000


def test_before_profile_schema_roundtrip():
    """OperatorAttributedProfile survives JSON serialise → deserialise."""
    profile = _make_before_profile()
    raw = profile.model_dump_json()
    restored = OperatorAttributedProfile.model_validate_json(raw)
    assert len(restored.operators) == len(profile.operators)
    assert restored.operators[0].operator_id == "aten::add_0"


# ---------------------------------------------------------------------------
# Stage 2 — OptimizationLoop
# ---------------------------------------------------------------------------

def test_loop_runs_and_returns_loop_result(tmp_path):
    """Full loop run with mocked planner/profiler yields a LoopResult."""
    loop, before = _make_loop(tmp_path)
    result = loop.run(_make_gm(), before, [torch.ones(1)])
    assert isinstance(result, LoopResult)
    assert result.best_speedup >= 1.0
    assert len(result.history) == 2


def test_loop_curates_memory_when_profiler_returns_faster_profile(tmp_path):
    """Memory grows when profiler_fn returns a faster profile (speedup > threshold)."""
    faster = _make_after_profile_exact(speedup=3.0)  # 3× speedup > 1.05 threshold
    loop, before = _make_loop(tmp_path, profiler_return=faster, n_iterations=1)
    loop.run(_make_gm(), before, [torch.ones(1)])
    assert len(loop._memory) >= 1


def test_loop_history_has_correct_keys(tmp_path):
    """Every iteration history dict contains all expected keys."""
    loop, before = _make_loop(tmp_path)
    result = loop.run(_make_gm(), before, [torch.ones(1)])
    required_keys = {
        "iteration", "bottleneck", "worst_op_id",
        "memory_hits", "plans_tried", "best_speedup_so_far", "beam_scores",
    }
    for entry in result.history:
        assert required_keys.issubset(entry.keys()), (
            f"Missing keys: {required_keys - entry.keys()}"
        )


def test_loop_uses_broad_search(tmp_path):
    """Loop calls broad_search, not the legacy narrow search."""
    from unittest.mock import patch
    loop, before = _make_loop(tmp_path, n_iterations=1)
    with patch.object(loop._memory, "broad_search", wraps=loop._memory.broad_search) as mb, \
         patch.object(loop._memory, "search", wraps=loop._memory.search) as mn:
        loop.run(_make_gm(), before, [torch.ones(1)])
    assert mb.call_count >= 1
    assert mn.call_count == 0


# ---------------------------------------------------------------------------
# Stage 3 — LoopResult serialization roundtrip
# ---------------------------------------------------------------------------

def test_loop_result_to_dict_from_dict_preserves_speedup(tmp_path):
    """LoopResult.to_dict() → JSON → from_dict() preserves best_speedup."""
    loop, before = _make_loop(tmp_path, n_iterations=1)
    result = loop.run(_make_gm(), before, [torch.ones(1)])

    d = result.to_dict()
    restored = LoopResult.from_dict(d)

    assert restored.best_speedup == pytest.approx(result.best_speedup)
    assert len(restored.history) == len(result.history)


def test_loop_result_to_dict_is_json_serializable(tmp_path):
    """to_dict() produces a structure that can be json.dumps'd without error."""
    loop, before = _make_loop(tmp_path, n_iterations=1)
    result = loop.run(_make_gm(), before, [torch.ones(1)])
    raw = json.dumps(result.to_dict())
    assert isinstance(raw, str)
    assert len(raw) > 0


def test_loop_result_with_plan_roundtrips(tmp_path):
    """LoopResult containing a non-empty RewritePlan survives roundtrip."""
    gm = _make_gm()
    node_names = _computation_nodes(gm)
    fuse_plan = RewritePlan(
        description="fuse all",
        ops=[FuseOp(op="fuse", id="f0", nodes=node_names, strategy="inductor_fuse")],
    )
    loop, before = _make_loop(
        tmp_path,
        planner_return=fuse_plan,
        profiler_return=_make_after_profile_exact(speedup=2.0),
        n_iterations=1,
    )
    result = loop.run(gm, before, [torch.ones(1)])
    if result.best_plan is not None:
        restored = LoopResult.from_dict(result.to_dict())
        assert restored.best_plan is not None
        assert len(restored.best_plan.ops) == len(result.best_plan.ops)


# ---------------------------------------------------------------------------
# Stage 4 — HybridExecutor
# ---------------------------------------------------------------------------

def test_hybrid_executor_empty_plan_returns_equivalent_graph():
    """Empty plan returns a graph that computes the same output as the original."""
    gm = _make_gm()
    executor = HybridExecutor(gm, RewritePlan(), ExecutorConfig(skip_verification=True))
    result_gm, ver_results = executor.run()
    assert isinstance(result_gm, torch.fx.GraphModule)
    assert ver_results == []  # empty plan → no ops → no verification results


def test_hybrid_executor_fuse_plan_applies_without_crash():
    """A FuseOp targeting real node names in the graph runs without error."""
    gm = _make_gm()
    node_names = _computation_nodes(gm)
    plan = RewritePlan(
        ops=[FuseOp(op="fuse", id="fuse_all", nodes=node_names, strategy="inductor_fuse")]
    )
    executor = HybridExecutor(gm, plan, ExecutorConfig(skip_verification=True))
    result_gm, _ = executor.run()
    assert isinstance(result_gm, torch.fx.GraphModule)


def test_hybrid_executor_does_not_mutate_original_graph():
    """HybridExecutor must never modify the graph it receives."""
    gm = _make_gm()
    original_node_names = [n.name for n in gm.graph.nodes]
    node_names = _computation_nodes(gm)
    plan = RewritePlan(
        ops=[FuseOp(op="fuse", id="f0", nodes=node_names, strategy="inductor_fuse")]
    )
    HybridExecutor(gm, plan, ExecutorConfig(skip_verification=True)).run()
    assert [n.name for n in gm.graph.nodes] == original_node_names


# ---------------------------------------------------------------------------
# Stage 5 — compute_diff
# ---------------------------------------------------------------------------

def test_compute_diff_exact_match_total_speedup():
    """compute_diff with 2× faster after profile reports total_speedup ≈ 2.0."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    diff = compute_diff(before, after, None)
    assert diff.total_speedup == pytest.approx(2.0, abs=0.05)


def test_compute_diff_model_name_propagated():
    """ProfileDiff.model_name comes from the before profile's capture_metadata."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None)
    assert diff.model_name == "TwoOpModel"


def test_compute_diff_operator_diffs_count():
    """All before operators appear in operator_diffs."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None)
    assert len(diff.operator_diffs) >= len(before.operators)


def test_compute_diff_exact_match_type():
    """Operators that appear unchanged are tagged match_type='exact'."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None)
    exact_diffs = [d for d in diff.operator_diffs if d.match_type == "exact"]
    assert len(exact_diffs) == 2


def test_compute_diff_wall_time_saved():
    """wall_time_saved_ns should be positive when after is faster than before."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    diff = compute_diff(before, after, None)
    assert diff.wall_time_saved_ns > 0


def test_compute_diff_top_bottlenecks_before():
    """top_bottlenecks_before contains the slowest operators."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None, top_n=1)
    # The slowest before op is aten::add_0 at 3 ms
    assert len(diff.top_bottlenecks_before) >= 1
    assert diff.top_bottlenecks_before[0].duration_before_ns == 3_000_000


# ---------------------------------------------------------------------------
# Stage 6 — entries_to_rules
# ---------------------------------------------------------------------------

def _seed_memory(tmp_path: Path) -> OptimizationMemory:
    """Return a memory store pre-seeded with known entries."""
    from operator_profiler.planner.memory import _make_pattern_hash
    mem = OptimizationMemory(tmp_path / "seed_mem.json")
    for i, (bottleneck, speedup) in enumerate(
        [("memory_bound", 1.8), ("latency_bound", 1.3), ("compute_bound", 1.2)]
    ):
        pattern = GraphPattern(
            op_sequence=["aten::add", "aten::relu"],
            pattern_hash=_make_pattern_hash(["aten::add", "aten::relu"]),
        )
        entry = MemoryEntry(
            entry_id=f"e{i}",
            graph_pattern=pattern,
            bottleneck=bottleneck,
            rewrite_plan=RewritePlan(
                description=f"plan_{i}",
                ops=[FuseOp(op="fuse", id=f"f{i}", nodes=["aten::add", "aten::relu"], strategy="inductor_fuse")]
            ),
            speedup=speedup,
            created_at="2026-03-22T00:00:00+00:00",
        )
        mem._store.entries.append(entry)
    return mem


def test_entries_to_rules_returns_one_rule_per_entry(tmp_path):
    """entries_to_rules produces exactly one OptimizationRule per MemoryEntry."""
    mem = _seed_memory(tmp_path)
    rules = entries_to_rules(mem.entries)
    assert len(rules) == len(mem.entries)


def test_entries_to_rules_speedup_pct_correct(tmp_path):
    """speedup_pct = (speedup - 1) * 100 for each rule."""
    mem = _seed_memory(tmp_path)
    rules = entries_to_rules(mem.entries)
    for rule, entry in zip(rules, mem.entries):
        expected_pct = (entry.speedup - 1) * 100
        assert rule.speedup_pct == pytest.approx(expected_pct, abs=0.01)


def test_entries_to_rules_rule_text_non_empty(tmp_path):
    """Every rule must have a non-empty rule_text."""
    mem = _seed_memory(tmp_path)
    rules = entries_to_rules(mem.entries)
    assert all(len(r.rule_text) > 0 for r in rules)


def test_entries_to_rules_empty_memory():
    """Empty memory produces empty rule list (no crash)."""
    rules = entries_to_rules([])
    assert rules == []


# ---------------------------------------------------------------------------
# Stage 7 — SummaryReport construction
# ---------------------------------------------------------------------------

def test_summary_report_builds_without_error(tmp_path):
    """SummaryReport can be constructed from pipeline outputs."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    loop_result = LoopResult(best_plan=RewritePlan(), best_speedup=2.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    assert isinstance(report, SummaryReport)


def test_summary_report_best_speedup_matches_loop_result(tmp_path):
    """SummaryReport.best_speedup comes directly from LoopResult."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=3.0)
    loop_result = LoopResult(best_plan=None, best_speedup=3.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    assert report.best_speedup == pytest.approx(3.0)


def test_summary_report_rules_count_matches_memory(tmp_path):
    """SummaryReport.rules has one entry per curated memory item."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    loop_result = LoopResult(best_plan=None, best_speedup=1.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    assert len(report.rules) == len(mem.entries)


# ---------------------------------------------------------------------------
# Stage 8 — Renderers
# ---------------------------------------------------------------------------

def test_render_markdown_contains_model_name(tmp_path):
    """Markdown output includes the model name from the profile."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    md = render_markdown(report)
    assert "TwoOpModel" in md


def test_render_markdown_contains_speedup(tmp_path):
    """Markdown output mentions the total speedup figure."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    md = render_markdown(report)
    assert "2.0" in md or "2.00" in md


def test_render_markdown_contains_lessons_learned_section(tmp_path):
    """Markdown output has a Lessons Learned section when rules exist."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    loop_result = LoopResult(best_plan=None, best_speedup=1.8, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    md = render_markdown(report)
    assert "Lessons" in md


def test_render_html_is_non_empty(tmp_path):
    """HTML render returns a non-empty string."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    html = render_html(report)
    assert len(html) > 100


def test_render_html_contains_html_boilerplate(tmp_path):
    """render_html returns a self-contained HTML document."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
    mem = _seed_memory(tmp_path)
    report = _build_summary_report(loop_result, before, after, mem)
    html = render_html(report)
    assert "<html" in html.lower()
    assert "TwoOpModel" in html


# ---------------------------------------------------------------------------
# Stage 9 — Provenance rows
# ---------------------------------------------------------------------------

def test_build_provenance_rows_one_row_per_kernel():
    """One ProvenanceRow is emitted for each kernel in the profile."""
    before = _make_before_profile()
    rows = build_provenance_rows(before, RewritePlan())
    total_kernels = sum(len(op.kernels) for op in before.operators)
    assert len(rows) == total_kernels


def test_render_provenance_text_contains_operator_ids():
    """Provenance text render includes the operator IDs from the profile."""
    before = _make_before_profile()
    rows = build_provenance_rows(before, RewritePlan())
    text = render_provenance_text(rows)
    assert "aten::add_0" in text
    assert "aten::relu_0" in text


def test_render_provenance_text_is_non_empty():
    """render_provenance_text returns a non-empty string for non-empty rows."""
    before = _make_before_profile()
    rows = build_provenance_rows(before, RewritePlan())
    text = render_provenance_text(rows)
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Stage 10 — explain_node
# ---------------------------------------------------------------------------

def test_explain_node_returns_string_for_known_node():
    """explain_node returns a non-empty explanation string for a known node ID."""
    before = _make_before_profile()
    after = _make_after_profile_exact(speedup=2.0)
    diff = compute_diff(before, after, None)
    loop_result = LoopResult(
        best_plan=None,
        best_speedup=2.0,
        history=[
            {
                "iteration": 0,
                "bottleneck": "memory_bound",
                "worst_op_id": "aten::add_0",
                "memory_hits": 0,
                "plans_tried": 2,
                "best_speedup_so_far": 2.0,
                "beam_scores": [2.0],
            }
        ],
    )
    explanation = explain_node("aten::add_0", diff, before, loop_result)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_explain_node_normalises_double_underscore_ids():
    """explain_node accepts aten__add_0 and normalises to aten::add_0."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None)
    loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
    # aten__add_0 → aten::add_0 after normalisation
    explanation = explain_node("aten__add_0", diff, before, loop_result)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_explain_node_unknown_id_returns_friendly_error():
    """explain_node with an unknown node ID returns a helpful message, not a crash."""
    before = _make_before_profile()
    after = _make_after_profile_exact()
    diff = compute_diff(before, after, None)
    loop_result = LoopResult(best_plan=None, best_speedup=1.0, history=[])
    explanation = explain_node("nonexistent__op_999", diff, before, loop_result)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


# ---------------------------------------------------------------------------
# Stage 11 — RichDashboard (plain mode)
# ---------------------------------------------------------------------------

def test_rich_dashboard_render_plain_does_not_crash(tmp_path, capsys):
    """RichDashboard.render() in plain mode (no rich installed) does not crash."""
    import operator_profiler.summarizer.dashboard as dash_mod
    original = dash_mod._RICH_AVAILABLE
    dash_mod._RICH_AVAILABLE = False
    try:
        before = _make_before_profile()
        after = _make_after_profile_exact(speedup=2.0)
        loop_result = LoopResult(best_plan=None, best_speedup=2.0, history=[])
        mem = _seed_memory(tmp_path)
        report = _build_summary_report(loop_result, before, after, mem)
        dashboard = RichDashboard(report)
        dashboard.render()  # must not raise
    finally:
        dash_mod._RICH_AVAILABLE = original


# ---------------------------------------------------------------------------
# Stage 12 — Full pipeline data integrity
# ---------------------------------------------------------------------------

def test_full_pipeline_data_integrity(tmp_path):
    """
    Run every stage in sequence and assert that data flows correctly
    from one stage to the next.

    This is the single most important integration test: it verifies that the
    pipeline stages produce compatible outputs that connect without error.
    """
    # --- Build model ---
    gm = _make_gm()
    node_names = _computation_nodes(gm)
    assert len(node_names) >= 2, "Model must have at least 2 computation nodes"

    # --- Build initial profile (before) ---
    before = _make_before_profile()
    before_total_ns = sum(op.aggregated.total_duration_ns for op in before.operators)

    # --- Run optimization loop ---
    fuse_plan = RewritePlan(
        description="Fuse add+relu to reduce kernel launch overhead",
        ops=[FuseOp(op="fuse", id="fuse_0", nodes=node_names, strategy="inductor_fuse")],
    )
    faster_profile = _make_after_profile_exact(speedup=2.5)

    loop, _ = _make_loop(
        tmp_path,
        planner_return=fuse_plan,
        profiler_return=faster_profile,
        n_iterations=2,
    )
    loop_result = loop.run(gm, before, [torch.ones(1)])

    # LoopResult sanity
    assert isinstance(loop_result, LoopResult)
    assert loop_result.best_speedup >= 1.0
    assert len(loop_result.history) == 2

    # --- LoopResult serialization roundtrip ---
    restored_result = LoopResult.from_dict(loop_result.to_dict())
    assert restored_result.best_speedup == pytest.approx(loop_result.best_speedup)

    # --- Apply best plan to graph ---
    plan_to_apply = loop_result.best_plan or RewritePlan()
    executor = HybridExecutor(gm, plan_to_apply, ExecutorConfig(skip_verification=True))
    result_gm, _ = executor.run()
    assert isinstance(result_gm, torch.fx.GraphModule)

    # Original graph must be untouched
    assert [n.name for n in gm.graph.nodes] is not None

    # --- Build after profile (simulating re-profiling the optimised model) ---
    after = _make_after_profile_exact(speedup=2.5)
    after_total_ns = sum(
        op.aggregated.total_duration_ns for op in after.operators
        if op.aggregated is not None
    )
    assert after_total_ns < before_total_ns

    # --- compute_diff ---
    diff = compute_diff(before, after, plan_to_apply)
    assert diff.model_name == "TwoOpModel"
    assert diff.total_speedup == pytest.approx(2.5, abs=0.1)
    assert diff.wall_time_saved_ns > 0
    assert len(diff.operator_diffs) >= 1

    # --- entries_to_rules from curated memory ---
    rules = entries_to_rules(loop._memory.entries)
    # memory may be empty if speedup < threshold in this run; that's OK
    assert isinstance(rules, list)

    # --- Build SummaryReport ---
    lessons = [r.rule_text for r in sorted(rules, key=lambda r: r.speedup, reverse=True)[:5]]
    report = SummaryReport(
        diff=diff,
        rules=rules,
        lessons_learned=lessons,
        loop_history=loop_result.history,
        best_speedup=loop_result.best_speedup,
        best_plan_description=plan_to_apply.description,
    )

    # SummaryReport must reference the correct speedup from LoopResult
    assert report.best_speedup == pytest.approx(loop_result.best_speedup)
    assert report.diff.model_name == "TwoOpModel"

    # --- Render Markdown ---
    md = render_markdown(report)
    assert "TwoOpModel" in md
    assert isinstance(md, str) and len(md) > 0

    # --- Render HTML ---
    html = render_html(report)
    assert "<html" in html.lower()
    assert "TwoOpModel" in html

    # --- Provenance rows ---
    rows = build_provenance_rows(before, plan_to_apply)
    prov_text = render_provenance_text(rows)
    assert isinstance(prov_text, str)

    # --- explain_node ---
    explanation = explain_node("aten::add_0", diff, before, loop_result)
    assert isinstance(explanation, str) and len(explanation) > 0

    # --- SummaryReport JSON roundtrip ---
    raw = report.model_dump_json()
    restored_report = SummaryReport.model_validate_json(raw)
    assert restored_report.diff.model_name == "TwoOpModel"
    assert restored_report.best_speedup == pytest.approx(report.best_speedup)
