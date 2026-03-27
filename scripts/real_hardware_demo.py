"""
real_hardware_demo.py — Full pipeline run on real CUDA hardware.

What is real here:
  - Model runs on the actual GPU (no mocking of tensor ops)
  - HybridExecutor numerical verification runs on CUDA (both graphs executed,
    outputs compared with torch.testing.assert_close)
  - GPU timing via torch.cuda.Event (hardware-accurate microsecond wall time)
  - torch.profiler CUDA activity trace → per-op durations for the profile
  - torch.compile (Inductor) runs for the optimized baseline — real kernel fusion
  - All four agents (DiagnosisAgent, VerifierAgent, RuleAgent, MemoryCuratorAgent)
    exercise real Anthropic API calls when ANTHROPIC_API_KEY is set

What is still mocked:
  - ThetaPlanner (the LLM API call); replaced by a pre-built RewritePlan
    so no Anthropic API key is needed for the core pipeline
  - VerifierAgent demo uses a synthetic failure scenario (labeled clearly)
    because FFBlock's FuseOp passes verification cleanly

Usage:
    # Core pipeline only (no API key needed):
    conda run -n ml_env python scripts/real_hardware_demo.py

    # Full pipeline including all four agents:
    ANTHROPIC_API_KEY=sk-ant-... conda run -n ml_env python scripts/real_hardware_demo.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.fx
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent.parent))

from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor
from operator_profiler.rewriter.verification import NodeDiff, VerificationResult
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    KernelRecord,
    KernelMetrics,
    OperatorAttributedProfile,
    OperatorRecord,
)
from operator_profiler.planner.schema import GraphPattern, MemoryEntry
from operator_profiler.planner.memory import OptimizationMemory, _make_pattern_hash
from operator_profiler.summarizer import (
    SummaryReport,
    build_provenance_rows,
    compute_diff,
    render_markdown,
    render_provenance_text,
)
from operator_profiler.summarizer.rules import entry_to_rule
from operator_profiler.aggregator.roofline import KNOWN_GPU_SPECS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE      = "cuda"
BATCH_SIZE  = 64
WARMUP      = 20
MEASURE     = 200
IN_FEATURES = 512
HIDDEN      = 2048

assert torch.cuda.is_available(), "No CUDA device found."
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB  = torch.cuda.get_device_properties(0).total_memory / 1e9

# Ridge point lookup
_specs = None
for key, val in KNOWN_GPU_SPECS.items():
    if key.lower() in GPU_NAME.lower() or GPU_NAME.lower() in key.lower():
        _specs = val
        break
RIDGE_POINT = (
    _specs["peak_compute_gflops"] / _specs["peak_bandwidth_gbs"]
    if _specs else None
)

# Agent enablement
AGENTS_ENABLED = bool(os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FFBlock(nn.Module):
    """Transformer-style feed-forward block: Linear → ReLU → Linear → GELU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, IN_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.fc2(torch.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# CUDA timing helpers
# ---------------------------------------------------------------------------

def cuda_time_ms(fn, n: int = MEASURE) -> float:
    """Return mean per-call CUDA wall time in ms over n iterations."""
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        for _ in range(n):
            fn()
        end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def profiler_op_times(fn, n: int = 10) -> dict[str, float]:
    """
    Run fn under torch.profiler and return per-op device (CUDA) time in µs.
    Both CPU and CUDA activities are needed to get aten:: dispatch events.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        acc_events=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(n):
                fn()
    torch.cuda.synchronize()

    times: dict[str, float] = {}
    for evt in prof.key_averages():
        device_us = getattr(evt, "self_device_time_total", 0)
        count = max(getattr(evt, "count", n), 1)
        if device_us > 0 and evt.key.startswith("aten::"):
            times[evt.key] = times.get(evt.key, 0.0) + device_us / count
    return times


# ---------------------------------------------------------------------------
# Build OperatorAttributedProfile from profiler data
# ---------------------------------------------------------------------------

def build_profile_from_measurements(
    op_times_us: dict[str, float],
    device_name: str,
    model_name: str,
) -> OperatorAttributedProfile:
    """
    Convert torch.profiler per-op CUDA times into an OperatorAttributedProfile.
    No per-kernel Nsight metrics (no ncu) — durations are real; AI/occupancy are None.
    """
    operators = []
    for call_idx, (op_name, duration_us) in enumerate(
        sorted(op_times_us.items(), key=lambda kv: -kv[1])
    ):
        duration_ns = int(duration_us * 1_000)
        operators.append(
            OperatorRecord(
                operator_id=f"{op_name}_{call_idx}",
                operator_name=op_name,
                call_index=call_idx,
                aggregated=AggregatedMetrics(
                    total_duration_ns=duration_ns,
                    kernel_count=1,
                    bottleneck_classification="memory_bound"
                    if op_name in ("aten::mm", "aten::linear", "aten::addmm")
                    else "latency_bound",
                ),
            )
        )
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name=model_name,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "unknown",
            compile_mode="eager",
            capture_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            device_name=device_name,
        ),
        operators=operators,
    )


def build_after_profile(
    op_times_us: dict[str, float],
    before_profile: OperatorAttributedProfile,
) -> OperatorAttributedProfile:
    """Build the after profile from compiled op times."""
    operators = []
    for call_idx, (op_name, duration_us) in enumerate(
        sorted(op_times_us.items(), key=lambda kv: -kv[1])
    ):
        duration_ns = int(duration_us * 1_000)
        operators.append(
            OperatorRecord(
                operator_id=f"{op_name}_{call_idx}",
                operator_name=op_name,
                call_index=call_idx,
                aggregated=AggregatedMetrics(
                    total_duration_ns=duration_ns,
                    kernel_count=1,
                    bottleneck_classification="compute_bound"
                    if op_name in ("aten::mm", "aten::linear", "aten::addmm")
                    else "latency_bound",
                ),
            )
        )
    meta = before_profile.capture_metadata
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name=meta.model_name,
            torch_version=meta.torch_version,
            cuda_version=meta.cuda_version,
            compile_mode="inductor",
            capture_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            device_name=meta.device_name,
        ),
        operators=operators,
    )


# ---------------------------------------------------------------------------
# Agent demo helpers
# ---------------------------------------------------------------------------

def _make_model_stats_for_demo(device_name: str):
    """
    Build a minimal ModelStats for DiagnosisAgent.  Since we have no ncu
    per-kernel AI/occupancy data, model-wide distributions are None — the
    agent will base its classification on duration rank order and kernel count.
    """
    from operator_profiler.agents.diagnosis import ModelStats
    return ModelStats(
        median_ai=None,
        p25_ai=None,
        p75_ai=None,
        median_occupancy=None,
        median_tensor_core_pct=None,
        ridge_point=RIDGE_POINT,
        device_name=device_name,
    )


def _make_memory_entry(
    op_sequence: list[str],
    bottleneck: str,
    plan: RewritePlan,
    speedup: float,
    model_name: str,
) -> MemoryEntry:
    """Construct a MemoryEntry for demo/synthetic use."""
    return MemoryEntry(
        entry_id=uuid.uuid4().hex,
        graph_pattern=GraphPattern(
            op_sequence=op_sequence,
            pattern_hash=_make_pattern_hash(op_sequence),
        ),
        bottleneck=bottleneck,
        rewrite_plan=plan,
        speedup=speedup,
        profile_id="1.0",
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 72

    print(sep)
    print("  OPERATOR PROFILER — REAL HARDWARE RUN")
    print(f"  GPU:   {GPU_NAME}")
    print(f"  VRAM:  {VRAM_GB:.1f} GB   |   CUDA: {torch.version.cuda}")
    print(f"  Torch: {torch.__version__}")
    if RIDGE_POINT:
        print(f"  Ridge point: {RIDGE_POINT:.0f} FLOP/byte  "
              f"(peak_compute={_specs['peak_compute_gflops']/1e3:.0f} TFLOP/s  "
              f"peak_bw={_specs['peak_bandwidth_gbs']:.0f} GB/s)")
    if AGENTS_ENABLED:
        print("  Agents: ENABLED (ANTHROPIC_API_KEY set)")
    else:
        print("  Agents: DISABLED (set ANTHROPIC_API_KEY to enable)")
    print(sep)

    # -----------------------------------------------------------------------
    # 1. Build model + trace
    # -----------------------------------------------------------------------
    model = FFBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE)

    gm = torch.fx.symbolic_trace(model)
    node_names = [n.name for n in gm.graph.nodes
                  if n.op not in ("placeholder", "output", "get_attr")]

    print(f"\n## Model: FFBlock  (batch={BATCH_SIZE}, in={IN_FEATURES}, hidden={HIDDEN})")
    print(f"   FX nodes: {node_names}")

    # -----------------------------------------------------------------------
    # 2. Baseline — eager timing + torch.profiler
    # -----------------------------------------------------------------------
    print(f"\n## Stage 1 — Eager baseline ({WARMUP} warmup, {MEASURE} timed iterations)")
    eager_ms = cuda_time_ms(lambda: model(x))
    print(f"   Eager mean: {eager_ms:.4f} ms/call")

    eager_op_times = profiler_op_times(lambda: model(x))
    print(f"   Per-op CUDA times (torch.profiler):")
    for op, us in sorted(eager_op_times.items(), key=lambda kv: -kv[1]):
        print(f"     {op:<30}  {us:>8.1f} µs")

    # -----------------------------------------------------------------------
    # 3. Build before profile from real measurements
    # -----------------------------------------------------------------------
    before_profile = build_profile_from_measurements(
        eager_op_times, GPU_NAME, "FFBlock"
    )
    before_total_ms = sum(
        op.aggregated.total_duration_ns for op in before_profile.operators
    ) / 1e6
    print(f"\n   OperatorAttributedProfile built:  {len(before_profile.operators)} ops, "
          f"{before_total_ms:.3f} ms total")

    # -----------------------------------------------------------------------
    # Agent Stage A — DiagnosisAgent: LLM bottleneck reclassification
    # -----------------------------------------------------------------------
    print(f"\n## Agent Stage A — DiagnosisAgent (LLM bottleneck reclassification)")
    if not AGENTS_ENABLED:
        print("   SKIPPED — set ANTHROPIC_API_KEY to enable agent stages")
    else:
        from operator_profiler.agents.diagnosis import DiagnosisAgent
        diagnosis_agent = DiagnosisAgent()
        model_stats = _make_model_stats_for_demo(GPU_NAME)

        print(f"   Classifying {len(before_profile.operators)} operators "
              f"(no ncu metrics — agent will use duration rank + kernel count):")
        for op in before_profile.operators:
            heuristic = op.aggregated.bottleneck_classification
            result = diagnosis_agent.diagnose(op, model_stats)
            # Overwrite with LLM classification
            op.aggregated.bottleneck_classification = result.classification
            changed = "→" if result.classification != heuristic else "="
            print(f"     {op.operator_id:<35}  "
                  f"{heuristic:<15} {changed} {result.classification}")
            if result.reasoning:
                print(f"       └─ {result.reasoning[:120]}")

    # -----------------------------------------------------------------------
    # 4. HybridExecutor — real CUDA verification
    # -----------------------------------------------------------------------
    print(f"\n## Stage 2 — HybridExecutor (real CUDA numerical verification)")
    relu_nodes = [n for n in node_names if "relu" in n]
    gelu_nodes = [n for n in node_names if "gelu" in n]
    fc_nodes   = [n for n in node_names if "fc" in n]

    fuse_ops = []
    if len(fc_nodes) >= 1 and relu_nodes:
        fuse_ops.append(FuseOp(
            op="fuse", id="fuse_fc1_relu",
            nodes=[fc_nodes[0], relu_nodes[0]],
            strategy="inductor_fuse",
            comment="Fuse fc1+relu to eliminate DRAM round-trip",
        ))
    if len(fc_nodes) >= 2 and gelu_nodes:
        fuse_ops.append(FuseOp(
            op="fuse", id="fuse_fc2_gelu",
            nodes=[fc_nodes[1], gelu_nodes[0]],
            strategy="inductor_fuse",
            comment="Fuse fc2+gelu to eliminate DRAM round-trip",
        ))
    if not fuse_ops and len(node_names) >= 2:
        fuse_ops.append(FuseOp(
            op="fuse", id="fuse_all",
            nodes=node_names, strategy="inductor_fuse",
        ))

    plan = RewritePlan(
        plan_version="1.0",
        source_profile_id=f"1.0/{before_profile.operators[0].operator_id}",
        description=(
            f"Fuse activation ops into adjacent linear layers using inductor_fuse. "
            f"Both relu and gelu are latency_bound on {GPU_NAME} "
            f"(ridge point {RIDGE_POINT:.0f} FLOP/byte)."
            if RIDGE_POINT else
            "Fuse activation ops into adjacent linear layers using inductor_fuse."
        ),
        ops=fuse_ops,
    )

    print(f"   Plan: {len(fuse_ops)} FuseOp(s)")
    for op in fuse_ops:
        print(f"     [{op.id}]  nodes={op.nodes}")

    cfg = ExecutorConfig(
        skip_verification=False,
        device=DEVICE,
        verification_atol=1e-4,
        verification_rtol=1e-4,
    )
    gm_cuda = torch.fx.symbolic_trace(model)
    executor = HybridExecutor(gm_cuda, plan, cfg)
    result_gm, ver_results = executor.run()

    print(f"\n   Verification results ({len(ver_results)} ops):")
    for vr in ver_results:
        status = "PASS" if vr.passed else "FAIL"
        err = f"  max_err={vr.max_abs_error:.2e}" if vr.max_abs_error is not None else ""
        print(f"     [{vr.op_id}]  {status}{err}")
    if not ver_results:
        print("     (empty plan — no ops to verify)")
    all_passed = all(vr.passed for vr in ver_results)
    print(f"\n   Overall verification: {'PASSED' if all_passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Agent Stage B — VerifierAgent: synthetic failure diagnosis
    # -----------------------------------------------------------------------
    print(f"\n## Agent Stage B — VerifierAgent (synthetic failure diagnosis)")
    if not AGENTS_ENABLED:
        print("   SKIPPED — set ANTHROPIC_API_KEY to enable agent stages")
    else:
        from operator_profiler.agents.verifier import VerifierAgent
        verifier_agent = VerifierAgent()

        # FFBlock's FuseOp passes verification cleanly (as shown above).
        # To demonstrate VerifierAgent, we construct a realistic synthetic failure
        # scenario: a hypothetical ChangeLayoutOp that caused a shape mismatch.
        # This is clearly labeled as simulated — no real failure occurred.
        print("   NOTE: FFBlock fuse passed verification. Demonstrating VerifierAgent")
        print("   with a SIMULATED failure scenario (ChangeLayoutOp shape mismatch).")

        synthetic_plan = RewritePlan(
            plan_version="1.0",
            description="Attempt to change fc1 weight layout from NCHW to NHWC",
            ops=[],  # not executed — just used for the message
        )
        # Realistic NodeDiff: shape went from (64, 512) to (64, 2048) — wrong
        synthetic_node_diff = NodeDiff(
            node_name="fc1",
            max_abs_error=14.37,
            original_shape=(64, 512),
            rewritten_shape=(64, 2048),
        )
        synthetic_ver_result = VerificationResult(
            op_id="change_layout_fc1",
            passed=False,
            max_abs_error=14.37,
            node_diffs=[synthetic_node_diff],
            error_message=(
                "Output shape mismatch after ChangeLayoutOp on fc1: "
                "expected (64, 512), got (64, 2048)"
            ),
        )

        repair_ctx = verifier_agent.diagnose(synthetic_plan, [synthetic_ver_result])
        print(f"\n   Failure category:  {repair_ctx.failure_category}")
        print(f"   Repair hint:       {repair_ctx.repair_hint[:300]}")
        if repair_ctx.avoid_ops:
            print(f"   Avoid ops:         {repair_ctx.avoid_ops}")
        print(f"\n   Prompt section (injected into ThetaPlanner on retry):")
        for line in repair_ctx.to_prompt_section().splitlines():
            print(f"     {line}")

    # -----------------------------------------------------------------------
    # 5. torch.compile (Inductor) — real kernel fusion
    # -----------------------------------------------------------------------
    print(f"\n## Stage 3 — torch.compile (Inductor backend, real kernel fusion)")
    print("   Compiling... ", end="", flush=True)
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    with torch.no_grad():
        _ = compiled_model(x)
    torch.cuda.synchronize()
    print("done")

    compiled_ms = cuda_time_ms(lambda: compiled_model(x))
    print(f"   Compiled mean: {compiled_ms:.4f} ms/call")
    speedup = eager_ms / compiled_ms
    print(f"   Speedup over eager: {speedup:.3f}×  ({(speedup-1)*100:.1f}% faster)")

    compiled_op_times = profiler_op_times(lambda: compiled_model(x))
    print(f"   Per-op CUDA times (torch.profiler, post-compile):")
    for op, us in sorted(compiled_op_times.items(), key=lambda kv: -kv[1]):
        print(f"     {op:<30}  {us:>8.1f} µs")

    # -----------------------------------------------------------------------
    # 6. Build after profile + diff
    # -----------------------------------------------------------------------
    after_profile = build_after_profile(compiled_op_times, before_profile)
    after_total_ms = sum(
        op.aggregated.total_duration_ns for op in after_profile.operators
    ) / 1e6
    print(f"\n   After profile: {len(after_profile.operators)} ops, {after_total_ms:.3f} ms total")

    diff = compute_diff(before_profile, after_profile, plan)

    print(f"\n## Profile Diff")
    print(f"   Profiler-based speedup:      {diff.total_speedup:.3f}×")
    print(f"   CUDA Event measured speedup: {speedup:.3f}×")
    print(f"   Wall time saved (profiler):  {diff.wall_time_saved_ns/1e6:.3f} ms")
    print(f"\n   {'Operator':<30} {'Before µs':>10} {'After µs':>9} {'Speedup':>8}  Match")
    print(f"   {'-'*30} {'-'*10} {'-'*9} {'-'*8}  {'-'*10}")
    for d in diff.operator_diffs:
        b = f"{d.duration_before_ns/1e3:.1f}" if d.duration_before_ns else "—"
        a = f"{d.duration_after_ns/1e3:.1f}"  if d.duration_after_ns  else "—"
        s = f"{d.speedup:.2f}×"               if d.speedup            else "—"
        print(f"   {d.operator_id_before:<30} {b:>10} {a:>9} {s:>8}  {d.match_type}")

    # -----------------------------------------------------------------------
    # Agent Stage C — Memory store + RuleAgent enrichment
    # -----------------------------------------------------------------------
    print(f"\n## Agent Stage C — OptimizationMemory + RuleAgent")
    if not AGENTS_ENABLED:
        print("   SKIPPED — set ANTHROPIC_API_KEY to enable agent stages")
    else:
        from operator_profiler.agents.rule import RuleAgent
        rule_agent = RuleAgent()

        # Use a temp file so the demo doesn't pollute the working directory
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            mem_path = tf.name
        memory = OptimizationMemory(mem_path)

        # Curate the real measured speedup into memory
        op_sequence = [op.operator_name for op in before_profile.operators]
        bottleneck = before_profile.operators[0].aggregated.bottleneck_classification
        real_entry = _make_memory_entry(
            op_sequence=op_sequence,
            bottleneck=bottleneck,
            plan=plan,
            speedup=speedup,
            model_name="FFBlock",
        )
        memory._store.entries.append(real_entry)
        memory.save_store()
        print(f"   Curated entry: speedup={speedup:.3f}×  bottleneck={bottleneck}")
        print(f"   Memory store: {len(memory)} entries at {mem_path}")

        # Convert to rule with template first, then enrich with LLM
        template_rule = entry_to_rule(real_entry)
        print(f"\n   Template rule_text:")
        print(f"     {template_rule.rule_text}")

        enriched_rule = entry_to_rule(real_entry, rule_agent=rule_agent)
        print(f"\n   Enriched rule_text (RuleAgent / Haiku):")
        for line in enriched_rule.rule_text.splitlines():
            print(f"     {line}")
        print(f"\n   Enriched conditions:")
        for cond in enriched_rule.conditions:
            print(f"     - {cond}")
        print(f"\n   Recommended action:")
        print(f"     {enriched_rule.recommended_action}")

    # -----------------------------------------------------------------------
    # Agent Stage D — MemoryCuratorAgent compaction
    # -----------------------------------------------------------------------
    print(f"\n## Agent Stage D — MemoryCuratorAgent (memory compaction)")
    if not AGENTS_ENABLED:
        print("   SKIPPED — set ANTHROPIC_API_KEY to enable agent stages")
    else:
        from operator_profiler.agents.curator import MemoryCuratorAgent

        # Add two near-duplicate synthetic entries so the curator has
        # something to reason about (same op sequence, lower speedup)
        dup_plan_a = RewritePlan(
            plan_version="1.0",
            description="Fuse fc1+relu (earlier run, lower speedup)",
            ops=[FuseOp(op="fuse", id="fuse_fc1_relu_v0",
                        nodes=fc_nodes[:1] + relu_nodes[:1] if fc_nodes and relu_nodes
                        else node_names[:2],
                        strategy="inductor_fuse")],
        )
        dup_plan_b = RewritePlan(
            plan_version="1.0",
            description="Fuse fc1+relu+fc2 (over-aggressive, stale)",
            ops=[FuseOp(op="fuse", id="fuse_all_v0",
                        nodes=node_names[:3] if len(node_names) >= 3 else node_names,
                        strategy="inductor_fuse")],
        )
        entry_dup_a = _make_memory_entry(
            op_sequence=op_sequence,
            bottleneck=bottleneck,
            plan=dup_plan_a,
            speedup=max(speedup * 0.92, 1.01),   # dominated: slightly lower
            model_name="FFBlock",
        )
        entry_dup_b = _make_memory_entry(
            op_sequence=op_sequence,
            bottleneck=bottleneck,
            plan=dup_plan_b,
            speedup=0.98,   # stale: no gain
            model_name="FFBlock",
        )
        memory._store.entries.append(entry_dup_a)
        memory._store.entries.append(entry_dup_b)
        memory.save_store()
        print(f"   Memory store before compaction: {len(memory)} entries")
        for e in memory.entries:
            print(f"     {e.entry_id[:12]}...  speedup={e.speedup:.3f}×  "
                  f"bottleneck={e.bottleneck}")

        curator_agent = MemoryCuratorAgent()
        result = memory.compact(curator_agent)

        print(f"\n   Curation result:")
        print(f"     Kept:    {len(result.entries_to_keep)}")
        print(f"     Removed: {result.removed_count}")
        print(f"     Reasoning: {result.reasoning}")
        print(f"   Memory store after compaction: {len(memory)} entries")

        # Cleanup temp file
        try:
            Path(mem_path).unlink(missing_ok=True)
            Path(mem_path).with_suffix(".tmp").unlink(missing_ok=True)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 7. Summary Report
    # -----------------------------------------------------------------------
    from operator_profiler.planner.loop import LoopResult
    loop_result = LoopResult(
        best_plan=plan,
        best_speedup=speedup,
        history=[{
            "iteration": 0,
            "bottleneck": "memory_bound",
            "worst_op_id": before_profile.operators[0].operator_id,
            "memory_hits": 0,
            "plans_tried": 1,
            "best_speedup_so_far": speedup,
            "beam_scores": [speedup],
        }],
    )

    report = SummaryReport(
        diff=diff,
        rules=[],
        lessons_learned=[],
        loop_history=loop_result.history,
        best_speedup=speedup,
        best_plan_description=plan.description,
    )

    # -----------------------------------------------------------------------
    # 8. Provenance table
    # -----------------------------------------------------------------------
    print(f"\n## Provenance Table (Before Profile)")
    rows = build_provenance_rows(before_profile, plan)
    for line in render_provenance_text(rows).splitlines():
        print("   " + line)

    # -----------------------------------------------------------------------
    # 9. Markdown report
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("  MARKDOWN REPORT")
    print(sep)
    print(render_markdown(report))

    # -----------------------------------------------------------------------
    # 10. Summary line
    # -----------------------------------------------------------------------
    print(sep)
    print(f"  RESULT: {eager_ms:.4f} ms (eager)  →  {compiled_ms:.4f} ms (compiled)")
    print(f"          {speedup:.3f}× speedup  |  {(speedup-1)*100:.1f}% faster")
    print(f"          Verification: {'PASSED' if all_passed else 'FAILED'}")
    if AGENTS_ENABLED:
        print(f"          Agents: DiagnosisAgent ✓  VerifierAgent ✓  "
              f"RuleAgent ✓  MemoryCuratorAgent ✓")
    else:
        print(f"          Agents: DISABLED (set ANTHROPIC_API_KEY to enable)")
    print(sep)


if __name__ == "__main__":
    main()
