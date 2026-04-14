"""
DiagnosisAgent — LLM-based GPU bottleneck diagnosis.

Replaces the 3-line heuristic in metric_aggregator._classify_bottleneck()
with reasoning over the full KernelMetrics set for an operator, relative to
model-wide metric distributions and the GPU ridge point.

Integrated via build_profile() in aggregator/profile_builder.py:
  - After all AggregatedMetrics are built, a second pass calls
    agent.diagnose(op, model_stats) for each operator and overwrites
    bottleneck_classification with the LLM result.
  - Falls back to the heuristic value on any API error.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from operator_profiler.schema.profile import OperatorRecord

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert GPU performance analyst. Given hardware metrics for one PyTorch
operator and the model-wide distribution of those metrics, classify the operator's
performance bottleneck and explain your reasoning.

Classification labels:
- "compute_bound"  : arithmetic intensity is near or above the GPU ridge point;
                     the GPU's math units are the bottleneck.
- "memory_bound"   : arithmetic intensity is far below the ridge point; DRAM
                     bandwidth is the bottleneck. Key signals: large DRAM transfer,
                     low L1 hit rate, AI well below ridge.
- "latency_bound"  : many small kernels per operator with occupancy far below the
                     model median; kernel dispatch overhead dominates execution time.
- "unknown"        : insufficient metrics to classify confidently.

Always reason RELATIVE to the model distribution (ridge point, model median
occupancy, model median AI) provided in the user message — never use fixed
thresholds like "occupancy < 50%". What matters is how this operator compares
to its peers.

Respond only by calling the diagnose_bottleneck tool."""

_DIAGNOSE_TOOL = {
    "name": "diagnose_bottleneck",
    "description": "Classify the GPU performance bottleneck for one operator.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["compute_bound", "memory_bound", "latency_bound", "unknown"],
                "description": "The bottleneck class.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "One to three sentences citing specific metric values relative to "
                    "the model distribution and ridge point. Name the dominant signal."
                ),
            },
            "key_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The 2–4 specific metric comparisons that drove the classification, "
                    "e.g. 'AI=2.1 vs ridge=156 (1.3% of ridge)' or "
                    "'occupancy=18% vs model median=62%'."
                ),
            },
        },
        "required": ["classification", "reasoning", "key_signals"],
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    classification: Literal["compute_bound", "memory_bound", "latency_bound", "unknown"]
    reasoning: str
    key_signals: list[str]


@dataclass
class ModelStats:
    """Model-wide metric distributions for relative comparison."""
    median_ai: float | None
    p25_ai: float | None
    p75_ai: float | None
    median_occupancy: float | None
    median_tensor_core_pct: float | None
    ridge_point: float | None   # peak_compute_gflops / peak_bandwidth_gbs
    device_name: str | None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DiagnosisAgent:
    """
    LLM-backed GPU bottleneck classifier.

    Parameters
    ----------
    model:
        Anthropic model string.
    api_key:
        Optional API key; falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = self._build_client()

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for DiagnosisAgent. "
                "Install with: pip install anthropic"
            ) from exc
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        op: "OperatorRecord",
        stats: ModelStats,
    ) -> DiagnosisResult:
        """
        Classify the bottleneck for a single OperatorRecord.

        Parameters
        ----------
        op:
            The operator to classify.  Must have ``op.aggregated`` filled.
        stats:
            Model-wide metric distributions for relative comparison.

        Returns
        -------
        DiagnosisResult
            Falls back to classification="unknown" on any API or parse error.
        """
        user_message = self._build_message(op, stats)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
                tools=[_DIAGNOSE_TOOL],
                tool_choice={"type": "tool", "name": "diagnose_bottleneck"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            log.warning(
                "DiagnosisAgent API call failed for %s: %s — returning unknown",
                op.operator_id, exc,
            )
            return DiagnosisResult(
                classification="unknown",
                reasoning=f"API call failed: {exc}",
                key_signals=[],
            )

        return self._parse_response(response, op.operator_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_message(self, op: "OperatorRecord", stats: ModelStats) -> str:
        kernel_metrics = [k.metrics for k in op.kernels]
        agg = op.aggregated

        ai_vals = [m.arithmetic_intensity for m in kernel_metrics if m.arithmetic_intensity is not None]
        occ_vals = [m.achieved_occupancy for m in kernel_metrics if m.achieved_occupancy is not None]
        tc_vals = [m.tensor_core_active_pct for m in kernel_metrics if m.tensor_core_active_pct is not None]
        dram_read = sum(m.dram_bytes_read for m in kernel_metrics if m.dram_bytes_read is not None)
        dram_write = sum(m.dram_bytes_written for m in kernel_metrics if m.dram_bytes_written is not None)
        l1_hits = [m.l1_hit_rate for m in kernel_metrics if m.l1_hit_rate is not None]

        duration_ms = (agg.total_duration_ns / 1e6) if agg else None

        lines = [
            f"## Operator: {op.operator_id}  ({op.operator_name})",
            f"Kernel count: {len(op.kernels)}",
            f"Total duration: {duration_ms:.3f} ms" if duration_ms is not None else "Total duration: unknown",
            "",
            "### Per-kernel metrics:",
            f"  Arithmetic intensity: {[f'{v:.2f}' for v in ai_vals] or 'n/a'} FLOP/byte",
            f"  Achieved occupancy:   {[f'{v:.1f}%' for v in occ_vals] or 'n/a'}",
            f"  Tensor core active %: {[f'{v:.1f}%' for v in tc_vals] or 'n/a'}",
            f"  DRAM read:  {dram_read / 1e6:.2f} MB",
            f"  DRAM write: {dram_write / 1e6:.2f} MB",
            f"  L1 hit rate: {[f'{v:.1f}%' for v in l1_hits] or 'n/a'}",
            "",
            "### Model-wide context (for relative comparison):",
        ]

        if stats.ridge_point is not None:
            lines.append(
                f"  GPU: {stats.device_name or 'unknown'}  "
                f"Ridge point: {stats.ridge_point:.1f} FLOP/byte"
            )
            if ai_vals:
                mean_ai = statistics.mean(ai_vals)
                pct_of_ridge = mean_ai / stats.ridge_point * 100
                lines.append(
                    f"  This op mean AI: {mean_ai:.2f} FLOP/byte "
                    f"({pct_of_ridge:.1f}% of ridge point)"
                )
        else:
            lines.append("  GPU ridge point: unknown")

        if stats.median_ai is not None:
            lines.append(
                f"  Model AI: p25={stats.p25_ai:.2f}  median={stats.median_ai:.2f}"
                f"  p75={stats.p75_ai:.2f} FLOP/byte"
            )
        if stats.median_occupancy is not None:
            lines.append(
                f"  Model median occupancy: {stats.median_occupancy:.1f}%"
                + (
                    f"  (this op: {statistics.mean(occ_vals):.1f}%)"
                    if occ_vals else ""
                )
            )
        if stats.median_tensor_core_pct is not None:
            lines.append(
                f"  Model median tensor core %: {stats.median_tensor_core_pct:.1f}%"
            )

        lines.append("\nClassify this operator's bottleneck.")
        return "\n".join(lines)

    def _parse_response(self, response, operator_id: str) -> DiagnosisResult:
        for block in response.content:
            if block.type == "tool_use" and block.name == "diagnose_bottleneck":
                try:
                    inp = block.input
                    return DiagnosisResult(
                        classification=inp["classification"],
                        reasoning=inp.get("reasoning", ""),
                        key_signals=inp.get("key_signals", []),
                    )
                except Exception as exc:
                    log.warning(
                        "DiagnosisAgent parse error for %s: %s", operator_id, exc
                    )
                    return DiagnosisResult(
                        classification="unknown",
                        reasoning=f"Parse error: {exc}",
                        key_signals=[],
                    )
        log.warning("DiagnosisAgent: no tool_use block for %s", operator_id)
        return DiagnosisResult(
            classification="unknown",
            reasoning="No tool_use block returned.",
            key_signals=[],
        )
