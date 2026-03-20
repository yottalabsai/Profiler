"""
Range Replay Orchestrator.

For each unique NVTX range in the mapping manifest, runs:
    ncu --nvtx --nvtx-include <range_text> --replay-mode range --metrics <list>

Then merges the resulting KernelMetrics back into the operator records.

Key design decisions (from architecture plan §4):
  - --replay-mode range: preserves cache state across fused kernel sequences.
  - One ncu subprocess per unique NVTX range (not per kernel).
  - ncu timestamps are NEVER used — only metric values (edge case #8).
  - Launch index within a range is used to join ncu rows to manifest entries,
    never absolute timestamps (edge case #1).
  - Input shapes are validated before replay to catch dynamic-shape mismatches
    (edge case #6).
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from operator_profiler.schema.manifest import MappingManifest
from operator_profiler.schema.metrics import DEFAULT_NCU_METRICS
from operator_profiler.schema.profile import KernelMetrics, OperatorRecord
from operator_profiler.mapper.ncu_runner import NcuRangeReplayConfig, run_range_replay, import_ncu_report
from operator_profiler.mapper.ncu_parser import parse_ncu_csv_by_id
from operator_profiler.aggregator.metric_aggregator import aggregate_fused_metrics
from operator_profiler.utils.validation import validate_input_shapes

log = logging.getLogger(__name__)


@dataclass
class ReplayTarget:
    """One ncu replay run — corresponds to one unique NVTX range."""
    range_text: str              # NVTX range glob, e.g. "aten::linear"
    range_depth: int             # depth to disambiguate parent vs child ranges
    kernel_ids: list[str]        # manifest kernel_ids within this range
    is_fused: bool
    fused_source_ops: list[str]


@dataclass
class RangeReplayConfig:
    """Top-level configuration for the RangeReplayOrchestrator."""
    replay_script: str | Path           # Python script to replay
    replay_script_args: list[str] = field(default_factory=list)
    output_dir: str | Path = ""
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    ncu_executable: str = "ncu"
    # Expected input shapes — validated before replay (edge case #6)
    expected_input_shapes: dict[str, list[int]] = field(default_factory=dict)


class RangeReplayOrchestrator:
    """
    Runs ncu range replays for all unique NVTX ranges in the manifest and
    populates KernelMetrics on OperatorRecord.kernels in-place.

    Usage:
        orch = RangeReplayOrchestrator(manifest, operator_records, config)
        orch.run()
        # operator_records[i].kernels[j].metrics is now populated
    """

    def __init__(
        self,
        manifest: MappingManifest,
        operator_records: list[OperatorRecord],
        config: RangeReplayConfig,
    ) -> None:
        self.manifest = manifest
        self.operator_records = operator_records
        self.config = config
        self._kernel_metrics: dict[str, KernelMetrics] = {}  # kernel_id → metrics

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run all range replays and merge metrics into operator_records in-place.
        """
        # Edge case #6: validate input shapes before replay
        if self.config.expected_input_shapes:
            validate_input_shapes(
                self.config.expected_input_shapes,
                self.manifest.capture_metadata.input_shapes,
            )

        targets = self._build_replay_targets()
        log.info("Running ncu range replay for %d unique NVTX ranges", len(targets))

        output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        if output_dir is None:
            # Use a temp dir — caller is responsible for cleanup if desired
            output_dir = Path(tempfile.mkdtemp(prefix="op_profiler_ncu_"))

        for target in targets:
            log.info("Replaying range '%s' (%d kernels)", target.range_text, len(target.kernel_ids))
            metrics_map = self._replay_one(target, output_dir)
            self._merge_metrics(target, metrics_map)

        # Write merged metrics into OperatorRecord.kernels
        self._apply_metrics_to_records()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_replay_targets(self) -> list[ReplayTarget]:
        """
        Collect unique NVTX ranges from the manifest.

        One ReplayTarget per unique (range_text, range_depth) pair.
        """
        seen: dict[tuple[str, int], ReplayTarget] = {}

        for entry in self.manifest.kernels:
            a = entry.attribution
            nvtx = a.nvtx_range
            if nvtx is None:
                continue
            key = (nvtx.text, nvtx.depth)
            if key not in seen:
                seen[key] = ReplayTarget(
                    range_text=nvtx.text,
                    range_depth=nvtx.depth,
                    kernel_ids=[],
                    is_fused=a.is_fused,
                    fused_source_ops=a.source_operators[1:] if a.is_fused else [],
                )
            seen[key].kernel_ids.append(entry.kernel_id)

        return list(seen.values())

    def _replay_one(
        self, target: ReplayTarget, output_dir: Path
    ) -> dict[tuple[str, str], KernelMetrics]:
        """Run ncu for one target, return (kernel_name, launch_id) → KernelMetrics."""
        safe_name = target.range_text.replace("::", "_").replace(" ", "_")
        ncu_rep_path = output_dir / f"{safe_name}.ncu-rep"

        ncu_config = NcuRangeReplayConfig(
            script=self.config.replay_script,
            script_args=self.config.replay_script_args,
            nvtx_include=target.range_text,
            metrics=self.config.metrics,
            replay_mode="range",
            output_path=ncu_rep_path,
            ncu_executable=self.config.ncu_executable,
        )
        run_range_replay(ncu_config)

        csv_text = import_ncu_report(ncu_rep_path, self.config.ncu_executable)
        return parse_ncu_csv_by_id(csv_text)

    def _merge_metrics(
        self,
        target: ReplayTarget,
        metrics_map: dict[tuple[str, str], KernelMetrics],
    ) -> None:
        """
        Merge ncu metrics into self._kernel_metrics.

        Join strategy (edge case #1): match by launch index within the range,
        NOT by absolute timestamp.  ncu rows are sorted by launch order;
        manifest kernel_ids within the range are also in launch order.
        """
        ordered_metrics = list(metrics_map.values())

        if target.is_fused and len(ordered_metrics) > 1:
            # Fused: aggregate all ncu rows into one combined metric
            agg = aggregate_fused_metrics(ordered_metrics)
            for kid in target.kernel_ids:
                self._kernel_metrics[kid] = agg
        else:
            # Non-fused: assign by launch index
            for i, kid in enumerate(target.kernel_ids):
                if i < len(ordered_metrics):
                    self._kernel_metrics[kid] = ordered_metrics[i]
                else:
                    log.warning(
                        "Kernel %s has no matching ncu row (index %d, range has %d rows)",
                        kid, i, len(ordered_metrics),
                    )

    def _apply_metrics_to_records(self) -> None:
        """Write collected metrics back into OperatorRecord.kernels in-place."""
        for op_record in self.operator_records:
            for kernel in op_record.kernels:
                if kernel.kernel_id in self._kernel_metrics:
                    kernel.metrics = self._kernel_metrics[kernel.kernel_id]
