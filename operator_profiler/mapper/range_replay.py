"""
Kernel Profile Orchestrator.

For each unique kernel name found in the mapping manifest, runs:
    ncu --kernel-name <name> --replay-mode kernel --metrics <list>

Then merges the resulting KernelMetrics back into the operator records by
matching on (kernel_name, invocation_index).

Design decisions
----------------
Kernel-name based profiling (instead of NVTX range filtering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Earlier designs used ``ncu --nvtx-include <range> --replay-mode range`` to
select kernels.  This approach requires the workload to expose NVTX push/pop
ranges through a *shared* libnvToolsExt.so at runtime.  Recent PyTorch builds
(2.x nightlies) statically link NVTX v3 inside libtorch_cuda.so, so ncu's
injection mechanism never sees any NVTX events — range mode silently captures
nothing.

Kernel-name based profiling avoids this dependency entirely:

  - Eager mode:   cuBLAS/cuDNN kernel names are stable identifiers.
  - Compiled mode (Inductor/Triton): each fused operation produces a uniquely
    named Triton kernel (e.g. ``triton_per_fused_addmm_relu_0``).  Profiling
    that kernel by name captures the entire fused unit — no grouping needed,
    and no cache-coherence concern because fusion already happened at compile
    time.

Invocation matching (edge case #1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ncu numbers every invocation of a kernel from 0 upward (in launch order).
The manifest lists kernels in the same launch order.  We match the i-th ncu
row for kernel name K to the i-th manifest entry with kernel_name == K,
never using absolute timestamps across tools.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from operator_profiler.schema.manifest import MappingManifest
from operator_profiler.schema.metrics import DEFAULT_NCU_METRICS
from operator_profiler.schema.profile import KernelMetrics, OperatorRecord
from operator_profiler.mapper.ncu_runner import NcuKernelProfileConfig, run_kernel_profile, import_ncu_report
from operator_profiler.mapper.ncu_parser import parse_ncu_csv_by_id
from operator_profiler.aggregator.metric_aggregator import aggregate_fused_metrics
from operator_profiler.utils.validation import validate_input_shapes

log = logging.getLogger(__name__)


@dataclass
class KernelReplayTarget:
    """One ncu profile run — one unique kernel name across the workload."""
    kernel_name: str
    # Ordered list of manifest kernel_ids that have this kernel_name,
    # in launch order.  Used to match ncu invocation index → kernel_id.
    kernel_ids: list[str] = field(default_factory=list)


@dataclass
class RangeReplayConfig:
    """Top-level configuration for the KernelProfileOrchestrator."""
    replay_script: str | Path           # Python script to replay
    replay_script_args: list[str] = field(default_factory=list)
    output_dir: str | Path = ""
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_NCU_METRICS))
    ncu_executable: str = "ncu"
    # Set True to prefix ncu with "sudo -E" (needed when perf counters are
    # restricted to root, e.g. ERR_NVGPUCTRPERM).
    ncu_sudo: bool = False
    # Extra environment variables forwarded to the ncu subprocess.
    # Useful for setting PYTHONPATH when running under sudo.
    ncu_extra_env: dict[str, str] = field(default_factory=dict)
    # Expected input shapes — validated before replay (edge case #6)
    expected_input_shapes: dict[str, list[int]] = field(default_factory=dict)


class RangeReplayOrchestrator:
    """
    Runs ncu kernel profiles for all unique kernel names in the manifest and
    populates KernelMetrics on OperatorRecord.kernels in-place.

    Despite the name (kept for API compatibility), this orchestrator no longer
    uses NVTX range replay.  It profiles by kernel name instead.

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
        Run all kernel profiles and merge metrics into operator_records in-place.
        """
        # Edge case #6: validate input shapes before replay
        if self.config.expected_input_shapes:
            validate_input_shapes(
                self.config.expected_input_shapes,
                self.manifest.capture_metadata.input_shapes,
            )

        targets = self._build_replay_targets()
        log.info("Running ncu kernel profile for %d unique kernel name(s)", len(targets))

        output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="op_profiler_ncu_"))

        for target in targets:
            log.info(
                "Profiling kernel '%s' (%d invocation(s))",
                target.kernel_name, len(target.kernel_ids),
            )
            metrics_map = self._profile_one(target, output_dir)
            self._merge_metrics(target, metrics_map)

        self._apply_metrics_to_records()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_replay_targets(self) -> list[KernelReplayTarget]:
        """
        Collect unique kernel names from the manifest in launch order.

        One KernelReplayTarget per unique kernel_name.  The kernel_ids list
        within each target preserves the launch order of that kernel across
        the whole workload, which is the index ncu uses when numbering
        invocations.
        """
        seen: dict[str, KernelReplayTarget] = {}

        for entry in self.manifest.kernels:
            name = entry.kernel_name
            if name not in seen:
                seen[name] = KernelReplayTarget(kernel_name=name)
            seen[name].kernel_ids.append(entry.kernel_id)

        return list(seen.values())

    def _profile_one(
        self, target: KernelReplayTarget, output_dir: Path
    ) -> dict[tuple[str, str], KernelMetrics]:
        """
        Run ncu for one kernel name, return (kernel_name, invocation_id) → KernelMetrics.
        """
        safe_name = target.kernel_name.replace("::", "_").replace(" ", "_")[:120]
        ncu_rep_path = output_dir / f"{safe_name}.ncu-rep"

        ncu_config = NcuKernelProfileConfig(
            script=self.config.replay_script,
            script_args=self.config.replay_script_args,
            kernel_name_filter=target.kernel_name,
            metrics=self.config.metrics,
            output_path=ncu_rep_path,
            ncu_executable=self.config.ncu_executable,
            use_sudo=self.config.ncu_sudo,
            extra_env=self.config.ncu_extra_env,
        )
        run_kernel_profile(ncu_config)

        csv_text = import_ncu_report(ncu_rep_path, self.config.ncu_executable)
        return parse_ncu_csv_by_id(csv_text)

    def _merge_metrics(
        self,
        target: KernelReplayTarget,
        metrics_map: dict[tuple[str, str], KernelMetrics],
    ) -> None:
        """
        Merge ncu metrics into self._kernel_metrics.

        Join strategy (edge case #1): match by invocation index within the
        workload for this kernel name, NOT by absolute timestamp.

        ncu numbers invocations of a given kernel name from 0 upward in
        launch order; the manifest kernel_ids in target.kernel_ids are also
        in launch order.  We sort the ncu rows for this kernel name by their
        numeric ID and zip them against kernel_ids positionally.
        """
        # Filter to rows for this kernel name and sort by numeric invocation id.
        #
        # nsys stores the *short* kernel name (e.g. "gemv2T_kernel_val") while
        # the ncu CSV reports the *full mangled* name (e.g. "void gemv2T_kernel_val
        # <int,...>(T13,T6,T6)").  We use a substring match so that the short name
        # from the manifest matches against the full name from ncu.
        name_rows = {
            kid: m for (kname, kid), m in metrics_map.items()
            if target.kernel_name in kname
        }
        ordered_metrics = [
            name_rows[k]
            for k in sorted(name_rows, key=lambda x: int(x) if x.isdigit() else 0)
        ]

        for i, manifest_kid in enumerate(target.kernel_ids):
            if i < len(ordered_metrics):
                self._kernel_metrics[manifest_kid] = ordered_metrics[i]
            else:
                log.warning(
                    "Kernel '%s' id %s has no matching ncu row "
                    "(invocation %d, ncu returned %d rows)",
                    target.kernel_name, manifest_kid, i, len(ordered_metrics),
                )

    def _apply_metrics_to_records(self) -> None:
        """Write collected metrics back into OperatorRecord.kernels in-place."""
        for op_record in self.operator_records:
            for kernel in op_record.kernels:
                if kernel.kernel_id in self._kernel_metrics:
                    kernel.metrics = self._kernel_metrics[kernel.kernel_id]
