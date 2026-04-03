"""
Attribution engine — applies the confidence fallback chain and handles all
eight edge cases when converting a MappingManifest into OperatorRecords.

Edge cases addressed here:
  #1  Clock domain mismatch    — join on (kernel_name, launch_index_within_range),
                                 never on absolute timestamps across tools.
  #2  CUDA graph replay        — graph manifest lookup for replay kernels.
  #3  Multi-stream             — already handled by interval tree; pass-through.
  #4  JIT warm-up inflation    — initialization kernels (pre-NVTX phase) are
                                 excluded from the operator-kernel mapping entirely.
  #5  Async kernel launch      — GPU-domain timestamps from CUPTI already used;
                                 no host-timestamp enclosure logic here.
  #6  Dynamic shapes           — input shape validation at replay start.
  #7  Fused kernel multi-NVTX  — is_fused=True; all enclosing ranges stored;
                                 provenance is authoritative.
  #8  ncu replay timing        — ncu timestamps NEVER cross-correlated with nsys.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from operator_profiler.schema.manifest import KernelManifestEntry, MappingManifest
from operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    KernelRecord,
    KernelMetrics,
    OperatorRecord,
    NvtxRangeInfo,
)

log = logging.getLogger(__name__)


@dataclass
class CudaGraphManifest:
    """
    Records the operator set observed during CUDAGraph capture (edge case #2).

    The replay phase emits new GPU timestamps that have no NVTX correspondence;
    we attribute all replay kernels to the manifest's operator list.
    """
    graph_id: str
    source_operators: list[str] = field(default_factory=list)
    kernel_names: list[str] = field(default_factory=list)


class AttributionEngine:
    """
    Converts a MappingManifest into a list of OperatorRecords + unattributed kernels.

    Parameters
    ----------
    manifest:
        The MappingManifest produced by ManifestBuilder.
    cuda_graph_manifests:
        Optional list of CUDAGraph manifests for edge case #2 (graph replay).
    warmup_kernel_ids:
        Set of kernel_id strings that are warm-up outliers (from manifest.warnings).
    """

    def __init__(
        self,
        manifest: MappingManifest,
        cuda_graph_manifests: list[CudaGraphManifest] | None = None,
        warmup_kernel_ids: set[str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.cuda_graph_manifests = {
            m.graph_id: m for m in (cuda_graph_manifests or [])
        }
        self.warmup_kernel_ids: set[str] = warmup_kernel_ids or set()
        self._extract_warmup_ids_from_warnings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[OperatorRecord], list[KernelRecord]]:
        """
        Returns (operator_records, unattributed_kernels).
        """
        # Group manifest entries by their primary operator name
        op_groups: dict[str, list[KernelManifestEntry]] = defaultdict(list)
        unattributed: list[KernelRecord] = []

        init_skipped = 0
        for entry in self.manifest.kernels:
            if entry.kernel_id in self.warmup_kernel_ids:
                init_skipped += 1
                continue

            method = entry.attribution.method
            ops = entry.attribution.source_operators

            if method == AttributionMethod.UNATTRIBUTED or not ops:
                # Edge case #2: try CUDA graph lookup before giving up
                graph_op = self._lookup_cuda_graph(entry.kernel_name)
                if graph_op:
                    op_groups[graph_op].append(entry)
                    log.debug(
                        "Kernel %s attributed via CUDAGraph manifest → %s",
                        entry.kernel_id,
                        graph_op,
                    )
                else:
                    unattributed.append(self._entry_to_kernel_record(entry))
            else:
                # Primary operator is the first source op
                op_groups[ops[0]].append(entry)

        operator_records = self._build_operator_records(op_groups)

        if init_skipped:
            log.info(
                "Skipped %d initialization kernel(s) from operator-kernel mapping",
                init_skipped,
            )
        log.info(
            "Attribution: %d operators, %d unattributed kernels",
            len(operator_records),
            len(unattributed),
        )
        return operator_records, unattributed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_warmup_ids_from_warnings(self) -> None:
        """Parse manifest warnings to find initialization kernel IDs."""
        for warning in self.manifest.warnings:
            if "initialization kernel" in warning or "warm-up outlier" in warning:
                # Format: "k_NNNNN (kernel_name): flagged as initialization kernel"
                kid = warning.split(" ")[0]
                self.warmup_kernel_ids.add(kid)

    def _lookup_cuda_graph(self, kernel_name: str) -> str | None:
        """
        Edge case #2: look up kernel_name in CUDAGraph manifests.
        Returns the first matching source operator, or None.
        """
        for gm in self.cuda_graph_manifests.values():
            if kernel_name in gm.kernel_names:
                return gm.source_operators[0] if gm.source_operators else None
        return None

    def _build_operator_records(
        self, op_groups: dict[str, list[KernelManifestEntry]]
    ) -> list[OperatorRecord]:
        records: list[OperatorRecord] = []
        call_counters: dict[str, int] = defaultdict(int)

        # Sort by first kernel start_ns for stable ordering
        sorted_ops = sorted(
            op_groups.items(),
            key=lambda kv: kv[1][0].start_ns if kv[1] else 0,
        )

        for op_name, entries in sorted_ops:
            call_idx = call_counters[op_name]
            call_counters[op_name] += 1
            operator_id = f"{op_name}_{call_idx}"

            kernel_records = [self._entry_to_kernel_record(e) for e in entries]

            # Determine if any kernel is fused
            is_fused = any(e.attribution.is_fused for e in entries)
            fused_with: list[str] = []
            for e in entries:
                if e.attribution.is_fused:
                    for op in e.attribution.source_operators[1:]:
                        if op not in fused_with:
                            fused_with.append(op)

            # NVTX range info from first attributed kernel
            nvtx_range: NvtxRangeInfo | None = None
            for e in entries:
                if e.attribution.nvtx_range:
                    nvtx_range = e.attribution.nvtx_range
                    break

            records.append(
                OperatorRecord(
                    operator_id=operator_id,
                    operator_name=op_name,
                    call_index=call_idx,
                    is_fused=is_fused,
                    fused_with=fused_with,
                    nvtx_range=nvtx_range,
                    kernels=kernel_records,
                    # aggregated metrics are filled in by metric_aggregator.py
                    aggregated=None,
                )
            )

        return records

    def _entry_to_kernel_record(self, entry: KernelManifestEntry) -> KernelRecord:
        a = entry.attribution
        # Edge case #8: no ncu timestamps here — metrics are empty until
        # kernel_profiler.py fills them in.
        return KernelRecord(
            kernel_id=entry.kernel_id,
            kernel_name=entry.kernel_name,
            stream_id=entry.stream_id,
            device_id=entry.device_id,
            start_ns=entry.start_ns,
            end_ns=entry.end_ns,
            duration_ns=entry.duration_ns,
            grid_dim=entry.grid_dim,
            block_dim=entry.block_dim,
            metrics=KernelMetrics(),   # populated later by ncu_parser
            attribution_method=a.method,
            confidence=a.confidence,
            nvtx_range=a.nvtx_range,
        )
