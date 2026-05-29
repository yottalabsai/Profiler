"""
Attribution engine — converts a MappingManifest into OperatorRecords.

Adapted from nvidia/operator_profiler/mapper/attribution_engine.py.

Changes from the NVIDIA version:
  - CudaGraphManifest removed (CUDA graph replay is not applicable to Trainium)
  - Warmup detection uses manifest warnings (same pattern as NVIDIA)
  - NvtxRangeInfo references replaced with NrtEventInfo
  - fused_ops logic removed (no Inductor fusion map on Trainium)
  - is_fused is always False (NRT does not expose intra-op fusion boundaries
    at this granularity; may be revisited when ntrace.pb schema is confirmed)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from trainium.operator_profiler.schema.manifest import KernelManifestEntry, MappingManifest
from trainium.operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    KernelMetrics,
    KernelRecord,
    NrtEventInfo,
    OperatorRecord,
)

log = logging.getLogger(__name__)


class AttributionEngine:
    """
    Converts a MappingManifest into a list of OperatorRecords + unattributed kernels.

    Parameters
    ----------
    manifest:
        The MappingManifest produced by ManifestBuilder.
    warmup_kernel_ids:
        Set of kernel_id strings that are warm-up events (from manifest.warnings).
    """

    def __init__(
        self,
        manifest: MappingManifest,
        warmup_kernel_ids: set[str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.warmup_kernel_ids: set[str] = warmup_kernel_ids or set()
        self._extract_warmup_ids_from_warnings()

    def run(self) -> tuple[list[OperatorRecord], list[KernelRecord]]:
        """Returns (operator_records, unattributed_kernels)."""
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
                unattributed.append(self._entry_to_kernel_record(entry))
            else:
                op_groups[ops[0]].append(entry)

        operator_records = self._build_operator_records(op_groups)

        if init_skipped:
            log.info("Skipped %d warm-up event(s)", init_skipped)
        log.info(
            "Attribution: %d operators, %d unattributed",
            len(operator_records),
            len(unattributed),
        )
        return operator_records, unattributed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_warmup_ids_from_warnings(self) -> None:
        for warning in self.manifest.warnings:
            if "warm-up event" in warning or "initialization kernel" in warning:
                kid = warning.split(" ")[0]
                self.warmup_kernel_ids.add(kid)

    def _build_operator_records(
        self, op_groups: dict[str, list[KernelManifestEntry]]
    ) -> list[OperatorRecord]:
        records: list[OperatorRecord] = []
        call_counters: dict[str, int] = defaultdict(int)

        sorted_ops = sorted(
            op_groups.items(),
            key=lambda kv: kv[1][0].start_ns if kv[1] else 0,
        )

        for op_name, entries in sorted_ops:
            call_idx = call_counters[op_name]
            call_counters[op_name] += 1
            operator_id = f"{op_name}_{call_idx}"

            kernel_records = [self._entry_to_kernel_record(e) for e in entries]
            nrt_event: NrtEventInfo | None = None
            if entries and entries[0].attribution.nrt_event:
                nrt_event = entries[0].attribution.nrt_event

            records.append(
                OperatorRecord(
                    operator_id=operator_id,
                    operator_name=op_name,
                    call_index=call_idx,
                    is_fused=False,
                    fused_with=[],
                    nrt_event=nrt_event,
                    kernels=kernel_records,
                    aggregated=None,
                )
            )

        return records

    def _entry_to_kernel_record(self, entry: KernelManifestEntry) -> KernelRecord:
        a = entry.attribution
        return KernelRecord(
            kernel_id=entry.kernel_id,
            kernel_name=entry.kernel_name,
            stream_id=entry.stream_id,
            device_id=entry.device_id,
            start_ns=entry.start_ns,
            end_ns=entry.end_ns,
            duration_ns=entry.duration_ns,
            metrics=KernelMetrics(),   # populated by ntrace metrics if available
            attribution_method=a.method,
            confidence=a.confidence,
            nrt_event=a.nrt_event,
        )
