"""
ManifestBuilder — builds a MappingManifest from trace.json + ntrace.pb outputs.

Trainium vs. NVIDIA manifest building
--------------------------------------
NVIDIA's ManifestBuilder exports a .nsys-rep to SQLite, queries kernel rows,
builds a per-stream NVTX interval tree, and runs a three-tier fallback chain
(torch.profiler External ID > NVTX enclosure > Inductor fusion map).

Trainium's ManifestBuilder is simpler because attribution is a direct lookup:
  - trace_correlator.py extracts {corr_id: op_name} from trace.json cpu_op events
  - trace_correlator.py also extracts [NrtDeviceEvent] from privateuse1_driver events
  - Attribution = corr_id_to_op.get(event.corr_id) — one step, no interval tree

Hardware metrics are overlaid from ntrace_parser.parse() (currently a stub).

Warm-up filtering
------------------
Warm-up device events have no corresponding aten:: op in the correlation map
because torch.profiler is not active during warm-up iterations.  These appear
as UNATTRIBUTED events at the start of the device_events list.  The first
attributed event marks the boundary; everything before it is flagged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from trainium.operator_profiler.capture.ntrace_parser import parse as parse_ntrace
from trainium.operator_profiler.capture.trace_correlator import (
    NrtDeviceEvent,
    build_attribution_maps,
)
from trainium.operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    KernelAttribution,
    KernelManifestEntry,
    MappingManifest,
)
from trainium.operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    NrtEventInfo,
)

log = logging.getLogger(__name__)


class ManifestBuilder:
    """
    Build a MappingManifest from a NRT session directory.

    Parameters
    ----------
    trace_json_path:
        Path to the Chrome trace exported by NeuronProfiler.export_trace().
    nrt_session_dir:
        NRT output directory (contains ntrace.pb, .ntff, trace_info.pb, etc.).
    metadata:
        Capture metadata to embed in the manifest.
    """

    def __init__(
        self,
        trace_json_path: Path,
        nrt_session_dir: Path,
        metadata: CaptureManifestMetadata,
    ) -> None:
        self.trace_json_path = trace_json_path
        self.nrt_session_dir = nrt_session_dir
        self.metadata = metadata

    def build(self) -> MappingManifest:
        # Step 1: extract attribution maps from trace.json
        corr_id_to_op, device_events = build_attribution_maps(self.trace_json_path)

        # Step 2: overlay hardware metrics from ntrace.pb (may return empty dict)
        ntrace_metrics = parse_ntrace(self.nrt_session_dir)
        if not ntrace_metrics:
            log.info("No hardware metrics available (ntrace_parser stub or missing ntrace.pb)")

        # Step 3: detect warm-up boundary
        warmup_ids = self._find_warmup_ids(device_events, corr_id_to_op)

        # Step 4: build KernelManifestEntry for each device event
        entries: list[KernelManifestEntry] = []
        warnings: list[str] = []

        for idx, ev in enumerate(device_events):
            kid = f"k_{idx:05d}"

            if kid in warmup_ids:
                warnings.append(
                    f"{kid} ({ev.event_name}): flagged as warm-up event "
                    f"(before first attributed operation)"
                )

            op_name = corr_id_to_op.get(ev.correlation_id)
            if op_name:
                method = AttributionMethod.NRT_CORRELATION
                confidence = Confidence.HIGH
                source_ops = [op_name]
            else:
                method = AttributionMethod.UNATTRIBUTED
                confidence = Confidence.UNATTRIBUTED
                source_ops = []

            nrt_event_info = NrtEventInfo(
                event_name=ev.event_name,
                neuroncore_id=ev.neuroncore_id,
                start_ns=ev.start_ns,
                end_ns=ev.end_ns,
                event_category="privateuse1_driver",
            )

            raw_metrics = ntrace_metrics.get(ev.correlation_id, {})

            entries.append(
                KernelManifestEntry(
                    kernel_id=kid,
                    kernel_name=ev.event_name,
                    stream_id=ev.neuroncore_id,
                    device_id=ev.device_id,
                    start_ns=ev.start_ns,
                    end_ns=ev.end_ns,
                    duration_ns=ev.duration_ns,
                    attribution=KernelAttribution(
                        method=method,
                        source_operators=source_ops,
                        nrt_event=nrt_event_info,
                        confidence=confidence,
                    ),
                    kineto_correlation_id=ev.correlation_id,
                    # _raw_metrics stored in attribution for downstream access;
                    # KernelManifestEntry does not carry raw metrics directly —
                    # profile_builder reads them from the NrtDeviceEvent via the
                    # correlation ID index built in attribution_engine.
                )
            )

        attributed = sum(1 for e in entries if e.attribution.method != AttributionMethod.UNATTRIBUTED)
        log.info(
            "ManifestBuilder: %d entries, %d attributed, %d unattributed, %d warm-up",
            len(entries),
            attributed,
            len(entries) - attributed - len(warmup_ids),
            len(warmup_ids),
        )

        return MappingManifest(
            capture_metadata=self.metadata,
            kernels=entries,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_warmup_ids(
        device_events: list[NrtDeviceEvent],
        corr_id_to_op: dict[int, str],
    ) -> set[str]:
        """
        Flag device events before the first attributed event as warm-up.

        The torch.profiler context is only active during measure_iters, not
        during warmup_iters, so warm-up operations have no correlation entry.
        The first attributed event marks the profiling-start boundary.
        """
        warmup_ids: set[str] = set()
        first_attributed_seen = False

        for idx, ev in enumerate(device_events):
            kid = f"k_{idx:05d}"
            if ev.correlation_id in corr_id_to_op:
                first_attributed_seen = True
            if not first_attributed_seen:
                warmup_ids.add(kid)

        return warmup_ids
