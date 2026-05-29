"""
trace_correlator — parse trace.json from NeuronProfiler.export_trace() and extract
two outputs used by manifest_builder:

  1. corr_id_to_op: dict[int, str]
       Kineto correlation ID → aten:: op name
       Source: "cpu_op" events in the Chrome trace that carry "External id"

  2. device_events: list[NrtDeviceEvent]
       All NRT device execution windows, ordered by start timestamp.
       Source: "privateuse1_driver" events (device execution) in the Chrome trace.
       Each event carries "External id" linking it back to the originating CPU op.

How the Kineto correlation ID works
-------------------------------------
torch_neuronx registers a ProfilerStubs implementation (TorchStubs.cpp) that
pushes a Kineto correlation ID before each aten:: RecordFunction and pops it after.
The NRT C++ layer (CorrelationTracker.cpp) reads this ID at operation submission
time and embeds it in the NRT event.  processTrace() then emits:

  - One PRIVATEUSE1_RUNTIME event (host dispatch window)
  - One PRIVATEUSE1_DRIVER event (device execution window)

Both carry the originating correlation ID as "External id" in the Chrome trace
args dict — the same field that CUDA kernels use for CUPTI External ID.

This means the SAME join logic as torch_profiler_correlator.py works here:
  ext_id_to_op[External id] → op_name  (from cpu_op events)
  driver_event.External id  → op_name  (join on above map)

Differences from nvidia/capture/torch_profiler_correlator.py
-------------------------------------------------------------
  1. GPU kernel events have cat="kernel"; Neuron device events have
     cat="privateuse1_driver".  We collect both categories here.
  2. NVIDIA only returns cpu_op→kernel attribution (kernel names needed for
     ncu invocation-order match). Here we return the full NrtDeviceEvent list
     because manifest_builder needs start/end timestamps and neuroncore_id.
  3. No kernel name normalisation needed — NRT operation names are stable
     identifiers, not demangled C++ signatures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from trainium.operator_profiler.utils.op_namespaces import is_attributed_op

log = logging.getLogger(__name__)

# Chrome trace categories emitted by torch_neuronx processTrace()
_CAT_CPU_OP       = "cpu_op"
_CAT_NRT_DRIVER   = "privateuse1_driver"   # device execution window
_CAT_NRT_RUNTIME  = "privateuse1_runtime"  # host dispatch window (not used for attribution)


@dataclass
class NrtDeviceEvent:
    """One NeuronCore operation execution window extracted from trace.json."""
    event_name: str
    correlation_id: int          # Kineto External id — join key to cpu_op events
    start_ns: int
    end_ns: int
    duration_ns: int
    neuroncore_id: int = 0       # parsed from tid or args; 0 when not determinable
    device_id: int = 0
    # Raw args dict from the Chrome event, preserved for ntrace_parser to join against
    args: dict = field(default_factory=dict)


def build_attribution_maps(
    trace_path: Path,
) -> tuple[dict[int, str], list[NrtDeviceEvent]]:
    """
    Parse a Chrome trace produced by NeuronProfiler.export_trace().

    Returns
    -------
    corr_id_to_op : dict[int, str]
        Kineto correlation ID → aten:: op name.  Only ops that pass
        is_attributed_op() are included.

    device_events : list[NrtDeviceEvent]
        All privateuse1_driver events, sorted by start timestamp (ascending).
        Events without a correlation ID that maps to an aten:: op will have
        correlation_id set to whatever the trace carried (may be 0 or -1).
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # ------------------------------------------------------------------
    # Step 1: build External id → cpu_op name
    # ------------------------------------------------------------------
    ext_id_to_op: dict[int, str] = {}
    for ev in events:
        if ev.get("cat") != _CAT_CPU_OP:
            continue
        ext_id = ev.get("args", {}).get("External id")
        name = ev.get("name", "")
        if ext_id is not None and name and is_attributed_op(name):
            ext_id_to_op[int(ext_id)] = name

    log.debug("Found %d attributed cpu_op entries", len(ext_id_to_op))

    # ------------------------------------------------------------------
    # Step 2: collect privateuse1_driver events
    # NRT emits timestamps in microseconds (Chrome trace standard); convert to ns.
    # ------------------------------------------------------------------
    raw_driver_events = [
        ev for ev in events
        if ev.get("cat") == _CAT_NRT_DRIVER and ev.get("name")
    ]
    raw_driver_events.sort(key=lambda e: e.get("ts", 0))

    device_events: list[NrtDeviceEvent] = []
    for ev in raw_driver_events:
        args = ev.get("args", {})
        ext_id = args.get("External id")
        if ext_id is None:
            # Some synthetic or annotation events lack a correlation ID — skip
            log.debug("Driver event '%s' has no External id, skipping", ev.get("name"))
            continue

        corr_id = int(ext_id)
        # Chrome trace timestamps are in microseconds; convert to nanoseconds
        ts_us: float = ev.get("ts", 0)
        dur_us: float = ev.get("dur", 0)
        start_ns = int(ts_us * 1000)
        dur_ns   = int(dur_us * 1000)
        end_ns   = start_ns + dur_ns

        # NeuronCore ID: torch_neuronx encodes it in the thread ID or in args.
        # Try args["device"] first (some versions), then fall back to 0.
        neuroncore_id = int(args.get("device", args.get("nc_id", 0)))
        device_id = int(args.get("device_id", 0))

        device_events.append(
            NrtDeviceEvent(
                event_name=ev["name"],
                correlation_id=corr_id,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=dur_ns,
                neuroncore_id=neuroncore_id,
                device_id=device_id,
                args=args,
            )
        )

    # Build the public-facing corr_id_to_op (only attributed ops)
    corr_id_to_op: dict[int, str] = {
        ev.correlation_id: ext_id_to_op[ev.correlation_id]
        for ev in device_events
        if ev.correlation_id in ext_id_to_op
    }

    attributed = sum(1 for ev in device_events if ev.correlation_id in corr_id_to_op)
    log.info(
        "trace_correlator: %d device events, %d attributed (%.0f%%)",
        len(device_events),
        attributed,
        100.0 * attributed / len(device_events) if device_events else 0,
    )

    return corr_id_to_op, device_events
