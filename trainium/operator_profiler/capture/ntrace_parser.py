"""
ntrace_parser — extract per-operation hardware metrics from ntrace.pb.

BLOCKER: ntrace.pb protobuf schema is not yet confirmed.
---------------------------------------------------------
This module ships as a documented stub.  It returns empty KernelMetrics for all
operations and emits a single warning so the pipeline can run end-to-end (Level 1
operator attribution) without hardware counters.

To unblock Level 2 (hardware metrics), run the following on a real Trainium
instance to discover the ntrace.pb schema:

  Option A — introspect protobuf directly:
    python3 -c "
    import sys; sys.path.insert(0, '/opt/aws/neuron/lib/python')
    import ntrace_pb2
    trace = ntrace_pb2.Trace()
    trace.ParseFromString(open('ntrace.pb','rb').read())
    for e in trace.events[:5]: print(e.DESCRIPTOR.fields_by_name.keys(), e)
    "

  Option B — use neuron-profile CLI:
    neuron-profile view --output-format json <session_dir> | python3 -m json.tool | head -200

  Option C — locate the .proto file:
    find /opt/aws/neuron -name '*.proto' 2>/dev/null
    find /usr/lib/python3 -path '*/neuronx*/*.proto' 2>/dev/null

Once the schema is confirmed:
  1. Replace _parse_ntrace_pb() with a real implementation.
  2. Update METRIC_POLICIES in schema/metrics.py with the actual counter names.
  3. Remove the stub warning from parse().

Expected ntrace.pb content (based on torch_neuronx C++ processTrace()):
  - Per NeuronCore operation events with:
      - operation name (matches NrtDeviceEvent.event_name from trace_correlator)
      - Kineto correlation ID (join key — same as trace.json External id)
      - start/end timestamps (may differ from Chrome trace due to device clock)
      - hardware counters: execution_cycles, dma_bytes_read, dma_bytes_written,
        tensor_engine_utilization_pct, stall_cycles, etc.
  - Aggregated per-NeuronCore summary metrics in trace_info.pb
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_STUB_WARNING_EMITTED = False


def parse(
    session_dir: Path,
) -> dict[int, dict[str, float | int | str]]:
    """
    Parse ntrace.pb from *session_dir* and return per-operation hardware metrics.

    Returns
    -------
    dict[correlation_id, metrics_raw_dict]
        Keys are Kineto correlation IDs (matching NrtDeviceEvent.correlation_id).
        Values are raw metric dicts suitable for KernelMetrics(raw=...).
        Returns an empty dict until the ntrace.pb schema is confirmed.
    """
    global _STUB_WARNING_EMITTED

    ntrace_path = _find_ntrace(session_dir)
    if ntrace_path is None:
        if not _STUB_WARNING_EMITTED:
            log.warning(
                "ntrace.pb not found in %s — hardware metrics will be empty. "
                "Ensure ProfileMode.RUNTIME was enabled during capture.",
                session_dir,
            )
            _STUB_WARNING_EMITTED = True
        return {}

    if not _STUB_WARNING_EMITTED:
        log.warning(
            "ntrace_parser is a stub — ntrace.pb found at %s but schema is not yet "
            "confirmed. Hardware metrics will be empty. See ntrace_parser.py for "
            "instructions on how to discover the schema on a Trainium instance.",
            ntrace_path,
        )
        _STUB_WARNING_EMITTED = True

    # TODO(blocker: ntrace.pb schema) Replace with real implementation:
    #   metrics_by_corr_id = _parse_ntrace_pb(ntrace_path)
    #   return metrics_by_corr_id
    return {}


def _find_ntrace(session_dir: Path) -> Path | None:
    """
    Locate ntrace.pb within the NRT session directory.

    NRT writes output to:
      {profile_output_dir}/{instance_id}_pid_{pid}/{timestamp}/ntrace.pb
    session_dir may point to any level of this hierarchy.
    """
    candidate = session_dir / "ntrace.pb"
    if candidate.exists():
        return candidate
    # Search one level deeper (session_dir is the instance_pid dir)
    for child in sorted(session_dir.iterdir()) if session_dir.is_dir() else []:
        deeper = child / "ntrace.pb"
        if deeper.exists():
            return deeper
    return None


def _parse_ntrace_pb(
    ntrace_path: Path,
) -> dict[int, dict[str, float | int | str]]:
    """
    TODO(blocker: ntrace.pb schema) Real implementation goes here.

    Expected logic once schema is confirmed:
      1. Import ntrace_pb2 (from aws-neuronx-runtime-lib or neuron-tools)
      2. Parse binary file into Trace proto
      3. For each event in trace.events:
           corr_id = event.kineto_correlation_id  # field name TBD
           metrics = {
               "TODO_execution_cycles":            event.execution_cycles,
               "TODO_dma_bytes_read":              event.dma_bytes_read,
               "TODO_dma_bytes_written":           event.dma_bytes_written,
               "TODO_tensor_engine_utilization_pct": event.te_util_pct,
               "TODO_stall_cycles":                event.stall_cycles,
               ...
           }
           result[corr_id] = metrics
      4. Return result
    """
    raise NotImplementedError(
        "ntrace_parser._parse_ntrace_pb is not implemented — "
        "discover the ntrace.pb schema first (see module docstring)."
    )
