"""
Metrics schema helpers — NRT metric policies and aggregation strategies.

Mirrors nvidia/operator_profiler/schema/metrics.py but for Neuron Runtime (NRT)
counter names instead of ncu counter names.

BLOCKER — ntrace.pb schema unknown
-----------------------------------
The exact NRT counter names inside ntrace.pb are not yet confirmed.  This file
contains the best-known candidate names based on:
  1. torch-neuronx C++ source: Session.cpp processTrace() emits
     PRIVATEUSE1_RUNTIME and PRIVATEUSE1_DRIVER GenericTraceActivity events.
  2. AWS Neuron documentation for neuron-profile (neuron-profile view --*).
  3. Known Trainium hardware units: TensorEngine (MXU), VectorEngine (VPE),
     DMA engine, SRAM (SBUF/PBUF/HBUF), DDR.

Steps to resolve the blocker:
  1. Run `NeuronConfig(modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME])` on a
     real Trainium instance and print all fields from ntrace.pb:
       python3 -c "
       import ntrace_pb2          # from neuron-tools or aws-neuronx-runtime-lib
       trace = ntrace_pb2.Trace()
       trace.ParseFromString(open('ntrace.pb','rb').read())
       for event in trace.events: print(event)
       "
  2. Alternatively, run `neuron-profile view --output-format json <session_dir>`
     to see what metrics the Neuron profiler exposes in its JSON output.
  3. Replace the TODO placeholders below with confirmed counter names.

Architecture compatibility
--------------------------
Trainium1 (trn1) and Trainium2 (trn2) may use different counter names in NRT.
Use nrt_name_fallbacks (same pattern as NVIDIA's ncu_name_fallbacks) to handle
renamed counters across hardware generations.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class AggregationOp(str, Enum):
    SUM  = "sum"
    MEAN = "mean"
    MAX  = "max"
    MIN  = "min"


class MetricPolicy(NamedTuple):
    nrt_name: str              # primary NRT counter name
    profile_field: str         # logical field name in KernelMetrics.raw
    aggregation: AggregationOp
    description: str
    nrt_name_fallbacks: tuple[str, ...] = ()

    @property
    def nrt_names(self) -> list[str]:
        return [self.nrt_name, *self.nrt_name_fallbacks]


# ---------------------------------------------------------------------------
# Candidate metric policies
#
# TODO(blocker: ntrace.pb schema) Replace "TODO_*" counter names with real
# NRT counter names once ntrace.pb is introspected on a Trainium instance.
#
# Best-guess names are annotated with their likely source in parentheses.
# ---------------------------------------------------------------------------
METRIC_POLICIES: list[MetricPolicy] = [
    # --- Execution time ---
    MetricPolicy(
        "TODO_execution_duration_ns",       # likely from trace event duration
        "execution_duration_ns",
        AggregationOp.SUM,
        "NeuronCore operation wall time (sum)",
    ),
    # --- DDR memory bandwidth ---
    MetricPolicy(
        "TODO_dma_bytes_read",              # DMA engine read bytes (trn1 name TBD)
        "dma_bytes_read",
        AggregationOp.SUM,
        "Total DDR bytes read by DMA engine",
        nrt_name_fallbacks=(
            "TODO_ddr_read_bytes",          # possible trn2 renamed counter
        ),
    ),
    MetricPolicy(
        "TODO_dma_bytes_written",
        "dma_bytes_written",
        AggregationOp.SUM,
        "Total DDR bytes written by DMA engine",
        nrt_name_fallbacks=(
            "TODO_ddr_write_bytes",
        ),
    ),
    # --- Memory throughput ---
    MetricPolicy(
        "TODO_memory_utilization_pct",
        "memory_utilization_pct",
        AggregationOp.MEAN,
        "DDR bandwidth utilization % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "TODO_ddr_throughput_pct",
        "ddr_throughput_pct",
        AggregationOp.MEAN,
        "DDR throughput % of peak (rate — use mean)",
    ),
    # --- Compute utilization ---
    MetricPolicy(
        "TODO_tensor_engine_utilization_pct",  # MXU (systolic array) busy %
        "tensor_engine_utilization_pct",
        AggregationOp.MEAN,
        "TensorEngine (MXU) utilization % of peak (rate — use mean)",
    ),
    MetricPolicy(
        "TODO_vector_engine_utilization_pct",  # VPE (vector processing) busy %
        "vector_engine_utilization_pct",
        AggregationOp.MEAN,
        "VectorEngine (VPE) utilization % of peak (rate — use mean)",
    ),
    # --- Stall / latency ---
    MetricPolicy(
        "TODO_stall_cycles",
        "stall_cycles",
        AggregationOp.SUM,
        "Pipeline stall cycles (DMA wait, dependency stalls)",
    ),
    MetricPolicy(
        "TODO_stall_cycles_pct",
        "stall_cycles_pct",
        AggregationOp.MEAN,
        "Stall fraction — fraction of cycles wasted waiting (use mean)",
    ),
    # --- Instruction / operation throughput ---
    MetricPolicy(
        "TODO_execution_cycles",
        "execution_cycles",
        AggregationOp.SUM,
        "Total NeuronCore execution cycles",
    ),
    MetricPolicy(
        "TODO_operations_per_cycle",
        "operations_per_cycle",
        AggregationOp.MEAN,
        "Effective throughput: operations dispatched per clock cycle (use mean)",
    ),
    # --- On-chip memory (scratchpad) ---
    # SBUF (state buffer / input scratchpad), PBUF (pool buffer), HBUF (hidden buffer)
    MetricPolicy(
        "TODO_sbuf_utilization_pct",
        "sbuf_utilization_pct",
        AggregationOp.MEAN,
        "SBUF (state buffer) utilization % — analogous to L1 hit rate",
    ),
    MetricPolicy(
        "TODO_hbuf_utilization_pct",
        "hbuf_utilization_pct",
        AggregationOp.MEAN,
        "HBUF (hidden buffer) utilization % — analogous to L2 hit rate",
    ),
]

# Fast lookups
NRT_NAME_TO_POLICY: dict[str, MetricPolicy] = {}
for _p in METRIC_POLICIES:
    for _name in _p.nrt_names:
        NRT_NAME_TO_POLICY[_name] = _p

PROFILE_FIELD_TO_NRT_NAMES: dict[str, list[str]] = {}
for _nrt_name, _p in NRT_NAME_TO_POLICY.items():
    PROFILE_FIELD_TO_NRT_NAMES.setdefault(_p.profile_field, []).append(_nrt_name)


def get_raw_value(
    raw: dict[str, "float | int | str"], profile_field: str
) -> "float | None":
    """
    Look up a metric from a KernelMetrics.raw dict by logical name.

    Checks all NRT name aliases so callers don't need to know which variant
    the NRT used.  Returns the first numeric match, or None.
    """
    for nrt_name in PROFILE_FIELD_TO_NRT_NAMES.get(profile_field, []):
        val = raw.get(nrt_name)
        if isinstance(val, (int, float)):
            return float(val)
    return None


# Explicit metric list to request when configuring NRT capture.
# TODO(blocker: ntrace.pb schema) Once real counter names are known, this list
# is passed to NeuronConfig or equivalent NRT configuration to ensure metrics
# are collected.  Currently all entries are placeholders.
_seen: set[str] = set()
AGGREGATE_NRT_METRICS: list[str] = []
for _p in METRIC_POLICIES:
    for _name in _p.nrt_names:
        if _name not in _seen:
            AGGREGATE_NRT_METRICS.append(_name)
            _seen.add(_name)
del _seen
