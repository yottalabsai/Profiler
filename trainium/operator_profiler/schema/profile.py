"""
Operator-Attributed Profile schema — the central data contract for all pipeline stages.

Mirrors nvidia/operator_profiler/schema/profile.py with Trainium-specific adaptations:

  AttributionMethod: NVTX and INDUCTOR_FUSION replaced by NRT_CORRELATION.
    Neuron uses Kineto correlation IDs embedded in ntrace.pb events to link
    device operations back to the originating PyTorch aten:: op — no NVTX
    ranges or Inductor debug artifacts are involved.

  KernelRecord: stream_id repurposed as neuroncore_id (which NeuronCore ran
    the operation).  grid_dim / block_dim are absent on Trainium (NeuronCore
    does not expose SIMT launch geometry) and default to None.

  CaptureMetadata: cuda_version / nsys_report_path / ncu_report_path replaced
    by neuron_sdk_version / nrt_session_dir.

  KernelMetrics.raw: same open dict design; keys are NRT counter names defined
    in trainium.operator_profiler.schema.metrics rather than ncu counter names.

Everything else (AggregatedMetrics, OperatorRecord, OperatorAttributedProfile)
is identical to the NVIDIA schema so that the same downstream analysis tools —
the profiler-plugin optimization strategist, comparison agent, and backend
engineer — can consume both backends' profile.json files without modification.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AttributionMethod(str, Enum):
    TORCH_PROFILER  = "torch_profiler"   # Kineto External-id join via trace.json
    NRT_CORRELATION = "nrt_correlation"  # Kineto correlation-ID stack embedded in ntrace.pb
    UNATTRIBUTED    = "unattributed"


class Confidence(str, Enum):
    HIGH         = "high"
    MEDIUM       = "medium"
    LOW          = "low"
    UNATTRIBUTED = "unattributed"


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class NrtEventInfo(BaseModel):
    """Neuron Runtime event metadata — parallel to nvidia's NvtxRangeInfo."""
    event_name: str
    neuroncore_id: int
    start_ns: int
    end_ns: int
    # NRT event category: "privateuse1_driver" (device execution) or
    # "privateuse1_runtime" (host dispatch window).  Set from Chrome trace cat.
    event_category: str = "privateuse1_driver"


class KernelMetrics(BaseModel):
    # NRT hardware counters stored as-collected — no hard-coded named fields.
    # Keys are NRT metric names defined in trainium.operator_profiler.schema.metrics.
    # Use get_raw_value() from that module to read by logical name.
    #
    # TODO(blocker: ntrace.pb schema) Fill this in once the NRT protobuf schema
    # is known.  Expected candidates from ntrace.pb/trace_info.pb:
    #   - dma_bytes_read / dma_bytes_written     (DDR bandwidth)
    #   - execution_cycles                        (NeuronCore cycles)
    #   - tensor_engine_utilization_pct           (MXU/VPE busy %)
    #   - memory_utilization_pct                  (DDR throughput %)
    #   - stall_cycles                            (pipeline stall %)
    #   See trainium/operator_profiler/schema/metrics.py for the full candidate list.
    raw: dict[str, float | int | str] = Field(default_factory=dict)

    @property
    def dma_bytes_read(self) -> int | None:
        from trainium.operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "dma_bytes_read")
        return int(v) if v is not None else None

    @property
    def dma_bytes_written(self) -> int | None:
        from trainium.operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "dma_bytes_written")
        return int(v) if v is not None else None

    @property
    def tensor_engine_utilization_pct(self) -> float | None:
        from trainium.operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "tensor_engine_utilization_pct")
        return float(v) if v is not None else None

    @property
    def execution_cycles(self) -> int | None:
        from trainium.operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "execution_cycles")
        return int(v) if v is not None else None


# ---------------------------------------------------------------------------
# Kernel record — one NeuronCore operation execution
# ---------------------------------------------------------------------------

class KernelRecord(BaseModel):
    kernel_id: str
    kernel_name: str            # NRT operation name (e.g. "nrt_op_matmul_0")
    demangled_name: str | None = None
    # neuroncore_id repurposes stream_id so downstream consumers that read
    # stream_id still work; semantics differ (NeuronCore index, not CUDA stream).
    stream_id: int              # = neuroncore_id
    device_id: int
    start_ns: int
    end_ns: int
    duration_ns: int
    grid_dim: tuple[int, int, int] | None = None   # always None on Trainium
    block_dim: tuple[int, int, int] | None = None  # always None on Trainium
    metrics: KernelMetrics = Field(default_factory=KernelMetrics)
    attribution_method: AttributionMethod = AttributionMethod.UNATTRIBUTED
    confidence: Confidence = Confidence.UNATTRIBUTED
    nrt_event: NrtEventInfo | None = None


# ---------------------------------------------------------------------------
# Aggregated metrics across all NeuronCore operations in one operator
# ---------------------------------------------------------------------------

class AggregatedMetrics(BaseModel):
    total_duration_ns: int
    kernel_count: int
    dominant_kernel_id: str | None = None

    # ------------------------------------------------------------------
    # Memory bandwidth (None = counter unavailable)
    # TODO(blocker: ntrace.pb schema) Verify NRT exposes these per-op
    # ------------------------------------------------------------------
    total_dma_bytes_read: int | None = None
    total_dma_bytes_written: int | None = None

    # ------------------------------------------------------------------
    # Memory subsystem throughput — % of peak (duration-weighted mean)
    # ------------------------------------------------------------------
    memory_utilization_pct: float | None = None   # DDR bandwidth utilization
    ddr_throughput_pct: float | None = None       # DDR specifically

    # ------------------------------------------------------------------
    # Compute utilization — % of peak (duration-weighted mean)
    # ------------------------------------------------------------------
    tensor_engine_utilization_pct: float | None = None  # MXU/VPE busy
    vector_engine_utilization_pct: float | None = None  # VPE specifically

    # ------------------------------------------------------------------
    # Stall / latency (duration-weighted mean)
    # ------------------------------------------------------------------
    stall_cycles_pct: float | None = None   # pipeline stall fraction

    # ------------------------------------------------------------------
    # Instruction throughput
    # ------------------------------------------------------------------
    total_execution_cycles: int | None = None
    operations_per_cycle: float | None = None  # analogous to IPC

    # ------------------------------------------------------------------
    # These fields mirror the NVIDIA schema so comparison tools work
    # transparently across backends.  Map semantics in comments.
    # ------------------------------------------------------------------
    total_dram_bytes_read: int | None = None    # alias → total_dma_bytes_read
    total_dram_bytes_written: int | None = None # alias → total_dma_bytes_written
    sm_throughput_pct: float | None = None      # alias → tensor_engine_utilization_pct
    achieved_occupancy: float | None = None     # NeuronCore equivalent if available
    l1_hit_rate: float | None = None            # scratchpad / SBUF hit rate if exposed
    l2_hit_rate: float | None = None            # HBUF / PBUF hit rate if exposed


# ---------------------------------------------------------------------------
# Operator record (one aten:: dispatch)
# ---------------------------------------------------------------------------

class OperatorRecord(BaseModel):
    operator_id: str          # e.g. "aten::linear_0"
    operator_name: str        # e.g. "aten::linear"
    call_index: int
    is_fused: bool = False
    fused_with: list[str] = Field(default_factory=list)
    nrt_event: NrtEventInfo | None = None
    kernels: list[KernelRecord] = Field(default_factory=list)
    aggregated: AggregatedMetrics | None = None


# ---------------------------------------------------------------------------
# Capture metadata
# ---------------------------------------------------------------------------

class CaptureMetadata(BaseModel):
    model_name: str
    torch_version: str
    neuron_sdk_version: str | None = None   # replaces cuda_version
    compile_mode: str = "neuron"            # "neuron" | "torch_neuronx" | "eager"
    nrt_session_dir: str | None = None      # replaces nsys_report_path + ncu_report_path
    capture_timestamp_utc: str
    device_name: str | None = None          # e.g. "trn1.2xlarge"
    neuroncore_count: int | None = None     # how many NeuronCores were active


# ---------------------------------------------------------------------------
# Top-level output — identical structure to NVIDIA OperatorAttributedProfile
# ---------------------------------------------------------------------------

class OperatorAttributedProfile(BaseModel):
    schema_version: Annotated[str, Field(default=SCHEMA_VERSION)]
    capture_metadata: CaptureMetadata
    operators: list[OperatorRecord] = Field(default_factory=list)
    unattributed_kernels: list[KernelRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
