"""
Operator-Attributed Profile schema — the central data contract for all stages.

Every CUDA kernel launch is represented as a KernelRecord attributed to a
PyTorch OperatorRecord.  OperatorAttributedProfile is the top-level output.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AttributionMethod(str, Enum):
    NVTX = "nvtx"
    NAME_HEURISTIC = "name_heuristic"
    UNATTRIBUTED = "unattributed"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNATTRIBUTED = "unattributed"


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class NvtxRangeInfo(BaseModel):
    text: str
    depth: int
    start_ns: int
    end_ns: int
    domain: str = "default"


class KernelMetrics(BaseModel):
    # All NCU metrics stored as-collected — no hard-coded named fields.
    # Use operator_profiler.schema.metrics.get_raw_value() to read by logical name,
    # or the convenience properties below.
    raw: dict[str, float | int | str] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience read-only properties (delegate to get_raw_value)
    # ------------------------------------------------------------------

    @property
    def dram_bytes_read(self) -> int | None:
        from operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "dram_bytes_read")
        return int(v) if v is not None else None

    @property
    def dram_bytes_written(self) -> int | None:
        from operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "dram_bytes_written")
        return int(v) if v is not None else None

    @property
    def achieved_occupancy(self) -> float | None:
        from operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "achieved_occupancy")
        return float(v) if v is not None else None

    @property
    def sm_active_cycles(self) -> float | None:
        from operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "sm_active_cycles")
        return float(v) if v is not None else None

    @property
    def tensor_core_active_pct(self) -> float | None:
        from operator_profiler.schema.metrics import get_raw_value
        v = get_raw_value(self.raw, "tensor_core_active_pct")
        return float(v) if v is not None else None

    @property
    def arithmetic_intensity(self) -> float | None:
        # Not a standard NCU column — stored directly under this key if available.
        v = self.raw.get("arithmetic_intensity")
        return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Kernel record (one CUDA kernel launch)
# ---------------------------------------------------------------------------

class KernelRecord(BaseModel):
    kernel_id: str
    kernel_name: str
    demangled_name: str | None = None
    stream_id: int
    device_id: int
    start_ns: int
    end_ns: int
    duration_ns: int
    grid_dim: tuple[int, int, int] | None = None
    block_dim: tuple[int, int, int] | None = None
    metrics: KernelMetrics = Field(default_factory=KernelMetrics)
    attribution_method: AttributionMethod = AttributionMethod.UNATTRIBUTED
    confidence: Confidence = Confidence.UNATTRIBUTED
    nvtx_range: NvtxRangeInfo | None = None


# ---------------------------------------------------------------------------
# Aggregated metrics across all kernels in one operator
# ---------------------------------------------------------------------------

class AggregatedMetrics(BaseModel):
    total_duration_ns: int
    kernel_count: int
    # Kernel that accounts for the most GPU time — primary optimization target
    dominant_kernel_id: str | None = None

    # ------------------------------------------------------------------
    # Memory bandwidth (None = counter unavailable on this hardware)
    # ------------------------------------------------------------------
    total_dram_bytes_read: int | None = None
    total_dram_bytes_written: int | None = None

    # ------------------------------------------------------------------
    # Memory subsystem throughput — % of peak (duration-weighted mean)
    # ------------------------------------------------------------------
    memory_throughput_pct: float | None = None   # overall memory subsystem
    dram_throughput_pct: float | None = None     # DRAM specifically
    mem_busy_pct: float | None = None            # memory pipeline busy %

    # ------------------------------------------------------------------
    # Cache efficiency — % hit rate (duration-weighted mean)
    # ------------------------------------------------------------------
    l1_hit_rate: float | None = None
    l2_hit_rate: float | None = None

    # ------------------------------------------------------------------
    # Compute utilization — % of peak (duration-weighted mean)
    # ------------------------------------------------------------------
    sm_throughput_pct: float | None = None       # overall SM compute
    tensor_core_active_pct: float | None = None  # Tensor Core pipe

    # ------------------------------------------------------------------
    # Occupancy & latency (duration-weighted mean)
    # ------------------------------------------------------------------
    achieved_occupancy: float | None = None
    warp_cycles_per_instruction: float | None = None  # high → latency-bound
    eligible_cycles_pct: float | None = None          # low → latency-bound

    # ------------------------------------------------------------------
    # Instruction throughput
    # ------------------------------------------------------------------
    total_executed_instructions: int = 0
    total_issued_instructions: int = 0           # backward compat; ≈ executed
    ipc_active: float | None = None              # IPC when SM is active

    # ------------------------------------------------------------------
    # Thread utilization
    # ------------------------------------------------------------------
    avg_threads_per_warp: float | None = None    # <32 → control-flow divergence

    # ------------------------------------------------------------------
    # Register / shared memory pressure (max across kernels in operator)
    # ------------------------------------------------------------------
    registers_per_thread: float | None = None    # high → limits occupancy
    local_memory_spills: int | None = None       # nonzero → costly DRAM spills
    dynamic_smem_per_block: float | None = None  # bytes; high → limits occupancy

    # ------------------------------------------------------------------
    # Set by DiagnosisAgent when present; None otherwise
    # ------------------------------------------------------------------
    bottleneck_classification: str | None = None


# ---------------------------------------------------------------------------
# Operator record (one aten:: dispatch)
# ---------------------------------------------------------------------------

class OperatorRecord(BaseModel):
    operator_id: str          # e.g. "aten::linear_0"
    operator_name: str        # e.g. "aten::linear"
    call_index: int
    is_fused: bool = False
    fused_with: list[str] = Field(default_factory=list)
    nvtx_range: NvtxRangeInfo | None = None
    kernels: list[KernelRecord] = Field(default_factory=list)
    aggregated: AggregatedMetrics | None = None


# ---------------------------------------------------------------------------
# Capture metadata
# ---------------------------------------------------------------------------

class CaptureMetadata(BaseModel):
    model_name: str
    torch_version: str
    cuda_version: str | None = None
    compile_mode: Literal["eager", "inductor", "cudagraphs"] = "eager"
    nsys_report_path: str | None = None
    ncu_report_path: str | None = None
    capture_timestamp_utc: str  # ISO 8601
    device_name: str | None = None


# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------

class OperatorAttributedProfile(BaseModel):
    schema_version: Annotated[str, Field(default=SCHEMA_VERSION)]
    capture_metadata: CaptureMetadata
    operators: list[OperatorRecord] = Field(default_factory=list)
    # Kernels that could not be attributed to any operator — never silently dropped
    unattributed_kernels: list[KernelRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
