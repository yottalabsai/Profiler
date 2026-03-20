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
    PROVENANCE = "provenance"
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

class SourceLocation(BaseModel):
    file: str
    line: int
    col: int | None = None
    op: str


class NvtxRangeInfo(BaseModel):
    text: str
    depth: int
    start_ns: int
    end_ns: int
    domain: str = "default"


class KernelMetrics(BaseModel):
    sm_active_cycles: float | None = None
    dram_bytes_read: int | None = None
    dram_bytes_written: int | None = None
    l1_hit_rate: float | None = None
    achieved_occupancy: float | None = None
    tensor_core_active_pct: float | None = None
    arithmetic_intensity: float | None = None  # FLOP/byte (roofline)
    achieved_gflops: float | None = None
    # Forward-compat: arbitrary ncu metrics across CUDA versions
    raw: dict[str, float | int | str] = Field(default_factory=dict)


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
    source_locations: list[SourceLocation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregated metrics across all kernels in one operator
# ---------------------------------------------------------------------------

class AggregatedMetrics(BaseModel):
    total_duration_ns: int
    kernel_count: int
    total_dram_bytes_read: int = 0
    total_dram_bytes_written: int = 0
    mean_achieved_occupancy: float | None = None
    mean_tensor_core_active_pct: float | None = None
    bottleneck_classification: Literal[
        "compute_bound", "memory_bound", "latency_bound", "unknown"
    ] = "unknown"


# ---------------------------------------------------------------------------
# Operator record (one aten:: dispatch)
# ---------------------------------------------------------------------------

class OperatorRecord(BaseModel):
    operator_id: str          # e.g. "aten::linear_0"
    operator_name: str        # e.g. "aten::linear"
    call_index: int
    source_location: SourceLocation | None = None
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
    provenance_log_path: str | None = None
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
