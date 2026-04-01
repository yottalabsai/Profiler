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
    # All NCU metrics stored as-collected — no hard-coded named fields.
    # Use operator_profiler.schema.metrics.get_raw_value() to read by logical name.
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
    # Kernel that accounts for the most GPU time — primary optimization target
    dominant_kernel_id: str | None = None
    # Additive quantities — sum correctly across heterogeneous kernels.
    # None means the metric was unavailable on this hardware (e.g. Blackwell);
    # 0 means the metric was collected and genuinely zero.
    total_dram_bytes_read: int | None = None
    total_dram_bytes_written: int | None = None
    total_executed_instructions: int = 0
    total_issued_instructions: int = 0
    # Rate/utilization metrics — duration-weighted mean across kernels
    mean_achieved_occupancy: float | None = None
    mean_tensor_core_active_pct: float | None = None
    # Set by DiagnosisAgent when present; None otherwise
    bottleneck_classification: str | None = None


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
