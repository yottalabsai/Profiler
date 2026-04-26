"""
Mapping Manifest schema — intermediate artifact produced by manifest_builder.py
and consumed by kernel_profiler.py and profile_builder.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .profile import AttributionMethod, Confidence, NvtxRangeInfo

MANIFEST_SCHEMA_VERSION = "1.0"


class KernelManifestEntry(BaseModel):
    kernel_id: str
    kernel_name: str
    stream_id: int
    device_id: int
    start_ns: int
    end_ns: int
    duration_ns: int
    grid_dim: tuple[int, int, int] | None = None
    block_dim: tuple[int, int, int] | None = None
    attribution: KernelAttribution


class KernelAttribution(BaseModel):
    method: AttributionMethod
    source_operators: list[str] = Field(default_factory=list)
    nvtx_range: NvtxRangeInfo | None = None
    confidence: Confidence
    is_fused: bool = False
    # All enclosing NVTX ranges, outermost first — needed for fused kernel attribution
    all_enclosing_ranges: list[NvtxRangeInfo] = Field(default_factory=list)
    # All aten ops fused into this kernel by Inductor (from debug artifacts).
    # Separate from source_operators: source_operators answers "what Python-level
    # op owns this kernel?" while fused_ops answers "what aten primitives did
    # Inductor fuse into it?" — they can differ in abstraction level.
    fused_ops: list[str] = Field(default_factory=list)


class CaptureManifestMetadata(BaseModel):
    model_name: str
    torch_version: str
    compile_mode: str
    nsys_report_path: str | None = None
    capture_timestamp_utc: str
    # Input shapes recorded at capture time — validated at replay to prevent
    # dynamic-shape kernel count mismatches (edge case #6)
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)


class MappingManifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    capture_metadata: CaptureManifestMetadata
    kernels: list[KernelManifestEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
