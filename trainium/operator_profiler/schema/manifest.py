"""
Mapping Manifest schema — intermediate artifact produced by manifest_builder.py
and consumed by attribution_engine.py and profile_builder.py.

Mirrors nvidia/operator_profiler/schema/manifest.py with Trainium-specific changes:
  - No nsys_report_path (no nsys on Trainium)
  - kineto_correlation_id added to KernelManifestEntry (primary attribution key)
  - layer_partition / is_unique_partition kept (transformer dedup still applies)
  - NRT session dir replaces nsys report path in metadata
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .profile import AttributionMethod, Confidence, NrtEventInfo

MANIFEST_SCHEMA_VERSION = "1.0"


class KernelAttribution(BaseModel):
    method: AttributionMethod
    source_operators: list[str] = Field(default_factory=list)
    nrt_event: NrtEventInfo | None = None
    confidence: Confidence
    is_fused: bool = False
    all_enclosing_ops: list[str] = Field(default_factory=list)


class KernelManifestEntry(BaseModel):
    kernel_id: str
    kernel_name: str            # NRT operation name
    # stream_id == neuroncore_id; kept as stream_id for downstream compatibility
    stream_id: int
    device_id: int
    start_ns: int
    end_ns: int
    duration_ns: int
    grid_dim: tuple[int, int, int] | None = None   # always None on Trainium
    block_dim: tuple[int, int, int] | None = None  # always None on Trainium
    attribution: KernelAttribution
    # Kineto correlation ID embedded in the trace.json device event.
    # Primary key for joining device execution → originating aten:: op.
    kineto_correlation_id: int | None = None
    # Transformer block deduplication (same semantics as NVIDIA)
    layer_partition: str | None = None
    is_unique_partition: bool = False


class CaptureManifestMetadata(BaseModel):
    model_name: str
    torch_version: str
    compile_mode: str = "neuron"
    nrt_session_dir: str | None = None   # replaces nsys_report_path
    capture_timestamp_utc: str
    device_name: str | None = None
    neuroncore_count: int | None = None
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)


class MappingManifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    capture_metadata: CaptureManifestMetadata
    kernels: list[KernelManifestEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
