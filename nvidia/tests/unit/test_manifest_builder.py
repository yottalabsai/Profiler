"""
Unit tests for ManifestBuilder — the two-way join logic (nsys + NVTX forest).

Uses mocked nsys SQLite so no real GPU hardware is needed.

Verifies:
  - NVTX enclosure → method=nvtx, confidence=medium
  - Name heuristic only → method=name_heuristic, confidence=low
  - No match → method=unattributed, confidence=unattributed
  - Fused kernel (multiple source_ops from heuristic) → is_fused=True
  - Warm-up outlier detection (edge case #4)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder, _WARMUP_OUTLIER_RATIO
from nvidia.operator_profiler.mapper.nsys_export import KernelRow, NvtxRow
from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata
from nvidia.operator_profiler.schema.profile import AttributionMethod, Confidence, NvtxRangeInfo


def make_metadata(**kwargs) -> CaptureManifestMetadata:
    defaults = dict(
        model_name="TestModel",
        torch_version="2.3.0",
        compile_mode="inductor",
        capture_timestamp_utc="2026-03-19T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return CaptureManifestMetadata(**defaults)


def make_kernel_row(**kwargs) -> KernelRow:
    defaults = dict(
        correlation_id=1,
        kernel_name="some_kernel",
        start_ns=1000,
        end_ns=1200,
        stream_id=7,
        device_id=0,
        grid_x=32, grid_y=1, grid_z=1,
        block_x=128, block_y=1, block_z=1,
    )
    defaults.update(kwargs)
    return KernelRow(**defaults)


def make_nvtx_row(**kwargs) -> NvtxRow:
    defaults = dict(
        text="aten::mm",
        start_ns=900,
        end_ns=1300,
        nesting_level=2,
        domain="default",
        stream_id=7,
        device_id=0,
    )
    defaults.update(kwargs)
    return NvtxRow(**defaults)


# ---------------------------------------------------------------------------
# ManifestBuilder with mocked nsys export
# ---------------------------------------------------------------------------

class TestManifestBuilder:
    def _build(
        self,
        kernel_rows: list[KernelRow],
        nvtx_rows: list[NvtxRow],
        correlation_map: dict | None = None,
    ):
        """Helper: run ManifestBuilder with mocked nsys I/O functions only."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata = make_metadata()
            builder = ManifestBuilder(
                nsys_rep_path=tmp_path / "fake.nsys-rep",
                metadata=metadata,
                correlation_map=correlation_map,
            )

            # Mock only the I/O boundary — let _build_forest and _attribute run normally
            with (
                patch("nvidia.operator_profiler.mapper.manifest_builder.export_to_sqlite",
                      return_value=tmp_path / "fake.sqlite"),
                patch("nvidia.operator_profiler.mapper.manifest_builder.query_kernels",
                      return_value=kernel_rows),
                patch("nvidia.operator_profiler.mapper.manifest_builder.query_nvtx_events",
                      return_value=nvtx_rows),
            ):
                manifest = builder.build()

        return manifest

    # --- NVTX attribution ---

    def test_nvtx_medium_confidence(self):
        kernel = make_kernel_row(kernel_name="unknown_cuda_kernel", start_ns=1000, end_ns=1200)
        nvtx = make_nvtx_row(text="aten::mm", start_ns=900, end_ns=1300, nesting_level=2)
        manifest = self._build([kernel], [nvtx])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NVTX
        assert entry.attribution.confidence == Confidence.MEDIUM
        assert entry.attribution.source_operators == ["aten::mm"]

    def test_nvtx_only_matches_same_stream(self):
        """NVTX on stream 8 should NOT attribute kernel on stream 7 (edge case #3)."""
        kernel = make_kernel_row(kernel_name="kernel_s7", start_ns=1000, end_ns=1200, stream_id=7)
        nvtx = make_nvtx_row(text="aten::relu", start_ns=900, end_ns=1300, stream_id=8)
        manifest = self._build([kernel], [nvtx])
        # Should fall back to heuristic or unattributed — not NVTX
        entry = manifest.kernels[0]
        assert entry.attribution.method != AttributionMethod.NVTX

    # --- Name heuristic ---

    def test_name_heuristic_low_confidence(self):
        kernel = make_kernel_row(kernel_name="triton_poi_fused_relu_0")
        manifest = self._build([kernel], [])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NAME_HEURISTIC
        assert entry.attribution.confidence == Confidence.LOW
        assert entry.attribution.source_operators == ["aten::relu"]

    # --- Unattributed fallback ---

    def test_unattributed_fallback(self):
        kernel = make_kernel_row(kernel_name="some_mystery_kernel_xyz")
        manifest = self._build([kernel], [])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.UNATTRIBUTED
        assert entry.attribution.confidence == Confidence.UNATTRIBUTED

    # --- Warm-up outlier detection (edge case #4) ---

    def test_warmup_outlier_flagged(self):
        # One kernel is >10× median duration
        median_dur = 100
        outlier_dur = int(median_dur * (_WARMUP_OUTLIER_RATIO + 1))
        kernels = [
            make_kernel_row(kernel_name=f"k_{i}", start_ns=i * 1000, end_ns=i * 1000 + median_dur)
            for i in range(10)
        ]
        # Replace first kernel with outlier
        kernels[0] = make_kernel_row(
            kernel_name="warm_up_kernel", start_ns=0, end_ns=outlier_dur
        )
        manifest = self._build(kernels, [])
        # At least one warning about initialization kernel
        assert any("initialization kernel" in w for w in manifest.warnings)

    def test_no_false_warmup_with_uniform_durations(self):
        kernels = [
            make_kernel_row(kernel_name=f"k_{i}", start_ns=i * 1000, end_ns=i * 1000 + 100)
            for i in range(10)
        ]
        manifest = self._build(kernels, [])
        warmup_warnings = [w for w in manifest.warnings if "warm-up" in w]
        assert len(warmup_warnings) == 0

    # --- All enclosing ranges stored for fused kernels (edge case #7) ---

    def test_all_enclosing_ranges_stored(self):
        kernel = make_kernel_row(kernel_name="unknown_k", start_ns=500, end_ns=600)
        outer = make_nvtx_row(text="outer_op", start_ns=0, end_ns=1000, nesting_level=1)
        inner = make_nvtx_row(text="aten::linear", start_ns=400, end_ns=700, nesting_level=2)
        manifest = self._build([kernel], [outer, inner])
        entry = manifest.kernels[0]
        assert len(entry.attribution.all_enclosing_ranges) == 2
        texts = [r.text for r in entry.attribution.all_enclosing_ranges]
        assert "outer_op" in texts
        assert "aten::linear" in texts

    # --- torch.profiler HIGH confidence tier ---

    def test_torch_profiler_high_confidence(self):
        kernel = make_kernel_row(kernel_name="triton_mm_0")
        corr_map = {("triton_mm_0", 0): "aten::mm"}
        manifest = self._build([kernel], [], correlation_map=corr_map)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.TORCH_PROFILER
        assert entry.attribution.confidence == Confidence.HIGH
        assert entry.attribution.source_operators == ["aten::mm"]

    def test_torch_profiler_falls_through_to_nvtx(self):
        """No correlation map entry → falls through to NVTX MEDIUM."""
        kernel = make_kernel_row(kernel_name="triton_mm_0")
        nvtx = make_nvtx_row(text="aten::mm", start_ns=900, end_ns=1300, nesting_level=2)
        corr_map = {("other_kernel", 0): "aten::relu"}  # doesn't match triton_mm_0
        manifest = self._build([kernel], [nvtx], correlation_map=corr_map)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NVTX
        assert entry.attribution.confidence == Confidence.MEDIUM

    def test_torch_profiler_beats_nvtx(self):
        """When both correlation map and NVTX match, HIGH confidence wins."""
        kernel = make_kernel_row(kernel_name="triton_mm_0")
        nvtx = make_nvtx_row(text="aten::addmm", start_ns=900, end_ns=1300, nesting_level=2)
        corr_map = {("triton_mm_0", 0): "aten::mm"}
        manifest = self._build([kernel], [nvtx], correlation_map=corr_map)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.TORCH_PROFILER
        assert entry.attribution.confidence == Confidence.HIGH
        assert entry.attribution.source_operators == ["aten::mm"]

    def test_invocation_counter_increments(self):
        """Same kernel name twice uses indices 0 and 1 correctly."""
        kernels = [
            make_kernel_row(kernel_name="triton_mm_0", correlation_id=1),
            make_kernel_row(kernel_name="triton_mm_0", correlation_id=2),
        ]
        corr_map = {
            ("triton_mm_0", 0): "aten::mm",
            ("triton_mm_0", 1): "aten::linear",
        }
        manifest = self._build(kernels, [], correlation_map=corr_map)
        assert manifest.kernels[0].attribution.source_operators == ["aten::mm"]
        assert manifest.kernels[1].attribution.source_operators == ["aten::linear"]
        assert manifest.kernels[0].attribution.confidence == Confidence.HIGH
        assert manifest.kernels[1].attribution.confidence == Confidence.HIGH

    def test_no_correlation_map_unchanged_behavior(self):
        """Without correlation_map, non-Triton kernels with no NVTX are UNATTRIBUTED."""
        kernel = make_kernel_row(kernel_name="volta_gemm_fp32_nn_128x128")
        manifest = self._build([kernel], [])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.UNATTRIBUTED
        assert entry.attribution.confidence == Confidence.UNATTRIBUTED

    # --- quantized:: and torch.library custom op attribution ---

    def test_nvtx_quantized_op_medium_confidence(self):
        """quantized:: NVTX range is accepted at MEDIUM confidence."""
        kernel = make_kernel_row(kernel_name="quantized_linear_kernel")
        nvtx = make_nvtx_row(text="quantized::linear", start_ns=900, end_ns=1300)
        manifest = self._build([kernel], [nvtx])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NVTX
        assert entry.attribution.confidence == Confidence.MEDIUM
        assert entry.attribution.source_operators == ["quantized::linear"]

    def test_nvtx_custom_op_medium_confidence(self):
        """Arbitrary torch.library namespace NVTX range is accepted at MEDIUM confidence."""
        kernel = make_kernel_row(kernel_name="flash_attn_fwd_kernel")
        nvtx = make_nvtx_row(text="flash_attn::fwd", start_ns=900, end_ns=1300)
        manifest = self._build([kernel], [nvtx])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NVTX
        assert entry.attribution.confidence == Confidence.MEDIUM
        assert entry.attribution.source_operators == ["flash_attn::fwd"]

    def test_nvtx_prims_range_skipped_in_favour_of_outer_aten(self):
        """prims:: is excluded — kernel enclosed by prims:: (inner) and aten::
        (outer) is attributed to the aten:: range, not prims::."""
        kernel = make_kernel_row(kernel_name="triton_mm_0", start_ns=500, end_ns=600)
        outer = make_nvtx_row(text="aten::linear", start_ns=400, end_ns=700, nesting_level=1)
        inner = make_nvtx_row(text="prims::mm", start_ns=450, end_ns=650, nesting_level=2)
        manifest = self._build([kernel], [outer, inner])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NVTX
        assert entry.attribution.source_operators == ["aten::linear"]

    def test_nvtx_non_namespace_text_still_rejected(self):
        """Non-namespace NVTX text (e.g. ProfilerStep#0) does not produce NVTX attribution."""
        kernel = make_kernel_row(kernel_name="some_kernel")
        nvtx = make_nvtx_row(text="ProfilerStep#0", start_ns=900, end_ns=1300)
        manifest = self._build([kernel], [nvtx])
        entry = manifest.kernels[0]
        assert entry.attribution.method != AttributionMethod.NVTX

    def test_torch_profiler_quantized_op_high_confidence(self):
        """quantized:: op in correlation map produces HIGH confidence."""
        kernel = make_kernel_row(kernel_name="quantized_linear_kernel")
        corr_map = {("quantized_linear_kernel", 0): "quantized::linear"}
        manifest = self._build([kernel], [], correlation_map=corr_map)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.TORCH_PROFILER
        assert entry.attribution.confidence == Confidence.HIGH
        assert entry.attribution.source_operators == ["quantized::linear"]

    def test_torch_profiler_custom_library_op_high_confidence(self):
        """torch.library custom op in correlation map produces HIGH confidence."""
        kernel = make_kernel_row(kernel_name="flash_attn_fwd_kernel")
        corr_map = {("flash_attn_fwd_kernel", 0): "flash_attn::fwd"}
        manifest = self._build([kernel], [], correlation_map=corr_map)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.TORCH_PROFILER
        assert entry.attribution.confidence == Confidence.HIGH
        assert entry.attribution.source_operators == ["flash_attn::fwd"]
