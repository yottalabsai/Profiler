"""
Unit tests for ManifestBuilder — the three-way join logic.

Uses mocked nsys SQLite + provenance JSONL so no real GPU hardware is needed.

Verifies:
  - Provenance hit → method=provenance, confidence=high (highest priority)
  - NVTX enclosure only → method=nvtx, confidence=medium
  - Name heuristic only → method=name_heuristic, confidence=low
  - No match → method=unattributed, confidence=unattributed
  - Fused kernel (multiple source_ops) → is_fused=True
  - Warm-up outlier detection (edge case #4)
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from operator_profiler.mapper.manifest_builder import ManifestBuilder, _WARMUP_OUTLIER_RATIO
from operator_profiler.mapper.nsys_export import KernelRow, NvtxRow
from operator_profiler.schema.manifest import CaptureManifestMetadata
from operator_profiler.schema.profile import AttributionMethod, Confidence, NvtxRangeInfo


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


def make_provenance_jsonl(entries: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# ManifestBuilder with mocked nsys export
# ---------------------------------------------------------------------------

class TestManifestBuilder:
    def _build(
        self,
        kernel_rows: list[KernelRow],
        nvtx_rows: list[NvtxRow],
        provenance_entries: list[dict] | None = None,
    ):
        """Helper: run ManifestBuilder with mocked nsys I/O functions only."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prov_path = None
            if provenance_entries is not None:
                prov_path = tmp_path / "provenance.jsonl"
                make_provenance_jsonl(provenance_entries, prov_path)

            metadata = make_metadata(provenance_log_path=str(prov_path) if prov_path else None)
            builder = ManifestBuilder(
                nsys_rep_path=tmp_path / "fake.nsys-rep",
                metadata=metadata,
                provenance_jsonl_path=prov_path,
            )

            # Mock only the I/O boundary — let _build_forest and _attribute run normally
            with (
                patch("operator_profiler.mapper.manifest_builder.export_to_sqlite",
                      return_value=tmp_path / "fake.sqlite"),
                patch("operator_profiler.mapper.manifest_builder.query_kernels",
                      return_value=kernel_rows),
                patch("operator_profiler.mapper.manifest_builder.query_nvtx_events",
                      return_value=nvtx_rows),
            ):
                manifest = builder.build()

        return manifest

    # --- Provenance attribution ---

    def test_provenance_high_confidence(self):
        kernel = make_kernel_row(kernel_name="triton_per_fused_linear_relu_0")
        nvtx = make_nvtx_row(text="aten::linear", start_ns=900, end_ns=1300, nesting_level=2)
        prov = [
            {
                "generated_kernel_name": "triton_per_fused_linear_relu_0",
                "source_ops": ["aten::linear", "aten::relu"],
                "source_locations": [{"file": "model.py", "line": 42, "op": "aten::linear"}],
            }
        ]
        manifest = self._build([kernel], [nvtx], prov)
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.PROVENANCE
        assert entry.attribution.confidence == Confidence.HIGH
        assert "aten::linear" in entry.attribution.source_operators
        assert "aten::relu" in entry.attribution.source_operators

    def test_provenance_fused_kernel_marked(self):
        kernel = make_kernel_row(kernel_name="triton_fused_add_relu_0")
        prov = [
            {
                "generated_kernel_name": "triton_fused_add_relu_0",
                "source_ops": ["aten::add", "aten::relu"],
                "source_locations": [],
            }
        ]
        manifest = self._build([kernel], [], prov)
        assert manifest.kernels[0].attribution.is_fused is True

    def test_provenance_single_op_not_fused(self):
        kernel = make_kernel_row(kernel_name="triton_mm_0")
        prov = [{"generated_kernel_name": "triton_mm_0", "source_ops": ["aten::mm"], "source_locations": []}]
        manifest = self._build([kernel], [], prov)
        assert manifest.kernels[0].attribution.is_fused is False

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
        kernel = make_kernel_row(kernel_name="volta_gemm_fp32_nn_128x128")
        manifest = self._build([kernel], [])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.NAME_HEURISTIC
        assert entry.attribution.confidence == Confidence.LOW
        assert entry.attribution.source_operators == ["aten::mm"]

    # --- Unattributed fallback ---

    def test_unattributed_fallback(self):
        kernel = make_kernel_row(kernel_name="some_mystery_kernel_xyz")
        manifest = self._build([kernel], [])
        entry = manifest.kernels[0]
        assert entry.attribution.method == AttributionMethod.UNATTRIBUTED
        assert entry.attribution.confidence == Confidence.UNATTRIBUTED

    # --- Priority order: provenance > NVTX > heuristic ---

    def test_provenance_beats_nvtx(self):
        kernel = make_kernel_row(kernel_name="triton_relu_0", start_ns=1000, end_ns=1200)
        nvtx = make_nvtx_row(text="aten::mm", start_ns=900, end_ns=1300)
        prov = [{"generated_kernel_name": "triton_relu_0", "source_ops": ["aten::relu"], "source_locations": []}]
        manifest = self._build([kernel], [nvtx], prov)
        entry = manifest.kernels[0]
        # Provenance must win
        assert entry.attribution.method == AttributionMethod.PROVENANCE
        assert "aten::relu" in entry.attribution.source_operators

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
        # At least one warning about warm-up
        assert any("warm-up outlier" in w for w in manifest.warnings)

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
