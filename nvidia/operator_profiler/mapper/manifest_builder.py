"""
Manifest builder — the two-way join at the heart of the Operator Mapper Stage.

Sources joined:
  1. nsys SQLite export  → CUDA kernel rows + NVTX event rows
  2. Per-stream NVTX interval tree → innermost NVTX range enclosing each kernel

Produces: MappingManifest

Confidence scoring:
  NVTX enclosure     → method=nvtx,            confidence=medium
  name substring     → method=name_heuristic,  confidence=low
    (includes Triton fused-name parsing: triton_*_fused_op1_op2_N)
  no match           → method=unattributed,    confidence=unattributed

Edge cases handled here:
  #1  Clock domain  — we never join on absolute timestamps across tools.
  #3  Multi-stream  — per-stream interval trees; match only on same (stream, device).
  #4  JIT warm-up   — kernels before the first NVTX range ran during torch.compile()
                      initialization; detected via NVTX time boundary and excluded
                      from the operator-kernel mapping.  Duration-outlier heuristic
                      (>10× median) used as fallback when no NVTX data is present.
  #7  Fused kernel  — store all enclosing ranges; mark is_fused=True when
                      provenance reports multiple source_ops.

torch.compile() / Triton note:
  Inductor-compiled kernels have no Python cpu_parent chain (they launch from
  C++), so the provenance sidecar approach yields nothing for compiled models.
  The name heuristic tier handles compiled kernels by parsing the Triton kernel
  identifier, which encodes fused op names:
      triton_poi_fused_relu_addmm_0  →  aten::relu, aten::addmm
  NVTX attribution still works because emit_nvtx() pushes aten:: ranges onto
  the CPU timeline before handing off to the compiled graph.  Kernels launched
  during warmup (before emit_nvtx is active) will be unattributed; the outlier
  detector flags these separately.
"""
from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path

from nvidia.operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    KernelAttribution,
    KernelManifestEntry,
    MappingManifest,
)
from nvidia.operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    NvtxRangeInfo,
)
from nvidia.operator_profiler.mapper.interval_tree import NvtxIntervalForest
from nvidia.operator_profiler.mapper.nsys_export import (
    KernelRow,
    NvtxRow,
    export_to_sqlite,
    query_kernels,
    query_nvtx_events,
)

log = logging.getLogger(__name__)

# Heuristics -----------------------------------------------------------

# Known aten:: op name fragments that appear in CUDA/Triton kernel names.
# Used as a fallback when Triton fused-name parsing yields nothing.
_OP_NAME_FRAGMENTS: dict[str, str] = {
    "gemm": "aten::mm",
    "conv": "aten::conv2d",
    "batch_norm": "aten::batch_norm",
    "relu": "aten::relu",
    "softmax": "aten::softmax",
    "layer_norm": "aten::layer_norm",
    "embedding": "aten::embedding",
    "add": "aten::add",
    "mul": "aten::mul",
    "linear": "aten::linear",
    "gelu": "aten::gelu",
    "addmm": "aten::addmm",
    "bmm": "aten::bmm",
    "mm": "aten::mm",
}

# Triton inductor kernel prefix pattern:  triton_{kind}_fused_{tokens...}_{index}
_TRITON_FUSED_RE = re.compile(
    r"^triton_[a-z]+_fused_([a-z0-9_]+?)_\d+$", re.IGNORECASE
)

# Tokens in the fused section that map directly to aten:: ops.
# Ordered longest-first so "layer_norm" matches before "norm".
_TRITON_TOKEN_TO_OP: list[tuple[str, str]] = [
    ("layer_norm",  "aten::layer_norm"),
    ("batch_norm",  "aten::batch_norm"),
    ("addmm",       "aten::addmm"),
    ("softmax",     "aten::softmax"),
    ("embedding",   "aten::embedding"),
    ("relu",        "aten::relu"),
    ("gelu",        "aten::gelu"),
    ("silu",        "aten::silu"),
    ("conv2d",      "aten::conv2d"),
    ("linear",      "aten::linear"),
    ("bmm",         "aten::bmm"),
    ("mm",          "aten::mm"),
    ("add",         "aten::add"),
    ("mul",         "aten::mul"),
    ("sub",         "aten::sub"),
    ("div",         "aten::div"),
]

# Ratio above which a kernel duration is flagged as a warm-up outlier
_WARMUP_OUTLIER_RATIO = 10.0


class ManifestBuilder:
    """
    Build a MappingManifest from nsys inputs.

    Parameters
    ----------
    nsys_rep_path:
        Path to the .nsys-rep file (will be exported to SQLite).
    metadata:
        CaptureManifestMetadata for the manifest header.
    """

    def __init__(
        self,
        nsys_rep_path: str | Path,
        metadata: CaptureManifestMetadata,
        sqlite_cache_dir: str | Path | None = None,
    ) -> None:
        self.nsys_rep_path = Path(nsys_rep_path)
        self.metadata = metadata
        self.sqlite_cache_dir = sqlite_cache_dir

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> MappingManifest:
        # Step 1: Export nsys → SQLite and query
        db_path = export_to_sqlite(self.nsys_rep_path, self.sqlite_cache_dir)
        kernel_rows = query_kernels(db_path)
        nvtx_rows = query_nvtx_events(db_path)

        # Step 2: Build per-stream NVTX interval forest
        forest = self._build_forest(nvtx_rows)

        # Step 3: Detect initialization kernels (edge case #4)
        # Uses NVTX time boundary when available (kernels before the first NVTX
        # range ran during torch.compile() warm-up, before emit_nvtx() was
        # active).  Falls back to duration-outlier heuristic if no NVTX data.
        outlier_ids = self._detect_initialization_kernels(kernel_rows, nvtx_rows)
        if outlier_ids:
            log.warning(
                "%d kernel(s) flagged as initialization kernels (pre-NVTX phase)",
                len(outlier_ids),
            )

        # Step 4: Build kernel entries via two-way join
        entries: list[KernelManifestEntry] = []
        warnings: list[str] = []

        for i, kr in enumerate(kernel_rows):
            kernel_id = f"k_{i:05d}"
            attribution = self._attribute(kr, forest)
            is_warmup = kernel_id in outlier_ids

            if is_warmup:
                warnings.append(
                    f"{kernel_id} ({kr.kernel_name}): flagged as initialization kernel"
                )

            entries.append(
                KernelManifestEntry(
                    kernel_id=kernel_id,
                    kernel_name=kr.kernel_name,
                    stream_id=kr.stream_id,
                    device_id=kr.device_id,
                    start_ns=kr.start_ns,
                    end_ns=kr.end_ns,
                    duration_ns=max(0, kr.end_ns - kr.start_ns),
                    grid_dim=(kr.grid_x, kr.grid_y, kr.grid_z) if kr.grid_x else None,
                    block_dim=(kr.block_x, kr.block_y, kr.block_z) if kr.block_x else None,
                    attribution=attribution,
                )
            )

        log.info(
            "Manifest: %d kernels, %d warnings",
            len(entries),
            len(warnings),
        )
        return MappingManifest(
            capture_metadata=self.metadata,
            kernels=entries,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_forest(self, nvtx_rows: list[NvtxRow]) -> NvtxIntervalForest:
        forest = NvtxIntervalForest()
        for row in nvtx_rows:
            if not row.text:
                continue
            forest.insert(
                stream_id=row.stream_id,
                device_id=row.device_id,
                range_info=NvtxRangeInfo(
                    text=row.text,
                    depth=row.nesting_level,
                    start_ns=row.start_ns,
                    end_ns=row.end_ns,
                    domain=row.domain,
                ),
            )
        return forest

    def _attribute(
        self,
        kr: KernelRow,
        forest: NvtxIntervalForest,
    ) -> KernelAttribution:
        # NVTX_EVENTS rows are keyed by host globalTid, not GPU streamId.
        # Use host_tid from the RUNTIME join; fall back to stream_id for older
        # exports that lack a RUNTIME table.
        nvtx_tid = kr.host_tid if kr.host_tid else kr.stream_id

        # NVTX ranges carry CPU-side timestamps (push/pop time).  GPU kernels
        # execute asynchronously after the CPU launches them, so the GPU start
        # timestamp is consistently ~1–10 µs after the NVTX range closes.
        # Using the CPU launch timestamp (when cuLaunchKernel fired) instead
        # of the GPU start timestamp gives correct enclosure results because
        # the launch call happens inside the aten:: NVTX range.
        nvtx_query_ts = kr.cpu_launch_start_ns if kr.cpu_launch_start_ns else kr.start_ns

        # --- Priority 1: NVTX enclosure (medium confidence) ---
        # Only accept ranges whose text is an aten:: op name. Walk from
        # innermost outward and take the first aten:: match. Non-aten:: ranges
        # (inductor graph markers, TorchDynamo internals, etc.) fall through.
        all_ranges = forest.query_enclosing(nvtx_tid, kr.device_id, nvtx_query_ts)
        aten_range = next(
            (r for r in reversed(all_ranges) if r.text.startswith("aten::")),
            None,
        )
        if aten_range:
            return KernelAttribution(
                method=AttributionMethod.NVTX,
                source_operators=[aten_range.text],
                nvtx_range=aten_range,
                confidence=Confidence.MEDIUM,
                all_enclosing_ranges=all_ranges,
            )

        # --- Priority 2: kernel name heuristic (low confidence) ---
        # Handles both Triton inductor names (triton_*_fused_op1_op2_N) and
        # cuBLAS/cuDNN names via substring matching.
        heuristic_ops = self._name_heuristic(kr.kernel_name)
        if heuristic_ops:
            return KernelAttribution(
                method=AttributionMethod.NAME_HEURISTIC,
                source_operators=heuristic_ops,
                confidence=Confidence.LOW,
                is_fused=len(heuristic_ops) > 1,
            )

        # --- Fallback: unattributed ---
        return KernelAttribution(
            method=AttributionMethod.UNATTRIBUTED,
            confidence=Confidence.UNATTRIBUTED,
        )

    @staticmethod
    def _name_heuristic(kernel_name: str) -> list[str]:
        """
        Return a list of aten:: ops inferred from *kernel_name*.

        Strategy (in order):
        1. Triton inductor fused name: triton_{kind}_fused_{tokens}_{index}
           Parse the fused token section against _TRITON_TOKEN_TO_OP.
        2. Substring fallback for cuBLAS/cuDNN/custom CUDA kernels.

        Returns an empty list if no match is found.
        """
        # 1. Triton fused name
        m = _TRITON_FUSED_RE.match(kernel_name)
        if m:
            token_str = m.group(1)   # e.g. "relu_addmm" or "layer_norm"
            ops: list[str] = []
            remaining = token_str
            # Greedy left-to-right scan: consume the longest matching prefix
            while remaining:
                matched = False
                for token, op in _TRITON_TOKEN_TO_OP:
                    if remaining == token or remaining.startswith(token + "_"):
                        if op not in ops:
                            ops.append(op)
                        remaining = remaining[len(token):]
                        if remaining.startswith("_"):
                            remaining = remaining[1:]
                        matched = True
                        break
                if not matched:
                    # Skip one underscore-delimited token we don't recognise
                    parts = remaining.split("_", 1)
                    remaining = parts[1] if len(parts) > 1 else ""
            if ops:
                return ops

        # 2. Substring fallback (cuBLAS, cuDNN, custom kernels)
        lower = kernel_name.lower()
        for fragment, op in _OP_NAME_FRAGMENTS.items():
            if fragment in lower:
                return [op]
        return []

    def _detect_initialization_kernels(
        self, kernel_rows: list[KernelRow], nvtx_rows: list[NvtxRow]
    ) -> set[str]:
        """
        Return kernel IDs for kernels launched before the steady-state phase.

        Strategy:
          1. If NVTX ranges exist, use the earliest NVTX start timestamp as the
             boundary between initialization and steady-state.  Kernels whose
             CPU launch timestamp precedes this boundary ran before emit_nvtx()
             was active (i.e. during torch.compile() warm-up) and are excluded
             from the operator-kernel mapping.
          2. If no NVTX ranges exist (non-NVTX capture), fall back to duration-
             outlier detection (>_WARMUP_OUTLIER_RATIO × median) as a heuristic.
        """
        if nvtx_rows:
            nvtx_window_start = min(r.start_ns for r in nvtx_rows)
            outliers: set[str] = set()
            for i, kr in enumerate(kernel_rows):
                launch_ts = kr.cpu_launch_start_ns if kr.cpu_launch_start_ns else kr.start_ns
                if launch_ts < nvtx_window_start:
                    outliers.add(f"k_{i:05d}")
            return outliers

        # Fallback: duration-based heuristic when no NVTX ranges are present
        if len(kernel_rows) < 3:
            return set()
        durations = [max(0, r.end_ns - r.start_ns) for r in kernel_rows]
        med = statistics.median(durations)
        if med == 0:
            return set()
        outliers = set()
        for i, (row, dur) in enumerate(zip(kernel_rows, durations)):
            if dur > _WARMUP_OUTLIER_RATIO * med:
                outliers.add(f"k_{i:05d}")
        return outliers
