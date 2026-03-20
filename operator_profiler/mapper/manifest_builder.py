"""
Manifest builder — the three-way join at the heart of the Operator Mapper Stage.

Sources joined:
  1. nsys SQLite export  → CUDA kernel rows + NVTX event rows
  2. Inductor provenance JSONL sidecar → kernel_name → {source_ops, locations}
  3. Per-stream NVTX interval tree → innermost NVTX range enclosing each kernel

Produces: MappingManifest

Confidence scoring:
  provenance hit     → method=provenance,      confidence=high
  NVTX enclosure     → method=nvtx,            confidence=medium
  name substring     → method=name_heuristic,  confidence=low
  no match           → method=unattributed,    confidence=unattributed

Edge cases handled here:
  #1  Clock domain  — we never join on absolute timestamps across tools.
  #3  Multi-stream  — per-stream interval trees; match only on same (stream, device).
  #4  JIT warm-up   — detect outlier durations (>10× median), emit warning.
  #7  Fused kernel  — store all enclosing ranges; mark is_fused=True when
                      provenance reports multiple source_ops.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    KernelAttribution,
    KernelManifestEntry,
    MappingManifest,
)
from operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    NvtxRangeInfo,
    SourceLocation,
)
from operator_profiler.mapper.interval_tree import NvtxIntervalForest
from operator_profiler.mapper.nsys_export import (
    KernelRow,
    NvtxRow,
    export_to_sqlite,
    query_kernels,
    query_nvtx_events,
)

log = logging.getLogger(__name__)

# Heuristics -----------------------------------------------------------

# Known aten:: op name fragments that appear in CUDA/Triton kernel names
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
}

# Ratio above which a kernel duration is flagged as a warm-up outlier
_WARMUP_OUTLIER_RATIO = 10.0


class ManifestBuilder:
    """
    Build a MappingManifest from nsys + provenance inputs.

    Parameters
    ----------
    nsys_rep_path:
        Path to the .nsys-rep file (will be exported to SQLite).
    provenance_jsonl_path:
        Path to the INDUCTOR_PROVENANCE=1 sidecar (optional — None for eager mode).
    metadata:
        CaptureManifestMetadata for the manifest header.
    """

    def __init__(
        self,
        nsys_rep_path: str | Path,
        metadata: CaptureManifestMetadata,
        provenance_jsonl_path: str | Path | None = None,
        sqlite_cache_dir: str | Path | None = None,
    ) -> None:
        self.nsys_rep_path = Path(nsys_rep_path)
        self.provenance_jsonl_path = (
            Path(provenance_jsonl_path) if provenance_jsonl_path else None
        )
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

        # Step 3: Load provenance sidecar
        provenance = self._load_provenance()

        # Step 4: Detect warm-up outliers (edge case #4)
        outlier_ids = self._detect_warmup_outliers(kernel_rows)
        if outlier_ids:
            log.warning(
                "%d kernel(s) flagged as warm-up outliers (duration >%g× median)",
                len(outlier_ids),
                _WARMUP_OUTLIER_RATIO,
            )

        # Step 5: Build kernel entries via three-way join
        entries: list[KernelManifestEntry] = []
        warnings: list[str] = []

        for i, kr in enumerate(kernel_rows):
            kernel_id = f"k_{i:05d}"
            attribution = self._attribute(kr, forest, provenance)
            is_warmup = kernel_id in outlier_ids

            if is_warmup:
                warnings.append(
                    f"{kernel_id} ({kr.kernel_name}): flagged as warm-up outlier"
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

    def _load_provenance(self) -> dict[str, dict]:
        """Return kernel_name → {source_ops, source_locations} from JSONL."""
        result: dict[str, dict] = {}
        if self.provenance_jsonl_path is None:
            return result
        if not self.provenance_jsonl_path.exists():
            log.warning("Provenance file not found: %s", self.provenance_jsonl_path)
            return result

        with open(self.provenance_jsonl_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    kernel_name = record.get("generated_kernel_name", "")
                    if kernel_name:
                        result[kernel_name] = record
                except json.JSONDecodeError as exc:
                    log.warning("Provenance JSONL parse error at line %d: %s", lineno, exc)

        log.info("Loaded provenance for %d kernels", len(result))
        return result

    def _attribute(
        self,
        kr: KernelRow,
        forest: NvtxIntervalForest,
        provenance: dict[str, dict],
    ) -> KernelAttribution:
        # --- Priority 1: provenance sidecar (high confidence) ---
        prov = provenance.get(kr.kernel_name)
        if prov:
            source_ops: list[str] = prov.get("source_ops", [])
            raw_locs: list[dict] = prov.get("source_locations", [])
            locs = [
                SourceLocation(
                    file=loc.get("file", ""),
                    line=loc.get("line", 0),
                    col=loc.get("col"),
                    op=loc.get("op", source_ops[0] if source_ops else ""),
                )
                for loc in raw_locs
            ]
            all_ranges = forest.query_enclosing(kr.stream_id, kr.device_id, kr.start_ns)
            innermost = all_ranges[-1] if all_ranges else None
            return KernelAttribution(
                method=AttributionMethod.PROVENANCE,
                source_operators=source_ops,
                source_locations=locs,
                nvtx_range=innermost,
                confidence=Confidence.HIGH,
                is_fused=len(source_ops) > 1,
                all_enclosing_ranges=all_ranges,
            )

        # --- Priority 2: NVTX enclosure (medium confidence) ---
        all_ranges = forest.query_enclosing(kr.stream_id, kr.device_id, kr.start_ns)
        innermost = all_ranges[-1] if all_ranges else None
        if innermost:
            return KernelAttribution(
                method=AttributionMethod.NVTX,
                source_operators=[innermost.text],
                nvtx_range=innermost,
                confidence=Confidence.MEDIUM,
                all_enclosing_ranges=all_ranges,
            )

        # --- Priority 3: kernel name substring heuristic (low confidence) ---
        heuristic_op = self._name_heuristic(kr.kernel_name)
        if heuristic_op:
            return KernelAttribution(
                method=AttributionMethod.NAME_HEURISTIC,
                source_operators=[heuristic_op],
                confidence=Confidence.LOW,
            )

        # --- Fallback: unattributed ---
        return KernelAttribution(
            method=AttributionMethod.UNATTRIBUTED,
            confidence=Confidence.UNATTRIBUTED,
        )

    @staticmethod
    def _name_heuristic(kernel_name: str) -> str | None:
        lower = kernel_name.lower()
        for fragment, op in _OP_NAME_FRAGMENTS.items():
            if fragment in lower:
                return op
        return None

    def _detect_warmup_outliers(self, kernel_rows: list[KernelRow]) -> set[str]:
        """
        Return kernel IDs whose duration is >_WARMUP_OUTLIER_RATIO × the
        median duration (edge case #4: JIT warm-up inflation).
        """
        if len(kernel_rows) < 3:
            return set()
        durations = [max(0, r.end_ns - r.start_ns) for r in kernel_rows]
        med = statistics.median(durations)
        if med == 0:
            return set()
        outliers: set[str] = set()
        for i, (row, dur) in enumerate(zip(kernel_rows, durations)):
            if dur > _WARMUP_OUTLIER_RATIO * med:
                outliers.add(f"k_{i:05d}")
        return outliers
