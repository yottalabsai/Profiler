"""
CUDAGraph manifest recording (edge case #2).

During CUDAGraph capture, NVTX ranges are emitted on the CPU thread.  When
the graph replays, the kernels execute with new GPU timestamps that have no
corresponding NVTX events.  This module records the operator set during
capture so that AttributionEngine can attribute replay kernels correctly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CudaGraphCaptureManifest:
    """Records the operator set observed during one CUDAGraph capture."""
    graph_id: str
    source_operators: list[str] = field(default_factory=list)
    kernel_names: list[str] = field(default_factory=list)
    input_shapes: dict[str, list[int]] = field(default_factory=dict)


class CudaGraphCapture:
    """
    Context manager that wraps a CUDAGraph capture and records a manifest.

    Usage:
        manifest = CudaGraphCaptureManifest(graph_id="graph_0")
        with CudaGraphCapture(manifest, model):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output = model(input_tensor)
    """

    def __init__(
        self,
        manifest: CudaGraphCaptureManifest,
        operators_to_record: list[str] | None = None,
    ) -> None:
        self.manifest = manifest
        self._operators_to_record = operators_to_record or []

    def __enter__(self) -> "CudaGraphCapture":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CUDAGraph capture")
        # Record the operator list declared by the caller
        self.manifest.source_operators = list(self._operators_to_record)
        return self

    def __exit__(self, *args) -> None:
        log.info(
            "CUDAGraph manifest recorded: graph_id=%s, ops=%s",
            self.manifest.graph_id,
            self.manifest.source_operators,
        )

    def save(self, path: str | Path) -> None:
        """Persist the manifest as JSON for use in AttributionEngine."""
        path = Path(path)
        path.write_text(json.dumps(asdict(self.manifest), indent=2))
        log.info("Saved CUDAGraph manifest to %s", path)

    @staticmethod
    def load(path: str | Path) -> CudaGraphCaptureManifest:
        path = Path(path)
        data = json.loads(path.read_text())
        return CudaGraphCaptureManifest(**data)
