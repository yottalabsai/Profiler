"""
emit_nvtx() wrapper with warm-up logic.

Provides a context manager that:
  1. Runs ≥2 warm-up iterations before capture (edge case #4: JIT warm-up inflation).
  2. Wraps the capture iteration in torch.autograd.profiler.emit_nvtx().

Usage:
    with NvtxCapture(warmup_iters=2) as ctx:
        output = model(input_tensor)
"""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.autograd.profiler as _profiler
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class NvtxCapture:
    """
    Context manager for NVTX-annotated PyTorch profiling.

    Parameters
    ----------
    warmup_iters:
        Number of warm-up iterations before the annotated capture run.
        Must be ≥ 2 to avoid JIT warm-up inflation (edge case #4).
    warmup_fn:
        Optional callable for warm-up iterations.  If None, the user is
        responsible for running warm-up before entering the context.
    record_shapes:
        Passed to emit_nvtx().
    """

    def __init__(
        self,
        warmup_iters: int = 2,
        warmup_fn: Callable | None = None,
        record_shapes: bool = True,
    ) -> None:
        if warmup_iters < 2:
            log.warning(
                "warmup_iters=%d is less than 2 — JIT warm-up inflation may "
                "distort kernel durations (edge case #4)",
                warmup_iters,
            )
        self.warmup_iters = warmup_iters
        self.warmup_fn = warmup_fn
        self.record_shapes = record_shapes
        self._ctx = None

    def __enter__(self) -> "NvtxCapture":
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed.  Install it with: pip install torch"
            )
        # Run warm-up iterations to let Triton/inductor finish compilation
        if self.warmup_fn is not None:
            log.info("Running %d warm-up iteration(s)", self.warmup_iters)
            for i in range(self.warmup_iters):
                self.warmup_fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self._ctx = torch.autograd.profiler.emit_nvtx(record_shapes=self.record_shapes)
        self._ctx.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*args)
