"""
Shared helper for identifying op names that directly dispatch CUDA kernels.

Used by ManifestBuilder (NVTX tier) and TorchProfilerCorrelator (torch.profiler
tier) to decide which op namespaces are valid attribution targets.

Accepted:
  aten::       — core ATen dispatch (vast majority of kernels)
  quantized::  — INT8/INT4 quantized ops
  <any>        — torch.library custom ops (flash_attn::, xformers::, etc.)

Rejected:
  prims::      — PrimTorch intermediate ops; decompose before kernel launch
  torch::      — higher-level wrappers that delegate to aten:: dispatch
  <no "::">    — non-op NVTX annotations (ProfilerStep#0, TorchDynamo text, etc.)
"""
from __future__ import annotations

import re

_NAMESPACE_OP_RE = re.compile(r"^[A-Za-z_]\w*::[A-Za-z_]\w*")

# Namespaces that appear in RecordFunction / emit_nvtx but do NOT directly
# dispatch CUDA kernels.  Accepting these would misattribute a kernel to the
# wrong (inner) NVTX range when they nest inside an aten:: range.
_NON_KERNEL_NAMESPACES: frozenset[str] = frozenset({"prims", "torch"})


def is_attributed_op(text: str) -> bool:
    """Return True if *text* is a kernel-dispatching op name (namespace::op).

    Accepts aten::, quantized::, and any torch.library custom namespace.
    Rejects prims:: and torch:: (no direct kernel dispatch) and any text
    that does not follow the namespace::op_name pattern.
    """
    if not _NAMESPACE_OP_RE.match(text):
        return False
    namespace = text[: text.index("::")]
    return namespace not in _NON_KERNEL_NAMESPACES
