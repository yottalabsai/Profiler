"""
Shared helper for identifying op names that directly dispatch NeuronCore operations.

Identical to nvidia/operator_profiler/utils/op_namespaces.py — the aten:: namespace
logic is the same regardless of backend hardware.

Accepted:
  aten::       — core ATen dispatch
  quantized::  — INT8/INT4 quantized ops
  <any>        — torch.library custom ops

Rejected:
  prims::      — PrimTorch intermediate ops; decompose before dispatch
  torch::      — higher-level wrappers that delegate to aten::
  <no "::">    — non-op annotations
"""
from __future__ import annotations

import re

_NAMESPACE_OP_RE = re.compile(r"^[A-Za-z_]\w*::[A-Za-z_]\w*")

_NON_KERNEL_NAMESPACES: frozenset[str] = frozenset({"prims", "torch"})


def is_attributed_op(text: str) -> bool:
    """Return True if *text* is a kernel-dispatching op name (namespace::op)."""
    if not _NAMESPACE_OP_RE.match(text):
        return False
    namespace = text[: text.index("::")]
    return namespace not in _NON_KERNEL_NAMESPACES
