"""
inductor_fusion_extractor — parse Inductor debug output_code.py artifacts.

Inductor writes output_code.py during torch.compile() when debug is enabled.
Each kernel call is preceded by a comment listing the fused aten ops:

    # Topologically Sorted Source Nodes: [...], Original ATen: [aten.relu, aten.addmm]
    triton_poi_fused_relu_addmm_0.run(...)

parse_inductor_debug_dir() walks a debug directory, extracts these pairs,
and returns {kernel_name: [aten::relu, aten::addmm]} — ground-truth fusion
metadata from the compiler, no heuristics involved.

Usage
-----
In run_workload.py (before torch.compile()):
    import os, torch._inductor.config as _ind_cfg
    _ind_cfg.debug = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(debug_dir)

After the nsys run:
    from nvidia.operator_profiler.capture.inductor_fusion_extractor import (
        parse_inductor_debug_dir,
    )
    fusion_map = parse_inductor_debug_dir(debug_dir)
    builder = ManifestBuilder(..., inductor_fusion_map=fusion_map)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from nvidia.operator_profiler.utils.op_namespaces import is_attributed_op

log = logging.getLogger(__name__)

_ORIG_ATEN_RE = re.compile(r"# .+Original ATen:\s*\[([^\]]+)\]")
_KERNEL_RUN_RE = re.compile(r"\b(\w+)\.run\(")
# PyTorch >= 2.x emits a stream-setup assignment between the Original ATen
# comment and the .run() call: "stream0 = get_raw_stream(0)".  This line
# must not reset pending_ops.
_STREAM_SETUP_RE = re.compile(r"^\s*\w+\s*=\s*get_raw_stream\(")


def _normalize_op(raw: str) -> str | None:
    """
    Convert Inductor op notation to aten:: form and filter non-kernel namespaces.

    Examples:
      "aten.relu"        → "aten::relu"
      "aten.add.Tensor"  → "aten::add"   (drop overload suffix)
      "prims.broadcast"  → None          (filtered by is_attributed_op)
    """
    raw = raw.strip()
    parts = raw.split(".", 1)
    if len(parts) < 2:
        return None
    ns_op = f"{parts[0]}::{parts[1].split('.')[0]}"
    return ns_op if is_attributed_op(ns_op) else None


def parse_inductor_debug_dir(debug_dir: str | Path) -> dict[str, list[str]]:
    """
    Walk debug_dir, find all output_code.py files, and return
    {kernel_short_name: [aten::op1, aten::op2, ...]} for every fused kernel.

    Multiple output_code.py files (one per compiled sub-graph) are merged.
    Kernels that appear in more than one file keep the first seen mapping.
    """
    debug_dir = Path(debug_dir)
    result: dict[str, list[str]] = {}

    if not debug_dir.exists():
        log.debug("Inductor debug dir does not exist: %s", debug_dir)
        return result

    code_files = list(debug_dir.rglob("output_code.py"))
    if not code_files:
        log.debug("No output_code.py files found in %s", debug_dir)
        return result

    for code_file in code_files:
        lines = code_file.read_text(encoding="utf-8", errors="replace").splitlines()
        pending_ops: list[str] | None = None

        for line in lines:
            m_aten = _ORIG_ATEN_RE.search(line)
            if m_aten:
                raw_ops = [s.strip() for s in m_aten.group(1).split(",")]
                ops = [o for r in raw_ops if (o := _normalize_op(r))]
                pending_ops = ops if ops else None
                continue

            if pending_ops is not None:
                m_run = _KERNEL_RUN_RE.search(line)
                if m_run:
                    kernel_name = m_run.group(1)
                    if kernel_name not in result:
                        result[kernel_name] = pending_ops
                    pending_ops = None
                elif _STREAM_SETUP_RE.search(line):
                    # Stream-setup assignment (e.g. "stream0 = get_raw_stream(0)")
                    # appears between the Original ATen comment and the .run()
                    # call in PyTorch >= 2.x generated code; skip without reset.
                    pass
                elif line.strip() and not line.strip().startswith("#"):
                    # Non-comment, non-empty line without a .run() call —
                    # the comment preceded an extern_kernel or other non-Triton
                    # call; discard and reset.
                    pending_ops = None

    log.info(
        "Inductor fusion map: %d kernel→ops entries from %d output_code.py file(s)",
        len(result),
        len(code_files),
    )
    return result
