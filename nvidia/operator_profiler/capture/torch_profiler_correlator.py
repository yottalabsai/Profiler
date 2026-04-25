"""
torch_profiler_correlator — build a kernel-name → aten-op attribution map.

Runs the workload under torch.profiler (CPU + CUDA activities), exports the
Chrome trace, then parses it using per-thread timestamp enclosure to find
which innermost aten:: RecordFunction scope was active when each CUDA kernel
was launched.

Returns {(kernel_name, nth_occurrence): aten_op_name}.

Join strategy
-------------
Kernel names are stable across runs of the same compiled model on the same
input.  By running this pass for the same number of iterations as the nsys
capture, the (name, nth_occurrence) index aligns exactly — matching the
invocation-order join used for ncu replay in kernel_profiler.py.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

from nvidia.operator_profiler.utils.op_namespaces import is_attributed_op

log = logging.getLogger(__name__)

# Match "void " or "<type> " prefix, then optional "ns::ns::" namespace prefix
_STRIP_RETTYPE = re.compile(r"^(?:[a-zA-Z_]\w*\s+)+")
_STRIP_NS      = re.compile(r"^(?:[a-zA-Z_]\w*::)+")
_STRIP_SUFFIX  = re.compile(r"[<(].*")


def _short_kernel_name(full_name: str) -> str:
    """
    Extract the nsys short name from a fully-demangled kernel name.

    Examples:
      "void gemmSN_TN_kernel<float,...>(...)" → "gemmSN_TN_kernel"
      "void cutlass::Kernel2<...>(...)"       → "Kernel2"
      "sm80_xmma_gemm_..._cublas"             → "sm80_xmma_gemm_..._cublas"
      "triton_per_fused_native_layer_norm_0"  → "triton_per_fused_native_layer_norm_0"
    """
    name = _STRIP_RETTYPE.sub("", full_name)
    name = _STRIP_NS.sub("", name)
    name = _STRIP_SUFFIX.sub("", name)
    return name.strip() or full_name


def build_correlation_map(
    workload_fn: Callable[[], None],
    n_iters: int = 1,
    trace_path: Path | None = None,
) -> dict[tuple[str, int], str]:
    """
    Run workload_fn n_iters times under torch.profiler.
    Return {(kernel_name, nth_occurrence): aten_op_name}.

    n_iters must equal --measure-iters used for the nsys capture so that
    invocation indices align across the two runs (same compiled kernel launch
    order, same input).
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as e:
        raise RuntimeError("torch is required for build_correlation_map") from e

    delete_trace = trace_path is None
    if trace_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        trace_path = Path(tmp.name)

    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            record_shapes=False,
        ) as prof:
            with torch.no_grad():
                for _ in range(n_iters):
                    workload_fn()

        prof.export_chrome_trace(str(trace_path))
        log.info("torch.profiler trace written to %s", trace_path)
        return _parse_chrome_trace(trace_path)
    finally:
        if delete_trace:
            try:
                trace_path.unlink(missing_ok=True)
            except Exception:
                pass


def _parse_chrome_trace(trace_path: Path) -> dict[tuple[str, int], str]:
    """
    Parse a Chrome trace JSON and return {(kernel_name, nth_occurrence): op_name}.

    torch.profiler emits:
      - "cpu_op" events with args["External id"] = N, carrying the RecordFunction name
      - "kernel" events (GPU side) with args["External id"] = N, same value

    The "External id" is the direct link — no timestamp enclosure needed.
    Kernels where the cpu_op name is an aten:: op are returned; others are skipped
    so that NVTX / name-heuristic tiers can handle them.

    Kernels are processed in GPU-timestamp order to ensure nth_occurrence counts
    match the launch order seen by nsys.
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # -----------------------------------------------------------------------
    # Step 1: build External id → cpu_op name
    # -----------------------------------------------------------------------
    ext_id_to_op: dict[int, str] = {}
    for ev in events:
        if ev.get("cat") != "cpu_op":
            continue
        ext_id = ev.get("args", {}).get("External id")
        name = ev.get("name", "")
        if ext_id is not None and name:
            ext_id_to_op[int(ext_id)] = name

    # -----------------------------------------------------------------------
    # Step 2: collect kernel events sorted by GPU timestamp
    # torch.profiler uses "cat": "kernel" for GPU kernel activity
    # -----------------------------------------------------------------------
    kernel_events = [
        ev for ev in events
        if ev.get("cat") == "kernel" and ev.get("name")
    ]
    kernel_events.sort(key=lambda e: e.get("ts", 0))

    # -----------------------------------------------------------------------
    # Step 3: for each kernel, resolve op via External id
    # -----------------------------------------------------------------------
    name_counter: dict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], str] = {}

    for ev in kernel_events:
        full_name = ev.get("name", "")
        # nsys stores the short (undecorated) name; torch.profiler emits the full
        # demangled name.  Normalise to the short form so the key matches nsys.
        kernel_name = _short_kernel_name(full_name)
        ext_id = ev.get("args", {}).get("External id")
        if ext_id is None:
            name_counter[kernel_name] += 1
            continue

        op_name = ext_id_to_op.get(int(ext_id), "")
        idx = name_counter[kernel_name]
        name_counter[kernel_name] += 1

        # Only store entries for kernel-dispatching op namespaces (aten::,
        # quantized::, torch.library custom ops).  Non-kernel namespaces and
        # non-op cpu_op names fall through to NVTX / name-heuristic tiers.
        if is_attributed_op(op_name):
            result[(kernel_name, idx)] = op_name

    log.info(
        "Parsed correlation map: %d aten:: kernel→op entries (%d unique kernel names)",
        len(result),
        len(name_counter),
    )
    return result
