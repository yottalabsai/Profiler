"""
inductor_workload.py — Inductor-mode workload for provenance sidecar testing.

When INDUCTOR_PROVENANCE=1 is set this script:
  1. Compiles the model with torch.compile(backend="inductor")
  2. Warms up to trigger all Triton JIT compilation
  3. Runs torch.profiler.profile() to collect actual CUDA kernel names
  4. Walks the cpu_parent chain to find source aten:: ops per kernel
  5. Writes a provenance JSONL to INDUCTOR_PROVENANCE_OUTPUT

The nsys-measured iterations use emit_nvtx so the trace carries both
aten:: NVTX ranges and CUDA kernel launches, exercising the full
PROVENANCE > NVTX > heuristic attribution priority stack.

Run via:
    INDUCTOR_PROVENANCE=1 \\
    INDUCTOR_PROVENANCE_OUTPUT=profile.provenance.jsonl \\
    nsys profile --trace=cuda,nvtx --output=<prefix> \\
    python scripts/inductor_workload.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.autograd.profiler as autograd_profiler
from torch.profiler import profile as TorchProfile, ProfilerActivity

DEVICE      = "cuda"
BATCH_SIZE  = 4
IN_FEATURES = 256
HIDDEN      = 1024
WARMUP      = 5
MEASURE     = 10


class FFBlock(nn.Module):
    """Feed-forward block: Linear → ReLU → Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=True)
        self.fc2 = nn.Linear(HIDDEN, IN_FEATURES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _normalize_to_short_name(name: str) -> str:
    """
    Convert a torch.profiler CUDA event name to the nsys shortName format.

    nsys stores the bare function identifier in CUPTI_ACTIVITY_KIND_KERNEL.shortName
    (no 'void' prefix, no template parameters, no namespace qualifiers), while
    torch.profiler exposes the full demangled name.  Examples:

      'void gemmSN_TN_kernel<float, 128, …>'  →  'gemmSN_TN_kernel'
      'triton_poi_fused_addmm_relu_0'          →  'triton_poi_fused_addmm_relu_0'
    """
    # Strip leading 'void '
    name = re.sub(r"^void\s+", "", name)
    # Take everything before the first template '<' or argument '('
    base = re.split(r"[<(]", name)[0].rstrip()
    # If the remaining string is namespace-qualified, keep only the last component
    if "::" in base:
        parts = [p for p in base.split("::") if p]
        return parts[-1] if parts else base
    return base


def _walk_cpu_parent_for_aten(evt) -> str | None:
    """
    Walk the cpu_parent chain of a profiler FunctionEvent upward and return
    the first aten:: name found, or None.
    """
    cursor = getattr(evt, "cpu_parent", None)
    while cursor is not None:
        name = getattr(cursor, "name", "") or ""
        if name.startswith("aten::"):
            return name
        cursor = getattr(cursor, "cpu_parent", None)
    return None


def _collect_kernel_provenance(model, x: torch.Tensor) -> dict[str, list[str]]:
    """
    Run one forward pass inside torch.profiler.profile and return a mapping
    of CUDA kernel name → list of source aten:: op names.

    The kernel names recorded by torch.profiler are identical to the
    shortName values stored by nsys in CUPTI_ACTIVITY_KIND_KERNEL, so the
    dict keys match exactly what _attribute() looks up via
    provenance.get(kr.kernel_name).
    """
    kernel_to_ops: dict[str, list[str]] = {}

    with TorchProfile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    for evt in prof.events():
        if evt.device_type != torch.profiler.DeviceType.CUDA:
            continue
        kname = _normalize_to_short_name(evt.name)
        if not kname:
            continue

        source_op = _walk_cpu_parent_for_aten(evt)
        if kname not in kernel_to_ops:
            kernel_to_ops[kname] = []
        if source_op and source_op not in kernel_to_ops[kname]:
            kernel_to_ops[kname].append(source_op)

    return kernel_to_ops


def _write_provenance_jsonl(
    kernel_to_ops: dict[str, list[str]],
    output_path: str | Path,
) -> int:
    """
    Write provenance JSONL in the format expected by
    ManifestBuilder._load_provenance().

    Returns the number of entries written.
    """
    output_path = Path(output_path)
    lines: list[str] = []

    for kname, source_ops in kernel_to_ops.items():
        if not kname:
            continue
        # Fall back to "unknown" if the cpu_parent chain had no aten:: op.
        # The kernel name still matches, so _attribute() will fire PROVENANCE.
        effective_ops = source_ops if source_ops else ["unknown"]
        locs = [
            {"file": "inductor_workload.py", "line": 43, "col": 4, "op": op}
            for op in effective_ops
        ]
        lines.append(json.dumps({
            "generated_kernel_name": kname,
            "source_ops": effective_ops,
            "source_locations": locs,
        }))

    content = "\n".join(lines) + ("\n" if lines else "")
    output_path.write_text(content, encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance-only", action="store_true",
        help="Collect provenance JSONL and exit — skip the NVTX capture phase. "
             "Run this WITHOUT nsys to avoid CUPTI conflicts, then run the full "
             "script under nsys (without INDUCTOR_PROVENANCE) for the trace.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "compile"):
        print(
            "[inductor_workload] ERROR: torch.compile not available — "
            "upgrade to PyTorch >= 2.0",
            flush=True,
        )
        sys.exit(1)

    assert torch.cuda.is_available(), "CUDA required"
    print(f"[inductor_workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    do_provenance = args.provenance_only or os.environ.get("INDUCTOR_PROVENANCE", "0") == "1"
    prov_output   = os.environ.get(
        "INDUCTOR_PROVENANCE_OUTPUT", "inductor_workload.provenance.jsonl"
    )

    model = FFBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE)

    # ------------------------------------------------------------------
    # Phase 1: compile
    # ------------------------------------------------------------------
    print("[inductor_workload] Compiling with inductor...", flush=True)
    compiled_model = torch.compile(model, backend="inductor")

    # ------------------------------------------------------------------
    # Phase 2: warmup — triggers all Triton JIT compilation
    # ------------------------------------------------------------------
    print(f"[inductor_workload] Warmup ({WARMUP} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = compiled_model(x)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Phase 3: provenance — collect Triton kernel names via profiler
    # ------------------------------------------------------------------
    if do_provenance:
        print("[inductor_workload] Collecting kernel provenance...", flush=True)
        kernel_to_ops = _collect_kernel_provenance(compiled_model, x)
        n_written = _write_provenance_jsonl(kernel_to_ops, prov_output)
        print(
            f"[inductor_workload] Provenance: {n_written} kernel entries "
            f"→ {prov_output}",
            flush=True,
        )
        if n_written == 0:
            print(
                "[inductor_workload] WARNING: no CUDA kernel events captured — "
                "torch.profiler may not have observed Triton kernel launches",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Phase 4: NVTX-annotated capture for nsys
    # ------------------------------------------------------------------
    if args.provenance_only:
        print("[inductor_workload] --provenance-only: skipping NVTX capture.", flush=True)
        return

    print(f"[inductor_workload] Capture ({MEASURE} iters with emit_nvtx)...",
          flush=True)
    with torch.no_grad():
        with autograd_profiler.emit_nvtx(record_shapes=True):
            for _ in range(MEASURE):
                _ = compiled_model(x)
    torch.cuda.synchronize()

    print("[inductor_workload] Done.", flush=True)


if __name__ == "__main__":
    main()
