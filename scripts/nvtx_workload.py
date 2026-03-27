"""
nvtx_workload.py — Workload script meant to be run UNDER nsys profile.

Runs a transformer-style FFBlock with torch.autograd.profiler.emit_nvtx()
so that every aten:: dispatch emits an NVTX range that the operator profiler
manifest_builder can match via NVTX enclosure attribution.

Schema contract:
  - emit_nvtx() writes NVTX_EVENTS rows with text = "aten::<op_name> [shapes]"
  - eventType 59 = NvtxRangeStart/End pairs (the common aten:: ranges)
  - globalTid is used as the stream_id proxy (no per-stream GPU timestamps in NVTX)

Run via:
    nsys profile --trace=cuda,nvtx --output=<path> python scripts/nvtx_workload.py
    nsys export --type sqlite --output=<path>.sqlite <path>.nsys-rep
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

DEVICE      = "cuda"
BATCH_SIZE  = 16
IN_FEATURES = 512
HIDDEN      = 2048
WARMUP      = 5
MEASURE     = 20


class FFBlock(nn.Module):
    """Transformer feed-forward block: Linear → ReLU → Linear → GELU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=True)
        self.fc2 = nn.Linear(HIDDEN, IN_FEATURES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.fc2(torch.relu(self.fc1(x))))


class AttentionBlock(nn.Module):
    """Single-head attention projection pair for GEMM coverage."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.v_proj = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.out_proj = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.relu(self.q_proj(x))
        v = self.v_proj(x)
        # Scaled dot-product (simplified, no mask)
        scores = torch.softmax(q @ v.transpose(-1, -2) / (IN_FEATURES ** 0.5), dim=-1)
        return self.out_proj(scores @ v)


class TransformerBlock(nn.Module):
    """Attention + FFN with layer-norm — representative of LLM inference."""
    def __init__(self):
        super().__init__()
        self.attn = AttentionBlock()
        self.ff   = FFBlock()
        self.ln1  = nn.LayerNorm(IN_FEATURES)
        self.ln2  = nn.LayerNorm(IN_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device_name = torch.cuda.get_device_name(0)
    print(f"[nvtx_workload] GPU: {device_name}", flush=True)

    model = TransformerBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE)

    # ------------------------------------------------------------------ warmup
    # Warm up without NVTX so JIT/Triton compilation does not appear in trace
    print(f"[nvtx_workload] Warmup ({WARMUP} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ capture
    # emit_nvtx wraps every aten:: dispatch with an NVTX range so nsys records:
    #   NVTX_EVENTS.text = "aten::addmm [...]"
    #   NVTX_EVENTS.start, end  (nanoseconds, CUPTI GPU clock domain)
    #   NVTX_EVENTS.globalTid   (thread id used as stream proxy)
    print(f"[nvtx_workload] Capture ({MEASURE} iters with emit_nvtx)...", flush=True)
    with torch.no_grad():
        with profiler.emit_nvtx(record_shapes=True):
            for _ in range(MEASURE):
                _ = model(x)
    torch.cuda.synchronize()

    print("[nvtx_workload] Done.", flush=True)


if __name__ == "__main__":
    main()
