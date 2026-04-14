"""
workload.py — Reference workload for the profiling pipeline.

Exposes the workload interface used by run_workload.py and the pipeline tools:

    def get_model_and_input() -> tuple[model, input_tensor]:
        ...

The returned model is an uncompiled nn.Module on CUDA. Compilation and warmup
are handled by run_workload.py via --compile-backend and --warmup-iters.

The model is a TransformerBlock (attention + FFN + layer norm), representative
of LLM inference and covering a broad range of kernel types:
  gemm, softmax, layer_norm, relu, gelu.

To profile this workload:
    python scripts/run_workload.py --workload scripts/workload.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEVICE      = "cuda"
BATCH_SIZE  = 16
IN_FEATURES = 512
HIDDEN      = 2048


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
        self.q_proj   = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.v_proj   = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.out_proj = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.relu(self.q_proj(x))
        v = self.v_proj(x)
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


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = TransformerBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE)
    return model, x
