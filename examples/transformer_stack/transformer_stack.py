"""
transformer_stack.py — GPT-2-style stack of 8 identical transformer layers.

Architecture: 8 TransformerLayer blocks (attention + FFN + pre-norm LayerNorm).
Representative of a mid-size LLM decoder at inference time.

Each layer:
    LayerNorm → Multi-head self-attention (Q/K/V projections + output projection)
    LayerNorm → Feed-forward (fc_up → GELU → fc_down)

Config: hidden=512, heads=8, ffn_dim=2048, seq_len=128, batch=4

Kernel types exercised:
  gemm (8 linear projections × 8 layers), layer_norm (× 2 × 8 layers),
  softmax, gelu

Deduplication opportunity:
  All 8 layers are structurally identical. With --layer-deduplicate, ncu
  profiles only layer_0 and propagates metrics to layers 1-7, giving ~8×
  speedup on the ncu replay step.

Profile size estimate: ~80–100 operators, ~20–30 min ncu without dedup,
  ~3–5 min with --layer-deduplicate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
HIDDEN   = 512
N_HEADS  = 8
FFN_DIM  = 2048
N_LAYERS = 8
HEAD_DIM = HIDDEN // N_HEADS


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj   = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj   = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj   = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_up   = nn.Linear(HIDDEN, FFN_DIM, bias=True)
        self.fc_down = nn.Linear(FFN_DIM, HIDDEN, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_down(F.gelu(self.fc_up(x)))


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(HIDDEN)
        self.attn = SelfAttention()
        self.ln2  = nn.LayerNorm(HIDDEN)
        self.ff   = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(N_LAYERS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def get_model_and_input() -> tuple:
    """Return (uncompiled model on CUDA, input tensor on CUDA)."""
    assert torch.cuda.is_available(), "CUDA required"
    model = TransformerStack().to(DEVICE).eval()
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
