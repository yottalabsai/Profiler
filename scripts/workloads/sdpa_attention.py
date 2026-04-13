"""
sdpa_attention.py — Multi-head self-attention via scaled_dot_product_attention.

Architecture: multi-head self-attention block using
torch.nn.functional.scaled_dot_product_attention (SDPA), which dispatches to
FlashAttention on Ampere/Hopper GPUs. Contrasts with the manually-decomposed
attention in transformer_block.py.

    QKV projections (fused or 3× Linear)
    F.scaled_dot_product_attention  → FlashAttention kernel
    Output projection
    Pre/post LayerNorm

Kernel types exercised:
  linear (GEMM), flash_attention (fused QK^T V), layer_norm

Bottleneck profile expected (inductor):
  - QKV and output Linear: compute-bound (large GEMM)
  - SDPA / FlashAttention: memory-bandwidth-bound for short sequences,
    compute-bound for long sequences (quadratic attention score matrix)
  - LayerNorm: memory-bound

Key teaching point: with inductor compilation the three separate QKV linear
layers are often fused into a single batched GEMM, and the softmax + masking
+ matmul in SDPA become a single FlashAttention kernel. The profiler will
attribute this single fused kernel back to the aten::scaled_dot_product_attention
NVTX range, demonstrating the "medium confidence / NVTX enclosure" attribution
path and the effect of operator fusion on the operator count.

Profile size estimate: ~20–30 operators, ~200–300 KB, ~4–6 min ncu time.

To profile:
    operator-profiler profile scripts/workloads/sdpa_attention.py \\
        --model-name SDPAAttention --compile-mode inductor \\
        --output runs/sdpa_attention
    operator-profiler map runs/sdpa_attention.manifest.json \\
        --script scripts/workloads/sdpa_attention.py \\
        --output runs/sdpa_attention_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE     = "cuda"
BATCH_SIZE = 8
SEQ_LEN    = 512
DIM        = 512
NUM_HEADS  = 8
HEAD_DIM   = DIM // NUM_HEADS   # 64


class SDPAAttentionBlock(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention.

    Uses separate Q, K, V projections (no bias) so the three linear layers are
    visible as distinct NVTX ranges. With inductor they may be fused.
    """
    def __init__(self):
        super().__init__()
        self.q_proj   = nn.Linear(DIM, DIM, bias=False)
        self.k_proj   = nn.Linear(DIM, DIM, bias=False)
        self.v_proj   = nn.Linear(DIM, DIM, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)
        self.ln_pre   = nn.LayerNorm(DIM)
        self.ln_post  = nn.LayerNorm(DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.ln_pre(x)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # Dispatches to FlashAttention on Ampere / Hopper when available.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        out = self.ln_post(out + residual)
        return out


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = SDPAAttentionBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    print(model(x).shape)
