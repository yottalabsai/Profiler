"""
embedding_projection.py — Embedding lookup + projection head workload.

Architecture: token embedding table lookup followed by a two-layer projection
head and a final logit layer. Representative of the input/output stages of an
LLM or encoder-only model (BERT, GPT-2 embedding + unembedding).

    nn.Embedding(32000, 512)       — gather from large table
    LayerNorm(512)
    Linear(512  → 2048) → GELU
    Linear(2048 → 512)
    Linear(512  → 32000)           — logit projection to vocabulary

Kernel types exercised:
  embedding (index-based gather), layer_norm, linear (GEMM), gelu

Bottleneck profile expected (inductor):
  - Embedding lookup: pure memory-bandwidth-bound. The table is
    32000 × 512 × 2 bytes ≈ 32 MB (fp16). Each forward pass reads a
    scattered subset of rows — irregular access pattern, no compute.
    Arithmetic intensity ≈ 0 FLOPs / byte; sits at the far-left of the
    roofline, far below the ridge point.
  - LayerNorm: memory-bound (reduction over 512 elements per token).
  - Linear(512→2048) + GELU: compute-bound (GEMM shape is
    (B·T, 512) × (512, 2048) — tensor-core friendly).
  - Linear(512→32000): the dominant operator by FLOP count. Shape is
    (B·T, 512) × (512, 32000) — a very wide GEMM. With B=64, T=128 this
    is 64·128 = 8192 rows × 32000 columns. This single matrix multiply
    can represent >80% of total FLOPs in language model decoding.

This workload is the canonical example of extreme imbalance: one operator
(embedding) is pure bandwidth with zero compute, while the logit projection
dominates FLOPs — opposite ends of the roofline within the same model.

Profile size estimate: ~20–30 operators, ~200–300 KB, ~3–5 min ncu time.

To profile:
    operator-profiler profile scripts/workloads/embedding_projection.py \\
        --model-name EmbeddingProjection --compile-mode inductor \\
        --output runs/embedding_projection
    operator-profiler map runs/embedding_projection.manifest.json \\
        --script scripts/workloads/embedding_projection.py \\
        --output runs/embedding_projection_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE     = "cuda"
BATCH_SIZE = 64
SEQ_LEN    = 128
VOCAB_SIZE = 32_000
DIM        = 512
DIM_FF     = 2048


class EmbeddingProjection(nn.Module):
    """
    Token embedding lookup + two-layer projection + logit head.

    The embedding table (32000 × 512) is ≈ 32 MB in fp16. The logit
    projection (512 × 32000) mirrors it as a large, compute-heavy GEMM.
    """
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, DIM)
        self.ln      = nn.LayerNorm(DIM)
        self.proj1   = nn.Linear(DIM,    DIM_FF, bias=True)
        self.proj2   = nn.Linear(DIM_FF, DIM,    bias=True)
        self.logits  = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) — integer indices
        x = self.embed(token_ids)          # (B, T, DIM)
        x = self.ln(x)
        x = F.gelu(self.proj1(x))          # (B, T, DIM_FF)
        x = self.proj2(x)                  # (B, T, DIM)
        return self.logits(x)              # (B, T, VOCAB_SIZE)


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.

    Input is integer token IDs in [0, VOCAB_SIZE).
    """
    assert torch.cuda.is_available(), "CUDA required"
    model     = EmbeddingProjection().to(DEVICE).eval()
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    return model, token_ids


if __name__ == "__main__":
    model, token_ids = get_model_and_input()
    print(model(token_ids).shape)
