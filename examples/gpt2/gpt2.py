"""
gpt2.py — GPT-2 small (117M) HuggingFace transformer workload.

Architecture: 12 identical transformer decoder blocks, each with:
    LayerNorm → Multi-head causal self-attention (Q/K/V + output projections)
    LayerNorm → Feed-forward (fc_up → GELU → fc_down)
Plus token + positional embeddings and a final LayerNorm.

Config (GPT-2 small): hidden=768, heads=12, ffn_dim=3072, n_layers=12
Workload config: batch=4, seq_len=128

Kernel types exercised:
  gemm (4 attention projections × 12 layers + 2 FFN projections × 12 layers),
  layer_norm (× 2 × 12 layers + final), scaled_dot_product_attention,
  gelu, embedding lookup

Deduplication opportunity:
  All 12 transformer blocks (h.0 … h.11) are structurally identical.
  With layer deduplication, ncu profiles only h.0 and propagates metrics
  to h.1–h.11, giving ~12× speedup on the ncu replay step — the primary
  motivation for this example.

Profile size estimate: ~120–150 operators, ~30–45 min ncu without dedup,
  ~3–5 min with --layer-deduplicate.

Requires:
  pip install transformers

To profile:
    python scripts/run_workload.py examples/gpt2/gpt2.py \\
        --warmup-iters 3 --measure-iters 10
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEVICE    = "cuda"
BATCH     = 4
SEQ_LEN   = 128
MODEL_ID  = "gpt2"


class GPT2Wrapper(nn.Module):
    """Thin wrapper so model(input_ids) returns the last hidden state tensor."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


def get_model_and_input() -> tuple:
    """Return (uncompiled model on CUDA, input_ids tensor on CUDA).

    Downloads GPT-2 weights from HuggingFace on first call (~500 MB).
    Subsequent calls use the local cache.
    """
    assert torch.cuda.is_available(), "CUDA required"

    from transformers import GPT2Model  # imported here to keep top-level import-free

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval()

    # Random token ids in [0, vocab_size)
    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    return model, input_ids


if __name__ == "__main__":
    model, input_ids = get_model_and_input()
    with torch.no_grad():
        y = model(input_ids)
    print(f"Output shape: {y.shape}")  # expect (4, 128, 768)
