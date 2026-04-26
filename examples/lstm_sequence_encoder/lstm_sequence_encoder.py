"""
lstm_sequence_encoder.py — Stacked LSTM sequence encoder with classification head.

Architecture: two-layer LSTM followed by mean-pool temporal reduction and a
linear classifier. Representative of sequence classification workloads
(sentiment analysis, time-series labeling, sequence tagging).

    nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)
    mean-pool over time dimension                    — (B, T, H) → (B, H)
    nn.Linear(512 → NUM_CLASSES)

Kernel types exercised:
  lstm_cell (cuDNN fused RNN or inductor-unrolled GEMM + sigmoid + tanh),
  sigmoid (input/forget/output gates), tanh (cell gate, hidden state),
  element-wise mul (gate application), mean (temporal reduction), linear (GEMM)

Bottleneck profile expected (inductor):
  - LSTM computation: without compilation, the full two-layer forward pass
    dispatches to a single cudnnRNNForwardInference kernel — highly optimized
    but opaque to per-operator attribution. With torch.compile + inductor the
    RNN is decomposed into per-timestep Triton fused kernels (gate GEMM +
    sigmoid/tanh + element-wise), making individual gate operations visible.
  - Gate GEMMs: compute-bound. Each timestep multiplies
    (B, INPUT_SIZE + HIDDEN_SIZE) × (INPUT_SIZE + HIDDEN_SIZE, 4 × HIDDEN_SIZE)
    for all four gates simultaneously — the dominant FLOP source.
  - sigmoid / tanh activations: memory-bound element-wise ops. In the
    inductor-unrolled path these are the dominant non-GEMM operators and sit
    near the bandwidth ceiling on the roofline.
  - Mean pooling: memory-bound temporal reduction over SEQ_LEN steps.
  - Linear head: small compute-bound GEMM.

Key teaching point: unlike most examples where torch.compile fuses ops into
fewer kernels, LSTM under inductor may *decompose* the cuDNN fused RNN kernel
into constituent ops. The profile reveals whether the gate GEMMs (compute-bound)
or the sigmoid/tanh activations (memory-bound) dominate after unrolling —
and whether inductor re-fuses them into a single Triton kernel per timestep.

Profile size estimate: ~30–50 operators (inductor-unrolled), ~300–500 KB, ~5–8 min ncu time.

To profile:
    operator-profiler profile examples/lstm_sequence_encoder/lstm_sequence_encoder.py \\
        --model-name LSTMSequenceEncoder --compile-mode inductor \\
        --output runs/lstm_sequence_encoder
    operator-profiler map runs/lstm_sequence_encoder.manifest.json \\
        --script examples/lstm_sequence_encoder/lstm_sequence_encoder.py \\
        --output runs/lstm_sequence_encoder_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEVICE      = "cuda"
BATCH_SIZE  = 32
SEQ_LEN     = 128
INPUT_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 2
NUM_CLASSES = 10


class LSTMSequenceEncoder(nn.Module):
    """
    Stacked 2-layer LSTM with mean-pool temporal reduction and linear classifier.

    Input:  (B, T, INPUT_SIZE)  float tensor
    Output: (B, NUM_CLASSES)    logit tensor
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, T, HIDDEN_SIZE)
        pooled = out.mean(dim=1)        # (B, HIDDEN_SIZE)
        return self.classifier(pooled)  # (B, NUM_CLASSES)


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    print(model(x).shape)
