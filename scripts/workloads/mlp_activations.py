"""
mlp_activations.py — Deep MLP with four different activation functions.

Architecture: four Linear layers, each followed by a different activation,
chosen to span the full spectrum of compute cost and arithmetic intensity.

    Linear(512 → 2048) → ReLU
    Linear(2048 → 2048) → GELU
    Linear(2048 → 2048) → SiLU
    Linear(2048 → 512)  → Tanh

Kernel types exercised:
  linear (GEMM), relu, gelu, silu (sigmoid-weighted linear), tanh

Bottleneck profile expected (inductor):
  - Linear layers: compute-bound (large GEMM, high arithmetic intensity)
  - ReLU: memory-bound (single comparison, trivial compute)
  - GELU / SiLU: memory-bound but compute-heavier than ReLU (polynomial /
    sigmoid approximation adds ALU work per element)
  - Tanh: memory-bound and slowest activation (special-function unit pressure)

This workload is useful for showing that operator choice within the same
layer type (activation functions) can meaningfully change the memory-to-
compute ratio of a block. All Linear layers are compute-bound while
activation functions span from trivial (ReLU) to expensive (Tanh).

Profile size estimate: ~20–25 operators, ~200 KB, ~3–5 min ncu time.

To profile:
    operator-profiler profile scripts/workloads/mlp_activations.py \\
        --model-name MLPActivations --compile-mode inductor \\
        --output runs/mlp_activations
    operator-profiler map runs/mlp_activations.manifest.json \\
        --script scripts/workloads/mlp_activations.py \\
        --output runs/mlp_activations_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE     = "cuda"
BATCH_SIZE = 256
DIM_IN     = 512
DIM_HIDDEN = 2048
DIM_OUT    = 512


class MLPActivations(nn.Module):
    """
    Four-layer MLP with heterogeneous activations.

    Layer 1: Linear + ReLU     — near-zero extra compute after GEMM
    Layer 2: Linear + GELU     — ~4 ops/element (erf approximation)
    Layer 3: Linear + SiLU     — sigmoid × input, fused in Triton
    Layer 4: Linear + Tanh     — special-function unit pressure
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN,     DIM_HIDDEN, bias=True)
        self.fc2 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc3 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc4 = nn.Linear(DIM_HIDDEN, DIM_OUT,    bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = MLPActivations().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    print(model(x).shape)
