"""
conv_block.py — VGG-style convolutional pipeline workload.

Architecture: three Conv2d → BatchNorm → ReLU stages with pooling, followed
by a linear classifier head. Representative of early CNN backbones (VGG, AlexNet).

Kernel types exercised:
  conv2d (cudnn/implicit GEMM), batch_norm, relu, max_pool2d,
  adaptive_avg_pool2d, linear (GEMM)

Bottleneck profile expected (inductor):
  - Conv2d stages: shift from memory-bound (early, small spatial maps) to
    compute-bound (middle, many channels) as depth increases
  - BatchNorm + ReLU: memory-bound (simple element-wise passes)
  - MaxPool, AdaptiveAvgPool: memory-bound (reduction, no compute)
  - Linear head: compute-bound (small GEMM)

This workload is useful for demonstrating that the same operator class
(Conv2d) can land in different bottleneck regimes depending on problem size —
a key teaching point for roofline analysis.

Profile size estimate: ~20–25 operators, ~200 KB, ~3–5 min ncu time.

To profile:
    operator-profiler profile scripts/workloads/conv_block.py \\
        --model-name ConvBlock --compile-mode inductor \\
        --output runs/conv_block
    operator-profiler map runs/conv_block.manifest.json \\
        --script scripts/workloads/conv_block.py \\
        --output runs/conv_block_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEVICE     = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT      = 64
WIDTH       = 64
NUM_CLASSES = 10


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    """
    Three-stage VGG-style conv pipeline.

    Stage 1: 3 → 64  channels, 64×64 spatial  (memory-bound conv)
    Stage 2: 64 → 128 channels, 32×32 spatial  (transitional)
    Stage 3: 128 → 256 channels, 16×16 spatial (compute-bound conv)
    Then: AdaptiveAvgPool → Linear classifier
    """
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBnRelu(3,   64,  kernel_size=3)
        self.stage2 = nn.Sequential(
            ConvBnRelu(64,  128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ConvBnRelu(128, 256, kernel_size=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = ConvBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    print(model(x).shape)
