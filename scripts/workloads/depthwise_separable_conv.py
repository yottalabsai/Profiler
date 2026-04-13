"""
depthwise_separable_conv.py — MobileNet-style depthwise-separable conv blocks.

Architecture: three depthwise-separable convolution blocks with channel doubling,
representative of MobileNetV1/V2 feature extraction.

Each block:
    DepthwiseConv2d (groups=C_in, 3×3) → BatchNorm → ReLU6
    PointwiseConv2d (1×1)              → BatchNorm → ReLU6

Channel progression:  32 → 64 → 128 → 256

Kernel types exercised:
  depthwise_conv2d (memory-bound), pointwise_conv2d (compute-bound),
  batch_norm, relu6 (clamped relu)

Bottleneck profile expected (inductor):
  - Depthwise Conv2d: memory-bound. Each output channel has only one input
    channel; arithmetic intensity is very low (≈ 9 FLOPs per element with
    3×3 kernel). Hardware sits near the bandwidth ceiling of the roofline.
  - Pointwise Conv2d (1×1): compute-bound. Equivalent to a batched GEMM
    with shape (B·H·W, C_in) × (C_in, C_out). Tensor cores engage; this
    is the same regime as a fully-connected layer.
  - BatchNorm + ReLU6: memory-bound.

This is a classic roofline teaching example: the depthwise and pointwise
convolutions in the same block land on opposite sides of the ridge point.
The profiler will surface this contrast in bottleneck_classification.

Profile size estimate: ~30–40 operators, ~300–400 KB, ~5–8 min ncu time.

To profile:
    operator-profiler profile scripts/workloads/depthwise_separable_conv.py \\
        --model-name DepthwiseSepConv --compile-mode inductor \\
        --output runs/depthwise_sep_conv
    operator-profiler map runs/depthwise_sep_conv.manifest.json \\
        --script scripts/workloads/depthwise_separable_conv.py \\
        --output runs/depthwise_sep_conv_profile.json
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEVICE     = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 32
HEIGHT      = 56
WIDTH       = 56


class DWSepBlock(nn.Module):
    """
    Depthwise-separable convolution block.

    Depthwise:  groups=in_ch, kernel 3×3 — each channel filtered independently
    Pointwise:  1×1 conv, projects to out_ch — mixes channel information
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise  = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False
        )
        self.bn_dw      = nn.BatchNorm2d(in_ch)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw      = nn.BatchNorm2d(out_ch)
        self.act        = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.act(self.bn_pw(self.pointwise(x)))
        return x


class DepthwiseSepConv(nn.Module):
    """
    Three stacked DWSepBlocks with channel doubling (32→64→128→256).
    Spatial size is kept constant at 56×56 to isolate channel-depth effects.
    """
    def __init__(self):
        super().__init__()
        self.block1 = DWSepBlock(32,  64)
        self.block2 = DWSepBlock(64,  128)
        self.block3 = DWSepBlock(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model on CUDA. Compilation and warmup
    are handled externally by run_workload.py.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = DepthwiseSepConv().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    print(model(x).shape)
