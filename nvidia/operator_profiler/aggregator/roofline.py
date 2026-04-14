"""
Roofline analysis helpers.

Computes arithmetic intensity and plots operator positions on the roofline
model to classify bottlenecks.

References:
  Williams et al., "Roofline: An Insightful Visual Performance Model for
  Floating-Point Programs and Multiprocessor Systems", CACM 2009.
"""
from __future__ import annotations

from dataclasses import dataclass

from nvidia.operator_profiler.schema.profile import AggregatedMetrics, KernelMetrics


@dataclass
class RooflinePoint:
    operator_id: str
    arithmetic_intensity: float   # FLOP / byte
    achieved_gflops: float
    peak_compute_gflops: float
    peak_bandwidth_gbs: float
    bottleneck: str               # "compute_bound" | "memory_bound"
    efficiency_pct: float         # % of roofline ceiling achieved


def compute_arithmetic_intensity(flops: float, dram_bytes: int) -> float | None:
    """FLOP/byte.  Returns None if dram_bytes is zero (avoid division by zero)."""
    if dram_bytes <= 0:
        return None
    return flops / dram_bytes


def classify_roofline(
    arithmetic_intensity: float,
    peak_compute_gflops: float,
    peak_bandwidth_gbs: float,
) -> tuple[str, float]:
    """
    Determine roofline bottleneck and ceiling.

    Returns (bottleneck_str, ceiling_gflops).
    The ridge point = peak_compute / peak_bandwidth.
    """
    ridge_point = peak_compute_gflops / peak_bandwidth_gbs
    if arithmetic_intensity >= ridge_point:
        return "compute_bound", peak_compute_gflops
    else:
        return "memory_bound", arithmetic_intensity * peak_bandwidth_gbs


def roofline_efficiency(achieved_gflops: float, ceiling_gflops: float) -> float:
    """Return achieved / ceiling as a percentage (0–100)."""
    if ceiling_gflops <= 0:
        return 0.0
    return min(100.0, 100.0 * achieved_gflops / ceiling_gflops)


