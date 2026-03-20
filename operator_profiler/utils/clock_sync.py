"""
GPU/CPU clock domain helpers.

The fundamental rule (edge case #1): NEVER join on absolute timestamps
across nsys and ncu.  nsys uses CUPTI GPU-domain timestamps; ncu uses its own
replay-time timestamps.  There is no shared epoch between them.

This module provides utilities to detect clock domain issues and document the
correct join strategy: (kernel_name, launch_index_within_range).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Documentation constants — used to make the constraint explicit in code
# ---------------------------------------------------------------------------

NSYS_TIMESTAMP_DOMAIN = "CUPTI_GPU"
"""
nsys timestamps in CUPTI_ACTIVITY_KIND_KERNEL rows are in the GPU clock domain,
synchronized with the host by nsys at session start.  Use these for NVTX
enclosure queries only.
"""

NCU_TIMESTAMP_DOMAIN = "NCU_REPLAY"
"""
ncu replay timestamps are internal to each replay run.  They share no epoch
with nsys and MUST NOT be used for NVTX enclosure queries or ordering vs. nsys.
"""

SAFE_JOIN_KEY = "(kernel_name, launch_index_within_range)"
"""
The only safe join key between nsys and ncu outputs.  Kernels within a given
NVTX range are launched in a deterministic order (provided input shapes are
locked), so launch index is stable across the nsys capture and the ncu replay.
"""


def warn_if_timestamp_join_attempted(
    source_a: str, source_b: str, join_key: str
) -> None:
    """
    Emit a warning if a timestamp-based join is attempted across tool boundaries.

    Call this before any join that involves timestamps from both nsys and ncu.
    """
    if "ns" in join_key.lower() and source_a != source_b:
        log.warning(
            "Potential cross-tool timestamp join detected: %s × %s on key '%s'. "
            "Timestamps from different tools share no common epoch — use '%s' instead.",
            source_a,
            source_b,
            join_key,
            SAFE_JOIN_KEY,
        )


def gpu_ns_to_us(ns: int) -> float:
    """Convert GPU nanoseconds to microseconds."""
    return ns / 1_000.0


def gpu_ns_to_ms(ns: int) -> float:
    """Convert GPU nanoseconds to milliseconds."""
    return ns / 1_000_000.0
