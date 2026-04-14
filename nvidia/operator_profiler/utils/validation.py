"""
Input shape validation for replay (edge case #6: dynamic shapes / kernel count mismatch).

If the model was captured with certain input shapes and the replay script
produces different shapes, Triton may compile different kernels, making the
(kernel_name, launch_index) join between nsys and ncu incorrect.

This module validates that replay shapes match the shapes recorded in the
mapping manifest metadata before any ncu replay is started.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class ShapeMismatchError(ValueError):
    """
    Raised when replay input shapes differ from manifest-recorded shapes.

    Proceeding with mismatched shapes would cause kernel count mismatches
    between the nsys capture and the ncu replay (edge case #6).
    """


def validate_input_shapes(
    replay_shapes: dict[str, list[int]],
    manifest_shapes: dict[str, list[int]],
    strict: bool = True,
) -> None:
    """
    Validate that *replay_shapes* match *manifest_shapes*.

    Parameters
    ----------
    replay_shapes:
        Input shapes as they will be used in the ncu replay run.
    manifest_shapes:
        Input shapes recorded in the mapping manifest metadata at capture time.
    strict:
        If True (default), raise ShapeMismatchError on any mismatch.
        If False, log a warning instead.

    Raises
    ------
    ShapeMismatchError if strict=True and shapes differ.
    """
    if not manifest_shapes:
        # No shapes recorded — can't validate; emit a notice
        log.info(
            "No input shapes recorded in manifest — cannot validate replay shapes. "
            "Set TORCH_COMPILE_CACHE_SIZE=0 to prevent stale kernel caches."
        )
        return

    mismatches: list[str] = []

    for name, expected in manifest_shapes.items():
        actual = replay_shapes.get(name)
        if actual is None:
            mismatches.append(f"  {name}: expected {expected}, not provided in replay")
        elif actual != expected:
            mismatches.append(f"  {name}: expected {expected}, got {actual}")

    for name in replay_shapes:
        if name not in manifest_shapes:
            mismatches.append(f"  {name}: present in replay but not in manifest")

    if mismatches:
        msg = (
            "Input shape mismatch between nsys capture and ncu replay — "
            "this will cause kernel count mismatches (edge case #6):\n"
            + "\n".join(mismatches)
            + "\nFix: lock input shapes and set TORCH_COMPILE_CACHE_SIZE=0."
        )
        if strict:
            raise ShapeMismatchError(msg)
        else:
            log.warning(msg)
    else:
        log.debug("Input shape validation passed (%d shapes checked)", len(manifest_shapes))
