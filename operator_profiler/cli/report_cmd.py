"""
operator-profiler report <profile.json>

Prints a summary of the operator-attributed profile to stdout.

Usage:
    operator-profiler report profile.json [--top N] [--sort duration|dram]
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "report",
        help="Print a summary of an operator-attributed profile.",
    )
    p.add_argument("profile", help="Path to operator_attributed_profile.json")
    p.add_argument("--top", type=int, default=20, help="Top N operators to show")
    p.add_argument(
        "--sort",
        choices=["duration", "dram_read", "dram_write", "kernel_count"],
        default="duration",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    from operator_profiler.schema.profile import OperatorAttributedProfile

    path = Path(args.profile)
    profile = OperatorAttributedProfile.model_validate_json(path.read_text())

    meta = profile.capture_metadata
    print(f"\n{'='*70}")
    print(f"  Operator-Attributed Profile")
    print(f"  Model:   {meta.model_name}")
    print(f"  Torch:   {meta.torch_version}")
    print(f"  Device:  {meta.device_name or 'unknown'}")
    print(f"  Mode:    {meta.compile_mode}")
    print(f"  Captured: {meta.capture_timestamp_utc}")
    print(f"{'='*70}\n")

    ops = [op for op in profile.operators if op.aggregated is not None]

    # Sort
    sort_key = {
        "duration": lambda o: o.aggregated.total_duration_ns,
        "dram_read": lambda o: o.aggregated.total_dram_bytes_read,
        "dram_write": lambda o: o.aggregated.total_dram_bytes_written,
        "kernel_count": lambda o: o.aggregated.kernel_count,
    }[args.sort]

    ops.sort(key=sort_key, reverse=True)
    top_ops = ops[: args.top]

    total_ns = sum(o.aggregated.total_duration_ns for o in ops) or 1

    print(
        f"{'Rank':<5} {'Operator':<40} {'Duration ms':>12} {'% Total':>8} "
        f"{'Kernels':>8} {'Bottleneck':<16} {'Confidence'}"
    )
    print("-" * 105)

    for rank, op in enumerate(top_ops, 1):
        agg = op.aggregated
        dur_ms = agg.total_duration_ns / 1_000_000
        pct = 100.0 * agg.total_duration_ns / total_ns
        conf = (
            op.kernels[0].confidence.value if op.kernels else "n/a"
        )
        fused_marker = " [fused]" if op.is_fused else ""
        print(
            f"{rank:<5} {op.operator_name + fused_marker:<40} {dur_ms:>12.3f} {pct:>8.1f}% "
            f"{agg.kernel_count:>8} {agg.bottleneck_classification:<16} {conf}"
        )

    if profile.unattributed_kernels:
        print(
            f"\n  Unattributed kernels: {len(profile.unattributed_kernels)} "
            f"(see profile JSON for details)"
        )

    if profile.warnings:
        print(f"\n  Warnings ({len(profile.warnings)}):")
        for w in profile.warnings[:10]:
            print(f"    - {w}")
        if len(profile.warnings) > 10:
            print(f"    ... ({len(profile.warnings) - 10} more)")

    print()
