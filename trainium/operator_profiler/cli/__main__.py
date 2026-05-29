"""
Entry point: python -m trainium.operator_profiler [command] [args]

Commands:
    profile     Run capture + attribution + aggregation → profile.json
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="trainium-profiler",
        description="Trainium operator-attributed profiling pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from trainium.operator_profiler.cli.profile_cmd import add_parser as add_profile
    add_profile(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
