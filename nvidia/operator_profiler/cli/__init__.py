"""CLI entry point — delegates to subcommand modules."""
import argparse
import logging
import sys
from pathlib import Path

from . import manifest_cmd, map_cmd, profile_cmd

# Ensure repo root is on sys.path so preflight can import nvidia.*
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nvidia.scripts.preflight import check_all as _preflight  # noqa: E402


def main() -> None:
    _preflight(label="operator-profiler")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="operator-profiler",
        description="Profile GPU workloads and attribute hardware metrics to PyTorch operators.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_cmd.add_parser(subparsers)
    manifest_cmd.add_parser(subparsers)
    map_cmd.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
