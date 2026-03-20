"""CLI entry point — delegates to subcommand modules."""
import argparse
import logging
import sys

from . import map_cmd, profile_cmd, report_cmd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="operator-profiler",
        description="Correlate GPU hardware metrics with PyTorch operators.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_cmd.add_parser(subparsers)
    map_cmd.add_parser(subparsers)
    report_cmd.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
