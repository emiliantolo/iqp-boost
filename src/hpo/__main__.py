"""Command-line entrypoint for HPO runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.hpo.study import run_hpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for IQP boosting.")
    parser.add_argument("--config", required=True, help="Path to HPO config JSON/TOML.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_hpo(Path(args.config))


if __name__ == "__main__":
    main()
