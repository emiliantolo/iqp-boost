"""Compatibility shim for the new :mod:`src.hpo` package."""

from __future__ import annotations

from src.hpo.__main__ import build_parser, main
from src.hpo.study import run_hpo

__all__ = ["build_parser", "main", "run_hpo"]


if __name__ == "__main__":
    main()
