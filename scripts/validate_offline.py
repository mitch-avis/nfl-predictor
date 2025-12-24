#!/usr/bin/env python
"""
Offline validation for the latest collected dataset.

Runs schema, range, and consistency checks against data/all_data.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import validation_utils

ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def main() -> int:
    """Run offline validation and report any issues."""
    _ensure_repo_root_on_path()

    data_path = Path(constants.DATA_PATH) / "all_data.csv"
    if not data_path.exists():
        print(f"Missing data file: {data_path}")
        return 2

    df = pl.read_csv(data_path)
    result = validation_utils.validate_dataframe(df)

    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"- {err}")
    if result.warnings:
        print("Warnings:")
        for warn in result.warnings:
            print(f"- {warn}")

    if result.errors:
        return 1

    print("Validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
