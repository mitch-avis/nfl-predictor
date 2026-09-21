#!/usr/bin/env python
"""Offline validation for the latest collected dataset.

Runs schema, range, and consistency checks against data/all_data.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

try:
    from nfl_predictor import constants
    from nfl_predictor.utils import validation_utils
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:
    # Allow running as a script: `python scripts/validate_offline.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.utils import validation_utils
    from nfl_predictor.utils.logger import log


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line for this script.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed arguments, with ``data_dir`` resolved to a directory.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(constants.DATA_PATH),
        help="Directory holding the collected datasets (default: the packaged data directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run offline validation and report any issues.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.

    """
    args = _parse_args(argv)
    data_path = args.data_dir / "all_data.csv"
    if not data_path.exists():
        log.error("Missing data file: %s", data_path)
        return 2

    df = pl.read_csv(data_path)
    result = validation_utils.validate_dataframe(df)

    if result.errors:
        log.error("Errors:")
        for err in result.errors:
            log.error("- %s", err)
    if result.warnings:
        log.warning("Warnings:")
        for warn in result.warnings:
            log.warning("- %s", warn)

    if result.errors:
        return 1

    log.info("Validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
