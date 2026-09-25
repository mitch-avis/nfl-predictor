"""Validate the latest collected dataset, data/all_data.csv.

By default this runs the offline schema, range, and consistency checks. With --live it instead
compares the latest completed week's scores against the season schedule from nflreadpy, which
may need network access depending on the nflreadpy backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import validation_utils
from nfl_predictor.utils.logger import log


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line for this command.

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
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compare the latest completed week's scores with the schedule instead.",
    )
    return parser.parse_args(argv)


def _validate_offline(df: pl.DataFrame) -> int:
    """Run the offline checks and report any issues; return the exit code."""
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


def _validate_live(df: pl.DataFrame) -> int:
    """Compare the latest completed week with the schedule; return the exit code."""
    mismatches = validation_utils.compare_latest_week_scores(df)

    if mismatches.height == 0:
        log.info("No mismatches found")
        return 0

    log.error("Score mismatches detected:")
    log.error("%s", mismatches)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the offline checks, or the live schedule comparison with ``--live``.

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
    if args.live:
        return _validate_live(df)
    return _validate_offline(df)


if __name__ == "__main__":
    raise SystemExit(main())
