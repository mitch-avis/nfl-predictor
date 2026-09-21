#!/usr/bin/env python
"""Live validation by comparing the latest completed week against the schedule.

This loads data/all_data.csv and pulls the latest season schedule via nflreadpy.
Network access may be required depending on the nflreadpy backend.
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
    # Allow running as a script: `python scripts/validate_live.py`.
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
    """Run live validation and report any score mismatches.

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
    mismatches = validation_utils.compare_latest_week_scores(df)

    if mismatches.height == 0:
        log.info("No mismatches found")
        return 0

    log.error("Score mismatches detected:")
    log.error("%s", mismatches)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
