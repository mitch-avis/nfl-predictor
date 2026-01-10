#!/usr/bin/env python
"""
Live validation by comparing the latest completed week against the schedule.

This loads data/all_data.csv and pulls the latest season schedule via nflreadpy.
Network access may be required depending on the nflreadpy backend.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from nfl_predictor import constants
    from nfl_predictor.utils import validation_utils
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script: `python scripts/validate_live.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.utils import validation_utils


def main() -> int:
    """Run live validation and report any score mismatches."""
    data_path = Path(constants.DATA_PATH) / "all_data.csv"
    if not data_path.exists():
        print(f"Missing data file: {data_path}")
        return 2

    df = pl.read_csv(data_path)
    mismatches = validation_utils.compare_latest_week_scores(df)

    if mismatches.height == 0:
        print("No mismatches found")
        return 0

    print("Score mismatches detected:")
    print(mismatches)
    return 1


if __name__ == "__main__":
    main()
    main()
