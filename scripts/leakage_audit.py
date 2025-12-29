#!/usr/bin/env python
"""Leakage audit runner.

Runs a lightweight leakage audit on a CSV dataset and writes a structured JSON report.

Example:
    python scripts/leakage_audit.py \
      --data-path data/completed_games_ml.csv \
      --out-json models/leakage_audit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nfl_predictor.ml import leakage_audit  # noqa: E402, pylint: disable=wrong-import-position
from nfl_predictor.utils.logger import log  # noqa: E402, pylint: disable=wrong-import-position


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage audit on an ML dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to a completed-games ML CSV (e.g., data/completed_games_ml.csv).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--feature-start",
        default=leakage_audit.LeakageAuditConfig.feature_start,
        help="First column in feature range (inclusive).",
    )
    parser.add_argument(
        "--feature-end",
        default=leakage_audit.LeakageAuditConfig.feature_end,
        help="Last column in feature range (inclusive).",
    )
    parser.add_argument(
        "--include-market",
        action="store_true",
        default=True,
        help="Include market columns as features when present.",
    )
    parser.add_argument(
        "--exclude-market",
        action="store_false",
        dest="include_market",
        help="Exclude market columns from the feature set.",
    )
    parser.add_argument(
        "--market-transform",
        action="store_true",
        default=False,
        help="Enable market-derived columns (market_home_margin, etc).",
    )
    parser.add_argument(
        "--max-cardinality-ratio",
        type=float,
        default=0.5,
        help="Drop categoricals with unique_ratio >= threshold.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the leakage audit and write a JSON report."""
    args = _parse_args()
    if not args.data_path.exists():
        log.error("Missing data file: %s", args.data_path)
        return 2

    df = pd.read_csv(args.data_path)
    config = leakage_audit.LeakageAuditConfig(
        feature_start=str(args.feature_start),
        feature_end=str(args.feature_end),
        include_market=bool(args.include_market),
        market_transform=bool(args.market_transform),
        max_cardinality_ratio=float(args.max_cardinality_ratio),
    )

    report = leakage_audit.run_leakage_audit(df, config)
    leakage_audit.write_report(report, args.out_json)

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
