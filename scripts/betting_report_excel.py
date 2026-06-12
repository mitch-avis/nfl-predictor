#!/usr/bin/env python
"""Generate an Excel betting template from predictions.

This produces a `.xlsx` workbook with:
- model outputs (probabilities, predicted margin/total)
- blank/live input columns for sportsbook moneylines/spread/total
- formulas that recompute implied probabilities, no-vig probabilities, edges,
  and action labels.

Note: writing `.xlsb` directly is not supported here. If you need `.xlsb`,
open the generated `.xlsx` in Excel and use "Save As" -> `.xlsb`.

Example:
  python scripts/betting_report_excel.py \
    --predictions models/<run_id>/predictions.csv \
    --out data/predict/week_19_betting_template.xlsx

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from nfl_predictor.reporting.betting_excel import write_betting_template_xlsx
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor.reporting.betting_excel import write_betting_template_xlsx
    from nfl_predictor.utils.logger import log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an Excel betting template")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.csv produced by training/prediction.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .xlsx path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the betting report Excel generator."""
    args = _parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {args.predictions}")

    df = pd.read_csv(args.predictions)
    write_betting_template_xlsx(predictions=df, out_path=args.out)
    log.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
