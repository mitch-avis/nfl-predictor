"""Check the own Elo columns a build publishes against the evaluation's numbers.

``integration_feature_values.py`` writes ``games.parquet``, every 1999-2025 game row as
``nfl-predictor data`` publishes it. From the commit that wires the own Elo in, those rows carry
both presets' eight columns. This script reports, for one such run (read-only):

- how many rows carry each column and its range;
- on the evaluation's window (completed regular-season games of 2007-2025, a tie is not a home
  win, ``.agents/sos_port/ELO_EVAL.md``), the games, the home wins and the Brier score of each
  preset's published home win probability, which must equal the evaluation's preset numbers;
- the build's own Elo record (``own_elo_metadata.json`` in the run directory): the win-total
  file it read and the win-total preset's fitted weights.

Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_own_elo_check.py \
        --run-dir /tmp/step6_int/values2/<label>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants

FIRST_WINDOW_SEASON = 2007
PROBABILITIES = ("own_elo_home_win_prob", "own_wt_elo_home_win_prob")


def main() -> None:
    """Print the coverage, the ranges and the window's Brier scores."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    games = pl.read_parquet(args.run_dir / "games.parquet")
    columns = [*constants.OWN_ELO_COLUMNS, *constants.OWN_WT_ELO_COLUMNS]
    summary = pl.DataFrame(
        {
            "column": columns,
            "rows": [games.height] * len(columns),
            "nulls": [games.get_column(name).null_count() for name in columns],
            "min": [games.get_column(name).min() for name in columns],
            "max": [games.get_column(name).max() for name in columns],
        }
    )
    window = games.filter(
        (pl.col("game_type") == "REG")
        & (pl.col("season") >= FIRST_WINDOW_SEASON)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    ).with_columns((pl.col("home_score") > pl.col("away_score")).cast(pl.Float64).alias("win"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=6):
        sys.stdout.write(f"{summary}\n")
    sys.stdout.write(
        f"window: {window.height} games, {int(window.get_column('win').sum())} home wins\n"
    )
    for probability in PROBABILITIES:
        brier = window.select(((pl.col(probability) - pl.col("win")) ** 2).mean()).item()
        sys.stdout.write(f"{probability}: Brier {brier:.5f}\n")
    record = json.loads((args.run_dir / "own_elo_metadata.json").read_text(encoding="utf-8"))
    sys.stdout.write(f"seasons: {record['seasons']}\n")
    win_total_file = {k: v for k, v in record["win_total_file"].items() if k != "snapshot"}
    sys.stdout.write(f"win-total file: {json.dumps(win_total_file, sort_keys=True)}\n")
    sys.stdout.write(
        f"win-total weights: {json.dumps(record['win_totals']['win_total_weights'])}\n"
    )


if __name__ == "__main__":
    main()
