"""Print how strongly the scoring-margin trend tracks the game margin, as published and if leaking.

Run from the repository root with the worktree's environment:

    .venv/bin/python .agents/sos_port/decisions/trend_parity_audit_reach.py \
        [--source ~/scratch/step6/data]

``nfl-predictor leakage-audit`` flags a feature whose correlation with the home margin reaches
0.995 in absolute value and fails one that reaches 0.999999. This reads the source's published
completed regular-season rows (``all_data_ml.csv``, read only) and prints the correlation of the
home margin with ``scoring_margin_4wk_trend_diff`` (away minus home) as published, and with the
same difference recomputed from the schedule scores so that each side's trend includes the game
itself: the leak the reviewer asked about, built on purpose to see whether the audit could see it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trend_parity_check as check


def main() -> None:
    """Print the two correlations over every published season."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=check.DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    published = pl.read_csv(
        source / "all_data_ml.csv",
        columns=[
            "game_id",
            "season",
            "game_type",
            "away_score",
            "home_score",
            "scoring_margin_4wk_trend_diff",
        ],
        infer_schema_length=None,
    ).filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())

    candidates: list[pl.DataFrame] = []
    for season in sorted(published.get_column("season").unique().to_list()):
        path = source / "cache" / "nflreadpy" / f"schedule_{season}.parquet"
        schedule = pl.read_parquet(path).filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        candidates.append(check.season_candidates(schedule))
    sides = pl.concat(candidates).pivot(on="side", index="game_id", values="inclusive")
    rows = published.join(sides, on="game_id", how="inner").with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("home_margin"),
        (pl.col("away") - pl.col("home")).alias("leaking_diff"),
    )
    published_r = rows.select(pl.corr("home_margin", "scoring_margin_4wk_trend_diff")).item()
    leaking_r = rows.select(pl.corr("home_margin", "leaking_diff")).item()
    print(f"Completed regular-season games: {rows.height}")
    print(f"Correlation with the home margin, published difference: {published_r:.4f}")
    print(f"Correlation with the home margin, difference including the game: {leaking_r:.4f}")
    print("The audit flags at |r| >= 0.995 and fails at |r| >= 0.999999.")


if __name__ == "__main__":
    main()
