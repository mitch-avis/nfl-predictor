"""Show, column by column and season by season, how a change moved the published red-zone rates.

``integration_feature_diff.py`` reports a published stat's ``away_``, ``home_`` and ``_diff``
columns together and only by absolute change. This script splits them, for every published
column whose name holds ``red_zone`` (``red_zone_td_pct``, its allowed mirror
``opponent_red_zone_td_pct``, each as ``away_``, ``home_`` and ``_diff``), and keeps the sign. It
reads two ``integration_feature_values.py`` outputs, the commit before the change and the commit
that makes it, and prints:

- per column and season: the games, the games where the value changed (beyond last-bit noise,
  or between null and a value), the nulls before and after, the mean before and after, the mean
  signed change (after minus before, over the games with a value on both sides) and the largest
  absolute change;
- per season: the league red-zone touchdown rate per trip (touchdown drives over trips, summed
  over the team-games of ``team_games.parquet``) before and after.

Both inputs are read only. Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_red_zone_movement.py \
        --before /tmp/step6_int/values/<before> --after /tmp/step6_int/values/<after>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# Values closer than this are the same number (last-bit noise from a different summation order).
TOLERANCE = 1e-12


def _column_movement(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """Return the per-column, per-season movement of every published red-zone column."""
    columns = [name for name in before.columns if "red_zone" in name]
    joined = before.select("game_id", "season", *columns).join(
        after.select("game_id", *columns), on="game_id", how="inner", suffix="__after"
    )
    tables: list[pl.DataFrame] = []
    for column in columns:
        old, new = pl.col(column), pl.col(f"{column}__after")
        both = old.is_not_null() & new.is_not_null()
        change = (new - old).filter(both)
        moved = (both & ((new - old).abs() > TOLERANCE)) | (old.is_null() != new.is_null())
        tables.append(
            joined.group_by("season")
            .agg(
                pl.len().alias("games"),
                moved.sum().alias("changed"),
                old.null_count().alias("null_before"),
                new.null_count().alias("null_after"),
                old.mean().alias("mean_before"),
                new.mean().alias("mean_after"),
                change.mean().alias("mean_change"),
                change.abs().max().alias("max_abs_change"),
            )
            .with_columns(pl.lit(column).alias("column"))
        )
    return (
        pl.concat(tables)
        .select(
            "column",
            "season",
            "games",
            "changed",
            "null_before",
            "null_after",
            "mean_before",
            "mean_after",
            "mean_change",
            "max_abs_change",
        )
        .sort("column", "season")
    )


def _league_rates(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """Return the league red-zone touchdown rate per trip by season, before and after."""

    def rates(frame: pl.DataFrame, label: str) -> pl.DataFrame:
        return frame.group_by("season").agg(
            pl.col("red_zone_trips").sum().alias(f"trips_{label}"),
            pl.col("red_zone_td_drives").sum().alias(f"td_drives_{label}"),
            (pl.col("red_zone_td_drives").sum() / pl.col("red_zone_trips").sum()).alias(
                f"rate_{label}"
            ),
        )

    return (
        rates(before, "before")
        .join(rates(after, "after"), on="season", how="full", coalesce=True)
        .with_columns((pl.col("rate_after") - pl.col("rate_before")).alias("rate_change"))
        .sort("season")
    )


def main() -> None:
    """Print the two tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    games_before = pl.read_parquet(args.before / "games.parquet")
    games_after = pl.read_parquet(args.after / "games.parquet")
    movement = _column_movement(games_before, games_after)
    totals = (
        movement.group_by("column")
        .agg(
            pl.col("games").sum(),
            pl.col("changed").sum(),
            pl.col("max_abs_change").max(),
        )
        .sort("column")
    )
    rates = _league_rates(
        pl.read_parquet(args.before / "team_games.parquet"),
        pl.read_parquet(args.after / "team_games.parquet"),
    )
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=250,
        tbl_hide_dataframe_shape=True,
        float_precision=4,
    ):
        sys.stdout.write(
            f"games before {games_before.height}, after {games_after.height}\n"
            f"\n## Published red-zone columns, all seasons\n{totals}\n"
            f"\n## Published red-zone columns, by season\n{movement}\n"
            f"\n## League touchdown rate per red-zone trip, by season\n{rates}\n"
        )


if __name__ == "__main__":
    main()
