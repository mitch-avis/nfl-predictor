"""Compare two ``integration_feature_values.py`` outputs and print what moved, by season.

Run from the repository root with the output of the commit before a change and of the commit
that makes it:

    .venv/bin/python -I .agents/sos_port/decisions/integration_feature_diff.py \
        --before /tmp/step6_int/values/<before> --after /tmp/step6_int/values/<after>

It prints:

- for the team-game counts, the box scores and the quarterback-game sums: every column whose
  values differ, per season: the team-games (or quarterback-games) that differ and the season
  total before and after; rows present on one side only are counted separately;
- for the published game columns: the columns added or removed, and every published stat that
  moved (a stat is its ``away_``, ``home_`` and ``_diff`` columns, or a game-level column on its
  own), per season: the games where any of its columns differs, the largest absolute change and
  the mean absolute change over those games; a value that turns null or non-null counts as a
  change and is listed separately. Every other published column is identical, which the last
  line states with the count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# Values closer than this are the same number (last-bit noise from a different summation order).
TOLERANCE = 1e-12
_SIDES = ("away_", "home_")
_DIFF = "_diff"


def _print(title: str, frame: pl.DataFrame) -> None:
    """Print one titled table in full."""
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=250,
        tbl_hide_dataframe_shape=True,
        float_precision=6,
    ):
        sys.stdout.write(f"\n## {title}\n{frame}\n")


def _numeric(frame: pl.DataFrame, keys: list[str]) -> list[str]:
    """Return the numeric columns of ``frame`` that are not keys."""
    return [
        name
        for name, dtype in frame.schema.items()
        if name not in keys and (dtype.is_numeric() or dtype == pl.Boolean)
    ]


def _changed(before: pl.Expr, after: pl.Expr) -> pl.Expr:
    """Return whether two values differ beyond the tolerance, a null against a value included."""
    both = before.is_not_null() & after.is_not_null()
    moved = (before.cast(pl.Float64) - after.cast(pl.Float64)).abs() > TOLERANCE
    return (both & moved) | (before.is_null() != after.is_null())


def compare_counts(name: str, before: pl.DataFrame, after: pl.DataFrame, keys: list[str]) -> None:
    """Print the count columns that differ, per season."""
    shared = [column for column in _numeric(before, keys) if column in after.columns]
    added = sorted(set(after.columns) - set(before.columns))
    removed = sorted(set(before.columns) - set(after.columns))
    joined = before.join(after, on=keys, how="full", coalesce=True, suffix="__after")
    only_before = joined.filter(pl.col(f"{shared[0]}__after").is_null() & pl.col(shared[0]).is_not_null())
    only_after = joined.filter(pl.col(shared[0]).is_null() & pl.col(f"{shared[0]}__after").is_not_null())
    sys.stdout.write(
        f"\n# {name}: {before.height} rows before, {after.height} after; "
        f"{only_before.height} only before, {only_after.height} only after; "
        f"columns added {added}, removed {removed}\n"
    )
    rows: list[pl.DataFrame] = []
    for column in shared:
        moved = _changed(pl.col(column), pl.col(f"{column}__after"))
        table = (
            joined.group_by("season", maintain_order=True)
            .agg(
                moved.sum().alias("rows_changed"),
                pl.col(column).cast(pl.Float64).sum().alias("total_before"),
                pl.col(f"{column}__after").cast(pl.Float64).sum().alias("total_after"),
            )
            .filter(pl.col("rows_changed") > 0)
            .with_columns(pl.lit(column).alias("column"))
        )
        rows.append(table)
    changed = pl.concat(rows).select("column", "season", "rows_changed", "total_before", "total_after")
    if changed.height == 0:
        sys.stdout.write("no column differs\n")
        return
    _print(f"{name}: columns that differ, by season", changed.sort("column", "season"))


def _stat(column: str) -> str:
    """Return the published stat a column belongs to."""
    for side in _SIDES:
        if column.startswith(side):
            return column.removeprefix(side)
    return column.removesuffix(_DIFF)


def compare_games(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Print the published stats that moved, per season."""
    added = sorted(set(after.columns) - set(before.columns))
    removed = sorted(set(before.columns) - set(after.columns))
    sys.stdout.write(
        f"\n# published game columns: {before.height} games before, {after.height} after; "
        f"{before.width} columns before, {after.width} after\n"
        f"columns added: {added}\ncolumns removed: {removed}\n"
    )
    shared = [column for column in _numeric(before, ["game_id", "season"]) if column in after.columns]
    joined = before.join(after, on=["game_id", "season"], how="inner", suffix="__after")
    by_stat: dict[str, list[str]] = {}
    for column in shared:
        by_stat.setdefault(_stat(column), []).append(column)
    tables: list[pl.DataFrame] = []
    unchanged = 0
    for stat, columns in sorted(by_stat.items()):
        moves = [_changed(pl.col(c), pl.col(f"{c}__after")) for c in columns]
        nullness = [pl.col(c).is_null() != pl.col(f"{c}__after").is_null() for c in columns]
        gaps = [(pl.col(c).cast(pl.Float64) - pl.col(f"{c}__after").cast(pl.Float64)).abs() for c in columns]
        frame = joined.select(
            "season",
            pl.any_horizontal(moves).alias("moved"),
            pl.any_horizontal(nullness).alias("null_changed"),
            pl.max_horizontal(gaps).alias("gap"),
        )
        table = (
            frame.filter(pl.col("moved"))
            .group_by("season", maintain_order=True)
            .agg(
                pl.len().alias("games_changed"),
                pl.col("null_changed").sum().alias("null_changed"),
                pl.col("gap").max().alias("max_abs_change"),
                pl.col("gap").mean().alias("mean_abs_change"),
            )
            .sort("season")
        )
        if table.height == 0:
            unchanged += len(columns)
            continue
        tables.append(table.with_columns(pl.lit(stat).alias("stat")))
    if tables:
        moved = pl.concat(tables).select(
            "stat", "season", "games_changed", "null_changed", "max_abs_change", "mean_abs_change"
        )
        summary = (
            moved.group_by("stat", maintain_order=True)
            .agg(
                pl.col("games_changed").sum(),
                pl.col("season").min().alias("first_season"),
                pl.col("season").max().alias("last_season"),
                pl.col("max_abs_change").max(),
            )
            .sort("stat")
        )
        _print("published stats that moved (all seasons)", summary)
        _print("published stats that moved, by season", moved)
    sys.stdout.write(f"\n{unchanged} published numeric columns are identical before and after\n")


def main() -> None:
    """Compare every frame of the two outputs."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    for name, keys in (
        ("team_games", ["season", "week", "team_abbr", "opponent_abbr"]),
        ("box_scores", ["season", "week", "team_abbr", "opponent_abbr"]),
        ("qb_games", ["season", "week", "team_abbr", "qb_id"]),
    ):
        compare_counts(
            name,
            pl.read_parquet(args.before / f"{name}.parquet"),
            pl.read_parquet(args.after / f"{name}.parquet"),
            keys,
        )
    compare_games(
        pl.read_parquet(args.before / "games.parquet"), pl.read_parquet(args.after / "games.parquet")
    )


if __name__ == "__main__":
    main()
