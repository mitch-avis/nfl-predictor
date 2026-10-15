"""Measure how far the trend parity fix moves the published game rows, season by season.

Run from the repository root with the worktree's environment:

    .venv/bin/python .agents/sos_port/decisions/trend_parity_movement.py --base <commit> \
        [--source ~/scratch/step6/data] [--nflreadpy-cache DIR] \
        [--min-season 1999] [--max-season 2025]

It copies the inputs of every season from ``--min-season`` through ``--max-season`` into one
temporary directory, exactly as ``trend_parity_check.py`` does and with the same input options,
and builds them twice through ``data_collection.collect_all_data`` with the network refused:
once with the package source as of ``--base`` (extracted with ``git archive`` and put ahead of
the installed package with ``PYTHONPATH``) and once with the working tree's source. It pairs
every game row of the two builds by ``game_id`` and prints:

1. which code built each side;
2. per season and row kind (completed regular-season game, completed playoff game, unplayed
   game), the rows, the rows whose team-stat trends moved, and the largest and the mean move of
   the two trend differences among the moved rows;
3. every other column that differs anywhere between the two builds;
4. per regular-season week over all seasons, the team-stat trends a completed row carries
   (share of sides with a non-zero trend, mean and largest absolute value). An upcoming row
   carried 0.0 before the fix and carries its completed row's value after it, so this is the
   move on a prediction row in that week.

The default range stops at the last completed season: the current season's schedule is always
downloaded, so it cannot be built offline. Nothing is written outside the temporary directory,
which is removed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trend_parity_check as check

KEY = "game_id"
KIND_ORDER = {"completed regular season": 0, "completed playoff": 1, "unplayed": 2}


def _row_kind() -> pl.Expr:
    return (
        pl.when(pl.col("home_score").is_null() | pl.col("away_score").is_null())
        .then(pl.lit("unplayed"))
        .when(pl.col("game_type") == "REG")
        .then(pl.lit("completed regular season"))
        .otherwise(pl.lit("completed playoff"))
        .alias("kind")
    )


def _trend_moves(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """Return one row per game with its kind and each trend column's absolute move."""
    columns = check.trend_columns()
    paired = after.select(KEY, "season", "week", _row_kind(), *columns).join(
        before.select(KEY, *columns), on=KEY, how="inner", suffix="__before"
    )
    moves = [
        (pl.col(column) - pl.col(f"{column}__before")).abs().fill_null(0.0).alias(column)
        for column in columns
    ]
    nulls_differ = pl.any_horizontal(
        pl.col(column).is_null() != pl.col(f"{column}__before").is_null() for column in columns
    )
    return paired.select(
        KEY,
        "season",
        "week",
        "kind",
        *moves,
        nulls_differ.alias("nulls_differ"),
    ).with_columns((pl.max_horizontal(columns) > 1e-9).or_(pl.col("nulls_differ")).alias("moved"))


def _movement_table(moves: pl.DataFrame) -> pl.DataFrame:
    """Summarize the moves per season and row kind."""
    moved = pl.col("moved")
    return (
        moves.group_by("season", "kind", maintain_order=True)
        .agg(
            pl.len().alias("rows"),
            moved.sum().alias("rows_moved"),
            pl.col("scoring_margin_4wk_trend_diff").filter(moved).max().alias("scoring_diff_max"),
            pl.col("scoring_margin_4wk_trend_diff").filter(moved).mean().alias("scoring_diff_mean"),
            pl.col("turnover_margin_4wk_trend_diff").filter(moved).max().alias("turnover_diff_max"),
            pl.col("turnover_margin_4wk_trend_diff")
            .filter(moved)
            .mean()
            .alias("turnover_diff_mean"),
        )
        .with_columns(pl.col("kind").replace_strict(KIND_ORDER).alias("_order"))
        .sort("season", "_order")
        .drop("_order")
    )


def _kind_totals(moves: pl.DataFrame) -> pl.DataFrame:
    """Summarize the moves per row kind over every season."""
    return (
        moves.group_by("kind", maintain_order=True)
        .agg(pl.len().alias("rows"), pl.col("moved").sum().alias("rows_moved"))
        .with_columns(pl.col("kind").replace_strict(KIND_ORDER).alias("_order"))
        .sort("_order")
        .drop("_order")
    )


def _by_week(after: pl.DataFrame) -> pl.DataFrame:
    """Summarize the trends completed regular-season rows carry, per week over all seasons."""
    sides = pl.concat(
        [
            after.filter(
                (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
            ).select(
                "week",
                pl.col(f"{side}_scoring_margin_4wk_trend").alias("scoring"),
                pl.col(f"{side}_turnover_margin_4wk_trend").alias("turnover"),
            )
            for side in ("away", "home")
        ]
    )
    return (
        sides.group_by("week", maintain_order=True)
        .agg(
            pl.len().alias("team_sides"),
            (pl.col("scoring") != 0).mean().round(3).alias("scoring_nonzero_share"),
            pl.col("scoring").abs().mean().round(3).alias("scoring_mean_abs"),
            pl.col("scoring").abs().max().alias("scoring_max_abs"),
            pl.col("turnover").abs().mean().round(3).alias("turnover_mean_abs"),
            pl.col("turnover").abs().max().alias("turnover_max_abs"),
        )
        .sort("week")
    )


def main() -> None:
    """Build the seasons with both versions of the code and compare the game rows."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    check.add_input_arguments(parser)
    parser.add_argument("--base", required=True, help="Commit whose source builds the before side.")
    parser.add_argument("--min-season", type=int, default=1999)
    parser.add_argument("--max-season", type=int, default=2025)
    args = parser.parse_args()
    check.resolve_input_arguments(args)
    check.configure_tables()
    years = list(range(args.min_season, args.max_season + 1))
    print(f"Source (read only): {args.source}")
    print(f"nflreadpy files first from (read only): {args.nflreadpy_cache}")
    print(f"Seasons: {years[0]}-{years[-1]}; before = {args.base}, after = the working tree")

    root = Path(tempfile.mkdtemp(prefix="trend-parity-movement-"))
    try:
        inputs = root / "data"
        provenance = check.prepare(args, inputs, years)
        missing = [line for line in provenance if line.endswith("missing")]
        print(f"   inputs copied: {len(provenance) - len(missing)} files, missing: {missing}")
        base_src = check.base_source(args.base, root / "base")
        before, before_package = check.run_build(
            inputs, years, out_csv=root / "before.csv", pythonpath=base_src
        )
        after, after_package = check.run_build(inputs, years, out_csv=root / "after.csv")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n1. Builds")
    print(f"   before: {before.height} rows, built by {before_package}")
    print(f"   after:  {after.height} rows, built by {after_package}")
    print(f"   games in both: {before.join(after.select(KEY), on=KEY).height}")

    moves = _trend_moves(before, after)
    print("\n2. Rows whose team-stat trends moved, per season and row kind")
    print("   (moves of the away-minus-home differences, among the moved rows)")
    print(_movement_table(moves))
    print(_kind_totals(moves))

    print("\n3. Other columns that differ between the two builds")
    others = [column for column in after.columns if column not in check.trend_columns()]
    print(
        check.column_differences(
            before.select([c for c in others if c in before.columns]),
            after.select(others),
            [KEY],
        )
    )

    print("\n4. Trends a completed regular-season row carries, per week (all seasons)")
    print(_by_week(after))


if __name__ == "__main__":
    main()
