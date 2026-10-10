"""Count the game sides whose team-stat trend is lost to a missing value in the game before.

Run from the repository root with the worktree's environment:

    .venv/bin/python .agents/sos_port/decisions/trend_parity_missing_values.py --base af66025 \
        [--source ~/scratch/step6/data] [--nflreadpy-cache DIR] \
        [--min-season 1999] [--max-season 2025]

A team-game whose statistic is missing (a game the team-stats source lacks) still fills a slot of
the four-game window, and the team's next row gets no trend: Polars' cumulative sum is null at a
null, so the season-to-date mean is null and the week builder publishes 0.0. The trend fix keeps
that behavior (``tests/test_trend_features.py``,
``test_build_team_stat_trends_keeps_a_game_without_a_value_in_the_window``). This counts the sides
it reaches.

It copies the inputs as ``trend_parity_check.py`` does (same options), loads the ETL's sources
offline in a separate process with the package source at ``--base`` (``git archive``, put ahead of
the installed package with ``PYTHONPATH``), and takes the per-team-game frame the trends are
computed from. A regular-season side loses its trend when the team's previous game of the season
has no value and an earlier game has one; a playoff side when the team's last regular-season game
has no value and an earlier game has one. For each such side it also computes the trend the side
would carry if the missing value counted in neither mean without erasing the season-to-date mean,
and counts as changed the sides where that trend is not 0.0 (the others publish 0.0 either way,
early in a season, where the last four games are the whole season). It prints the counts per
season and statistic, and the sides themselves. Nothing is written outside the temporary
directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trend_parity_check as check

STATS = ("scoring_margin", "turnover_margin")
TEAM = ("season", "team_abbr")


def load(data_dir: Path, seasons: list[int], out_parquet: Path) -> None:
    """Load the sources from ``data_dir`` offline and write the per-team-game margins."""
    check._refuse_network()
    from nfl_predictor import constants

    constants.DATA_PATH = data_dir
    constants.NFLREADPY_CACHE_DIR = data_dir / "cache" / "nflreadpy"
    constants.NFELO_CACHE_DIR = data_dir / "cache" / "nfelo"

    from nfl_predictor import data_collection

    config = data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=min(seasons),
        max_season=max(seasons),
    )
    sources = data_collection._load_sources(seasons, config)
    sources.team_stats_df.filter(pl.col("season").is_in(seasons)).select(
        "season", "week", "team_abbr", *STATS
    ).write_parquet(out_parquet)


def _run_load(data_dir: Path, seasons: list[int], pythonpath: Path | None) -> pl.DataFrame:
    out = data_dir / "team_games.parquet"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--load",
        str(data_dir),
        "--min-season",
        str(min(seasons)),
        "--max-season",
        str(max(seasons)),
        "--out",
        str(out),
    ]
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        sys.stderr.write((result.stdout + result.stderr)[-check.LOG_TAIL :])
        msg = f"loading the sources in {data_dir} failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    return pl.read_parquet(out)


def _states(team_games: pl.DataFrame, stat: str) -> pl.DataFrame:
    """Return each team-game's state for ``stat``: whether the next row loses its trend, and
    the trend it would carry if the missing value counted in neither mean without erasing the
    season-to-date mean (the cumulative sum skipping nulls instead of turning null).
    """
    count = pl.col(stat).cum_count()
    recent = pl.col(stat).rolling_mean(window_size=check.WINDOW, min_samples=1)
    season_mean = pl.col(stat).fill_null(0.0).cum_sum() / count
    return team_games.sort(*TEAM, "week").select(
        "season",
        "week",
        "team_abbr",
        (pl.col(stat).is_null() & (count > 0)).over(*TEAM).alias("lost"),
        pl.when(count == 0)
        .then(pl.lit(0.0))
        .otherwise(recent - season_mean)
        .over(*TEAM)
        .alias("would_be"),
    )


def _sides(team_games: pl.DataFrame, playoff_sides: pl.DataFrame) -> pl.DataFrame:
    """Return every side whose trend a missing value in the game before it removes."""
    frames: list[pl.DataFrame] = []
    labels = [pl.lit(stat).alias("statistic") for stat in STATS]
    for stat, label in zip(STATS, labels, strict=True):
        states = _states(team_games, stat)
        regular = states.with_columns(pl.col("lost", "would_be").shift(1).over(*TEAM)).select(
            "season",
            "week",
            "team_abbr",
            pl.lit("regular season").alias("kind"),
            label,
            "lost",
            "would_be",
        )
        after_season = states.group_by(*TEAM, maintain_order=True).agg(
            pl.col("lost", "would_be").last()
        )
        playoff = playoff_sides.join(after_season, on=list(TEAM), how="inner").select(
            "season",
            "week",
            "team_abbr",
            pl.lit("playoff").alias("kind"),
            label,
            "lost",
            "would_be",
        )
        frames.extend([regular, playoff])
    return pl.concat(frames, how="vertical_relaxed").filter(pl.col("lost").fill_null(value=False))


def _playoff_sides(args: argparse.Namespace, seasons: list[int]) -> pl.DataFrame:
    """Return each completed playoff game's two sides from the schedule cache."""
    frames = []
    for season in seasons:
        name = f"schedule_{season}.parquet"
        path = args.nflreadpy_cache / name
        if not path.exists():
            path = args.source / "cache" / "nflreadpy" / name
        schedule = pl.read_parquet(path).filter(
            (pl.col("game_type") != "REG") & pl.col("home_score").is_not_null()
        )
        frames.extend(
            schedule.select(
                pl.col("season").cast(pl.Int64), "week", pl.col(side).alias("team_abbr")
            )
            for side in ("away_abbr", "home_abbr")
        )
    return pl.concat(frames)


def main() -> None:
    """Count the sides, or load the sources when called with ``--load``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    check.add_input_arguments(parser)
    parser.add_argument("--base", help="Load with the package source as of this commit.")
    parser.add_argument("--min-season", type=int, default=1999)
    parser.add_argument("--max-season", type=int, default=2025)
    parser.add_argument("--load", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    seasons = list(range(args.min_season, args.max_season + 1))
    if args.load is not None:
        load(args.load, seasons, args.out)
        return

    check.resolve_input_arguments(args)
    check.configure_tables()
    print(f"Source (read only): {args.source}")
    print(f"nflreadpy files first from (read only): {args.nflreadpy_cache}")
    print(f"Seasons: {seasons[0]}-{seasons[-1]}; code: {args.base or 'the working tree'}")
    root = Path(tempfile.mkdtemp(prefix="trend-parity-missing-"))
    try:
        inputs = root / "data"
        check.prepare(args, inputs, seasons)
        package_source = check.base_source(args.base, root / "base") if args.base else None
        team_games = _run_load(inputs, seasons, package_source)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    sides = _sides(
        team_games.with_columns(pl.col("season").cast(pl.Int64)), _playoff_sides(args, seasons)
    )
    print(f"\nTeam-games loaded: {team_games.height}")
    for stat in STATS:
        missing = team_games.filter(pl.col(stat).is_null()).height
        print(f"   {stat}: {missing} team-games without a value")
    print("\nSides that lose their trend to a missing value in the game before, per season")
    print("(changed: the sides whose trend without the loss would not be 0.0)")
    counts = (
        sides.group_by("season", "statistic", "kind", maintain_order=True)
        .agg(
            pl.len().alias("sides"),
            (pl.col("would_be").abs() > 1e-9).sum().alias("changed"),
        )
        .sort("season", "statistic", "kind")
    )
    print(counts)
    print(
        f"   total: {counts['sides'].sum()} sides, {counts['changed'].sum()} with a changed value"
    )
    print("\nThe sides, with the trend each would carry without the loss")
    print(sides.sort("season", "week", "team_abbr").drop("lost"))


if __name__ == "__main__":
    main()
