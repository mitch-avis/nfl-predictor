"""Explain why this repository's red-zone trips differ from nfl-sos-ratings' (read-only).

This repository counts a team's red-zone trips as its drives (``fixed_drive``) with any play
at ``yardline_100 <= 20`` (``pbp._red_zone_drive_aggregations``, over the plays of
``pbp.regular_season_plays`` grouped by ``posteam``). ``nfl-sos-ratings`` counts the drives
nflverse flags as reaching the 20 (``drive_inside20``, its ``team_stats_expanded``
``_aggregate_drive_stats``). This script counts both on the measurement cache, compares both
with the source's published ``{season}_team_game_logs.parquet`` ``red_zone_trips``, and sorts
every drive the two definitions disagree on into a named cause:

- ``try_only``: the drive's only plays inside the 20 belong to the try after its touchdown:
  the extra point or two-point attempt, or a penalty-wiped play in kick formation or on a
  two-point attempt (a touchdown drive that never snapped inside the 20 before scoring);
- ``at_the_20``: apart from the try, its only such plays are at exactly the 20
  (``yardline_100 == 20``);
- ``wiped_only``: apart from the try, its only plays inside the 20 are plays a penalty wiped
  out (``no_play``);
- ``other``: a live play inside the 20 that nflverse does not flag;
- ``flag_without_snap``: nflverse flags the drive inside the 20 but no play of it is there.

Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_red_zone.py \
        --cache-dir /tmp/step6_int/cache --sos-data /home/mitch/workspace/nfl-sos-ratings/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import pbp

FIRST_SEASON = 1999
LAST_SEASON = 2025
RED_ZONE = 20
# The causes in the order the docstring lists them, which fixes the by-season table's columns.
CAUSES = ("try_only", "at_the_20", "wiped_only", "other", "flag_without_snap")


def _drives(plays: pl.DataFrame) -> pl.DataFrame:
    """Return one row per team drive with both definitions' flags and the cause facts."""
    in_zone = (pl.col("yardline_100") <= RED_ZONE).fill_null(value=False)
    wiped = (pl.col("play_type") == "no_play").fill_null(value=False)
    on_try = pl.col("desc").str.contains(r"Kick formation|TWO-POINT").fill_null(value=False)
    is_try = (
        (pl.col("play_type") == "extra_point").fill_null(value=False)
        | (pl.col("two_point_attempt").fill_null(0) > 0)
        | (wiped & on_try)
    )
    before_try = in_zone & ~is_try
    inside = before_try & (pl.col("yardline_100") < RED_ZONE)
    here, there = pl.col("here"), pl.col("there")
    return (
        plays.filter(pl.col("fixed_drive").is_not_null())
        .group_by("season", "week", "game_id", "posteam", "fixed_drive", maintain_order=True)
        .agg(
            in_zone.any().alias("here"),
            (pl.col("drive_inside20").fill_null(0).max() > 0).alias("there"),
            (pl.col("fixed_drive_result") == "Touchdown").fill_null(value=False).any().alias("td"),
            before_try.any().alias("before_try"),
            inside.any().alias("inside"),
            (inside & ~wiped).any().alias("inside_live"),
        )
        .with_columns(
            pl.when(here & ~there & ~pl.col("before_try"))
            .then(pl.lit("try_only"))
            .when(here & ~there & ~pl.col("inside"))
            .then(pl.lit("at_the_20"))
            .when(here & ~there & ~pl.col("inside_live"))
            .then(pl.lit("wiped_only"))
            .when(here & ~there)
            .then(pl.lit("other"))
            .when(~here & there)
            .then(pl.lit("flag_without_snap"))
            .otherwise(pl.lit(None))
            .alias("cause")
        )
    )


def main() -> None:
    """Print the counts, the comparison with the published logs and the causes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sos-data", type=Path, required=True)
    args = parser.parse_args()
    columns = [
        "season",
        "week",
        "game_id",
        "season_type",
        "posteam",
        "defteam",
        "play_type",
        "yardline_100",
        "two_point_attempt",
        "fixed_drive",
        "fixed_drive_result",
        "drive_inside20",
        "desc",
    ]
    frames = [
        pl.read_parquet(args.cache_dir / f"pbp_{season}_reg.parquet", columns=columns)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
    ]
    plays = pbp.regular_season_plays(pl.concat(frames, how="vertical_relaxed"))
    drives = _drives(plays)
    team_games = drives.group_by("season", "week", "posteam", maintain_order=True).agg(
        pl.col("here").sum().alias("trips_here"), pl.col("there").sum().alias("trips_there")
    )
    published = pl.concat(
        [
            pl.read_parquet(args.sos_data / f"{season}_team_game_logs.parquet").select(
                pl.lit(season, dtype=pl.Int64).alias("season"),
                pl.col("week").cast(pl.Int64),
                pl.col("team").replace(constants.ALIAS_TO_CANONICAL).alias("posteam"),
                pl.col("red_zone_trips").cast(pl.Int64).alias("trips_source"),
            )
            for season in range(FIRST_SEASON, LAST_SEASON + 1)
        ]
    )
    compared = team_games.with_columns(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)
    ).join(published, on=["season", "week", "posteam"], how="inner")
    sys.stdout.write(
        f"team-games compared: {compared.height}\n"
        f"this repository vs the source's published logs, team-games that differ: "
        f"{compared.filter(pl.col('trips_here') != pl.col('trips_source')).height}\n"
        f"drive_inside20 on these plays vs the published logs, team-games that differ: "
        f"{compared.filter(pl.col('trips_there') != pl.col('trips_source')).height}\n"
        f"trips 1999-2025: this repository {int(compared['trips_here'].sum())}, "
        f"drive_inside20 {int(compared['trips_there'].sum())}, "
        f"published {int(compared['trips_source'].sum())}\n"
    )
    for label, flag in (("this repository", "here"), ("drive_inside20", "there")):
        trips = drives.filter(pl.col(flag))
        sys.stdout.write(
            f"league-wide touchdown rate per trip, {label}: {int(trips['td'].sum())} touchdown "
            f"drives of {trips.height} trips, {trips['td'].mean():.4f}\n"
        )
    causes = (
        drives.filter(pl.col("cause").is_not_null())
        .group_by("cause")
        .agg(pl.len().alias("drives"))
        .sort(["drives", "cause"], descending=[True, False])
    )
    by_season = (
        drives.filter(pl.col("cause").is_not_null())
        .group_by("season", "cause")
        .agg(pl.len().alias("drives"))
        .pivot(on="cause", index="season", values="drives")
        .sort("season")
        .fill_null(0)
    )
    by_season = by_season.select(
        "season",
        *(
            pl.col(cause) if cause in by_season.columns else pl.lit(0, dtype=pl.UInt32).alias(cause)
            for cause in CAUSES
        ),
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        sys.stdout.write(f"\n## Drives the definitions disagree on, by cause\n{causes}\n")
        sys.stdout.write(f"\n## By season\n{by_season}\n")


if __name__ == "__main__":
    main()
