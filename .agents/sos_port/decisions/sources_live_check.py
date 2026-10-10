"""Run the nflverse player-source loaders against live nflverse files (a check by hand).

Run from the repository root:

    .venv/bin/python .agents/sos_port/decisions/sources_live_check.py

Every loader writes its cache into a temporary directory that is removed at the end, so nothing
under ``data/`` is read or written. It prints the numbers ``sources.md`` quotes for the loaders
themselves (the Jacksonville repair, the league roster codes, the quarterback identity, ESPN QBR
and the current-season behavior). nflverse revises its files, so the output can drift from the
copy in ``sources.md``, which records the date and versions.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import logging
import tempfile
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import espn_qbr, nflverse_sources, player_identity

CURRENT = 2026
JACKSONVILLE_SEASONS = [2001, 2002]


def heading(text: str) -> None:
    """Print a section heading."""
    print(f"\n== {text}")


def main() -> None:
    """Print every section."""
    logging.getLogger("nfl_predictor").setLevel(logging.WARNING)
    heading("Environment")
    print("date:", datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"))
    for package in ("nflreadpy", "polars"):
        print(package, importlib.metadata.version(package))

    with tempfile.TemporaryDirectory(prefix="sources-live-check-") as directory:
        cache = Path(directory)

        heading("Weekly player stats 2001-2002: the Jacksonville home games after the repair")
        stats = nflverse_sources.load_weekly_player_stats(
            JACKSONVILLE_SEASONS, cache_dir=cache, current_season=CURRENT
        )
        jacksonville = stats.filter(pl.col("game_id").str.ends_with("_JAX"))
        print(
            jacksonville.group_by("game_id", maintain_order=True)
            .agg(
                pl.len().alias("rows"),
                pl.col("team").n_unique().alias("teams"),
                pl.col("team").is_null().sum().alias("without_team"),
            )
            .sort("game_id")
        )
        print("rows without a team:", jacksonville["team"].null_count())
        print(
            "games naming one team after the repair:",
            stats.group_by("game_id", maintain_order=True)
            .agg(pl.col("team").drop_nulls().n_unique().alias("teams"), pl.len().alias("rows"))
            .filter((pl.col("teams") == 1) & (pl.col("rows") >= constants.ONE_TEAM_GAME_MIN_VALUES))
            .height,
        )
        print(
            "columns as requested:", tuple(stats.columns) == nflverse_sources.PLAYER_STATS.columns
        )

        heading("Season rosters: team codes after normalization, 2002 and 2016")
        rosters = nflverse_sources.load_rosters(
            [2002, 2016], cache_dir=cache, current_season=CURRENT
        )
        print(
            rosters.group_by("season", maintain_order=True)
            .agg(pl.col("team").n_unique().alias("teams"), pl.col("team").unique().sort())
            .sort("season")
            .rows()
        )

        heading("Quarterback identity 2022: Taysom Hill, and the season's quarterbacks")
        identity = nflverse_sources.load_season_qb_identity(
            2022, cache_dir=cache, current_season=CURRENT
        )
        print(identity.filter(pl.col("qb_id") == "00-0033357").rows())
        print("players:", identity.height)
        print("quarterbacks:", len(player_identity.season_quarterback_ids(identity)))

        heading("Snap counts 2012-2013 (2012's file holds no rows)")
        snaps = nflverse_sources.load_snap_counts(
            [2012, 2013], cache_dir=cache, current_season=CURRENT
        )
        print(snaps.group_by("season", maintain_order=True).len().sort("season").rows())
        print("teams:", sorted(snaps["team"].unique().to_list()))

        heading("ESPN QBR, season level 2005-2007 and week level 2016")
        qbr = espn_qbr.load_espn_qbr([2005, 2006, 2007], cache_dir=cache, current_season=CURRENT)
        print(qbr.group_by("season", maintain_order=True).len().sort("season").rows())
        print("season types:", qbr["season_type"].unique().to_list())
        weeks = espn_qbr.load_espn_qbr(
            [2016], level="week", cache_dir=cache, current_season=CURRENT
        )
        print("week rows:", weeks.height, "teams:", sorted(weeks["team_abb"].unique().to_list()))

        heading(f"The current season ({CURRENT}) and the next ({CURRENT + 1})")
        for name, frame in (
            (
                "weekly player stats",
                nflverse_sources.load_weekly_player_stats(
                    [CURRENT, CURRENT + 1], cache_dir=cache, current_season=CURRENT
                ),
            ),
            (
                "snap counts",
                nflverse_sources.load_snap_counts(
                    [CURRENT, CURRENT + 1], cache_dir=cache, current_season=CURRENT
                ),
            ),
            (
                "weekly rosters",
                nflverse_sources.load_weekly_rosters(
                    [CURRENT, CURRENT + 1], cache_dir=cache, current_season=CURRENT
                ),
            ),
            (
                "season rosters",
                nflverse_sources.load_rosters(
                    [CURRENT, CURRENT + 1], cache_dir=cache, current_season=CURRENT
                ),
            ),
            (
                "ESPN QBR season level",
                espn_qbr.load_espn_qbr(
                    [CURRENT, CURRENT + 1], cache_dir=cache, current_season=CURRENT
                ),
            ),
        ):
            print(name, frame.group_by("season", maintain_order=True).len().sort("season").rows())

        heading("Cache files written")
        print(sorted(path.name for path in cache.iterdir()))


if __name__ == "__main__":
    main()
