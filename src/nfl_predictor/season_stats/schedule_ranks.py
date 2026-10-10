"""All-time strength-of-schedule leaderboard across every season's ratings.

``sos`` is the average ``team_rating`` of a team's opponents, in points per game, with each
opponent rated without its games against that team (the ratings package). This module ranks
every team-season on it, so one season's schedule can be placed in the full history, as
``nfl-sos-ratings``' ``schedules`` command does over its data files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

# Every completed season since 1999 has at least 16 games per team; fewer means still in progress.
COMPLETE_SEASON_MIN_GAMES = 16
_SCHEMA = pl.Schema(
    {
        "season": pl.Int64(),
        "team": pl.String(),
        "sos": pl.Float64(),
        "softest_rank": pl.UInt32(),
        "hardest_rank": pl.UInt32(),
    }
)


def rank_schedules(ratings_by_season: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
    """Return every team-season with its ``sos`` and its softest and hardest rank overall.

    Rank 1 is the softest (``softest_rank``) or hardest (``hardest_rank``) schedule. Ratings
    tables without an ``sos`` column and seasons still in progress (no team with a full season of
    games) are skipped.

    Args:
        ratings_by_season: Each season's team ratings (``team``, ``sos``, ``games_played``).

    Returns:
        One row per team-season, softest first

    """
    frames: list[pl.DataFrame] = []
    for season, ratings in sorted(ratings_by_season.items()):
        if "sos" not in ratings.columns:
            continue
        if (
            "games_played" in ratings.columns
            and ratings.select(pl.col("games_played").max()).item() < COMPLETE_SEASON_MIN_GAMES
        ):
            continue
        frames.append(ratings.select("team", "sos").with_columns(pl.lit(season).alias("season")))
    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("sos").rank("ordinal").alias("softest_rank"),
            pl.col("sos").rank("ordinal", descending=True).alias("hardest_rank"),
        )
        .select("season", "team", "sos", "softest_rank", "hardest_rank")
        .sort("softest_rank")
        .cast(_SCHEMA)
    )


__all__ = ["COMPLETE_SEASON_MIN_GAMES", "rank_schedules"]
