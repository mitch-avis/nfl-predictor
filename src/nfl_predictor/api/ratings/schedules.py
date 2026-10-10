"""The all-time schedule-strength leaderboard over every season's ratings table.

``sos`` is the average ``team_rating`` of a team's opponents, in points per game, each opponent
rated without its games against that team. The ranking itself is
:func:`nfl_predictor.season_stats.schedule_ranks.rank_schedules`, the port of the
``nfl-sos-ratings`` ``schedules`` command; this module reads the tables and describes the columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfl_predictor.api.ratings.columns import metric_table
from nfl_predictor.api.ratings.files import SEASON_FILE, read_table
from nfl_predictor.api.schemas.ratings import MetricColumnOut, ScheduleRanksOut
from nfl_predictor.season_stats.schedule_ranks import rank_schedules

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

RATINGS_KIND = "ratings"
_LOCAL_COLUMNS = {
    "season": MetricColumnOut(
        key="season",
        label="Season",
        description="The season the schedule was played in.",
        group="identity",
        kind="text",
        full_name="Season",
        shape="id",
    ),
    "softest_rank": MetricColumnOut(
        key="softest_rank",
        label="Softest Rank",
        description=(
            "Where this schedule ranks among every completed team-season, 1 being the softest "
            "(the lowest SoS)."
        ),
        group="ranks",
        kind="int",
        full_name="Softest-Schedule Rank, All Seasons",
        shape="score",
    ),
    "hardest_rank": MetricColumnOut(
        key="hardest_rank",
        label="Hardest Rank",
        description=(
            "Where this schedule ranks among every completed team-season, 1 being the hardest "
            "(the highest SoS)."
        ),
        group="ranks",
        kind="int",
        full_name="Hardest-Schedule Rank, All Seasons",
        shape="score",
    ),
}


def _ratings_by_season(ratings_dir: Path) -> dict[int, pl.DataFrame]:
    """Return every season's team ratings table in ``ratings_dir``."""
    if not ratings_dir.is_dir():
        return {}
    seasons = []
    for path in ratings_dir.glob(f"*_{RATINGS_KIND}.parquet"):
        match = SEASON_FILE.match(path.name)
        if match is not None and match["kind"] == RATINGS_KIND:
            seasons.append(int(match["season"]))
    return {season: read_table(ratings_dir, season, RATINGS_KIND) for season in sorted(seasons)}


def schedule_ranks(ratings_dir: Path) -> ScheduleRanksOut:
    """Return every completed team-season ranked on ``sos``, softest first.

    A season in which no team has played a full schedule is still in progress
    (``schedule_ranks.COMPLETE_SEASON_MIN_GAMES``): the ranking leaves it out, and it is listed in
    ``skipped_seasons``.
    """
    ratings = _ratings_by_season(ratings_dir)
    ranked = rank_schedules(ratings)
    seasons = sorted({int(season) for season in ranked.get_column("season").to_list()})
    skipped = [
        season
        for season, frame in ratings.items()
        if season not in seasons and "sos" in frame.columns
    ]
    table = metric_table(
        ranked,
        {
            "identity": ["season", "team"],
            "schedule": ["sos"],
            "ranks": ["softest_rank", "hardest_rank"],
        },
        local=_LOCAL_COLUMNS,
    )
    return ScheduleRanksOut(
        count=ranked.height, seasons=seasons, skipped_seasons=skipped, table=table
    )
