"""The ETL's season statistics: each season's tables of the statistics ported from nfl-sos-ratings.

The season statistics (:mod:`nfl_predictor.season_stats`) are built for every season of an ETL
run from the frames the run already holds, so no nflverse source is read twice in a run: the
play-by-play the ETL loaded for its seasons (the games credited to one team already repaired from
the rosters), and the schedules, weekly player stats, snap counts, players file and weekly rosters
the ratings loaded (``ratings_build.LoadedSources``). The official weekly team stats are the one
source of their own, loaded once for every season of the run (:func:`load_season_stats_sources`).

A season's inputs are those ``season_stats.sources.load_season_inputs`` reads, cut from these
frames (:meth:`SeasonStatsBuild.season_inputs`): the play-by-play's columns the builders read
(``SEASON_STATS_PBP_COLUMNS``), prepared as that loader prepares them, the regular-season
schedule, and the season's quarterback identity. Each slice of a frame that spans seasons is
rechunked, so a later season's rows cannot move where a sum's chunks split.

The tables are built in the two stages of ``season_stats.season``: the game rows
(``build_season_games``), then the tables (``build_season_tables``), whose ``combined`` and
``qb_combined`` take their rating columns from the season's fitted ratings, as
``nfl-sos-ratings``' ``main.run_season`` fed them (``ratings_build.RatingsBuild`` passes them).
The tables read nothing else, so a finished season's tables are kept in the season cache with its
ratings (``utils.season_cache``), keyed on exactly those frames
(:meth:`SeasonStatsBuild.key_frames`). The game rows are built again for the key on every
incremental run: they read the players file, which an in-season run downloads again and which
gains players every week, and only a change to a player the season's sources name moves them.

Before kickoff the current season has no game played, so no tables. A season with plays but
without official team stats, weekly player stats or snap counts (the current season before
nflverse publishes them) is built without them, with a warning naming what its tables lack.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats import sources as season_stats_sources
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import nflverse_sources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nfl_predictor.utils.polars.ratings_build import LoadedSources

# The season tables, named as nfl-sos-ratings names its files, in the order they are built.
SEASON_STATS_KINDS: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(season_tables.SeasonTables)
)
# The first season nflverse's snap counts hold a row: it publishes them from 2012, but its 2012
# file is empty.
SNAP_COUNTS_FIRST_SEASON = 2013
# What a season's tables lack without each source, for the warning.
_WITHOUT_OFFICIAL_TEAM_STATS = (
    "the play-by-play's passing, rushing, first-down, sack, interception and lost-fumble totals "
    "stand in for the official ones"
)
_WITHOUT_PLAYER_STATS = (
    "its tables leave out the defense box score and the passers' official passing and rushing "
    "figures"
)
_WITHOUT_SNAP_COUNTS = (
    "its tables leave out the passers' offensive snaps and the passers who played without a "
    "dropback"
)
# The rating columns of a season without ratings: null in ``combined`` and ``qb_combined``.
_NO_TEAM_RATINGS = pl.DataFrame(
    schema={
        "team": pl.String(),
        **dict.fromkeys(season_tables.TEAM_RATING_INPUT_COLUMNS[1:], pl.Float64()),
    }
)
_NO_QB_RATINGS = pl.DataFrame(
    schema={
        "qb_id": pl.String(),
        **dict.fromkeys(season_tables.QB_RATING_INPUT_COLUMNS[1:], pl.Float64()),
    }
)


@dataclass(frozen=True)
class SeasonStatsSources:
    """Every season's sources of the season statistics, as an ETL run holds them.

    Attributes:
        plays: Regular-season play-by-play (``loaders.load_pbp``). The run's seasons, the ones
            whose tables are built, come from the ETL's own load, whose games credited to one
            team are repaired (``nflverse_sources.repair_one_team_pbp_games``); the ratings'
            earlier seasons beside them are read from the cache as they are.
        schedule: The schedules (``loaders.load_schedule``), every game type.
        player_stats: Weekly player stats (``nflverse_sources.load_weekly_player_stats``).
        snap_counts: Snap counts (``nflverse_sources.load_snap_counts``).
        official_team_stats: Official weekly team stats of the run's seasons
            (``season_stats.sources.load_official_team_stats``).
        identity: The players file and weekly rosters, which give each season's quarterback
            identity.

    """

    plays: pl.DataFrame
    schedule: pl.DataFrame
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    official_team_stats: pl.DataFrame
    identity: nflverse_sources.QbIdentitySources


def load_season_stats_sources(
    seasons: Sequence[int],
    loaded: LoadedSources,
    *,
    current_season: int,
    force_refresh: bool,
) -> SeasonStatsSources:
    """Return the season statistics' sources: the ratings' frames and the official team stats.

    Every source but the official weekly team stats is one the ratings already loaded, so none is
    asked for again. The official team stats are loaded once, for ``seasons``, under the cache
    contract of the other per-season nflverse sources (``nflverse_sources.SeasonSource``): a
    finished season from its cache file, the current season fetched again and, when that fails,
    read from the cache or left out. The current season's file is the one the team-stat loader
    fetched earlier in the run, which nflreadpy serves again from its in-memory cache.

    Args:
        seasons: The seasons of the run, the ones whose tables are built.
        loaded: The ratings' sources of the run (``ratings_build.load_ratings_sources``).
        current_season: The current season, for the loader's cache decisions.
        force_refresh: Whether the run downloads its sources again (``--refresh-nflreadpy``).

    Returns:
        The sources of every season of the run.

    Raises:
        ValueError: If the play-by-play lacks a column the season statistics read
            (``season_stats.sources.SEASON_STATS_PBP_COLUMNS``), which would turn every stat
            built on it into zeros.

    """
    missing = season_stats_sources.missing_pbp_columns(loaded.plays)
    if missing:
        msg = "the play-by-play lacks columns the season statistics read: " + ", ".join(missing)
        raise ValueError(msg)
    return SeasonStatsSources(
        plays=loaded.plays,
        schedule=loaded.schedule,
        player_stats=loaded.player_stats,
        snap_counts=loaded.snap_counts,
        official_team_stats=season_stats_sources.load_official_team_stats(
            seasons, force_refresh=force_refresh, current_season=current_season
        ),
        identity=nflverse_sources.QbIdentitySources(loaded.players, loaded.weekly_rosters),
    )


def _of_season(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return the rows of ``season``, rechunked so later seasons cannot reach their sums."""
    return frame.filter(pl.col("season") == season).rechunk()


def log_missing_sources(inputs: season_stats_sources.SeasonInputs) -> None:
    """Warn for each source a season with plays lacks, naming what its tables leave out.

    A source counts as missing from the first season nflverse publishes rows of it: official
    weekly team stats and weekly player stats from the first play-by-play season, snap counts from
    ``SNAP_COUNTS_FIRST_SEASON``.
    """
    if inputs.pbp.is_empty():
        return
    for name, frame, first_season, lacking in (
        (
            "official weekly team stats",
            inputs.official_team_stats,
            season_stats_sources.PBP_START_SEASON,
            _WITHOUT_OFFICIAL_TEAM_STATS,
        ),
        (
            "weekly player stats",
            inputs.player_stats,
            constants.NFLREADPY_PLAYER_STATS_MIN_SEASON,
            _WITHOUT_PLAYER_STATS,
        ),
        ("snap counts", inputs.snap_counts, SNAP_COUNTS_FIRST_SEASON, _WITHOUT_SNAP_COUNTS),
    ):
        if inputs.season >= first_season and frame.is_empty():
            log.warning(
                "Season statistics: season %d has no %s, so %s", inputs.season, name, lacking
            )


class SeasonStatsBuild:
    """The season statistics part of one ETL run: each season's tables from the run's frames.

    :meth:`tables` builds a season's tables from its fitted ratings; :meth:`key_frames` returns
    what else they are built from, for the season cache's key, and keeps the season's game rows
    for :meth:`tables`.
    """

    def __init__(self, sources: SeasonStatsSources) -> None:
        """Remember the run's sources."""
        self._sources = sources
        self._games: dict[int, season_tables.SeasonGames] = {}

    def season_inputs(self, season: int) -> season_stats_sources.SeasonInputs:
        """Return one season's inputs, as ``season_stats.sources.load_season_inputs`` returns them.

        The play-by-play keeps the columns the builders read and is prepared as that loader
        prepares it (``prepare_pbp``); its games credited to one team were repaired when the ETL
        loaded it.
        """
        sources = self._sources
        plays = _of_season(
            sources.plays.select(season_stats_sources.SEASON_STATS_PBP_COLUMNS), season
        )
        return season_stats_sources.SeasonInputs(
            season=season,
            pbp=season_stats_sources.prepare_pbp(plays, season),
            player_stats=_of_season(sources.player_stats, season),
            schedule=season_stats_sources.prepare_schedule(_of_season(sources.schedule, season)),
            official_team_stats=_of_season(sources.official_team_stats, season),
            snap_counts=_of_season(sources.snap_counts, season),
            qb_identity=sources.identity.season_identity(season),
        )

    def key_frames(self, season: int) -> dict[str, pl.DataFrame]:
        """Return what a season's tables are built from beside its ratings, by name.

        The tables read only the season's play-by-play, its game rows and its played schedule
        (``build_season_tables``' arguments, the ratings aside), so those are their part of the
        season cache's key. The game rows are built here and kept for :meth:`tables`.
        """
        inputs = self.season_inputs(season)
        games = season_tables.build_season_games(inputs)
        self._games[season] = games
        return {
            "season_stats_plays": inputs.pbp,
            "season_stats_team_games": games.team_games,
            "season_stats_qb_games": games.qb_games,
            "season_stats_schedule": games.schedule,
        }

    def tables(
        self,
        season: int,
        *,
        team_ratings: pl.DataFrame | None,
        qb_ratings: pl.DataFrame | None,
    ) -> dict[str, pl.DataFrame]:
        """Return one season's tables by kind, or none before the season's first game.

        Args:
            season: The season.
            team_ratings: The season's team ratings (the ratings build's ``ratings`` table), whose
                ``TEAM_RATING_INPUT_COLUMNS`` ``combined`` takes; None leaves them null.
            qb_ratings: Every passer's rating and faced pass defense (the quarterback fit's
                ``qb_season``), whose ``QB_RATING_INPUT_COLUMNS`` ``qb_combined`` takes; None,
                for a season without a qualified passer and so without a quarterback fit, leaves
                them null.

        Returns:
            The season's tables by kind (``SEASON_STATS_KINDS``; the opponent profiles are absent
            when no opponent game supports one), each in its published row order.

        """
        start = time.perf_counter()
        inputs = self.season_inputs(season)
        games = self._games.pop(season, None)
        if games is None:
            games = season_tables.build_season_games(inputs)
        if not games.played:
            log.info("Season statistics: season %d has no game played yet, so no tables", season)
            return {}
        log_missing_sources(inputs)
        if team_ratings is None:
            log.warning(
                "Season statistics: season %d has no team ratings, so combined leaves them null",
                season,
            )
        if qb_ratings is None:
            log.info(
                "Season statistics: season %d has no quarterback ratings (no passer qualified "
                "for the rankings), so qb_combined leaves them null",
                season,
            )
        tables = season_tables.build_season_tables(
            games,
            inputs.pbp,
            team_ratings=_NO_TEAM_RATINGS if team_ratings is None else team_ratings,
            qb_ratings=_NO_QB_RATINGS if qb_ratings is None else qb_ratings,
        ).tables()
        log.info(
            "Season statistics for season %d: %d tables in %.1fs",
            season,
            len(tables),
            time.perf_counter() - start,
        )
        return tables


__all__ = [
    "SEASON_STATS_KINDS",
    "SNAP_COUNTS_FIRST_SEASON",
    "SeasonStatsBuild",
    "SeasonStatsSources",
    "load_season_stats_sources",
    "log_missing_sources",
]
