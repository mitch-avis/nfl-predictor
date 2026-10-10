"""The ETL's ratings: each season's published ratings tables and the pre-week rating features.

The ratings ported from ``nfl-sos-ratings`` (:mod:`nfl_predictor.ratings`) are fit here on this
repository's play-by-play, schedules and player sources, through the adapter
(:mod:`nfl_predictor.ratings.pbp_inputs`), which keeps that project's definitions so the tables
match its published files. For each season an ETL run builds:

- the published tables, written by ``nfl-predictor data`` to
  ``data/ratings/{season}_{kind}.parquet`` with that project's kind names (``PUBLISHED_KINDS``),
  and for the current season each week's rank ranges as well (``IN_PROGRESS_KINDS``);
- the pre-week tables the features read: for every scheduled week, each team's ratings and each
  passer's rating fit only on the season's games before that week (with earlier seasons for the
  penalties and the preseason prior), the passers as the identity knew them before the week
  (:meth:`nfl_predictor.ratings.pre_week.RatingsHistory.pre_week`).

:meth:`RatingsBuild.attach` joins them onto the game rows as the ``ratings`` feature group
(``constants.RATINGS_FEATURE_STATS``): both teams' offense, defense, special-teams and team
ratings and ``sos`` for the row's week, and the expected starter's rating with his dropbacks; the
starter is the one every quarterback feature reads (``qb_stats.expected_starter_ids``). Early on:

- week 1: every team at its preseason prior (from 2003, the first season with three earlier
  season pairs for the carryover slopes; null before), ``sos`` null, and no quarterback rating,
  with 0 dropbacks;
- week 2: ``sos`` still null, since no opponent has yet played anyone but the team rated;
- 1999, the first play-by-play season: every rating null, since no earlier season supplies the
  penalties;
- a postseason week: the fit of the whole regular season;
- a starter without a dropback this season: no rating and 0 dropbacks; an unidentified one:
  both null.

The fits run with one BLAS thread (``threadpoolctl``): with more, the quarterback fits can differ
in their last bits. Their rows come from the adapter in a fixed order, and each season's slices
are rechunked, so two runs on the same inputs write the same bytes. A season's outputs read only
that season and earlier ones, so the season cache (``utils.season_cache``) stores them with the
season's build and an incremental run rebuilds only the seasons whose inputs changed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
from threadpoolctl import threadpool_limits

from nfl_predictor import constants
from nfl_predictor.ratings.inputs import TEAM_GAME_SCHEMA, team_game_rows
from nfl_predictor.ratings.pbp_inputs import (
    PASSER_GAME_SCHEMA,
    SeasonSources,
    passer_games_from_pbp,
    quarterback_rows,
    team_games_from_pbp,
)
from nfl_predictor.ratings.pre_week import (
    QB_PRE_WEEK_COLUMNS,
    TEAM_PRE_WEEK_COLUMNS,
    RatingsHistory,
)
from nfl_predictor.ratings.qb_rating import (
    QB_DROPBACKS_COLUMN,
    QB_ID_COLUMN,
    QB_RATING_COLUMN,
    qb_rating_rows,
    qb_season_summary,
)
from nfl_predictor.ratings.season import (
    PUBLISHED_BOOTSTRAP,
    BootstrapSettings,
    build_qb_season,
    build_team_season,
)
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import nflverse_sources, qb_stats
from nfl_predictor.utils.polars.teamrankings import calculate_stat_differentials

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

# The season tables nfl-sos-ratings publishes, by its file names, and the two it writes only for
# the season in progress.
PUBLISHED_KINDS = (
    "ratings",
    "team_prior",
    "ratings_by_week",
    "rating_ranges",
    "rating_pairs",
    "qb_ratings",
    "qb_ratings_by_week",
    "qb_rating_ranges",
    "qb_rating_pairs",
)
IN_PROGRESS_KINDS = ("rating_ranges_by_week", "qb_rating_ranges_by_week")
# The pre-week tables a season's build carries beside its published tables.
PRE_WEEK_TEAMS = "pre_week_teams"
PRE_WEEK_QUARTERBACKS = "pre_week_quarterbacks"

# Pre-week columns and the feature names they publish as.
_TEAM_FEATURES = {
    "offense_rating": "ratings_offense",
    "defense_rating": "ratings_defense",
    "special_teams_rating": "ratings_special_teams",
    "team_rating": "ratings_team",
    "sos": "ratings_sos",
}
_QB_FEATURES = {QB_RATING_COLUMN: "ratings_qb", QB_DROPBACKS_COLUMN: "ratings_qb_dropbacks"}
_SIDES = ("away", "home")
_ROW = "_ratings_row"
_WEEK_KEYS = {"season": pl.Int64(), "week": pl.Int64()}
_TEAM_PRE_WEEK_SCHEMA = pl.Schema(
    {
        **_WEEK_KEYS,
        "team": pl.String(),
        "games_played": pl.Int64(),
        **{column: pl.Float64() for column in TEAM_PRE_WEEK_COLUMNS[2:]},
    }
)
_QB_PRE_WEEK_SCHEMA = pl.Schema(
    {
        **_WEEK_KEYS,
        QB_ID_COLUMN: pl.String(),
        "qb_games_played": pl.Int64(),
        QB_DROPBACKS_COLUMN: pl.Int64(),
        **{column: pl.Float64() for column in QB_PRE_WEEK_COLUMNS[3:]},
    }
)


@dataclass(frozen=True)
class RatingsInputs:
    """Every season's rows for the ratings fits, and what the features read beside them.

    Attributes:
        team_games: ``season`` and the team-game rows (``team_games_from_pbp``).
        passer_games: Every passer-game row (``passer_games_from_pbp``), season by season.
        schedule: The scheduled games' ``season``, ``week``, ``game_type`` and teams, for each
            season's teams and the weeks that get pre-week ratings.
        starter_games: Quarterback-game sums (``qb_stats.aggregate_qb_game_stats``), for the
            expected starters' ids.
        starter_identity: The quarterback name-to-id map (``qb_stats.load_qb_identity``).

    """

    team_games: pl.DataFrame
    passer_games: pl.DataFrame
    schedule: pl.DataFrame
    starter_games: pl.DataFrame
    starter_identity: pl.DataFrame


def _with_earlier_seasons(
    loaded: pl.DataFrame, seasons: list[int], load: Callable[[list[int]], pl.DataFrame]
) -> pl.DataFrame:
    """Return ``loaded``'s rows of ``seasons`` with the seasons it lacks from ``load``."""
    present = (
        set(loaded.get_column("season").unique(maintain_order=True).to_list())
        if "season" in loaded.columns
        else set()
    )
    missing = [season for season in seasons if season not in present]
    parts = [loaded.filter(pl.col("season").is_in(seasons))] if present else []
    if missing:
        parts.append(load(missing))
    return pl.concat(parts, how="diagonal")


def load_ratings_sources(
    max_season: int,
    *,
    pbp_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    current_season: int,
    force_refresh: bool,
) -> LoadedSources | None:
    """Load every source the ratings read, from the first play-by-play season to ``max_season``.

    Every earlier season counts: the penalties come from the season before and the preseason
    prior from every season before. The play-by-play and schedules the ETL already loaded are
    reused, and the seasons they lack are read from the per-season caches without a refresh, as
    for the quarterback family's history. The player sources follow their loaders' cache
    contract (``nflverse_sources``), each asked only for the seasons it publishes; the players
    file is downloaded again when the run reaches the current season and then survives an outage
    with its cached copy. The quarterback identity is the quarterback family's file. The
    play-by-play and schedules are read through ``polars_utils``, as the rest of the ETL reads
    them; without any play there is nothing to rate, and no other source is read.

    Args:
        max_season: The last season of the run.
        pbp_df: The play-by-play the ETL loaded (any seasons, possibly none).
        schedule_df: The schedules the ETL loaded (any seasons, possibly none).
        current_season: The current season, for the loaders' cache decisions.
        force_refresh: Whether the run downloads its sources again (``--refresh-nflreadpy``).

    Returns:
        The sources of every season from ``constants.NFLREADPY_MIN_SEASON`` to ``max_season``,
        or ``None`` when no season has a play.

    """
    seasons = list(range(constants.NFLREADPY_MIN_SEASON, max_season + 1))
    plays = _with_earlier_seasons(
        pbp_df,
        seasons,
        lambda missing: polars_utils.load_pbp(
            missing, force_refresh=False, current_season=current_season
        ),
    )
    if plays.is_empty():
        log.warning("No play-by-play for the ratings; their columns stay null")
        return None

    def published(first: int) -> list[int]:
        return [season for season in seasons if season >= first]

    is_current = max_season >= current_season
    return LoadedSources(
        plays=plays,
        schedule=_with_earlier_seasons(
            schedule_df,
            seasons,
            lambda missing: polars_utils.load_schedule(
                missing, force_refresh=False, current_season=current_season
            ),
        ),
        player_stats=nflverse_sources.load_weekly_player_stats(
            published(constants.NFLREADPY_PLAYER_STATS_MIN_SEASON),
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        snap_counts=nflverse_sources.load_snap_counts(
            published(constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON),
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        players=nflverse_sources.load_players(
            force_refresh=force_refresh or is_current, degrade_on_failure=is_current
        ),
        weekly_rosters=nflverse_sources.load_weekly_rosters(
            published(constants.NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON),
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        starter_identity=qb_stats.load_qb_identity(
            constants.DATA_PATH / f"{constants.QB_META_DATA_NAME}.csv"
        ),
    )


def load_ratings_build(
    max_season: int,
    *,
    pbp_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    current_season: int,
    force_refresh: bool,
) -> RatingsBuild | None:
    """Return the ratings build of an ETL run, or ``None`` when no season has a play.

    The arguments are those of :func:`load_ratings_sources`; ``current_season`` is also the
    season the build gives weekly rank ranges.
    """
    sources = load_ratings_sources(
        max_season,
        pbp_df=pbp_df,
        schedule_df=schedule_df,
        current_season=current_season,
        force_refresh=force_refresh,
    )
    if sources is None:
        return None
    return RatingsBuild(ratings_inputs(sources), current_season=current_season)


def _of_season(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return the rows of ``season``, rechunked so later seasons cannot reach their sums."""
    return frame.filter(pl.col("season") == season).rechunk()


@dataclass(frozen=True)
class LoadedSources:
    """Every source the ratings read, for every season, as this repository's loaders return them.

    Attributes:
        plays: Regular-season play-by-play (``loaders.load_pbp``).
        schedule: The schedules (``loaders.load_schedule``).
        player_stats: Weekly player stats (``nflverse_sources.load_weekly_player_stats``).
        snap_counts: Snap counts (``nflverse_sources.load_snap_counts``).
        players: The players file (``nflverse_sources.load_players``).
        weekly_rosters: Regular-season weekly rosters (``nflverse_sources.load_weekly_rosters``).
        starter_identity: The quarterback name-to-id map (``qb_stats.load_qb_identity``).

    """

    plays: pl.DataFrame
    schedule: pl.DataFrame
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    players: pl.DataFrame
    weekly_rosters: pl.DataFrame
    starter_identity: pl.DataFrame


def ratings_inputs(sources: LoadedSources) -> RatingsInputs:
    """Return the ratings inputs of every season in ``sources.plays``."""
    plays, schedule = sources.plays, sources.schedule
    team_games = team_games_from_pbp(plays, schedule)
    seasons = sorted(team_games.get_column("season").unique(maintain_order=True).to_list())
    passers = [
        passer_games_from_pbp(
            SeasonSources(
                season=season,
                plays=_of_season(plays, season),
                player_stats=_of_season(sources.player_stats, season),
                snap_counts=_of_season(sources.snap_counts, season),
                players=sources.players,
                weekly_rosters=_of_season(sources.weekly_rosters, season),
            ),
            _of_season(team_games, season),
        )
        for season in seasons
    ]
    return RatingsInputs(
        team_games=team_games.rechunk(),
        passer_games=(
            pl.concat(passers).rechunk() if passers else pl.DataFrame(schema=PASSER_GAME_SCHEMA)
        ),
        schedule=schedule.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("game_type").cast(pl.String),
            pl.col("away_abbr").cast(pl.String),
            pl.col("home_abbr").cast(pl.String),
        ),
        starter_games=qb_stats.aggregate_qb_game_stats(plays),
        starter_identity=sources.starter_identity,
    )


class RatingsBuild:
    """The ratings part of one ETL run: each season's tables and pre-week ratings, and features.

    Call :meth:`build` (or :meth:`reuse` with a cached build's frames) for each season of the
    run, then :meth:`published_tables` for the files and :meth:`attach` for the game rows.
    """

    def __init__(
        self,
        inputs: RatingsInputs,
        *,
        current_season: int,
        first_season: int = constants.NFLREADPY_MIN_SEASON,
        bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
    ) -> None:
        """Remember the inputs and the run's current season.

        Args:
            inputs: Every season's rows (:func:`ratings_inputs`).
            current_season: The season in progress, the one that gets weekly rank ranges.
            first_season: The first season of rows, which has no earlier one for penalties.
            bootstrap: The game bootstrap behind the rank ranges (the published one).

        """
        self._inputs = inputs
        self._current_season = current_season
        self._first_season = first_season
        self._bootstrap = bootstrap
        self._team_rows = {
            int(key[0]): part.rechunk()
            for key, part in inputs.team_games.partition_by(
                "season", as_dict=True, maintain_order=True
            ).items()
        }
        self._passer_rows = {
            int(key[0]): part.rechunk()
            for key, part in inputs.passer_games.partition_by(
                "season", as_dict=True, maintain_order=True
            ).items()
        }
        self._history = RatingsHistory(
            self._season_team_games,
            self._season_quarterbacks,
            first_season=first_season,
            load_qb_games_before=self._quarterbacks_before,
        )
        self._frames: dict[int, dict[str, pl.DataFrame]] = {}

    def _season_team_games(self, season: int) -> pl.DataFrame:
        """Return one season's team-game rows, empty for a season without games."""
        rows = self._team_rows.get(season)
        return pl.DataFrame(schema=TEAM_GAME_SCHEMA) if rows is None else team_game_rows(rows)

    def _season_passers(self, season: int) -> pl.DataFrame:
        """Return one season's passer-game rows, empty for a season without games."""
        rows = self._passer_rows.get(season)
        return pl.DataFrame(schema=PASSER_GAME_SCHEMA) if rows is None else rows

    def _season_quarterbacks(self, season: int) -> pl.DataFrame:
        """Return one season's quarterback-game rows with the season's identity."""
        return quarterback_rows(self._season_passers(season))

    def _quarterbacks_before(self, season: int, week: int) -> pl.DataFrame:
        """Return one season's quarterback-game rows as the identity knew them before ``week``."""
        return quarterback_rows(self._season_passers(season), before_week=week)

    def key_frames(self) -> dict[str, pl.DataFrame]:
        """Return the rows a season's build reads, by name, each with its ``season``.

        A season's build reads these rows of that season and earlier ones (and the schedule,
        which the ETL's season cache keys on already), so they belong in its cache key.
        """
        return {
            "ratings_team_games": self._inputs.team_games,
            "ratings_passer_games": self._inputs.passer_games,
        }

    def build(self, season: int) -> dict[str, pl.DataFrame]:
        """Fit one season's tables and pre-week ratings, record them, and return them.

        Returns:
            The season's published tables by kind (none for a season without games) and its
            ``PRE_WEEK_TEAMS`` and ``PRE_WEEK_QUARTERBACKS`` tables.

        """
        start = time.perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            frames = {
                **self._tables(season),
                PRE_WEEK_TEAMS: self._pre_week_teams(season),
                PRE_WEEK_QUARTERBACKS: self._pre_week_quarterbacks(season),
            }
        self._frames[season] = frames
        log.info(
            "Ratings for season %d: %d tables and %d weeks of pre-week ratings in %.1fs",
            season,
            len(frames) - 2,
            frames[PRE_WEEK_TEAMS].get_column("week").n_unique(),
            time.perf_counter() - start,
        )
        return frames

    def reuse(self, season: int, frames: Mapping[str, pl.DataFrame]) -> None:
        """Record a season's frames from a cached build instead of fitting them again.

        Raises:
            ValueError: If ``frames`` lacks a pre-week table, so it is not a ratings build.

        """
        missing = [name for name in (PRE_WEEK_TEAMS, PRE_WEEK_QUARTERBACKS) if name not in frames]
        if missing:
            msg = f"season {season}'s cached ratings lack {', '.join(missing)}"
            raise ValueError(msg)
        self._frames[season] = dict(frames)

    def build_frames(self, season: int) -> dict[str, pl.DataFrame]:
        """Return the frames recorded for ``season`` by :meth:`build` or :meth:`reuse`."""
        return self._frames[season]

    def published_tables(self) -> dict[str, pl.DataFrame]:
        """Return every recorded season's published tables, by file stem ``{season}_{kind}``."""
        kinds = (*PUBLISHED_KINDS, *IN_PROGRESS_KINDS)
        return {
            f"{season}_{kind}": frame
            for season in sorted(self._frames)
            for kind, frame in self._frames[season].items()
            if kind in kinds
        }

    def _tables(self, season: int) -> dict[str, pl.DataFrame]:
        """Return one season's published tables, or none for a season without games."""
        team_games = self._season_team_games(season)
        if team_games.is_empty():
            return {}
        weekly = season == self._current_season
        team = build_team_season(
            season,
            team_games,
            self._history.team_priors,
            weekly_ranges=weekly,
            bootstrap=self._bootstrap,
        )
        tables = {
            "ratings": team.ratings,
            "team_prior": team.team_prior,
            "ratings_by_week": team.ratings_by_week,
            "rating_ranges": team.rating_ranges,
            "rating_pairs": team.rating_pairs,
        }
        quarterbacks = self._history.qb_games(season)
        if _has_qualified_passer(quarterbacks, team_games):
            qb = build_qb_season(
                quarterbacks, team_games, weekly_ranges=weekly, bootstrap=self._bootstrap
            )
            tables |= {
                "qb_ratings": qb.qb_ratings,
                "qb_ratings_by_week": qb.qb_ratings_by_week,
                "qb_rating_ranges": qb.qb_rating_ranges,
                "qb_rating_pairs": qb.qb_rating_pairs,
            }
            if qb.qb_rating_ranges_by_week is not None:
                tables["qb_rating_ranges_by_week"] = qb.qb_rating_ranges_by_week
        if team.rating_ranges_by_week is not None:
            tables["rating_ranges_by_week"] = team.rating_ranges_by_week
        return tables

    def _season_schedule(self, season: int) -> pl.DataFrame:
        return self._inputs.schedule.filter(pl.col("season") == season)

    def _teams(self, season: int) -> list[str]:
        """Return the teams of the season's regular-season schedule, known before kickoff."""
        schedule = self._season_schedule(season).filter(pl.col("game_type") == "REG")
        teams = pl.concat([schedule.get_column("away_abbr"), schedule.get_column("home_abbr")])
        return sorted(teams.drop_nulls().unique(maintain_order=True).to_list())

    def _weeks(self, season: int) -> list[tuple[int, int]]:
        """Return each scheduled week with the week whose pre-week ratings it reads.

        A postseason week reads the week after the regular season: the whole regular season.
        """
        after_regular_season = constants.get_regular_season_weeks(season) + 1
        weeks = self._season_schedule(season).get_column("week").unique(maintain_order=True)
        return [(week, min(week, after_regular_season)) for week in sorted(weeks.to_list())]

    def _pre_week_teams(self, season: int) -> pl.DataFrame:
        """Return every team's pre-week ratings for each scheduled week of the season."""
        teams = self._teams(season)
        fits: dict[int, pl.DataFrame] = {}
        parts = []
        for week, read_week in self._weeks(season):
            if read_week not in fits:
                fits[read_week] = self._history.pre_week_teams(season, read_week, teams)
            parts.append(_stamped(fits[read_week], season, week))
        return pl.concat(parts) if parts else pl.DataFrame(schema=_TEAM_PRE_WEEK_SCHEMA)

    def _pre_week_quarterbacks(self, season: int) -> pl.DataFrame:
        """Return every passer's pre-week rating for each scheduled week of the season."""
        fits: dict[int, pl.DataFrame] = {}
        parts = []
        for week, read_week in self._weeks(season):
            if read_week not in fits:
                fits[read_week] = self._history.pre_week_quarterbacks(season, read_week)
            parts.append(_stamped(fits[read_week], season, week))
        return pl.concat(parts) if parts else pl.DataFrame(schema=_QB_PRE_WEEK_SCHEMA)

    def _feature_table(self, name: str, key: str, features: Mapping[str, str]) -> pl.DataFrame:
        """Return one pre-week table of every recorded season, renamed to the feature names."""
        schema = _TEAM_PRE_WEEK_SCHEMA if name == PRE_WEEK_TEAMS else _QB_PRE_WEEK_SCHEMA
        frames = [self._frames[season][name] for season in sorted(self._frames)]
        table = pl.concat(frames) if frames else pl.DataFrame(schema=schema)
        return table.select(
            "season",
            "week",
            key,
            *(pl.col(column).alias(name) for column, name in features.items()),
        )

    def attach(self, games: pl.DataFrame) -> pl.DataFrame:
        """Return ``games`` with the ``ratings`` feature group, in the same row order.

        Args:
            games: Game rows with ``season``, ``week``, ``away_abbr`` and ``home_abbr``, and the
                expected starters as ``away_qb`` / ``home_qb``.

        Returns:
            ``games`` with ``away_<stat>``, ``home_<stat>`` and ``<stat>_diff`` (away minus
            home) for every stat of ``constants.RATINGS_FEATURE_STATS``, replacing any already
            there; null for a season not built.

        """
        stats = list(constants.RATINGS_FEATURE_STATS)
        new_columns = [f"{side}_{stat}" for stat in stats for side in _SIDES]
        new_columns += [f"{stat}_diff" for stat in stats]
        base = games.drop([column for column in new_columns if column in games.columns])
        starters = qb_stats.expected_starter_ids(
            base, self._inputs.starter_games, self._inputs.starter_identity
        )
        keys = (
            base.select(
                pl.col("season").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                *(pl.col(f"{side}_abbr").cast(pl.String) for side in _SIDES),
            )
            .hstack(starters)
            .with_row_index(_ROW)
        )
        teams = self._feature_table(PRE_WEEK_TEAMS, "team", _TEAM_FEATURES)
        quarterbacks = self._feature_table(PRE_WEEK_QUARTERBACKS, QB_ID_COLUMN, _QB_FEATURES)
        rated = [season for season in self._frames if season > self._first_season]
        dropbacks = _QB_FEATURES[QB_DROPBACKS_COLUMN]
        out = keys.select(_ROW)
        for side in _SIDES:
            part = (
                keys.select(
                    _ROW,
                    "season",
                    "week",
                    pl.col(f"{side}_abbr").alias("team"),
                    pl.col(f"{side}_qb_id").alias(QB_ID_COLUMN),
                )
                .join(teams, on=["season", "week", "team"], how="left")
                .join(quarterbacks, on=["season", "week", QB_ID_COLUMN], how="left")
                .with_columns(
                    pl.when(
                        pl.col(dropbacks).is_null()
                        & pl.col(QB_ID_COLUMN).is_not_null()
                        & pl.col("season").is_in(rated)
                    )
                    .then(pl.lit(0, dtype=pl.Int64))
                    .otherwise(pl.col(dropbacks))
                    .alias(dropbacks)
                )
                .select(_ROW, *(pl.col(stat).alias(f"{side}_{stat}") for stat in stats))
            )
            out = out.join(part, on=_ROW, how="left")
        out = calculate_stat_differentials(out.sort(_ROW), stats)
        return base.hstack(out.select(new_columns))


def _has_qualified_passer(quarterbacks: pl.DataFrame, team_games: pl.DataFrame) -> bool:
    """Return whether a passer qualifies for the season's quarterback rankings.

    The quarterback tables rank the qualified passers (14 attempts a game of their primary team),
    so a season without one (a season in progress after its first game, at worst) gets none.
    """
    if qb_rating_rows(quarterbacks).is_empty():
        return False
    return bool(qb_season_summary(quarterbacks, team_games).get_column("qb_is_eligible").any())


def _stamped(frame: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return a pre-week table with its ``season`` and ``week`` in front."""
    return frame.select(
        pl.lit(season, dtype=pl.Int64).alias("season"),
        pl.lit(week, dtype=pl.Int64).alias("week"),
        pl.all(),
    )


def write_published_tables(tables: Mapping[str, pl.DataFrame], directory: Path) -> None:
    """Write each table as ``{name}.parquet`` under ``directory``, creating it as needed."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(directory / f"{name}.parquet")
    log.info("Saved %d ratings tables to %s", len(tables), directory)


__all__ = [
    "IN_PROGRESS_KINDS",
    "PRE_WEEK_QUARTERBACKS",
    "PRE_WEEK_TEAMS",
    "PUBLISHED_KINDS",
    "LoadedSources",
    "RatingsBuild",
    "RatingsInputs",
    "load_ratings_build",
    "load_ratings_sources",
    "ratings_inputs",
    "write_published_tables",
]
