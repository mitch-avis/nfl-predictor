"""Routes for the team and quarterback pages: season tables, sections, schedules and glossary.

Every route reads the season tables the ETL writes to ``data/ratings/`` (``Settings.ratings_path``)
and none writes anything. A missing season or table is a 404 with code ``no_ratings_table``
naming the file, and a team or quarterback without rows a 404 with code ``no_ratings_entity``,
so a page can hide a section the season was built without.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from nfl_predictor.api.deps import CurrentUser, SettingsDep
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.ratings import files, glossary, schedules, sections, tables
from nfl_predictor.api.schemas.ratings import (
    EntityKind,
    EntityTableOut,
    MetricRegistryOut,
    RatingsSeasonsOut,
    ScheduleRanksOut,
    SeasonTableOut,
)
from nfl_predictor.utils import clock

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from nfl_predictor.api.schemas.ratings import MetricTableOut

router = APIRouter(prefix="/ratings", tags=["ratings"])

INDEX_KIND: dict[EntityKind, str] = {"teams": "combined", "qbs": "qb_combined"}
GAME_LOG_KIND: dict[EntityKind, str] = {"teams": "team_game_logs", "qbs": "qb_game_logs"}
RATING_HISTORY_KIND: dict[EntityKind, str] = {
    "teams": "ratings_by_week",
    "qbs": "qb_ratings_by_week",
}
RANK_RANGE_KIND: dict[EntityKind, str] = {"teams": "rating_ranges", "qbs": "qb_rating_ranges"}
RANK_HISTORY_KIND: dict[EntityKind, str] = {
    "teams": "rating_ranges_by_week",
    "qbs": "qb_rating_ranges_by_week",
}
PAIR_KIND: dict[EntityKind, str] = {"teams": "rating_pairs", "qbs": "qb_rating_pairs"}
ID_COLUMN: dict[EntityKind, str] = {"teams": "team", "qbs": "qb_id"}
TEAM_RATINGS_KIND = "ratings"
TEAM_INDEX_KIND = INDEX_KIND["teams"]


def current_season() -> int:
    """Return the NFL season today's date falls in, by the ETL's own calendar."""
    return clock.nfl_season(clock.local_today())


def _read(ratings_dir: Path, season: int, kind: str) -> pl.DataFrame:
    """Return one season's table, or raise the 404 that names the missing file."""
    try:
        return files.read_table(ratings_dir, season, kind)
    except files.RatingsTableMissingError as exc:
        raise NotFoundError(str(exc), code="no_ratings_table") from exc


def _read_optional(ratings_dir: Path, season: int, kind: str) -> pl.DataFrame | None:
    """Return one season's table when the season has it."""
    if not files.has_table(ratings_dir, season, kind):
        return None
    return files.read_table(ratings_dir, season, kind)


def _entity_rows(
    ratings_dir: Path, season: int, kind: EntityKind, table_kind: str, entity_id: str
) -> pl.DataFrame:
    """Return one team's or quarterback's rows of a season table, or raise a 404."""
    frame = _read(ratings_dir, season, table_kind)
    try:
        return files.entity_rows(frame, ID_COLUMN[kind], entity_id)
    except files.EntityNotFoundError as exc:
        msg = f"Season {season} has no {table_kind} rows for {entity_id}"
        raise NotFoundError(msg, code="no_ratings_entity") from exc


def _section(
    kind: EntityKind, season: int, entity_id: str | None, table: MetricTableOut
) -> SeasonTableOut:
    """Wrap a section table with what it describes."""
    return SeasonTableOut(kind=kind, season=season, entity_id=entity_id, table=table)


@router.get("/seasons", response_model=RatingsSeasonsOut)
def seasons(_user: CurrentUser, settings: SettingsDep) -> RatingsSeasonsOut:
    """List the seasons with team and quarterback tables, newest first."""
    return RatingsSeasonsOut(
        seasons=files.available_seasons(settings.ratings_path), current_season=current_season()
    )


@router.get("/metrics", response_model=MetricRegistryOut)
def metrics(_user: CurrentUser) -> MetricRegistryOut:
    """Return the metric registry the glossary is built from."""
    return glossary.metric_registry()


@router.get("/schedules", response_model=ScheduleRanksOut)
def schedule_ranks(_user: CurrentUser, settings: SettingsDep) -> ScheduleRanksOut:
    """Rank every completed team-season on schedule strength, softest first."""
    return schedules.schedule_ranks(settings.ratings_path)


@router.get("/{season}/{kind}", response_model=EntityTableOut)
def entity_table(
    season: int, kind: EntityKind, _user: CurrentUser, settings: SettingsDep
) -> EntityTableOut:
    """Return one season's team or quarterback table with its metric categories."""
    ratings_dir = settings.ratings_path
    frame = _read(ratings_dir, season, INDEX_KIND[kind])
    team_frame = frame if kind == "teams" else _read(ratings_dir, season, TEAM_INDEX_KIND)
    in_progress, games_played = tables.season_progress(team_frame, season, current_season())
    return EntityTableOut(
        kind=kind,
        season=season,
        in_progress=in_progress,
        games_played=games_played,
        games_per_team=tables.games_per_team(season),
        table=tables.index_table(kind, frame),
        categories=tables.categories(kind),
    )


@router.get("/{season}/{kind}/rank-ranges", response_model=SeasonTableOut)
def rank_ranges(
    season: int, kind: EntityKind, _user: CurrentUser, settings: SettingsDep
) -> SeasonTableOut:
    """Return every team's or qualifying quarterback's rank range, by published rank."""
    frame = _read(settings.ratings_path, season, RANK_RANGE_KIND[kind])
    return _section(kind, season, None, sections.rank_range_table(kind, frame))


@router.get("/{season}/{kind}/{entity_id}/game-logs", response_model=SeasonTableOut)
def game_logs(
    season: int, kind: EntityKind, entity_id: str, _user: CurrentUser, settings: SettingsDep
) -> SeasonTableOut:
    """Return one team's or quarterback's games, each with its opponent's season ratings."""
    ratings_dir = settings.ratings_path
    rows = _entity_rows(ratings_dir, season, kind, GAME_LOG_KIND[kind], entity_id)
    team_ratings = _read_optional(ratings_dir, season, TEAM_RATINGS_KIND)
    return _section(kind, season, entity_id, sections.game_log_table(kind, rows, team_ratings))


@router.get("/{season}/{kind}/{entity_id}/rating-history", response_model=SeasonTableOut)
def rating_history(
    season: int, kind: EntityKind, entity_id: str, _user: CurrentUser, settings: SettingsDep
) -> SeasonTableOut:
    """Return one team's or quarterback's rating after each week of the season."""
    rows = _entity_rows(settings.ratings_path, season, kind, RATING_HISTORY_KIND[kind], entity_id)
    return _section(kind, season, entity_id, sections.rating_history_table(kind, rows))


@router.get("/{season}/{kind}/{entity_id}/rank-history", response_model=SeasonTableOut)
def rank_history(
    season: int, kind: EntityKind, entity_id: str, _user: CurrentUser, settings: SettingsDep
) -> SeasonTableOut:
    """Return one team's or quarterback's rank range after each week of a season in progress.

    Weeks before every team has played three games are left out
    (``sections.first_rank_history_week``), quarterbacks' included.
    """
    ratings_dir = settings.ratings_path
    rows = _entity_rows(ratings_dir, season, kind, RANK_HISTORY_KIND[kind], entity_id)
    team_history = _read_optional(ratings_dir, season, RATING_HISTORY_KIND["teams"])
    first_week = sections.first_rank_history_week(team_history)
    return _section(
        kind, season, entity_id, sections.rank_history_table(kind, rows, first_week=first_week)
    )


@router.get("/{season}/{kind}/{entity_id}/head-to-head", response_model=SeasonTableOut)
def head_to_head(
    season: int, kind: EntityKind, entity_id: str, _user: CurrentUser, settings: SettingsDep
) -> SeasonTableOut:
    """Return how often one team or quarterback was rated above each other one."""
    rows = _entity_rows(settings.ratings_path, season, kind, PAIR_KIND[kind], entity_id)
    return _section(kind, season, entity_id, sections.pair_table(kind, rows))
