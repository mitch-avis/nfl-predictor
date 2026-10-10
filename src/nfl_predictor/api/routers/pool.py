"""Pool helper routes: the confidence sheet, the tiebreaker card and the survivor planner.

Each reads the predictions file the Predictions page shows for the same week (see
``routers/resolve.py``) and the final scores in the ETL's dataset; none writes anything.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Query, Request

from nfl_predictor import constants
from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.errors import NotFoundError, UnprocessableEntityError
from nfl_predictor.api.pool.results import final_scores
from nfl_predictor.api.pool.sheet import build_sheet
from nfl_predictor.api.pool.survivor import plan_survivor, week_options
from nfl_predictor.api.pool.tiebreakers import build_tiebreakers
from nfl_predictor.api.readers.predictions import load_frame
from nfl_predictor.api.routers.resolve import resolve_predictions
from nfl_predictor.api.schemas.data import PredictionQuery, WeekQuery
from nfl_predictor.api.schemas.pool import (
    ConfidenceSheetOut,
    FutureValueOut,
    SurvivorOut,
    SurvivorPickOut,
    SurvivorPlanOut,
    SurvivorRequest,
    TiebreakersOut,
)

if TYPE_CHECKING:
    from nfl_predictor.api.db import Database
    from nfl_predictor.api.pool.survivor import Option, Plan
    from nfl_predictor.api.routers.resolve import PredictionSource
    from nfl_predictor.api.settings import Settings

router = APIRouter(prefix="/pool", tags=["pool"])


def _source_fields(src: PredictionSource) -> dict[str, Any]:
    """Return the fields every week payload shares: which file, which week, written when."""
    return {
        "run_id": src.run_id,
        "source": src.source,
        "season": src.season,
        "week": src.week,
        "generated_at": src.generated_at,
    }


@router.get("/confidence", response_model=ConfidenceSheetOut)
def confidence(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    query: Annotated[PredictionQuery, Query()],
) -> ConfidenceSheetOut:
    """Return the week's confidence pool sheet, scored against any final scores."""
    src = resolve_predictions(request, db, settings, query)
    written = datetime.fromtimestamp(src.path.stat().st_mtime, tz=UTC)
    sheet = build_sheet(load_frame(src.path), final_scores(settings.data_path), written)
    return ConfidenceSheetOut.model_validate({**asdict(sheet), **_source_fields(src)})


@router.get("/tiebreakers", response_model=TiebreakersOut)
def tiebreakers(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    query: Annotated[PredictionQuery, Query()],
) -> TiebreakersOut:
    """Return the week's tiebreaker card."""
    src = resolve_predictions(request, db, settings, query)
    card = build_tiebreakers(load_frame(src.path))
    return TiebreakersOut.model_validate({**asdict(card), **_source_fields(src)})


def _used_teams(teams: list[str]) -> list[str]:
    """Return the canonical abbreviations of ``teams``, or reject one that is not a team."""
    canonical: set[str] = set()
    for team in teams:
        abbr = constants.normalize_team_abbr(team.strip().upper())
        if abbr not in constants.TEAM_ABBR:
            msg = f"Unknown team {team!r}; use a team abbreviation such as KC or WSH."
            raise UnprocessableEntityError(msg, code="unknown_team")
        canonical.add(abbr)
    return sorted(canonical)


def _season_options(
    request: Request, db: Database, settings: Settings, body: SurvivorRequest, last_week: int
) -> tuple[dict[int, list[Option]], list[int]]:
    """Return the open options of every week from the start week on, and the weeks without any."""
    finals = final_scores(settings.data_path)
    weeks: dict[int, list[Option]] = {}
    missing: list[int] = []
    for week in range(body.start_week, last_week + 1):
        try:
            src = resolve_predictions(
                request, db, settings, WeekQuery(season=body.season, week=week)
            )
        except NotFoundError:
            missing.append(week)
            continue
        weeks[week] = week_options(load_frame(src.path), week, finals)
    return weeks, missing


def _plan_out(rank: int, plan: Plan) -> SurvivorPlanOut:
    """Convert a plan to its payload."""
    return SurvivorPlanOut(
        rank=rank,
        picks=[
            SurvivorPickOut(
                **asdict(pick.option),
                later=pick.later,
                cumulative_survival=pick.cumulative_survival,
            )
            for pick in plan.picks
        ],
        survival=plan.survival,
        later_survival=plan.later_survival,
        total_survival=plan.total_survival,
    )


@router.post("/survivor", response_model=SurvivorOut)
def survivor(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    body: SurvivorRequest,
) -> SurvivorOut:
    """Rank survivor plans from the start week over every regular-season week with predictions."""
    last_week = constants.get_regular_season_weeks(body.season)
    if body.start_week > last_week:
        msg = f"Survivor pools run through the regular season, which ends in week {last_week}."
        raise UnprocessableEntityError(msg, code="past_regular_season")
    used = _used_teams(body.used_teams)
    weeks, missing = _season_options(request, db, settings, body, last_week)
    if body.start_week in missing:
        msg = (
            f"No predictions for season {body.season} week {body.start_week}; predict the week "
            "before planning from it."
        )
        raise NotFoundError(msg, code="no_predictions")
    horizon_end = body.start_week + body.horizon_weeks - 1
    planned_weeks = [week for week, options in weeks.items() if week <= horizon_end and options]
    later_weeks = [week for week, options in weeks.items() if week > horizon_end and options]
    result = plan_survivor(
        [weeks[week] for week in planned_weeks],
        [weeks[week] for week in later_weeks],
        used,
    )
    values = sorted(result.future_values.items(), key=lambda item: (-item[1], item[0]))
    return SurvivorOut(
        season=body.season,
        start_week=body.start_week,
        planned_weeks=planned_weeks,
        later_weeks=later_weeks,
        missing_weeks=missing,
        used_teams=used,
        teams=sorted(constants.TEAM_ABBR),
        plans=[_plan_out(rank, plan) for rank, plan in enumerate(result.plans, 1)],
        future_values=[FutureValueOut(team=team, value=value) for team, value in values],
    )
