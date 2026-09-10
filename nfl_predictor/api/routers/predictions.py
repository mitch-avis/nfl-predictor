"""Predictions and confidence-picks routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.readers import predictions as reader
from nfl_predictor.api.routers.resolve import available_weeks, get_index, resolve_predictions
from nfl_predictor.api.runs import active
from nfl_predictor.api.schemas.data import PicksOut, PredictionsOut, WeekRef

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionsOut)
def predictions(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    run: str | None = None,
    season: int | None = None,
    week: int | None = None,
    source: str | None = None,
) -> PredictionsOut:
    """Return a week of predictions with market context."""
    src = resolve_predictions(
        request, db, settings, run_id=run, season=season, week=week, source=source
    )
    table, summary = reader.read_predictions(src.path)
    index = get_index(request)
    return PredictionsOut(
        run_id=src.run_id,
        source=src.source,
        season=src.season,
        week=src.week,
        generated_at=src.generated_at,
        table=table,
        summary=summary.__dict__,
        weeks=available_weeks(settings, index, active.resolve_active_run(db, index)),
    )


@router.get("/weeks", response_model=list[WeekRef])
def weeks(_user: CurrentUser, db: DbDep, settings: SettingsDep, request: Request) -> list[WeekRef]:
    """List every week that has predictions."""
    index = get_index(request)
    return available_weeks(settings, index, active.resolve_active_run(db, index))


@router.get("/picks", response_model=PicksOut)
def picks(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    run: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> PicksOut:
    """Return the confidence picks, most confident first."""
    src = resolve_predictions(
        request, db, settings, run_id=run, season=season, week=week, source=None
    )
    picks_path = src.run.run_files.picks if src.run else None
    table = (
        reader.read_picks(picks_path)
        if picks_path is not None and picks_path.is_file()
        else reader.picks_from_predictions(src.path)
    )
    return PicksOut(run_id=src.run_id, season=src.season, week=src.week, table=table)
