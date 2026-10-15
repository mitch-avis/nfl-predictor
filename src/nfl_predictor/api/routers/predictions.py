"""Predictions and confidence-picks routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import polars as pl
from fastapi import APIRouter, Query, Request, Response

from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.readers import predictions as reader
from nfl_predictor.api.routers.resolve import (
    available_weeks,
    current_picks_file,
    get_index,
    resolve_predictions,
)
from nfl_predictor.api.runs import active
from nfl_predictor.api.schemas.data import (
    PicksOut,
    PicksQuery,
    PredictionQuery,
    PredictionsOut,
    WeekRef,
)

if TYPE_CHECKING:
    from nfl_predictor.api.schemas.common import TablePayload

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionsOut)
def predictions(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    query: Annotated[PredictionQuery, Query()],
) -> PredictionsOut:
    """Return a week of predictions with market context."""
    src = resolve_predictions(request, db, settings, query)
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


def _picks_csv(table: TablePayload, season: int | None, week: int | None) -> Response:
    """Return the picks table as a CSV download named after its season and week."""
    columns = table.visible_columns
    frame = pl.DataFrame(
        [[row.get(column) for column in columns] for row in table.rows],
        schema=columns,
        orient="row",
    )
    name = (
        f"season_{season}_week_{week:02d}_confidence_picks.csv"
        if season is not None and week is not None
        else "confidence_picks.csv"
    )
    return Response(
        content=frame.write_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/picks", response_model=PicksOut)
def picks(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    query: Annotated[PicksQuery, Query()],
) -> PicksOut | Response:
    """Return the confidence picks, most confident first; ``format=csv`` downloads them."""
    src = resolve_predictions(request, db, settings, query)
    picks_path = current_picks_file(src)
    table = (
        reader.read_picks(picks_path)
        if picks_path is not None
        else reader.picks_from_predictions(src.path)
    )
    if query.format == "csv":
        return _picks_csv(table, src.season, src.week)
    return PicksOut(run_id=src.run_id, season=src.season, week=src.week, table=table)
