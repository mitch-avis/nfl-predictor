"""Betting report routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.readers import betting as reader
from nfl_predictor.api.readers import market
from nfl_predictor.api.routers.resolve import resolve_predictions
from nfl_predictor.api.schemas.data import BettingOut, LadderStep

router = APIRouter(prefix="/betting", tags=["betting"])

NOTES = [
    "Moneyline edges compare the model probability with the implied probability of the offered "
    "price, vig included.",
    "Spread and total probabilities assume a normal distribution centered on the prediction "
    "with σ from the p10/p90 quantiles.",
    "Totals are informational only: the total model has not shown an edge against the market line.",
    "Lines are as of the last data refresh; refresh lines before acting on them.",
]


def ladder() -> list[LadderStep]:
    """Return the action ladder from the lowest to the highest rung."""
    steps = [LadderStep(action="PASS", min_edge=0.0)]
    steps.extend(
        LadderStep(action=label, min_edge=edge) for edge, label in reversed(market.ACTION_LADDER)
    )
    return steps


@router.get("", response_model=BettingOut)
def betting(
    _user: CurrentUser,
    db: DbDep,
    settings: SettingsDep,
    request: Request,
    run: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> BettingOut:
    """Return the betting table derived from the week's predictions."""
    src = resolve_predictions(
        request, db, settings, run_id=run, season=season, week=week, source=None
    )
    xlsx = src.run.run_files.betting_xlsx if src.run else None
    return BettingOut(
        run_id=src.run_id,
        season=src.season,
        week=src.week,
        generated_at=src.generated_at,
        table=reader.read_betting(src.path),
        ladder=ladder(),
        xlsx_available=bool(xlsx and xlsx.is_file()),
        notes=NOTES,
    )


@router.get("/xlsx")
def betting_xlsx(
    _user: CurrentUser, db: DbDep, settings: SettingsDep, request: Request, run: str | None = None
) -> FileResponse:
    """Download the run's betting workbook."""
    src = resolve_predictions(
        request, db, settings, run_id=run, season=None, week=None, source=None
    )
    xlsx = src.run.run_files.betting_xlsx if src.run else None
    if xlsx is None or not xlsx.is_file():
        raise NotFoundError(
            "No betting workbook for this run; generate one from the Jobs page",
            code="artifact_missing",
        )
    return FileResponse(xlsx, filename=f"{src.run_id}_betting_report.xlsx")
