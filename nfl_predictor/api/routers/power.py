"""Power rankings routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nfl_predictor.api.deps import CurrentUser, DbDep
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.readers import power as reader
from nfl_predictor.api.routers.resolve import get_index
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.indexer import RunIndex, RunSummary
from nfl_predictor.api.schemas.data import PowerOut

router = APIRouter(prefix="/power", tags=["power"])


def previous_rankings(index: RunIndex, current: RunSummary) -> RunSummary | None:
    """Find the newest other run whose rankings are stamped one week before ``current``."""
    files = current.run_files
    if files.power_season is None or files.power_week is None:
        return None
    for run in index.runs():
        other = run.run_files
        if run.run_id == current.run_id or other.power_rankings is None:
            continue
        if other.power_season == files.power_season and other.power_week == files.power_week - 1:
            return run
    return None


@router.get("", response_model=PowerOut)
def power(_user: CurrentUser, db: DbDep, request: Request, run: str | None = None) -> PowerOut:
    """Return rankings, movement, and projected standings for a run."""
    index = get_index(request)
    current = active.require_run(db, index, run)
    files = current.run_files
    if files.power_rankings is None:
        raise NotFoundError(
            f"Run {current.run_id!r} has no power rankings", code="no_power_rankings"
        )
    previous = previous_rankings(index, current)
    previous_path = previous.run_files.power_rankings if previous else None
    return PowerOut(
        run_id=current.run_id,
        season=files.power_season,
        through_week=files.power_week,
        previous_run_id=previous.run_id if previous else None,
        rankings=reader.read_rankings(files.power_rankings, previous_path),
        standings=reader.read_standings(files.standings) if files.standings else None,
        division_standings=reader.read_standings(files.division_standings)
        if files.division_standings
        else None,
    )
