"""Model metadata and metrics routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from nfl_predictor.api.deps import CurrentUser, DbDep
from nfl_predictor.api.readers.model import model_payload
from nfl_predictor.api.routers.resolve import get_index
from nfl_predictor.api.runs import active
from nfl_predictor.api.schemas.data import ModelOut

router = APIRouter(tags=["model"])


def _payload(request: Request, db: DbDep, run_id: str | None) -> ModelOut:
    """Build the model payload for ``run_id`` or the active run."""
    run = active.require_run(db, get_index(request), run_id)
    return ModelOut(run_id=run.run_id, kind=run.kind, **model_payload(run.run_files))


@router.get("/model", response_model=ModelOut)
def model(_user: CurrentUser, db: DbDep, request: Request, run: str | None = None) -> ModelOut:
    """Return the model page payload for the active (or given) run."""
    return _payload(request, db, run)


@router.get("/runs/{run_id}/model", response_model=ModelOut)
def run_model(run_id: str, _user: CurrentUser, db: DbDep, request: Request) -> ModelOut:
    """Return the model page payload for one run."""
    return _payload(request, db, run_id)
