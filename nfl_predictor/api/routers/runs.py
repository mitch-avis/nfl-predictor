"""Run listing, detail, activation, and artifact download routes."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from nfl_predictor.api.deps import AdminUser, CurrentUser, DbDep
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.indexer import RunIndex, RunSummary
from nfl_predictor.api.schemas.runs import RunDetailOut, RunListOut, RunOut

router = APIRouter(prefix="/runs", tags=["runs"])

DOWNLOADABLE = {
    "betting_xlsx": ("betting_xlsx", "betting_report.xlsx"),
    "predictions": ("predictions", "predictions.csv"),
    "picks": ("picks", "confidence_picks.csv"),
    "betting_csv": ("betting_csv", "betting_report.csv"),
    "power_rankings": ("power_rankings", "power_rankings.csv"),
    "standings": ("standings", "projected_standings.csv"),
    "division_standings": ("division_standings", "projected_division_standings.csv"),
    "metadata": ("metadata", "metadata.json"),
    "metrics": ("metrics", "metrics_report.json"),
}


def get_index(request: Request) -> RunIndex:
    """Return the application's run index."""
    index: RunIndex = request.app.state.run_index
    return index


def to_run_out(run: RunSummary, active_run_id: str | None) -> RunOut:
    """Convert a summary into its API shape."""
    return RunOut(
        run_id=run.run_id,
        created_at=run.created_at,
        kind=run.kind,
        season=run.season,
        week=run.week,
        stages=run.stages,
        complete=run.complete,
        git_commit_hash=run.git_commit_hash,
        dataset_hash=run.dataset_hash,
        model_kind=run.model_kind,
        holdout=run.holdout,
        files=run.files,
        is_active=run.run_id == active_run_id,
    )


@router.get("", response_model=RunListOut)
def list_runs(
    _user: CurrentUser,
    db: DbDep,
    request: Request,
    kind: Literal["weekly", "training", "walk_forward", "all"] = "all",
) -> RunListOut:
    """List runs newest first, optionally filtered by kind."""
    index = get_index(request)
    current = active.resolve_active_run(db, index)
    active_id = current.run_id if current else None
    runs = [run for run in index.runs() if kind == "all" or run.kind == kind]
    return RunListOut(
        active_run_id=active_id,
        pinned_run_id=active.pinned_run_id(db),
        runs=[to_run_out(run, active_id) for run in runs],
    )


@router.get("/{run_id}", response_model=RunDetailOut)
def run_detail(run_id: str, _user: CurrentUser, db: DbDep, request: Request) -> RunDetailOut:
    """Return one run with its metadata config."""
    index = get_index(request)
    run = index.get(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id!r} not found", code="run_not_found")
    current = active.resolve_active_run(db, index)
    metadata = json.loads(run.run_files.metadata.read_text(encoding="utf-8"))
    features = metadata.get("feature_list")
    base = to_run_out(run, current.run_id if current else None)
    return RunDetailOut(
        **base.model_dump(),
        config=metadata.get("config"),
        splits=metadata.get("splits"),
        library_versions=metadata.get("library_versions"),
        feature_count=len(features) if isinstance(features, list) else None,
    )


@router.post("/{run_id}/activate", response_model=RunOut)
def activate_run(run_id: str, _admin: AdminUser, db: DbDep, request: Request) -> RunOut:
    """Pin ``run_id`` as the active run."""
    index = get_index(request)
    index.invalidate()
    run = active.set_active_run(db, index, run_id)
    return to_run_out(run, run.run_id)


@router.delete("/active", status_code=204)
def clear_active(_admin: AdminUser, db: DbDep) -> None:
    """Remove the pin so the newest complete weekly run is used."""
    active.clear_active_run(db)


@router.get("/{run_id}/files/{name}")
def download_file(run_id: str, name: str, _user: CurrentUser, request: Request) -> FileResponse:
    """Download one allow-listed artifact of a run."""
    run = get_index(request).get(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id!r} not found", code="run_not_found")
    if name not in DOWNLOADABLE:
        raise NotFoundError(f"Unknown artifact {name!r}", code="unknown_artifact")
    attr, download_name = DOWNLOADABLE[name]
    path = getattr(run.run_files, attr)
    if path is None or not path.is_file():
        raise NotFoundError(f"Run {run_id!r} has no {name}", code="artifact_missing")
    return FileResponse(path, filename=f"{run_id}_{download_name}")
