"""The admin-chosen active run, with a fallback to the newest complete weekly run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfl_predictor.api.errors import ConflictError, NotFoundError
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from nfl_predictor.api.db import Database
    from nfl_predictor.api.runs.indexer import RunIndex, RunSummary

ACTIVE_RUN_KEY = "active_run_id"


def pinned_run_id(db: Database) -> str | None:
    """Return the pinned run id, if any."""
    return db.get_value(ACTIVE_RUN_KEY)


def set_active_run(db: Database, index: RunIndex, run_id: str) -> RunSummary:
    """Pin ``run_id`` as the active run; it must exist and carry a model."""
    run = index.get(run_id)
    if run is None:
        msg = f"Run {run_id!r} not found"
        raise NotFoundError(msg, code="run_not_found")
    if not run.has_model:
        msg = f"Run {run_id!r} has no model.joblib"
        raise ConflictError(msg, code="run_has_no_model")
    db.set_value(ACTIVE_RUN_KEY, run_id)
    return run


def clear_active_run(db: Database) -> None:
    """Remove the pin so the fallback applies again."""
    db.set_value(ACTIVE_RUN_KEY, None)


def resolve_active_run(db: Database, index: RunIndex) -> RunSummary | None:
    """Return the pinned run, else the newest complete weekly run, else ``None``.

    A pin pointing at a directory that no longer exists is ignored (and logged) rather than
    raised, so a deleted run never breaks every page.
    """
    pinned = pinned_run_id(db)
    if pinned is not None:
        run = index.get(pinned)
        if run is not None:
            return run
        log.warning("Pinned active run %r no longer exists; falling back", pinned)
    for run in index.runs():
        if run.kind == "weekly" and run.complete and run.has_model:
            return run
    return None


def require_run(db: Database, index: RunIndex, run_id: str | None) -> RunSummary:
    """Return ``run_id`` when given, else the active run; raise 404 when neither resolves."""
    if run_id is not None:
        run = index.get(run_id)
        if run is None:
            msg = f"Run {run_id!r} not found"
            raise NotFoundError(msg, code="run_not_found")
        return run
    run = resolve_active_run(db, index)
    if run is None:
        msg = "No active run; pin one from the Runs page"
        raise NotFoundError(msg, code="no_active_run")
    return run
