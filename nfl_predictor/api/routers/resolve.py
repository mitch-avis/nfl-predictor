"""Resolve which predictions file a request refers to: a run's, or an unattached one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Request

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.readers.data_status import unattached_files
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.indexer import RunIndex, RunSummary
from nfl_predictor.api.schemas.data import WeekRef
from nfl_predictor.api.settings import Settings


@dataclass(frozen=True)
class PredictionSource:
    """A predictions CSV and where it came from."""

    path: Path
    source: str
    run: RunSummary | None
    season: int | None
    week: int | None

    @property
    def run_id(self) -> str | None:
        """Return the owning run id, if any."""
        return self.run.run_id if self.run else None

    @property
    def generated_at(self) -> str:
        """Return the file's modification time in ISO-8601 UTC."""
        return datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC).isoformat()


def get_index(request: Request) -> RunIndex:
    """Return the application's run index."""
    index: RunIndex = request.app.state.run_index
    return index


def available_weeks(
    settings: Settings, index: RunIndex, active_run: RunSummary | None
) -> list[WeekRef]:
    """List every week with predictions: the active run's, other runs', and unattached files."""
    refs: list[WeekRef] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    if active_run and active_run.run_files.predictions is not None:
        refs.append(
            WeekRef(
                season=active_run.season,
                week=active_run.week,
                source="active",
                run_id=active_run.run_id,
                label=f"Week {active_run.week} (active run)",
            )
        )
        seen.add(("run", active_run.season, active_run.week))
    for run in index.runs():
        if run.run_files.predictions is None or run is active_run:
            continue
        key = ("run", run.season, run.week)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            WeekRef(
                season=run.season,
                week=run.week,
                source="run",
                run_id=run.run_id,
                label=f"Week {run.week} ({run.run_id})",
            )
        )
    for item in unattached_files(settings.data_path, settings.reports_path):
        if item["kind"] != "predictions":
            continue
        refs.append(
            WeekRef(
                season=item["season"],
                week=item["week"],
                source="unattached",
                run_id=None,
                label=f"Week {item['week']} (data/predict)",
            )
        )
    return refs


def resolve_predictions(
    request: Request,
    db: Database,
    settings: Settings,
    *,
    run_id: str | None,
    season: int | None,
    week: int | None,
    source: str | None,
) -> PredictionSource:
    """Pick the predictions file for the query.

    Precedence: an explicit ``run_id``; else the active run when it matches (or no week was
    asked for); else any run predicting that week; else an unattached ``data/predict`` file.
    """
    index = get_index(request)
    if run_id is not None:
        run = active.require_run(db, index, run_id)
        if run.run_files.predictions is None:
            raise NotFoundError(f"Run {run_id!r} has no predictions", code="no_predictions")
        return PredictionSource(run.run_files.predictions, "run", run, run.season, run.week)
    current = active.resolve_active_run(db, index)
    wants_week = season is not None or week is not None

    def _matches(run: RunSummary) -> bool:
        return (season is None or run.season == season) and (week is None or run.week == week)

    if (
        source != "unattached"
        and current
        and current.run_files.predictions is not None
        and (not wants_week or _matches(current))
    ):
        return PredictionSource(
            current.run_files.predictions, "active", current, current.season, current.week
        )
    if source != "unattached":
        for run in index.runs():
            if run.run_files.predictions is not None and _matches(run):
                return PredictionSource(run.run_files.predictions, "run", run, run.season, run.week)
    for item in unattached_files(settings.data_path, settings.reports_path):
        if item["kind"] != "predictions":
            continue
        if (season is None or item["season"] == season) and (week is None or item["week"] == week):
            return PredictionSource(
                Path(item["path"]), "unattached", None, item["season"], item["week"]
            )
    if wants_week:
        raise NotFoundError(
            f"No predictions for season {season} week {week}", code="no_predictions"
        )
    raise NotFoundError(
        "No predictions found; run a weekly run or a predict job", code="no_predictions"
    )
