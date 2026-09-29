"""Resolve which predictions file a request refers to: a run's, or an unattached one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nfl_predictor import week_builder
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.readers.data_status import current_season_week, unattached_files
from nfl_predictor.api.runs import active
from nfl_predictor.api.schemas.data import PredictionQuery, WeekQuery, WeekRef

if TYPE_CHECKING:
    from fastapi import Request

    from nfl_predictor.api.db import Database
    from nfl_predictor.api.runs.indexer import RunIndex, RunSummary
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
    """List the weeks a client can select.

    Weeks that already have predictions come first, from the active run, other runs, and then
    unattached ``data/predict`` files. Weeks of the current season that are still unplayed and have
    no predictions follow with source ``available``, so the selector can offer to generate them.
    """
    refs: list[WeekRef] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    predicted: set[tuple[int | None, int | None]] = set()
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
        predicted.add((active_run.season, active_run.week))
    for run in index.runs():
        if run.run_files.predictions is None or run is active_run:
            continue
        key = ("run", run.season, run.week)
        if key in seen:
            continue
        seen.add(key)
        predicted.add((run.season, run.week))
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
        predicted.add((item["season"], item["week"]))
        refs.append(
            WeekRef(
                season=item["season"],
                week=item["week"],
                source="unattached",
                run_id=None,
                label=f"Week {item['week']} (data/predict)",
            )
        )
    refs.extend(generatable_weeks(settings, predicted))
    return refs


def generatable_weeks(
    settings: Settings, predicted: set[tuple[int | None, int | None]]
) -> list[WeekRef]:
    """Return the current season's unplayed weeks that have no predictions yet."""
    season, _current_week = current_season_week()
    return [
        WeekRef(
            season=season,
            week=week,
            source="available",
            run_id=None,
            label=f"Week {week} (not predicted yet)",
        )
        for week in week_builder.available_weeks(season, data_dir=settings.data_path)
        if (season, week) not in predicted
    ]


def _matches(query: WeekQuery, season: int | None, week: int | None) -> bool:
    """Return whether a season and week satisfy the query's season and week, where given."""
    return (query.season is None or season == query.season) and (
        query.week is None or week == query.week
    )


def _explicit_run_source(db: Database, index: RunIndex, run_id: str) -> PredictionSource:
    """Return the named run's predictions, or 404 when it has none."""
    run = active.require_run(db, index, run_id)
    if run.run_files.predictions is None:
        msg = f"Run {run_id!r} has no predictions"
        raise NotFoundError(msg, code="no_predictions")
    return PredictionSource(run.run_files.predictions, "run", run, run.season, run.week)


def _run_source(db: Database, index: RunIndex, query: WeekQuery) -> PredictionSource | None:
    """Return the active run's predictions when they fit the query, else any run's that do."""
    current = active.resolve_active_run(db, index)
    wants_week = query.season is not None or query.week is not None
    if (
        current
        and current.run_files.predictions is not None
        and (not wants_week or _matches(query, current.season, current.week))
    ):
        return PredictionSource(
            current.run_files.predictions, "active", current, current.season, current.week
        )
    for run in index.runs():
        if run.run_files.predictions is not None and _matches(query, run.season, run.week):
            return PredictionSource(run.run_files.predictions, "run", run, run.season, run.week)
    return None


def _unattached_source(settings: Settings, query: WeekQuery) -> PredictionSource | None:
    """Return the first unattached ``data/predict`` file that fits the query."""
    for item in unattached_files(settings.data_path, settings.reports_path):
        if item["kind"] == "predictions" and _matches(query, item["season"], item["week"]):
            return PredictionSource(
                Path(item["path"]), "unattached", None, item["season"], item["week"]
            )
    return None


def resolve_predictions(
    request: Request, db: Database, settings: Settings, query: WeekQuery
) -> PredictionSource:
    """Pick the predictions file for the query.

    Precedence: an explicit ``run``; else the active run when it matches (or no week was
    asked for); else any run predicting that week; else an unattached ``data/predict`` file.
    """
    index = get_index(request)
    if query.run is not None:
        return _explicit_run_source(db, index, query.run)
    unattached_only = isinstance(query, PredictionQuery) and query.source == "unattached"
    found = None if unattached_only else _run_source(db, index, query)
    found = found or _unattached_source(settings, query)
    if found is not None:
        return found
    if query.season is not None or query.week is not None:
        msg = f"No predictions for season {query.season} week {query.week}"
        raise NotFoundError(msg, code="no_predictions")
    msg = "No predictions found; run a weekly run or a predict job"
    raise NotFoundError(msg, code="no_predictions")
