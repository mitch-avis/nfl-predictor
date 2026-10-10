"""Resolve which predictions file a request refers to: a run's, or an unattached one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nfl_predictor import week_builder
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.pool.kickoff import first_kickoff
from nfl_predictor.api.readers.data_status import current_season_week, unattached_files
from nfl_predictor.api.readers.predictions import load_frame
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.files import week_of
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

    Weeks that already have predictions come first: the active run's own week, the later weeks
    predicted into it, other runs, and then unattached ``data/predict`` files. Weeks of the current
    season that are still unplayed and have no predictions follow with source ``available``, so the
    selector can offer to generate them.
    """
    refs: list[WeekRef] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    predicted: set[tuple[int | None, int | None]] = set()
    if active_run and active_run.run_files.predictions is not None:
        own = (active_run.season, active_run.week)
        others = [key for key in active_run.run_files.week_predictions if key != own]
        for season, week in [own, *others]:
            refs.append(
                WeekRef(
                    season=season,
                    week=week,
                    source="active",
                    run_id=active_run.run_id,
                    label=f"Week {week} (active run)",
                )
            )
            seen.add(("run", season, week))
            predicted.add((season, week))
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


def _active_source(current: RunSummary, query: WeekQuery) -> PredictionSource | None:
    """Return the active run's file for the query: its own week, else a later week it holds."""
    own = current.run_files.predictions
    if own is None:
        return None
    if (query.season is None and query.week is None) or _matches(
        query, current.season, current.week
    ):
        return PredictionSource(own, "active", current, current.season, current.week)
    for (season, week), path in current.run_files.week_predictions.items():
        if _matches(query, season, week):
            return PredictionSource(path, "active", current, season, week)
    return None


def _run_source(db: Database, index: RunIndex, query: WeekQuery) -> PredictionSource | None:
    """Return the active run's predictions when they fit the query, else any run's that do."""
    current = active.resolve_active_run(db, index)
    found = _active_source(current, query) if current else None
    if found is not None:
        return found
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


def _week_picks_file(src: PredictionSource) -> Path | None:
    """Return the run's picks file for ``src``'s week, if the run wrote one."""
    picks = src.run.run_files.picks if src.run else None
    if picks is None or not picks.is_file() or week_of(picks) != (src.season, src.week):
        return None
    return picks


def pre_lock_picks_file(src: PredictionSource) -> Path | None:
    """Return the week's picks file when it was written before the lock and ``src`` after it.

    Pool picks are single-shot before the week's first kickoff. When the week's predictions are
    rewritten after that kickoff (a lines refresh chains a new prediction), the picks file the
    weekly run wrote before it still holds the picks that count.
    """
    picks = _week_picks_file(src)
    if picks is None:
        return None
    lock = first_kickoff(load_frame(src.path).to_dicts())
    if lock is None:
        return None
    locked_at = lock.timestamp()
    if picks.stat().st_mtime < locked_at < src.path.stat().st_mtime:
        return picks
    return None


def current_picks_file(src: PredictionSource) -> Path | None:
    """Return the run's picks file when it still describes ``src``, else ``None``.

    The weekly run writes the picks file beside its own week's predictions. It describes ``src``
    when it is for the same week and no older than ``src.path``, or when it was written before the
    week's lock and ``src`` after it (the single-shot picks). Otherwise, predicting the week again
    before the lock or predicting a later week leaves it behind, and the picks then come from the
    predictions themselves.
    """
    picks = _week_picks_file(src)
    if picks is None:
        return None
    if picks.stat().st_mtime >= src.path.stat().st_mtime:
        return picks
    return pre_lock_picks_file(src)


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
