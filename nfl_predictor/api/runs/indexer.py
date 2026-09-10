"""Scan ``models/`` for run directories and summarize them.

A run is any directory holding a ``metadata.json``. Its kind is inferred from what else is there:
``weekly`` when it carries predictions or orchestration stage markers, ``walk_forward`` when it
holds a walk-forward report but no model, and ``training`` otherwise. Runs are ordered newest
first by ``metadata.created_at``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nfl_predictor.api.runs.files import RunFiles, resolve_run_files
from nfl_predictor.utils.logger import log

RunKind = Literal["weekly", "training", "walk_forward"]
PREDICT_PATH_WEEK_RE = re.compile(r"week_(\d{2})")
CACHE_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class RunSummary:
    """What the run list shows for one run directory."""

    run_id: str
    run_dir: Path
    created_at: str
    kind: RunKind
    season: int | None
    week: int | None
    stages: dict[str, bool]
    complete: bool
    git_commit_hash: str | None
    dataset_hash: str | None
    model_kind: str | None
    holdout: dict[str, float] | None
    files: dict[str, bool]
    run_files: RunFiles

    @property
    def has_model(self) -> bool:
        """Return whether a model checkpoint exists."""
        return self.files["model"]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path`` or ``None`` when unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Skipping unreadable JSON %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _holdout_metrics(metrics_path: Path) -> dict[str, float] | None:
    """Return the holdout metric block of a training ``metrics_report.json`` when present."""
    report = _read_json(metrics_path) if metrics_path.is_file() else None
    if report is None:
        return None
    block: Any = report.get("metrics")
    if isinstance(block, dict):
        inner = block.get("metrics")
        block = inner.get("holdout") if isinstance(inner, dict) else block.get("overall")
    if not isinstance(block, dict):
        return None
    return {k: float(v) for k, v in block.items() if isinstance(v, int | float)}


def _infer_kind(files: RunFiles, metadata: dict[str, Any]) -> RunKind:
    """Classify a run directory from its contents."""
    if files.predictions is not None or any(files.stages.values()):
        return "weekly"
    splits = metadata.get("splits")
    if not files.model.is_file() and isinstance(splits, dict) and "eval_window" in splits:
        return "walk_forward"
    return "training"


def _season_week(files: RunFiles, metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return the season/week a run predicted, from its files or its config."""
    if files.season is not None:
        return files.season, files.week
    config = metadata.get("config")
    if not isinstance(config, dict):
        return None, None
    season = config.get("power_rankings_season")
    predict_path = config.get("predict_path")
    week: int | None = None
    if isinstance(predict_path, str):
        match = PREDICT_PATH_WEEK_RE.search(predict_path)
        week = int(match.group(1)) if match else None
    if season is None and week is not None:
        created = str(metadata.get("created_at", ""))
        season = int(created[:4]) if created[:4].isdigit() else None
    return (int(season) if season is not None else None), week


def summarize_run(run_dir: Path) -> RunSummary | None:
    """Build a :class:`RunSummary` for ``run_dir`` or ``None`` when it is not a run."""
    files = resolve_run_files(run_dir)
    if not files.metadata.is_file():
        return None
    metadata = _read_json(files.metadata)
    if metadata is None:
        return None
    config = metadata.get("config")
    season, week = _season_week(files, metadata)
    return RunSummary(
        run_id=run_dir.name,
        run_dir=run_dir,
        created_at=str(metadata.get("created_at", "")),
        kind=_infer_kind(files, metadata),
        season=season,
        week=week,
        stages=files.stages,
        complete=files.complete,
        git_commit_hash=metadata.get("git_commit_hash"),
        dataset_hash=metadata.get("dataset_hash"),
        model_kind=config.get("model_kind") if isinstance(config, dict) else None,
        holdout=_holdout_metrics(files.metrics),
        files=files.presence(),
        run_files=files,
    )


def scan_runs(models_dir: Path) -> list[RunSummary]:
    """Return every run under ``models_dir``, newest first."""
    if not models_dir.is_dir():
        return []
    runs: list[RunSummary] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = summarize_run(child)
        if summary is not None:
            runs.append(summary)
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return runs


class RunIndex:
    """A cached view of :func:`scan_runs` that refreshes after a short TTL."""

    def __init__(self, models_dir: Path, ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        """Remember the directory and cache lifetime."""
        self.models_dir = models_dir
        self.ttl_seconds = ttl_seconds
        self._runs: list[RunSummary] | None = None
        self._scanned_at = 0.0

    def invalidate(self) -> None:
        """Force the next read to rescan."""
        self._runs = None

    def runs(self) -> list[RunSummary]:
        """Return the cached run list, rescanning when stale."""
        now = time.monotonic()
        if self._runs is None or now - self._scanned_at > self.ttl_seconds:
            self._runs = scan_runs(self.models_dir)
            self._scanned_at = now
        return self._runs

    def get(self, run_id: str) -> RunSummary | None:
        """Return one run by id."""
        for run in self.runs():
            if run.run_id == run_id:
                return run
        return None
