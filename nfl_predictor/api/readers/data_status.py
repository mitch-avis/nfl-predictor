"""Describe the ETL state.

Dataset files, freshness, cache coverage, and the latest leakage audit.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_predictor.api.db import Database
from nfl_predictor.api.readers.cache import cached
from nfl_predictor.utils.fingerprints import dataset_fingerprint
from nfl_predictor.utils.logger import log

DATASET_FILES: tuple[tuple[str, str], ...] = (
    ("all_data_ml.csv", "Full ML matrix: every game 1999 to now with engineered features."),
    ("all_data.csv", "Same games without the diff columns."),
    ("completed_games_ml.csv", "Training set: played games with features and targets."),
    ("completed_games.csv", "Played games without the diff columns."),
    ("nfl_elo.csv", "Team Elo input."),
    ("qb_elos.csv", "Quarterback Elo input (copied from nfeloqb)."),
)
CACHE_FILE_RE = re.compile(r"^(pbp|schedule)_(\d{4})(?:_reg)?\.parquet$")
PREDICT_FILE_RE = re.compile(r"^week_(\d{2})_(games_to_predict|predictions)\.csv$")
FINGERPRINT_KEY_PREFIX = "fingerprint:"


@dataclass(frozen=True)
class FileStatus:
    """One dataset file."""

    name: str
    description: str
    exists: bool
    size: int | None
    modified_at: str | None
    rows: int | None
    seasons: tuple[int, int] | None


def _iso(ts: float) -> str:
    """Format a POSIX timestamp as ISO-8601 UTC."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _csv_shape(path: Path) -> tuple[int, tuple[int, int] | None]:
    """Return ``(row_count, (min_season, max_season) | None)`` for a CSV, scanning lazily."""
    lazy = pl.scan_csv(path, infer_schema_length=10000)
    columns = lazy.collect_schema().names()
    if "season" in columns:
        stats = lazy.select(
            pl.len().alias("n"),
            pl.col("season").min().alias("lo"),
            pl.col("season").max().alias("hi"),
        ).collect()
        row = stats.row(0)
        seasons = (int(row[1]), int(row[2])) if row[1] is not None else None
        return int(row[0]), seasons
    n = lazy.select(pl.len()).collect().item()
    return int(n), None


def file_status(data_dir: Path, name: str, description: str) -> FileStatus:
    """Describe ``data_dir/name``."""
    path = data_dir / name
    if not path.is_file():
        return FileStatus(name, description, False, None, None, None, None)
    stat = path.stat()
    rows, seasons = cached(path, _csv_shape, "shape")
    return FileStatus(name, description, True, stat.st_size, _iso(stat.st_mtime), rows, seasons)


def current_season_week(today: date | None = None) -> tuple[int, int]:
    """Return the current NFL season and week using the ETL's own calendar rules."""
    from nfl_predictor import constants  # noqa: PLC0415 - avoid import cost at module load
    from nfl_predictor.data_collection import _determine_nfl_week  # noqa: PLC0415

    today = today or date.today()
    season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1
    return season, _determine_nfl_week(today)


def cache_coverage(cache_dir: Path) -> dict[str, list[int]]:
    """Return the seasons present per nflreadpy cache family."""
    coverage: dict[str, list[int]] = {"schedule": [], "pbp": []}
    if not cache_dir.is_dir():
        return coverage
    for path in cache_dir.iterdir():
        match = CACHE_FILE_RE.match(path.name)
        if match:
            coverage[match.group(1)].append(int(match.group(2)))
    return {k: sorted(v) for k, v in coverage.items()}


def latest_leakage_audit(*roots: Path) -> dict[str, Any] | None:
    """Return the newest ``*leakage_audit*.json`` under ``roots`` (two levels deep).

    The payload gains ``path`` and ``modified_at`` keys.
    """
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob("*leakage_audit*.json"))
            candidates.extend(root.glob("*/*leakage_audit*.json"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    payload["path"] = str(newest)
    payload["modified_at"] = _iso(newest.stat().st_mtime)
    return payload


class FingerprintCache:
    """Compute the training-set SHA-256 in the background and remember it in the ``kv`` table."""

    def __init__(self, db: Database) -> None:
        """Remember the database used for persistence."""
        self._db = db
        self._lock = threading.Lock()
        self._pending: set[str] = set()

    def get(self, path: Path) -> dict[str, Any] | None:
        """Return the cached fingerprint for ``path`` or start computing it and return ``None``."""
        if not path.is_file():
            return None
        stat = path.stat()
        key = f"{FINGERPRINT_KEY_PREFIX}{path}"
        stored = self._db.get_value(key)
        if stored:
            try:
                payload = json.loads(stored)
            except ValueError:
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("size") == stat.st_size
                and payload.get("mtime") == stat.st_mtime
            ):
                return payload
        self._start(path, key)
        return None

    def _start(self, path: Path, key: str) -> None:
        """Kick off a background hash unless one is already running for ``key``."""
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
        threading.Thread(target=self._compute, args=(path, key), daemon=True).start()

    def _compute(self, path: Path, key: str) -> None:
        """Hash ``path`` and store the result."""
        try:
            payload = dataset_fingerprint(path)
            self._db.set_value(key, json.dumps(payload))
        except OSError as exc:
            log.warning("Fingerprint of %s failed: %s", path, exc)
        finally:
            with self._lock:
                self._pending.discard(key)


def unattached_files(data_dir: Path, reports_dir: Path) -> list[dict[str, Any]]:
    """List ad-hoc outputs in ``data/predict`` and ``reports`` that no run directory owns."""
    items: list[dict[str, Any]] = []
    predict_dir = data_dir / "predict"
    if predict_dir.is_dir():
        for path in sorted(predict_dir.glob("*.csv")):
            match = PREDICT_FILE_RE.match(path.name)
            if not match:
                continue
            season: int | None = None
            week = int(match.group(1))
            try:
                head = pl.read_csv(path, n_rows=1, columns=["season", "week"])
                if head.height:
                    season, week = int(head["season"][0]), int(head["week"][0])
            except pl.exceptions.PolarsError, OSError, ValueError:
                pass
            stat = path.stat()
            items.append(
                {
                    "name": path.name,
                    "kind": match.group(2),
                    "season": season,
                    "week": week,
                    "size": stat.st_size,
                    "modified_at": _iso(stat.st_mtime),
                    "path": str(path),
                }
            )
    if reports_dir.is_dir():
        for path in sorted(reports_dir.glob("*.xlsx")):
            stat = path.stat()
            items.append(
                {
                    "name": path.name,
                    "kind": "xlsx",
                    "season": None,
                    "week": None,
                    "size": stat.st_size,
                    "modified_at": _iso(stat.st_mtime),
                    "path": str(path),
                }
            )
    return items
