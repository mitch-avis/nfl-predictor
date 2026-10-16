"""Describe the ETL state.

Dataset files, freshness, cache coverage, and the latest leakage audit.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from nfl_predictor import constants
from nfl_predictor.api.readers.cache import cached
from nfl_predictor.utils import clock
from nfl_predictor.utils.fingerprints import dataset_fingerprint
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.narrowing import as_str_dict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from nfl_predictor.api.db import Database

# What the files card lists under the data directory: the ETL's outputs, its build records and
# the inputs copied in by hand. A name ending in "/" is a directory, reported as a file count.
DATASET_FILES: tuple[tuple[str, str], ...] = (
    (
        "all_data_ml.csv",
        (
            "Every game the ETL built, played and upcoming, with its features and diff columns. "
            "Each week's games to predict are cut from it; the lines refresh rewrites its lines."
        ),
    ),
    (
        "all_data.csv",
        (
            "The same games without the diff columns. The pool page scores picks against it and "
            "the power rankings read the schedule from it."
        ),
    ),
    (
        "completed_games_ml.csv",
        (
            "Training set: the played games with features and targets. Training, backtests and "
            "the leakage audit read it; its SHA-256 is the training-set hash."
        ),
    ),
    (
        "completed_games.csv",
        "The played games without the diff columns. The ETL writes it; nothing reads it back.",
    ),
    (
        f"{constants.STRENGTH_SNAPSHOTS_NAME}.csv",
        (
            "Each team's adjusted strength before every week, bye teams included, written by the "
            "ETL. The composite power ranking reads it."
        ),
    ),
    (
        f"{constants.RATINGS_DIRNAME}/",
        (
            "The ported team and quarterback ratings tables, one Parquet file per season and "
            "table, written by the ETL. The power rankings read the ported ratings from it."
        ),
    ),
    (
        f"{constants.MARKET_LINES_METADATA_NAME}.json",
        (
            "The build's line record: line source, nfelo snapshot, match rates, fallback rows and "
            "moneyline maps. The lines refresh reads it to refresh lines the same way."
        ),
    ),
    (
        f"{constants.OWN_ELO_METADATA_NAME}.json",
        (
            "The own Elo's build record: both presets and the win-total file it read. Written by "
            "the ETL as provenance; nothing reads it back."
        ),
    ),
    (
        "qb_elos.csv",
        (
            "Quarterback Elo input, copied by hand from nfeloqb. The ETL reads its quarterback "
            "adjustment."
        ),
    ),
    (
        f"{constants.QB_META_DATA_NAME}.csv",
        (
            "Quarterback identity file, copied by hand from nfeloqb. The ETL reads it to match "
            "starters to their play-by-play ids."
        ),
    ),
)
# A season file in the nflreadpy cache: `<source>_<season>[_reg|_all].parquet`.
CACHE_FILE_RE = re.compile(
    r"^(?P<source>[a-z][a-z_]*?)_(?P<season>\d{4})(?:_(?:reg|all))?\.parquet$"
)
# Sources the page has always shown; they lead the coverage even when nothing is cached.
CORE_CACHE_SOURCES = ("schedule", "pbp")
SEASON_PREFIX_RE = re.compile(r"^(\d{4})_")
SEASON_DIR_RE = re.compile(r"^\d{4}$")
ETL_SEASON_DIR_RE = re.compile(r"^season_(\d{4})$")
TEAM_RANKINGS_GLOB = "*_team_rankings.csv"
# The files a directory entry on the files card counts: its top-level tables.
DIRECTORY_TABLE_GLOB = "*.parquet"
PREDICT_FILE_RE = re.compile(r"^week_(\d{2})_(games_to_predict|predictions)\.csv$")
FINGERPRINT_KEY_PREFIX = "fingerprint:"


@dataclass(frozen=True, kw_only=True)
class FileStatus:
    """One dataset file, or one directory summarized by ``file_count``."""

    name: str
    description: str
    exists: bool
    size: int | None
    modified_at: str | None
    rows: int | None
    seasons: tuple[int, int] | None
    file_count: int | None = None


@dataclass(frozen=True, kw_only=True)
class CacheFile:
    """One top-level entry of a cache that is not a season file; a directory counts its files."""

    cache: str
    name: str
    file_count: int
    size: int
    modified_at: str | None


def iso_time(ts: float) -> str:
    """Format a POSIX timestamp as ISO-8601 UTC."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _tree_summary(directory: Path, pattern: str = "**/*") -> tuple[list[Path], int, str | None]:
    """Return the files under ``directory`` matching ``pattern``, their size and newest time."""
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    stats = [path.stat() for path in files]
    newest = max((stat.st_mtime for stat in stats), default=None)
    return (
        files,
        sum(stat.st_size for stat in stats),
        iso_time(newest) if newest is not None else None,
    )


def _directory_status(data_dir: Path, name: str, description: str) -> FileStatus:
    """Describe the directory ``data_dir/name`` by its top-level Parquet tables.

    Only ``*.parquet`` files directly inside count, so a table being written
    (``*.parquet.<hex>.tmp``), a README or a nested backup folder changes none of the figures.
    """
    directory = data_dir / name.rstrip("/")
    if not directory.is_dir():
        return FileStatus(
            name=name,
            description=description,
            exists=False,
            size=None,
            modified_at=None,
            rows=None,
            seasons=None,
        )
    files, size, modified_at = _tree_summary(directory, DIRECTORY_TABLE_GLOB)
    seasons = sorted(
        int(match.group(1)) for path in files if (match := SEASON_PREFIX_RE.match(path.name))
    )
    return FileStatus(
        name=name,
        description=description,
        exists=True,
        size=size,
        modified_at=modified_at,
        rows=None,
        seasons=(seasons[0], seasons[-1]) if seasons else None,
        file_count=len(files),
    )


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
    """Describe ``data_dir/name``: a CSV with its rows and seasons, a directory by its files."""
    if name.endswith("/"):
        return _directory_status(data_dir, name, description)
    path = data_dir / name
    if not path.is_file():
        return FileStatus(
            name=name,
            description=description,
            exists=False,
            size=None,
            modified_at=None,
            rows=None,
            seasons=None,
        )
    stat = path.stat()
    rows: int | None = None
    seasons: tuple[int, int] | None = None
    if path.suffix == ".csv":
        try:
            rows, seasons = cached(path, _csv_shape, "shape")
        except (pl.exceptions.PolarsError, OSError, ValueError) as exc:
            # One unreadable file must not take the whole status page down.
            log.warning("Could not read the rows and seasons of %s: %s", path, exc)
    return FileStatus(
        name=name,
        description=description,
        exists=True,
        size=stat.st_size,
        modified_at=iso_time(stat.st_mtime),
        rows=rows,
        seasons=seasons,
    )


def current_season_week(today: date | None = None) -> tuple[int, int]:
    """Return the current NFL season and week using the ETL's own calendar rules."""
    today = today or clock.local_today()
    return clock.nfl_season(today), clock.nfl_week(today)


def cache_coverage(cache_dir: Path) -> dict[str, list[int]]:
    """Return the seasons present per season-partitioned nflreadpy source found on disk.

    ``schedule`` and ``pbp`` always come first, empty when nothing is cached; the other sources
    follow by name, each only when it has a season file.
    """
    found: dict[str, set[int]] = {source: set() for source in CORE_CACHE_SOURCES}
    if cache_dir.is_dir():
        for path in cache_dir.iterdir():
            match = CACHE_FILE_RE.match(path.name)
            if match and path.is_file():
                found.setdefault(match["source"], set()).add(int(match["season"]))
    others = sorted(source for source in found if source not in CORE_CACHE_SOURCES)
    return {source: sorted(found[source]) for source in (*CORE_CACHE_SOURCES, *others)}


def cache_files(caches: Mapping[str, Path]) -> list[CacheFile]:
    """List each cache's top-level entries that are not nflreadpy season files, in name order.

    Args:
        caches: Cache directories by the name the page shows, in display order.

    Returns:
        One entry per file or directory; a directory reports its files, total size and newest
        modified time. A cache directory that does not exist lists nothing.

    """
    entries: list[CacheFile] = []
    for cache_name, directory in caches.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.is_file() and CACHE_FILE_RE.match(path.name):
                continue
            if path.is_dir():
                files, size, modified_at = _tree_summary(path)
                entries.append(
                    CacheFile(
                        cache=cache_name,
                        name=f"{path.name}/",
                        file_count=len(files),
                        size=size,
                        modified_at=modified_at,
                    )
                )
            elif path.is_file():
                stat = path.stat()
                entries.append(
                    CacheFile(
                        cache=cache_name,
                        name=path.name,
                        file_count=1,
                        size=stat.st_size,
                        modified_at=iso_time(stat.st_mtime),
                    )
                )
    return entries


def other_seasons(data_dir: Path) -> dict[str, list[int]]:
    """Return the seasons of the season-partitioned caches outside nflreadpy.

    ``team_rankings`` lists the ``data/<season>/`` directories that hold TeamRankings scrapes;
    ``etl_seasons`` lists the finished seasons' builds kept for the incremental ETL.
    """
    team_rankings = sorted(
        int(path.name)
        for path in (data_dir.iterdir() if data_dir.is_dir() else ())
        if path.is_dir()
        and SEASON_DIR_RE.match(path.name)
        and next(path.glob(TEAM_RANKINGS_GLOB), None) is not None
    )
    etl_dir = data_dir / "cache" / constants.ETL_SEASON_CACHE_DIRNAME
    etl_seasons = sorted(
        int(match.group(1))
        for path in (etl_dir.iterdir() if etl_dir.is_dir() else ())
        if path.is_dir() and (match := ETL_SEASON_DIR_RE.match(path.name))
    )
    return {"team_rankings": team_rankings, "etl_seasons": etl_seasons}


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
        payload = as_str_dict(json.loads(newest.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return None
    if payload is None:
        return None
    payload["path"] = str(newest)
    payload["modified_at"] = iso_time(newest.stat().st_mtime)
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
                payload = as_str_dict(json.loads(stored))
            except ValueError:
                payload = None
            if (
                payload is not None
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
                    "modified_at": iso_time(stat.st_mtime),
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
                    "modified_at": iso_time(stat.st_mtime),
                    "path": str(path),
                }
            )
    return items
