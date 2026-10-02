"""Getter for nfelo's market-lines file, on the nflreadpy cache-then-degrade pattern.

nfelo's ``nfelomarket_data`` repository publishes ``Data/lines.csv``: one row per game since
1999 with the opening and the latest spread, moneylines and total. The file is rewritten
several times a day, so a run must say exactly which copy it read. ``load_nfelo_lines``
downloads it, keeps the last good copy as ``lines.csv`` in the cache directory, keeps every
copy a run used under ``snapshots/lines_<sha256>.csv``, and falls back to the cached copy when
the download fails or returns something that is not the lines file. With neither, the run
continues with no nfelo rows, and every game keeps its stored line.

Team codes are normalized to the canonical abbreviations (nfelo writes ``OAK``, ``WAS`` and
``STL`` for those franchises' whole history), so games join the schedule on
``(season, week, away_abbr, home_abbr)``; nfelo's own ``game_id`` differs from nflverse's
whenever a franchise moved or renamed and is not used as a key.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import requests

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.scraping_utils import normalize_team_column

if TYPE_CHECKING:
    from collections.abc import Callable

# The join keys, in the schedule's names.
NFELO_KEY_COLUMNS: tuple[str, ...] = ("season", "week", "away_abbr", "home_abbr")
_SPREAD_AND_TOTAL_COLUMNS: tuple[str, ...] = (
    "home_spread_open",
    "home_spread_last",
    "total_line_open",
    "total_line_last",
)
_MONEYLINE_COLUMNS: tuple[str, ...] = (
    "home_ml_open",
    "away_ml_open",
    "home_ml_last",
    "away_ml_last",
)
# The line columns a normalized frame carries, home spread negative when the home team is
# favored (the schedule's ``home_spread`` convention).
NFELO_LINE_COLUMNS: tuple[str, ...] = (*_SPREAD_AND_TOTAL_COLUMNS, *_MONEYLINE_COLUMNS)
_RAW_TEAM_COLUMNS = {"home_team": "home_abbr", "away_team": "away_abbr"}
_REQUIRED_RAW_COLUMNS = frozenset({"season", "week", *_RAW_TEAM_COLUMNS, *NFELO_LINE_COLUMNS})
_CACHE_NAME = "lines.csv"
_SNAPSHOT_DIRECTORY = "snapshots"
_FETCH_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class NfeloLinesSnapshot:
    """The nfelo lines a run used and which copy of the file they came from.

    Attributes:
        frame: Normalized lines, one row per game (empty when the source is unavailable).
        url: Where the file is downloaded from.
        origin: ``download``, ``cache`` (the download failed) or ``unavailable``.
        sha256: Hash of the exact bytes read, or ``None`` when nothing was read.
        snapshot_path: The kept copy of those bytes, or ``None`` when nothing was read.

    """

    frame: pl.DataFrame
    url: str
    origin: str
    sha256: str | None
    snapshot_path: Path | None

    def metadata(self) -> dict[str, object]:
        """Return the run-metadata record of this snapshot."""
        return {
            "url": self.url,
            "origin": self.origin,
            "sha256": self.sha256,
            "snapshot": None if self.snapshot_path is None else str(self.snapshot_path),
            "rows": self.frame.height,
        }


def _download(url: str) -> bytes:
    """Return the file at ``url``; raises ``requests.RequestException`` on failure."""
    response = requests.get(url, timeout=_FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _empty_frame() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "season": pl.Int64(),
        "week": pl.Int64(),
        "away_abbr": pl.Utf8(),
        "home_abbr": pl.Utf8(),
    }
    schema.update(dict.fromkeys(_SPREAD_AND_TOTAL_COLUMNS, pl.Float64()))
    schema.update(dict.fromkeys(_MONEYLINE_COLUMNS, pl.Int64()))
    return pl.DataFrame(schema=schema)


def normalize_nfelo_lines(raw: pl.DataFrame) -> pl.DataFrame:
    """Return nfelo's lines keyed like the schedule, with canonical team codes.

    Spreads and totals are floats and moneylines whole numbers. A game listed twice keeps its
    first row, with a warning.

    Args:
        raw: The lines file as read.

    Returns:
        One row per game: the key columns of ``NFELO_KEY_COLUMNS`` and ``NFELO_LINE_COLUMNS``.

    """
    frame = raw.rename(_RAW_TEAM_COLUMNS)
    for column in ("away_abbr", "home_abbr"):
        frame = normalize_team_column(frame, column)
    frame = frame.select(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("away_abbr").cast(pl.Utf8),
        pl.col("home_abbr").cast(pl.Utf8),
        *(pl.col(column).cast(pl.Float64) for column in _SPREAD_AND_TOTAL_COLUMNS),
        *(pl.col(column).cast(pl.Float64).round(0).cast(pl.Int64) for column in _MONEYLINE_COLUMNS),
    )
    unique = frame.unique(subset=list(NFELO_KEY_COLUMNS), keep="first", maintain_order=True)
    if unique.height != frame.height:
        log.warning(
            "nfelo lines list %d games more than once; keeping the first row of each.",
            frame.height - unique.height,
        )
    return unique


def _parse(content: bytes) -> pl.DataFrame:
    """Parse and normalize the file; raises ``ValueError`` when it is not the lines file."""
    try:
        raw = pl.read_csv(io.BytesIO(content), infer_schema_length=None)
    except (pl.exceptions.PolarsError, OSError) as exc:
        msg = f"unreadable nfelo lines file: {exc}"
        raise ValueError(msg) from exc
    missing = sorted(_REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        msg = f"nfelo lines file lacks the columns {missing}"
        raise ValueError(msg)
    return normalize_nfelo_lines(raw)


def _keep_snapshot(content: bytes, cache_dir: Path) -> tuple[str, Path]:
    """Write the bytes under their hash, once, and return the hash and the path."""
    digest = hashlib.sha256(content).hexdigest()
    path = cache_dir / _SNAPSHOT_DIRECTORY / f"lines_{digest}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".tmp")
        staged.write_bytes(content)
        staged.replace(path)
    return digest, path


def _write_cache(content: bytes, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged = cache_dir / f"{_CACHE_NAME}.tmp"
    staged.write_bytes(content)
    staged.replace(cache_dir / _CACHE_NAME)


def _from_cache(cache_dir: Path, url: str) -> NfeloLinesSnapshot:
    """Return the cached copy, or an empty frame marked unavailable."""
    cached = cache_dir / _CACHE_NAME
    if cached.is_file():
        content = cached.read_bytes()
        try:
            frame = _parse(content)
        except ValueError as exc:
            log.warning("Cached nfelo lines %s are unreadable: %s", cached, exc)
        else:
            digest, path = _keep_snapshot(content, cache_dir)
            log.warning("Using the cached nfelo lines from %s (sha256 %s)", cached, digest)
            return NfeloLinesSnapshot(frame, url, "cache", digest, path)
    log.warning("No nfelo lines available; every game keeps its stored line.")
    return NfeloLinesSnapshot(_empty_frame(), url, "unavailable", None, None)


def load_nfelo_lines(
    *,
    cache_dir: Path | None = None,
    url: str = constants.NFELO_LINES_URL,
    fetch: Callable[[str], bytes] | None = None,
) -> NfeloLinesSnapshot:
    """Download nfelo's lines, cache them, and fall back to the cache without failing.

    Args:
        cache_dir: Cache directory; defaults to ``constants.NFELO_CACHE_DIR``.
        url: The file's address.
        fetch: Returns the bytes at a URL, raising ``requests.RequestException`` or
            ``OSError`` on failure; defaults to an HTTP GET.

    Returns:
        The normalized lines and the record of which copy they came from.

    """
    directory = Path(cache_dir) if cache_dir is not None else constants.NFELO_CACHE_DIR
    fetcher = fetch or _download
    try:
        content = fetcher(url)
        frame = _parse(content)
    except (requests.RequestException, OSError, ValueError) as exc:
        log.warning("nfelo lines download failed (%s); trying the cached copy.", exc)
        return _from_cache(directory, url)
    _write_cache(content, directory)
    digest, path = _keep_snapshot(content, directory)
    log.info("Downloaded nfelo lines: %d games (sha256 %s)", frame.height, digest)
    return NfeloLinesSnapshot(frame, url, "download", digest, path)
