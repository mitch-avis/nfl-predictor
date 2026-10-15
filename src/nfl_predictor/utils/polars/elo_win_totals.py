"""Getter for nfelosrs's preseason win-total ratings, on the cache-then-degrade pattern.

greerreNFL's ``nfelosrs`` repository publishes ``wt_ratings.csv``: one row per team and season
from 2003, rating each team by the sportsbooks' preseason win totals (the over/under on its
season wins) and its schedule. ``wt_rating_elo`` is that rating on the Elo scale, every season
averaging 1505, so it is market information: the own Elo's win-total pull (``elo_ratings``, the
win-total preset the ETL builds with ``own_elo_build``) is the only reader, and a market-free
rating never reads it.

``load_win_totals`` downloads the file, keeps the last good copy as ``wt_ratings.csv`` in the
cache directory (``NFELO_CACHE_DIR / "win_totals"`` by default), keeps every copy a run used
under ``snapshots/wt_ratings_<sha256>.csv``, and falls back to the cached copy when the download
fails or returns something that is not the file. With neither, it returns no rows; the ETL then
stops rather than publish the win-total preset without its pull
(``own_elo_build.require_win_totals``). Team codes are normalized to the canonical abbreviations
(the file writes ``OAK`` and ``WAS`` for those franchises' whole history).
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

WIN_TOTALS_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "team_abbr": pl.String(),
    "win_total_elo": pl.Float64(),
}
_REQUIRED_RAW_COLUMNS = frozenset({"season", "team", "wt_rating_elo"})
_CACHE_DIRECTORY = "win_totals"
_CACHE_NAME = "wt_ratings.csv"
_SNAPSHOT_DIRECTORY = "snapshots"
_FETCH_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class WinTotalsSnapshot:
    """The win-total ratings a run used and which copy of the file they came from.

    Attributes:
        frame: ``WIN_TOTALS_SCHEMA`` rows, one per team and season (empty when unavailable).
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


def parse_win_totals(content: bytes) -> pl.DataFrame:
    """Parse the file and return one rating per team and season with canonical team codes.

    Args:
        content: The bytes of ``wt_ratings.csv``.

    Returns:
        ``WIN_TOTALS_SCHEMA`` rows. A team listed twice in a season keeps its first row, with a
        warning.

    Raises:
        ValueError: When the bytes are not a readable CSV with ``season``, ``team`` and
            ``wt_rating_elo``.

    """
    try:
        raw = pl.read_csv(io.BytesIO(content), infer_schema_length=None)
    except (pl.exceptions.PolarsError, OSError) as exc:
        msg = f"unreadable win-total file: {exc}"
        raise ValueError(msg) from exc
    missing = sorted(_REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        msg = f"win-total file lacks the columns {missing}"
        raise ValueError(msg)
    frame = normalize_team_column(raw.rename({"team": "team_abbr"}), "team_abbr").select(
        pl.col("season").cast(pl.Int64),
        pl.col("team_abbr").cast(pl.String),
        pl.col("wt_rating_elo").cast(pl.Float64).alias("win_total_elo"),
    )
    unique = frame.unique(subset=["season", "team_abbr"], keep="first", maintain_order=True)
    if unique.height != frame.height:
        log.warning(
            "Win-total file lists %d team-seasons more than once; keeping the first row of each.",
            frame.height - unique.height,
        )
    return unique


def _keep_snapshot(content: bytes, cache_dir: Path) -> tuple[str, Path]:
    """Write the bytes under their hash, once, and return the hash and the path."""
    digest = hashlib.sha256(content).hexdigest()
    path = cache_dir / _SNAPSHOT_DIRECTORY / f"wt_ratings_{digest}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".tmp")
        staged.write_bytes(content)
        staged.replace(path)
    return digest, path


def _write_cache(content: bytes, cache_dir: Path) -> None:
    """Replace the cached copy with ``content``, atomically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged = cache_dir / f"{_CACHE_NAME}.tmp"
    staged.write_bytes(content)
    staged.replace(cache_dir / _CACHE_NAME)


def _from_cache(cache_dir: Path, url: str) -> WinTotalsSnapshot:
    """Return the cached copy, or an empty frame marked unavailable."""
    cached = cache_dir / _CACHE_NAME
    if cached.is_file():
        content = cached.read_bytes()
        try:
            frame = parse_win_totals(content)
        except ValueError as exc:
            log.warning("Cached win totals %s are unreadable: %s", cached, exc)
        else:
            digest, path = _keep_snapshot(content, cache_dir)
            log.warning("Using the cached win totals from %s (sha256 %s)", cached, digest)
            return WinTotalsSnapshot(frame, url, "cache", digest, path)
    log.warning("No win totals available; every preseason rating is the plain regression.")
    return WinTotalsSnapshot(pl.DataFrame(schema=WIN_TOTALS_SCHEMA), url, "unavailable", None, None)


def load_win_totals(
    *,
    cache_dir: Path | None = None,
    url: str = constants.NFELOSRS_WIN_TOTALS_URL,
    fetch: Callable[[str], bytes] | None = None,
) -> WinTotalsSnapshot:
    """Download the win-total ratings, cache them, and fall back to the cache without failing.

    Args:
        cache_dir: Cache directory; defaults to ``constants.NFELO_CACHE_DIR / "win_totals"``.
        url: The file's address.
        fetch: Returns the bytes at a URL, raising ``requests.RequestException`` or ``OSError``
            on failure; defaults to an HTTP GET with a 30-second timeout.

    Returns:
        The normalized ratings and the record of which copy they came from.

    """
    directory = (
        Path(cache_dir) if cache_dir is not None else constants.NFELO_CACHE_DIR / _CACHE_DIRECTORY
    )
    fetcher = fetch or _download
    try:
        content = fetcher(url)
        frame = parse_win_totals(content)
    except (requests.RequestException, OSError, ValueError) as exc:
        log.warning("Win-total download failed (%s); trying the cached copy.", exc)
        return _from_cache(directory, url)
    _write_cache(content, directory)
    digest, path = _keep_snapshot(content, directory)
    log.info("Downloaded win totals: %d team-seasons (sha256 %s)", frame.height, digest)
    return WinTotalsSnapshot(frame, url, "download", digest, path)
