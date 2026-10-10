"""Turn the snapshot of ``nfl-sos-ratings``' nflreadpy cache into this repository's cache files.

Shared by ``adapter_parity.py`` and ``ratings_etl_timing.py`` beside it. The snapshot
(``~/scratch/nflreadpy_cache_snapshot_20261008/``, with a ``sha256sum`` manifest beside it) holds
the raw nflverse files that project read on 2026-10-08, under nflreadpy's hashed names: one
play-by-play, weekly player stats, weekly roster, snap counts and team stats file per season, the
2001 and 2002 season rosters, the players file and the schedules of every season. Each file is
recognized by its columns and its ``season`` values, checked against the manifest, prepared
exactly as this repository's loaders prepare a download (``loaders._prepare_pbp`` and
``_prepare_schedule``, ``nflverse_sources.SeasonSource.prepare``) and written where they cache it,
with the requested columns they record. The real loaders then read those files as cache hits, so
a check runs this repository's whole loading path on the snapshot's data. Nothing is written
outside the cache directory given, and ``block_downloads`` makes any download attempt fail
loudly instead of reaching the network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, nflverse_sources

SNAPSHOT = Path.home() / "scratch" / "nflreadpy_cache_snapshot_20261008"
SNAPSHOT_MANIFEST = SNAPSHOT.parent / f"{SNAPSHOT.name}.sha256"
# Every snapshot season counts as historical, so the loaders read the cache and never refresh.
CURRENT_SEASON = 2027
_PBP_MARKERS = frozenset({"play_id", "posteam", "epa"})
_PLAYER_STATS_MARKERS = frozenset({"player_id", "passing_epa"})
_SNAP_COUNT_MARKERS = frozenset({"pfr_player_id", "offense_snaps"})
_ROSTER_MARKERS = frozenset({"gsis_id", "full_name"})
_PLAYERS_MARKERS = frozenset({"gsis_id", "display_name"})
_SCHEDULE_MARKERS = frozenset({"home_team", "away_team", "home_score"})
# More rows per player than this makes a roster file a weekly roster.
_WEEKLY_ROWS_PER_PLAYER = 2
_NFLREADPY_LOADERS = (
    "load_pbp",
    "load_player_stats",
    "load_players",
    "load_rosters",
    "load_rosters_weekly",
    "load_schedules",
    "load_snap_counts",
    "load_team_stats",
)


class DownloadBlockedError(RuntimeError):
    """Raised when code under check tries to download from nflverse."""


def block_downloads() -> None:
    """Replace nflreadpy's loaders with ones that raise, so a cache miss cannot download."""

    def refuse(*_args: object, **_kwargs: object) -> pl.DataFrame:
        message = "a check tried to download from nflverse; the snapshot cache missed a file"
        raise DownloadBlockedError(message)

    for name in _NFLREADPY_LOADERS:
        setattr(nfl, name, refuse)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SnapshotFiles:
    """The snapshot's files by kind and season, each checked against the manifest."""

    pbp: dict[int, Path] = field(default_factory=dict)
    player_stats: dict[int, Path] = field(default_factory=dict)
    snap_counts: dict[int, Path] = field(default_factory=dict)
    weekly_rosters: dict[int, Path] = field(default_factory=dict)
    season_rosters: dict[int, Path] = field(default_factory=dict)
    players: Path | None = None
    schedules: Path | None = None
    hashes: dict[str, str] = field(default_factory=dict)


def _seasons(path: Path) -> list[int]:
    return pl.read_parquet(path, columns=["season"]).get_column("season").unique().sort().to_list()


def _is_weekly_roster(path: Path) -> bool:
    """Return whether a roster file lists players week by week.

    A season roster has about one row per player (one per team he was on); a weekly roster has
    one per week he was on a roster, about sixteen on average. Both carry a ``week`` from 2002.
    """
    if "week" not in pl.read_parquet_schema(path):
        return False
    players = pl.read_parquet(path, columns=["gsis_id"]).get_column("gsis_id")
    return players.len() > _WEEKLY_ROWS_PER_PLAYER * players.n_unique()


def _kind(columns: set[str]) -> str | None:
    """Return what a snapshot file holds, judged by its columns, or None for other files."""
    if columns >= _PBP_MARKERS:
        return "pbp"
    if columns >= _SCHEDULE_MARKERS:
        return "schedules"
    if columns >= _PLAYERS_MARKERS and "season" not in columns:
        return "players"
    for kind, markers in (
        ("player_stats", _PLAYER_STATS_MARKERS),
        ("snap_counts", _SNAP_COUNT_MARKERS),
        ("rosters", _ROSTER_MARKERS),
    ):
        if columns >= markers:
            return kind
    return None


def find_snapshot_files(snapshot: Path = SNAPSHOT) -> SnapshotFiles:
    """Return the snapshot's files by kind and season.

    Raises:
        SystemExit: If a file does not match the manifest, a kind has two files for one season,
            or a file holds several seasons where one is expected.

    """
    manifest = {
        name: digest
        for digest, _, name in (
            line.strip().partition("  ") for line in SNAPSHOT_MANIFEST.read_text().splitlines()
        )
    }
    found = SnapshotFiles()
    for path in sorted(snapshot.glob("*.parquet")):
        digest = _sha256(path)
        if manifest.get(path.name) != digest:
            message = f"{path.name} does not match the snapshot manifest"
            raise SystemExit(message)
        found.hashes[path.name] = digest
        columns = set(pl.read_parquet_schema(path))
        kind = _kind(columns)
        if kind in {"schedules", "players"}:
            if getattr(found, kind) is not None:
                message = f"two {kind} files: {getattr(found, kind).name}, {path.name}"
                raise SystemExit(message)
            setattr(found, kind, path)
            continue
        if kind is None or "season" not in columns:
            continue
        seasons = _seasons(path)
        if not seasons:
            continue
        if len(seasons) != 1:
            message = f"{path.name} ({kind}) holds seasons {seasons}, not one"
            raise SystemExit(message)
        season = int(seasons[0])
        if kind == "rosters":
            kind = "weekly_rosters" if _is_weekly_roster(path) else "season_rosters"
        target: dict[int, Path] = getattr(found, kind)
        if season in target:
            message = f"{kind} has two files for {season}: {target[season].name}, {path.name}"
            raise SystemExit(message)
        target[season] = path
    return found


def write_cache(files: SnapshotFiles, cache_dir: Path, seasons: list[int]) -> None:
    """Write this repository's cache files for ``seasons`` from the snapshot into ``cache_dir``.

    Raises:
        SystemExit: If the snapshot lacks a season's play-by-play, player stats or schedule, or
            the players file.

    """
    if files.schedules is None or files.players is None:
        message = "the snapshot has no schedules or no players file"
        raise SystemExit(message)
    schedules = pl.read_parquet(files.schedules)
    for season in seasons:
        if season not in files.pbp or season not in files.player_stats:
            message = f"the snapshot has no play-by-play or player stats for {season}"
            raise SystemExit(message)
        pbp = loaders._prepare_pbp(pl.read_parquet(files.pbp[season]), regular_season_only=True)
        loaders._write_cached_frame(
            pbp,
            loaders._pbp_cache_path(cache_dir, season, regular_season_only=True),
            tuple(constants.PBP_COLUMNS),
        )
        schedule = loaders._prepare_schedule(schedules.filter(pl.col("season") == season))
        loaders._write_cached_frame(
            schedule,
            loaders._schedule_cache_path(cache_dir, season),
            loaders._schedule_requested_columns(),
        )
        for source, by_season in (
            (nflverse_sources.PLAYER_STATS, files.player_stats),
            (nflverse_sources.SNAP_COUNTS, files.snap_counts),
            (nflverse_sources.WEEKLY_ROSTERS, files.weekly_rosters),
        ):
            if season < source.first_season:
                continue
            raw = pl.read_parquet(by_season[season]) if season in by_season else source.empty()
            loaders._write_cached_frame(
                source.prepare(raw, regular_season_only=True),
                source.cache_path(cache_dir, season, regular_season_only=True),
                source.columns,
            )
    for season, path in files.season_rosters.items():
        source = nflverse_sources.ROSTERS
        loaders._write_cached_frame(
            source.prepare(pl.read_parquet(path), regular_season_only=False),
            source.cache_path(cache_dir, season, regular_season_only=False),
            source.columns,
        )
    players = nflverse_sources.conform(
        pl.read_parquet(files.players), nflverse_sources._PLAYERS_SCHEMA
    )
    loaders._write_cached_frame(
        players,
        cache_dir / nflverse_sources._PLAYERS_CACHE_NAME,
        tuple(constants.NFLREADPY_PLAYERS_COLUMNS),
    )


__all__ = [
    "CURRENT_SEASON",
    "SNAPSHOT",
    "SNAPSHOT_MANIFEST",
    "DownloadBlockedError",
    "SnapshotFiles",
    "block_downloads",
    "find_snapshot_files",
    "write_cache",
]
