"""Build a read-only measurement cache in this checkout's format from saved nflverse downloads.

The integration measurements need inputs the main checkout's cache does not hold yet: the
play-by-play with every column of ``constants.PBP_COLUMNS`` (the cached files hold the 55
columns written before the column extension), the players file and the weekly and season
rosters. They exist, as nflverse published them, in a saved nflreadpy file cache (hash-named
Parquet files, one per download). This script reads that directory and the main checkout's
cached schedules and team stats, both read-only, and writes a cache directory the package's own
loaders read as cache hits:

- ``pbp_<season>_reg.parquet``: each raw season prepared with ``loaders._prepare_pbp`` (regular
  season only) and written with the loader's own cache writer, as a download would be;
- ``players.parquet``, ``rosters_weekly_<season>_reg.parquet`` and ``rosters_<season>.parquet``:
  prepared with ``nflverse_sources``' own source definitions;
- ``schedule_<season>.parquet`` and ``team_stats_<season>_reg.parquet``: byte copies of the main
  checkout's cached files.

Each saved file is classified by its columns and the season it holds, not by its name. A season
with no saved file of a kind is reported and left out. Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_cache.py \
        --snapshot /home/mitch/scratch/nflreadpy_cache_snapshot_20261008 \
        --main-cache /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy \
        --out-dir /tmp/step6_int/cache --first-season 1999 --last-season 2025

It writes only under ``--out-dir`` and prints a manifest: every output file with its SHA-256 and
the input it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, nflverse_sources


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kind(columns: list[str]) -> str | None:
    """Return which nflverse download a saved file holds, judged by its columns."""
    names = set(columns)
    if {"play_id", "epa", "posteam"} <= names:
        return "pbp"
    if {"gsis_id", "display_name", "position"} <= names and "season" not in names:
        return "players"
    if {"gsis_id", "full_name", "team", "season"} <= names:
        return "rosters"
    return None


def classify(snapshot: Path) -> dict[tuple[str, int | None], list[Path]]:
    """Return the saved files by kind and season (``None`` for the players file)."""
    found: dict[tuple[str, int | None], list[Path]] = {}
    for path in sorted(snapshot.glob("*.parquet")):
        columns = list(pl.read_parquet_schema(path))
        kind = _kind(columns)
        if kind is None:
            continue
        season: int | None = None
        if "season" in columns:
            seasons = pl.scan_parquet(path).select(pl.col("season").unique()).collect()
            if seasons.height != 1:
                continue
            season = int(seasons.item())
        found.setdefault((kind, season), []).append(path)
    return found


def _rows_per_player(path: Path) -> float:
    """Return a roster download's rows per distinct player id."""
    counts = (
        pl.scan_parquet(path)
        .select(pl.len().alias("rows"), pl.col("gsis_id").n_unique().alias("players"))
        .collect()
    )
    return counts["rows"].item() / max(counts["players"].item(), 1)


# A weekly roster lists each player once per week he was on it, a season roster about once.
_WEEKLY_ROWS_PER_PLAYER = 2.0


def _roster_files(paths: list[Path]) -> tuple[Path | None, Path | None]:
    """Split one season's roster downloads into the season roster and the weekly roster."""
    season_roster: Path | None = None
    weekly_roster: Path | None = None
    for path in paths:
        if _rows_per_player(path) > _WEEKLY_ROWS_PER_PLAYER:
            weekly_roster = path
        else:
            season_roster = path
    return season_roster, weekly_roster


def build(
    snapshot: Path, main_cache: Path, out_dir: Path, seasons: list[int]
) -> list[tuple[str, str, str]]:
    """Write the measurement cache and return (output, sha256, source) for every file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = classify(snapshot)
    manifest: list[tuple[str, str, str]] = []

    def record(path: Path, source: str) -> None:
        manifest.append((path.name, _sha256(path), source))

    players = found.get(("players", None), [])
    if len(players) == 1:
        prepared = nflverse_sources.conform(
            pl.read_parquet(players[0]), nflverse_sources._PLAYERS_SCHEMA
        )
        target = out_dir / "players.parquet"
        loaders.write_cached_frame(prepared, target, tuple(constants.NFLREADPY_PLAYERS_COLUMNS))
        record(target, players[0].name)
    for season in seasons:
        pbp_paths = found.get(("pbp", season), [])
        if len(pbp_paths) == 1:
            prepared = loaders._prepare_pbp(pl.read_parquet(pbp_paths[0]), regular_season_only=True)
            target = out_dir / f"pbp_{season}_reg.parquet"
            loaders.write_cached_frame(prepared, target, tuple(constants.PBP_COLUMNS))
            record(target, pbp_paths[0].name)
        else:
            sys.stdout.write(f"season {season}: {len(pbp_paths)} play-by-play files, skipped\n")
        season_roster, weekly_roster = _roster_files(found.get(("rosters", season), []))
        if season_roster is not None:
            source = nflverse_sources.ROSTERS
            target = source.cache_path(out_dir, season, regular_season_only=False)
            frame = source.prepare(pl.read_parquet(season_roster), regular_season_only=False)
            loaders.write_cached_frame(frame, target, source.columns)
            record(target, season_roster.name)
        if weekly_roster is not None:
            source = nflverse_sources.WEEKLY_ROSTERS
            target = source.cache_path(out_dir, season, regular_season_only=True)
            frame = source.prepare(pl.read_parquet(weekly_roster), regular_season_only=True)
            loaders.write_cached_frame(frame, target, source.columns)
            record(target, weekly_roster.name)
        for name in (f"schedule_{season}.parquet", f"team_stats_{season}_reg.parquet"):
            copied = main_cache / name
            if copied.is_file():
                shutil.copyfile(copied, out_dir / name)
                record(out_dir / name, str(copied))
    return manifest


def main() -> None:
    """Build the cache and print its manifest."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--main-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    seasons = list(range(args.first_season, args.last_season + 1))
    manifest = build(args.snapshot, args.main_cache, args.out_dir, seasons)
    for name, digest, source in manifest:
        sys.stdout.write(f"{name} | {digest} | {source}\n")
    sys.stdout.write(f"{len(manifest)} files written to {args.out_dir}\n")


if __name__ == "__main__":
    main()
