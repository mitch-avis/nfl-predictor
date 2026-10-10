"""Compare a fresh nflverse play-by-play download with a play-by-play cache, season by season.

Run from the root of an ``nfl-predictor`` checkout, naming the cache to compare (read only) and a
scratch directory outside every checkout for the files whose size is measured:

    .venv/bin/python .agents/sos_port/decisions/pbp_release_check.py \
        --cache-dir /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy \
        --work-dir "$(mktemp -d)" --first-season 1999 --last-season 2025

For each season it downloads the play-by-play with nflreadpy (into memory, or reads a raw file
saved earlier with ``--raw-dir``) and prepares it with this checkout's loader
(``loaders._prepare_pbp``, regular season only), then:

- compares the prepared frame with the cached ``pbp_<season>_reg.parquet`` on every column both
  hold: the row counts, and per column the number of rows whose values differ (a null against a
  value counts; two nulls do not), the largest absolute difference of each numeric column, and
  the largest EPA difference. Rows are matched by position, which is valid when the row counts
  are equal; a season whose counts differ is matched game by game, and games whose counts
  differ are listed instead;
- writes the prepared frame twice with the loader's own cache writer, once holding only the
  cached file's columns and once holding every column this checkout requests, and reports both
  sizes (the cache file size before and after the column change, from the same download).

It writes only under ``--work-dir`` (and ``--raw-dir`` with ``--save-raw``) and prints a report
to stdout (``pbp_release_check_output.txt`` beside this file holds the run ``pbp_columns.md``
quotes). With ``--asset-dates`` it also lists when nflverse last updated each season's release
file (one unauthenticated GitHub API request), beside the cached file's modification time.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders

RELEASE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/pbp"
REQUEST_TIMEOUT_SECONDS = 60


@dataclass
class SeasonResult:
    """One season's comparison and sizes."""

    season: int
    cached_rows: int
    fresh_rows: int
    shared_columns: int
    differing: dict[str, int] = field(default_factory=dict)
    largest: dict[str, float] = field(default_factory=dict)
    dtype_changes: dict[str, str] = field(default_factory=dict)
    unmatched_games: list[str] = field(default_factory=list)
    size_cached_file: int = 0
    size_before: int = 0
    size_after: int = 0


def _raw_season(season: int, raw_dir: Path | None, *, save_raw: bool) -> pl.DataFrame:
    """Return one season's raw play-by-play, from ``raw_dir`` when saved there, else nflreadpy."""
    path = raw_dir / f"play_by_play_{season}.parquet" if raw_dir else None
    if path is not None and path.exists():
        return pl.read_parquet(path)
    frame = nfl.load_pbp(seasons=[season])
    if path is not None and save_raw:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    return frame


def _differs(left: pl.Series, right: pl.Series) -> pl.Series:
    """Return where two aligned columns differ; two nulls, or two NaNs, are equal."""
    different = left.ne_missing(right)
    if left.dtype.is_float() and right.dtype.is_float():
        both_nan = (left.is_nan() & right.is_nan()).fill_null(value=False)
        different = different & ~both_nan
    return different


def _compare_aligned(result: SeasonResult, cached: pl.DataFrame, fresh: pl.DataFrame) -> None:
    """Add the per-column differences of two frames whose rows correspond by position."""
    for column in cached.columns:
        left, right = cached.get_column(column), fresh.get_column(column)
        if left.dtype != right.dtype:
            result.dtype_changes[column] = f"{left.dtype} -> {right.dtype}"
            if left.dtype.is_numeric() and right.dtype.is_numeric():
                left, right = left.cast(pl.Float64), right.cast(pl.Float64)
            else:
                left, right = left.cast(pl.String), right.cast(pl.String)
        count = int(_differs(left, right).sum())
        if count:
            result.differing[column] = result.differing.get(column, 0) + count
        if left.dtype.is_numeric() and right.dtype.is_numeric():
            gap = (left.cast(pl.Float64) - right.cast(pl.Float64)).abs().max()
            if isinstance(gap, float) and gap > result.largest.get(column, 0.0):
                result.largest[column] = gap


def _compare(season: int, cached: pl.DataFrame, fresh: pl.DataFrame) -> SeasonResult:
    """Compare the cached and fresh frames on every column the cache holds."""
    fresh = fresh.select(cached.columns)
    result = SeasonResult(season, cached.height, fresh.height, len(cached.columns))
    if cached.height == fresh.height:
        _compare_aligned(result, cached, fresh)
        return result
    cached_counts = dict(cached.group_by("game_id").len().iter_rows())
    fresh_counts = dict(fresh.group_by("game_id").len().iter_rows())
    for game in sorted(set(cached_counts) | set(fresh_counts)):
        if cached_counts.get(game) != fresh_counts.get(game):
            result.unmatched_games.append(
                f"{game} ({cached_counts.get(game, 0)} cached, {fresh_counts.get(game, 0)} fresh)"
            )
            continue
        _compare_aligned(
            result,
            cached.filter(pl.col("game_id") == game),
            fresh.filter(pl.col("game_id") == game),
        )
    return result


def _measure_sizes(
    result: SeasonResult, fresh: pl.DataFrame, cached_columns: list[str], work_dir: Path
) -> None:
    """Write the fresh frame in the cache's old and new shapes and record their sizes."""
    before = work_dir / f"pbp_{result.season}_reg_cached_columns.parquet"
    after = work_dir / f"pbp_{result.season}_reg_requested_columns.parquet"
    loaders._write_cached_frame(fresh.select(cached_columns), before, cached_columns)
    loaders._write_cached_frame(fresh, after, list(constants.PBP_COLUMNS))
    result.size_before = before.stat().st_size
    result.size_after = after.stat().st_size


def _asset_dates() -> dict[int, str]:
    """Return each season's release file update time from the GitHub API."""
    request = urllib.request.Request(RELEASE_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        release = json.load(response)
    dates: dict[int, str] = {}
    for asset in release.get("assets", []):
        name = str(asset.get("name", ""))
        if name.startswith("play_by_play_") and name.endswith(".parquet"):
            season = name.removeprefix("play_by_play_").removesuffix(".parquet")
            if season.isdigit():
                dates[int(season)] = str(asset.get("updated_at"))
    return dates


def _report(results: list[SeasonResult], dates: dict[int, str], cache_dir: Path) -> None:
    """Print the per-season comparison, the sizes and the totals."""
    out = sys.stdout
    out.write("## Fresh download against the cache, every shared column\n\n")
    out.write(
        "season | rows cached | rows fresh | shared columns | columns differing | "
        "values differing | largest EPA difference | games unmatched\n"
    )
    for r in results:
        epa_gap = max(r.largest.get("epa", 0.0), 0.0)
        out.write(
            f"{r.season} | {r.cached_rows} | {r.fresh_rows} | {r.shared_columns} | "
            f"{len(r.differing)} | {sum(r.differing.values())} | {epa_gap:.6g} | "
            f"{len(r.unmatched_games)}\n"
        )
    out.write("\n## Differing values per column (seasons with any)\n\n")
    for r in results:
        if r.differing or r.dtype_changes or r.unmatched_games:
            parts = [
                f"{column} {count} (largest {r.largest[column]:.6g})"
                if column in r.largest
                else f"{column} {count}"
                for column, count in sorted(r.differing.items())
            ]
            out.write(f"{r.season}: {', '.join(parts) or 'no value differs'}\n")
            if r.dtype_changes:
                out.write(f"    dtype changes: {r.dtype_changes}\n")
            for game in r.unmatched_games:
                out.write(f"    unmatched game: {game}\n")
    out.write("\n## Cache file size (bytes)\n\n")
    out.write(
        "season | cached file today | fresh, cached columns | fresh, every requested column | "
        "after / before\n"
    )
    for r in results:
        ratio = r.size_after / r.size_before if r.size_before else float("nan")
        out.write(
            f"{r.season} | {r.size_cached_file} | {r.size_before} | {r.size_after} | {ratio:.2f}\n"
        )
    total_before = sum(r.size_before for r in results)
    total_after = sum(r.size_after for r in results)
    out.write(
        f"total | {sum(r.size_cached_file for r in results)} | {total_before} | {total_after} | "
        f"{total_after / total_before:.2f}\n"
    )
    if dates:
        out.write("\n## Release file update time against the cached file's modification time\n\n")
        for r in results:
            path = cache_dir / f"pbp_{r.season}_reg.parquet"
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
                timespec="seconds"
            )
            out.write(f"{r.season}: release {dates.get(r.season, 'unknown')}, cached {mtime}\n")
    identical = [r.season for r in results if not r.differing and not r.unmatched_games]
    out.write(
        f"\nSeasons identical on every shared column: {len(identical)} of {len(results)}\n"
        f"Requested columns: {len(constants.PBP_COLUMNS)}\n"
    )


def main() -> int:
    """Run the comparison and return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache to compare (read).")
    parser.add_argument("--work-dir", type=Path, required=True, help="Scratch directory (written).")
    parser.add_argument("--raw-dir", type=Path, help="Raw season files to reuse.")
    parser.add_argument("--save-raw", action="store_true", help="Save downloads in --raw-dir.")
    parser.add_argument("--first-season", type=int, default=constants.NFLREADPY_MIN_SEASON)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--asset-dates", action="store_true", help="List release update times.")
    arguments = parser.parse_args()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    results: list[SeasonResult] = []
    for season in range(arguments.first_season, arguments.last_season + 1):
        cache_path = arguments.cache_dir / f"pbp_{season}_reg.parquet"
        if not cache_path.exists():
            sys.stderr.write(f"error: no cache file {cache_path}\n")
            return 1
        cached = pl.read_parquet(cache_path)
        raw = _raw_season(season, arguments.raw_dir, save_raw=arguments.save_raw)
        fresh = loaders._prepare_pbp(raw, regular_season_only=True)
        missing = [column for column in cached.columns if column not in fresh.columns]
        if missing:
            sys.stderr.write(f"error: {season}: fresh frame lacks cached columns {missing}\n")
            return 1
        result = _compare(season, cached, fresh)
        result.size_cached_file = cache_path.stat().st_size
        _measure_sizes(result, fresh, cached.columns, arguments.work_dir)
        results.append(result)
    dates = _asset_dates() if arguments.asset_dates else {}
    _report(results, dates, arguments.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
