"""Compare the season statistics port with every table ``nfl-sos-ratings`` published, cell by cell.

Run from the repository root (single-threaded unless the environment sets the thread counts; about
a minute a season):

    .venv/bin/python .agents/sos_port/season_stats_parity.py [--source PATH] [--snapshot DIR]
        [--manifest FILE] [--seasons S ...] [--work-dir DIR] [--out-dir DIR]

``--source`` is the ``nfl-sos-ratings`` checkout (default: the one beside this repository's main
checkout), whose published Parquet files are read in place and never written; ``--snapshot`` and
``--manifest`` are the copy of nflreadpy's disk cache that project's build read and its SHA-256
manifest; ``--seasons`` limits the run to some seasons (default 1999-2025); ``--work-dir`` holds
this repository's loader caches for the run (default: a new directory under ``~/scratch``, never a
``data/`` directory); ``--out-dir`` writes the reports elsewhere (default: beside this script).

For each season the script:

1. serves every nflverse download from the snapshot: nflreadpy's downloader is replaced by one
   that reads the snapshot file nflreadpy's own cache would hold for the request (the MD5 of the
   URL and parameters), checks the file's SHA-256 against the manifest and its content against
   the request (the season it holds), and refuses anything the snapshot lacks, so nothing reaches
   the network;
2. loads the season through this repository's loaders (``season_stats.sources``), with the
   play-by-play loader's column list set to the columns the season statistics read
   (``sources.SEASON_STATS_PBP_COLUMNS``), so the build reads nothing else;
3. builds the season's tables (``season_stats.season``), taking the rating columns of ``combined``
   and ``qb_combined`` from the published ratings (the ratings port reproduces those);
4. compares every table with the published file: the same columns in the same order, the same rows
   in the same order, integers, text and flags equal, and floats within ``FLOAT_TOLERANCE``,
   counting the bit-identical cells. Washington's code is ``WSH`` here and ``WAS`` there, so the
   published team-code columns are read with ``WAS`` as ``WSH`` (counted). A column whose type
   differs is reported; its values are still compared, exactly when the published column holds
   whole numbers.

It writes ``SEASON_STATS_PARITY.md`` and ``season_stats_parity.json`` and exits 1 if any cell is
outside the tolerance or any table differs in columns or rows.
"""

from __future__ import annotations

import os

# One Polars thread unless the caller set the count, as the tests run.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
from nflreadpy.downloader import NflverseDownloader

from nfl_predictor import constants
from nfl_predictor.season_stats import published_logs, sources
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats.pooled_rates import rate_parts
from nfl_predictor.season_stats.team_stats_expanded import TEAM_FORMULA_RATES

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE_COMMIT = "9081bced89b0d51c99e28f80f8f62b9fcd0118e0"
FIRST_SEASON = 1999
LAST_SEASON = 2025
# Floats agree within this absolute difference; the report also counts bit-identical cells.
FLOAT_TOLERANCE = 1e-9
TEAM_CODE_COLUMNS = ("team", "opponent_team")
PUBLISHED_WASHINGTON = "WAS"
WASHINGTON = "WSH"
KINDS = (
    "team_game_logs",
    "qb_game_logs",
    "team_per_game_stats",
    "qb_per_game_stats",
    "qb_opponent_profiles",
    "opponent_profiles",
    "combined",
    "qb_combined",
    "team_wp_bins",
    "qb_wp_bins",
)
# The season tables the published game logs can feed (every table but the logs and the bins).
PUBLISHED_LOG_KINDS = (
    "team_per_game_stats",
    "qb_per_game_stats",
    "qb_opponent_profiles",
    "opponent_profiles",
    "combined",
    "qb_combined",
)
# Columns the combined tables take from the published ratings rather than compute.
SUPPLIED_COLUMNS = {
    "combined": season_tables.TEAM_RATING_INPUT_COLUMNS[1:],
    "qb_combined": season_tables.QB_RATING_INPUT_COLUMNS[1:],
}
MAX_LISTED_DIFFERENCES = 40


def default_source() -> Path:
    """Return the ``nfl-sos-ratings`` checkout beside this repository's main checkout."""
    git_common = REPO_ROOT / ".git"
    if git_common.is_file():  # a worktree: .git names the main checkout's git directory
        gitdir = Path(git_common.read_text(encoding="utf-8").split(":", 1)[1].strip())
        main_checkout = gitdir.parents[2]
    else:
        main_checkout = REPO_ROOT
    return main_checkout.parent / "nfl-sos-ratings"


def source_head(source: Path) -> str:
    """Return the commit the checkout's HEAD names, read from its git files (no git command)."""
    git_dir = source / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
        if line.endswith(f" {ref}"):
            return line.split(" ", 1)[0]
    msg = f"cannot resolve {ref} in {git_dir}"
    raise SystemExit(msg)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_commit() -> str:
    """Return this checkout's commit and whether its tracked files differ from it."""
    git = "git"
    head = subprocess.run(  # noqa: S603  # fixed argv in this repository
        [git, "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(  # noqa: S603  # fixed argv in this repository
        [git, "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{head}{' (with uncommitted changes)' if dirty else ''}"


@dataclass
class SnapshotDownloads:
    """nflreadpy downloads served from the snapshot of its disk cache, checked file by file."""

    snapshot: Path
    manifest: dict[str, str]
    served: dict[str, dict[str, Any]] = field(default_factory=dict)

    def install(self) -> None:
        """Replace nflreadpy's download (and its cache) with reads of the snapshot."""
        snapshot = self

        def download_file(
            downloader: NflverseDownloader, url: str, **kwargs: object
        ) -> pl.DataFrame:
            return snapshot.read(downloader, url, **kwargs)

        NflverseDownloader._download_file = download_file  # type: ignore[method-assign]

    def read(self, downloader: NflverseDownloader, url: str, **kwargs: object) -> pl.DataFrame:
        """Return the snapshot file nflreadpy's cache would hold for ``url`` and ``kwargs``."""
        key = downloader.cache._get_cache_key(url, **kwargs)
        path = self.snapshot / f"{key}.parquet"
        if not path.exists():
            msg = f"the snapshot has no file for {url} {kwargs}"
            raise ConnectionError(msg)
        if path.name not in self.served:
            digest = sha256_file(path)
            if self.manifest.get(path.name) != digest:
                msg = f"{path.name} does not match the snapshot manifest"
                raise SystemExit(msg)
            frame = pl.read_parquet(path)
            season = kwargs.get("season")
            seasons = (
                sorted(frame.get_column("season").unique().to_list())
                if "season" in frame.columns
                else []
            )
            if season is not None and frame.height and seasons != [season]:
                msg = f"{path.name} holds seasons {seasons}, not the requested {season}"
                raise SystemExit(msg)
            self.served[path.name] = {
                "url": url,
                "parameters": {name: str(value) for name, value in sorted(kwargs.items())},
                "sha256": digest,
                "rows": frame.height,
                "columns": frame.width,
                "seasons": [seasons[0], seasons[-1]] if seasons else [],
            }
            return frame
        return pl.read_parquet(path)


def read_manifest(path: Path) -> dict[str, str]:
    """Return the snapshot manifest: file name to SHA-256."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return entries


def relabel_washington(frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Return ``frame`` with ``WAS`` read as ``WSH`` in its team-code columns, and the count."""
    present = [column for column in TEAM_CODE_COLUMNS if column in frame.columns]
    count = sum(frame.get_column(column).eq(PUBLISHED_WASHINGTON).sum() for column in present)
    relabeled = frame.with_columns(
        pl.col(column).replace(PUBLISHED_WASHINGTON, WASHINGTON) for column in present
    )
    return relabeled, int(count)


@dataclass
class TableComparison:
    """How one rebuilt table compares with its published file."""

    season: int
    kind: str
    rows: int = 0
    columns: int = 0
    cells: int = 0
    identical: int = 0
    within_tolerance: int = 0
    outside: int = 0
    supplied_cells: int = 0
    unchecked_cells: int = 0
    relabeled_cells: int = 0
    max_abs_difference: float = 0.0
    structure: list[str] = field(default_factory=list)
    dtype_differences: list[str] = field(default_factory=list)
    differences: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every cell agrees and the table has the published columns and rows."""
        return self.outside == 0 and not self.structure


def _row_label(frame: pl.DataFrame, index: int) -> str:
    """Return a short label of row ``index``: its key columns' values."""
    keys = [
        key
        for key in ("season", "game_id", "week", "team", "qb_id", "wp_unit", "wp_bin")
        if key in frame.columns
    ]
    return ", ".join(f"{key}={frame.get_column(key)[index]}" for key in keys)


def _is_float(dtype: pl.DataType) -> bool:
    return dtype.is_float()


def compare_tables(
    season: int,
    kind: str,
    published: pl.DataFrame,
    rebuilt: pl.DataFrame,
    unchecked: frozenset[str] = frozenset(),
) -> TableComparison:
    """Compare a rebuilt table with its published file cell by cell, in published row order.

    Columns in ``unchecked`` are counted, not compared.
    """
    result = TableComparison(season=season, kind=kind)
    published, result.relabeled_cells = relabel_washington(published)
    result.rows = published.height
    result.columns = published.width
    if published.columns != rebuilt.columns:
        missing = [c for c in published.columns if c not in rebuilt.columns]
        extra = [c for c in rebuilt.columns if c not in published.columns]
        if missing or extra:
            result.structure.append(f"columns missing {missing}, extra {extra}")
        else:
            result.structure.append("columns in another order")
    if published.height != rebuilt.height:
        result.structure.append(f"{rebuilt.height} rows, published {published.height}")
        return result
    supplied = set(SUPPLIED_COLUMNS.get(kind, ()))
    for column in published.columns:
        if column not in rebuilt.columns:
            continue
        mine = rebuilt.get_column(column)
        theirs = published.get_column(column)
        if mine.dtype != theirs.dtype:
            result.dtype_differences.append(f"{column}: {mine.dtype}, published {theirs.dtype}")
        if column in unchecked:
            result.unchecked_cells += published.height
            continue
        if column in supplied:
            result.supplied_cells += published.height
        _compare_column(result, column, theirs, mine, published)
    return result


def _compare_column(
    result: TableComparison,
    column: str,
    theirs: pl.Series,
    mine: pl.Series,
    published: pl.DataFrame,
) -> None:
    """Add one column's cells to ``result``."""
    result.cells += theirs.len()
    numeric = theirs.dtype.is_numeric() and mine.dtype.is_numeric()
    if numeric and (_is_float(theirs.dtype) or _is_float(mine.dtype)):
        exact = not _is_float(theirs.dtype)
        a = theirs.cast(pl.Float64)
        b = mine.cast(pl.Float64)
        null_a = a.is_null()
        null_b = b.is_null()
        nan_a = a.is_nan().fill_null(value=False)
        nan_b = b.is_nan().fill_null(value=False)
        identical = (null_a & null_b) | (nan_a & nan_b) | (a == b).fill_null(value=False)
        diff = (a - b).abs()
        tolerance = 0.0 if exact else FLOAT_TOLERANCE
        close = ~null_a & ~null_b & ~nan_a & ~nan_b & (diff <= tolerance).fill_null(value=False)
        agrees = identical | close
        finite = diff.filter(agrees & ~identical)
        if finite.len():
            result.max_abs_difference = max(result.max_abs_difference, float(finite.max()))
        result.identical += int(identical.sum())
        result.within_tolerance += int((agrees & ~identical).sum())
    else:
        agrees = (theirs.is_null() & mine.is_null()) | (
            mine.cast(theirs.dtype, strict=False) == theirs
        ).fill_null(value=False)
        if not numeric and theirs.dtype != mine.dtype:
            agrees = (theirs.is_null() & mine.is_null()) | (
                mine.cast(pl.String) == theirs.cast(pl.String)
            ).fill_null(value=False)
        result.identical += int(agrees.sum())
    bad = (~agrees).arg_true().to_list()
    result.outside += len(bad)
    for index in bad[: max(0, MAX_LISTED_DIFFERENCES - len(result.differences))]:
        result.differences.append(
            {
                "column": column,
                "row": _row_label(published, index),
                "published": _jsonable(theirs[index]),
                "rebuilt": _jsonable(mine[index]),
            }
        )


def _jsonable(value: object) -> object:
    """Return ``value`` as JSON can hold it (NaN as text)."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


@dataclass
class SeasonRun:
    """One season's build and comparisons."""

    season: int
    load_seconds: float
    build_seconds: float
    compare_seconds: float
    comparisons: list[TableComparison]
    published_log_comparisons: list[TableComparison]
    unrecovered_team_rates: list[str]
    unrecovered_qb_rates: list[str]


def _unchecked_columns(
    kind: str, columns: list[str], team_rates: set[str], qb_rates: set[str]
) -> frozenset[str]:
    """Return the columns of a table built from published logs that need unrecovered parts.

    A team rate without its parts, a formula rate built from one, or either as an opponent
    profile's ``opp_`` column; a quarterback rate without its parts, or its ``qopp_`` column.
    """
    team = set(team_rates) | {
        rate
        for rate, formula in TEAM_FORMULA_RATES.items()
        if team_rates.intersection(formula.inputs)
    }
    unchecked: set[str] = set()
    for column in columns:
        if kind.startswith("qb_"):
            if column.removeprefix("qopp_") in qb_rates:
                unchecked.add(column)
        elif column.removeprefix("opp_") in team:
            unchecked.add(column)
    return frozenset(unchecked)


def compare_from_published_logs(
    season: int,
    published: dict[str, pl.DataFrame],
    games: season_tables.SeasonGames,
    ratings: tuple[pl.DataFrame, pl.DataFrame],
) -> tuple[list[TableComparison], list[str], list[str]]:
    """Build the season tables from the published game logs and compare what they determine.

    The schedule is the loaders' played schedule, as the source's build read it (the logs lack the
    three games without play-by-play, whose teams still count as opponents). ``games`` (the
    play-by-play build) says which rates carry hidden parts; the ones the adapter cannot rebuild
    from the published columns are reported, and their columns are not compared.
    """
    team_logs, _ = relabel_washington(published["team_game_logs"])
    qb_logs, _ = relabel_washington(published["qb_game_logs"])
    from_logs = season_tables.SeasonGames(
        team_games=published_logs.team_games_from_published(team_logs),
        qb_games=published_logs.qb_games_from_published(qb_logs),
        schedule=games.schedule,
    )
    team_rates = set(rate_parts(games.team_games.columns)) - set(
        published_logs.recovered_team_rates(team_logs.columns)
    )
    qb_rates = set(rate_parts(games.qb_games.columns))
    tables = season_tables.build_season_tables(
        from_logs, pl.DataFrame(), team_ratings=ratings[0], qb_ratings=ratings[1]
    ).tables()
    comparisons = []
    for kind in PUBLISHED_LOG_KINDS:
        frame = published.get(kind)
        if frame is None or kind not in tables:
            comparison = TableComparison(season=season, kind=kind)
            comparison.structure.append("absent from the published files or the rebuild")
            comparisons.append(comparison)
            continue
        unchecked = _unchecked_columns(kind, frame.columns, team_rates, qb_rates)
        comparisons.append(compare_tables(season, kind, frame, tables[kind], unchecked))
    return comparisons, sorted(team_rates), sorted(qb_rates)


def run_season(season: int, source: Path, cache_dir: Path, hashes: dict[str, str]) -> SeasonRun:
    """Build one season through the loaders and compare every table with its published file."""
    started = time.perf_counter()
    inputs = sources.load_season_inputs(season, cache_dir=cache_dir, current_season=LAST_SEASON + 1)
    loaded = time.perf_counter()

    def published(kind: str) -> pl.DataFrame:
        path = source / "data" / f"{season}_{kind}.parquet"
        hashes[path.name] = sha256_file(path)
        return pl.read_parquet(path)

    published_tables = {
        kind: published(kind)
        for kind in KINDS
        if (source / "data" / f"{season}_{kind}.parquet").exists()
    }

    team_ratings, _ = relabel_washington(published("ratings"))
    qb_ratings = published("qb_combined").select(season_tables.QB_RATING_INPUT_COLUMNS)
    games = season_tables.build_season_games(inputs)
    tables = season_tables.build_season_tables(
        games, inputs.pbp, team_ratings=team_ratings, qb_ratings=qb_ratings
    ).tables()
    built = time.perf_counter()
    comparisons: list[TableComparison] = []
    for kind in KINDS:
        if kind not in tables or kind not in published_tables:
            comparison = TableComparison(season=season, kind=kind)
            comparison.structure.append(
                f"rebuilt {'present' if kind in tables else 'absent'}, "
                f"published {'present' if kind in published_tables else 'absent'}"
            )
            comparisons.append(comparison)
            continue
        comparisons.append(compare_tables(season, kind, published_tables[kind], tables[kind]))
    from_logs, team_rates, qb_rates = compare_from_published_logs(
        season, published_tables, games, (team_ratings, qb_ratings)
    )
    compared = time.perf_counter()
    return SeasonRun(
        season=season,
        load_seconds=loaded - started,
        build_seconds=built - loaded,
        compare_seconds=compared - built,
        comparisons=comparisons,
        published_log_comparisons=from_logs,
        unrecovered_team_rates=team_rates,
        unrecovered_qb_rates=qb_rates,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path.home() / "scratch" / "nflreadpy_cache_snapshot_20261008",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "scratch" / "nflreadpy_cache_snapshot_20261008.sha256",
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=list(range(FIRST_SEASON, LAST_SEASON + 1))
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    return parser.parse_args(argv)


def _summary(comparisons: list[TableComparison], kinds: tuple[str, ...]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        rows = [c for c in comparisons if c.kind == kind]
        by_kind[kind] = {
            "tables": len(rows),
            "rows": sum(c.rows for c in rows),
            "cells": sum(c.cells for c in rows),
            "identical": sum(c.identical for c in rows),
            "within_tolerance": sum(c.within_tolerance for c in rows),
            "outside": sum(c.outside for c in rows),
            "supplied_cells": sum(c.supplied_cells for c in rows),
            "unchecked_cells": sum(c.unchecked_cells for c in rows),
            "relabeled_cells": sum(c.relabeled_cells for c in rows),
            "max_abs_difference": max((c.max_abs_difference for c in rows), default=0.0),
            "tables_with_structure_differences": sum(1 for c in rows if c.structure),
            "tables_with_dtype_differences": sum(1 for c in rows if c.dtype_differences),
        }
    return {
        "tables": len(comparisons),
        "cells": sum(c.cells for c in comparisons),
        "identical": sum(c.identical for c in comparisons),
        "within_tolerance": sum(c.within_tolerance for c in comparisons),
        "outside": sum(c.outside for c in comparisons),
        "supplied_cells": sum(c.supplied_cells for c in comparisons),
        "unchecked_cells": sum(c.unchecked_cells for c in comparisons),
        "relabeled_cells": sum(c.relabeled_cells for c in comparisons),
        "tables_with_structure_differences": sum(1 for c in comparisons if c.structure),
        "tables_with_dtype_differences": sum(1 for c in comparisons if c.dtype_differences),
        "max_abs_difference": max((c.max_abs_difference for c in comparisons), default=0.0),
        "by_kind": by_kind,
    }


def _code_list(names: list[str]) -> str:
    """Return names as a code-formatted list wrapped to the Markdown line limit, or None."""
    if not names:
        return "None."
    return textwrap.fill(
        ", ".join(f"`{name}`" for name in names),
        width=100,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _season_list(seasons: list[int]) -> str:
    """Return seasons as runs of consecutive years, such as ``1999-2012, 2014``."""
    runs: list[list[int]] = []
    for season in sorted(seasons):
        if runs and season == runs[-1][-1] + 1:
            runs[-1].append(season)
        else:
            runs.append([season])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs)


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Season statistics parity with nfl-sos-ratings",
        "",
        "Generated by `.agents/sos_port/season_stats_parity.py`; do not edit by hand. Every table",
        "below was rebuilt by `nfl_predictor.season_stats` from the nflreadpy disk-cache snapshot",
        "the published files were built from, read through this repository's loaders, and",
        "compared cell by cell with the published file of `nfl-sos-ratings` at",
        f"`{report['source_commit']}`.",
        "",
        "## Run",
        "",
        "Command, from the repository root:",
        "",
        "```bash",
        f"{report['command']}",
        "```",
        "",
        f"- Code: `{report['code_commit']}`",
        f"- Finished: {report['finished']}",
        f"- Seasons: {report['seasons'][0]}-{report['seasons'][-1]} ({len(report['seasons'])})",
        (
            f"- Polars {report['versions']['polars']}, nflreadpy {report['versions']['nflreadpy']},"
            f" `POLARS_MAX_THREADS={report['polars_threads']}`"
        ),
        (
            f"- Snapshot files served: {len(report['snapshot_files'])}, every one matching the"
            " manifest"
        ),
        f"- Float tolerance: {report['float_tolerance']:g} absolute",
        (
            f"- Play-by-play columns read: {len(report['pbp_columns']['read'])}; of them, missing"
            " from this code's `constants.PBP_COLUMNS`:"
            f" {len(report['pbp_columns']['missing_from_constants'])}"
        ),
        "",
        "## Headline counts",
        "",
        f"- Tables compared: {summary['tables']}",
        f"- Cells compared: {summary['cells']:,}",
        f"- Bit-identical: {summary['identical']:,}",
        (
            f"- Within tolerance but not bit-identical: {summary['within_tolerance']:,}"
            f" (largest difference {summary['max_abs_difference']:.3g})"
        ),
        f"- Outside tolerance: {summary['outside']:,}",
        f"- Tables with column or row differences: {summary['tables_with_structure_differences']}",
        f"- Tables with a column type that differs: {summary['tables_with_dtype_differences']}",
        (
            f"- Cells taken from the published ratings (rating columns of `combined` and"
            f" `qb_combined`): {summary['supplied_cells']:,}"
        ),
        f"- Published team codes read as `WSH` instead of `WAS`: {summary['relabeled_cells']:,}",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## By table",
        "",
        (
            "| table | seasons | rows | cells | bit-identical | within tolerance | outside |"
            " largest difference |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, row in summary["by_kind"].items():
        lines.append(
            f"| `{kind}` | {row['tables']} | {row['rows']:,} | {row['cells']:,} |"
            f" {row['identical']:,} | {row['within_tolerance']:,} | {row['outside']:,} |"
            f" {row['max_abs_difference']:.3g} |"
        )
    logs = report["published_log_summary"]
    lines += [
        "",
        "## Season tables from the published game logs",
        "",
        "The same season tables built a second way, from the published team and quarterback game",
        "logs (`season_stats.published_logs`), independent of the play-by-play build. The logs",
        "leave out each rate's hidden numerator and denominator; the adapter rebuilds those the",
        "published columns determine, and a column that needs any other is counted, not",
        "compared (its rates are listed below).",
        "",
        f"- Tables compared: {logs['tables']}",
        (
            f"- Cells compared: {logs['cells']:,}; bit-identical {logs['identical']:,}; within"
            f" tolerance {logs['within_tolerance']:,}; outside {logs['outside']:,}"
        ),
        f"- Cells not determined by the published logs (not compared): {logs['unchecked_cells']:,}",
        f"- Tables with column or row differences: {logs['tables_with_structure_differences']}",
        "",
        (
            "| table | seasons | cells compared | bit-identical | within tolerance | outside |"
            " not compared |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, row in logs["by_kind"].items():
        lines.append(
            f"| `{kind}` | {row['tables']} | {row['cells']:,} | {row['identical']:,} |"
            f" {row['within_tolerance']:,} | {row['outside']:,} | {row['unchecked_cells']:,} |"
        )
    lines += [
        "",
        "Team rates whose parts the published logs do not determine, in every season:",
        "",
        _code_list(report["unrecovered_rates"]["team"]),
        "",
        "Quarterback rates whose parts the published logs do not determine:",
        "",
        _code_list(report["unrecovered_rates"]["qb"]),
    ]
    lines += [
        "",
        "## Timings",
        "",
        (
            f"Full run: {report['total_seconds']:.0f} s"
            f" (load {report['timings']['load']:.0f} s, build {report['timings']['build']:.0f} s,"
            f" compare {report['timings']['compare']:.0f} s)."
        ),
        "",
        "| season | load (s) | build (s) | compare (s) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {run['season']} | {run['load_seconds']:.1f} | {run['build_seconds']:.1f} |"
        f" {run['compare_seconds']:.1f} |"
        for run in report["runs"]
    )
    seasons_by_difference: dict[tuple[str, str], list[int]] = {}
    for run in report["runs"]:
        for c in run["comparisons"]:
            for difference in c["dtype_differences"]:
                seasons_by_difference.setdefault((c["kind"], difference), []).append(c["season"])
    dtype_lines = [
        f"- `{kind}` `{difference}` ({_season_list(seasons)})"
        for (kind, difference), seasons in sorted(seasons_by_difference.items())
    ]
    lines += [
        "",
        "## Column types that differ",
        "",
        "Rebuilt type first, published second; every value in these columns is compared.",
        "",
    ]
    lines += dtype_lines or ["None."]
    structure = [
        f"- {c['season']} `{c['kind']}`{label}: {'; '.join(c['structure'])}"
        for run in report["runs"]
        for key, label in (
            ("comparisons", ""),
            ("published_log_comparisons", " (from the published logs)"),
        )
        for c in run[key]
        if c["structure"]
    ]
    lines += ["", "## Column and row differences", ""]
    lines += structure or ["None."]
    differences = [
        f"- {c['season']} `{c['kind']}`{label} `{d['column']}` ({d['row']}): published"
        f" {d['published']!r}, rebuilt {d['rebuilt']!r}"
        for run in report["runs"]
        for key, label in (
            ("comparisons", ""),
            ("published_log_comparisons", " (from the published logs)"),
        )
        for c in run[key]
        for d in c["differences"]
    ]
    lines += ["", "## Cells outside tolerance (first ones per table)", ""]
    lines += differences or ["None."]
    lines += [
        "",
        "## Play-by-play columns",
        "",
        "The build read only `sources.SEASON_STATS_PBP_COLUMNS`. Missing from this code's",
        "`constants.PBP_COLUMNS` (the play-by-play cache's list; the run set it to the columns",
        "above):",
        "",
        _code_list(report["pbp_columns"]["missing_from_constants"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write the reports; return 1 when any table fails."""
    args = _parse_args(argv)
    head = source_head(args.source)
    if head != SOURCE_COMMIT:
        msg = f"{args.source} is at {head}, not the frozen commit {SOURCE_COMMIT}"
        raise SystemExit(msg)
    work_dir = args.work_dir or Path(
        tempfile.mkdtemp(prefix="season_stats_parity_", dir=Path.home() / "scratch")
    )
    if "data" in work_dir.resolve().parts:
        msg = f"the work directory must not be under a data directory: {work_dir}"
        raise SystemExit(msg)
    snapshot = SnapshotDownloads(args.snapshot, read_manifest(args.manifest))
    snapshot.install()
    original_columns = list(constants.PBP_COLUMNS)
    constants.PBP_COLUMNS = list(sources.SEASON_STATS_PBP_COLUMNS)
    hashes: dict[str, str] = {}
    runs: list[SeasonRun] = []
    started = time.perf_counter()
    for season in args.seasons:
        run = run_season(season, args.source, work_dir / "cache", hashes)
        failed = [
            c.kind for c in (*run.comparisons, *run.published_log_comparisons) if not c.passed
        ]
        sys.stderr.write(
            f"{season}: load {run.load_seconds:.1f}s build {run.build_seconds:.1f}s"
            f" compare {run.compare_seconds:.1f}s, failed: {failed or 'none'}\n"
        )
        runs.append(run)
    total = time.perf_counter() - started
    summary = _summary([c for run in runs for c in run.comparisons], KINDS)
    log_summary = _summary(
        [c for run in runs for c in run.published_log_comparisons], PUBLISHED_LOG_KINDS
    )
    report = {
        "command": " ".join(["python", ".agents/sos_port/season_stats_parity.py", *(argv or [])]),
        "code_commit": code_commit(),
        "source": str(args.source),
        "source_commit": head,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seasons": args.seasons,
        "versions": {"polars": version("polars"), "nflreadpy": version("nflreadpy")},
        "polars_threads": os.environ.get("POLARS_MAX_THREADS"),
        "float_tolerance": FLOAT_TOLERANCE,
        "work_dir": str(work_dir),
        "total_seconds": total,
        "timings": {
            "load": sum(r.load_seconds for r in runs),
            "build": sum(r.build_seconds for r in runs),
            "compare": sum(r.compare_seconds for r in runs),
        },
        "pbp_columns": {
            "read": list(sources.SEASON_STATS_PBP_COLUMNS),
            "missing_from_constants": [
                c for c in sources.SEASON_STATS_PBP_COLUMNS if c not in original_columns
            ],
        },
        "snapshot_files": snapshot.served,
        "published_files": dict(sorted(hashes.items())),
        "summary": summary,
        "published_log_summary": log_summary,
        "unrecovered_rates": {
            "team": sorted({rate for run in runs for rate in run.unrecovered_team_rates}),
            "qb": sorted({rate for run in runs for rate in run.unrecovered_qb_rates}),
        },
        "passed": all(
            part["outside"] == 0 and part["tables_with_structure_differences"] == 0
            for part in (summary, log_summary)
        ),
        "runs": [
            {
                "season": r.season,
                "load_seconds": r.load_seconds,
                "build_seconds": r.build_seconds,
                "compare_seconds": r.compare_seconds,
                "comparisons": [asdict(c) for c in r.comparisons],
                "published_log_comparisons": [asdict(c) for c in r.published_log_comparisons],
                "unrecovered_team_rates": r.unrecovered_team_rates,
                "unrecovered_qb_rates": r.unrecovered_qb_rates,
            }
            for r in runs
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "season_stats_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "SEASON_STATS_PARITY.md").write_text(_markdown(report), encoding="utf-8")
    for label, part in (("play-by-play build", summary), ("published logs", log_summary)):
        sys.stderr.write(
            f"{label}: {part['cells']:,} cells: {part['identical']:,} identical,"
            f" {part['within_tolerance']:,} within tolerance, {part['outside']:,} outside,"
            f" {part['unchecked_cells']:,} not compared;"
            f" {part['tables_with_structure_differences']} tables with structure differences\n"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
