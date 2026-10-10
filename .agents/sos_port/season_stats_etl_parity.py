"""Compare the season statistics the ETL writes with every table ``nfl-sos-ratings`` published.

Run from the repository root (single-threaded unless the environment sets the thread counts; about
ten minutes on a busy machine; nothing is written outside a temporary directory under
``~/scratch``):

    .venv/bin/python .agents/sos_port/season_stats_etl_parity.py [--source PATH]
        [--seasons S ...] [--out-dir DIR]

``--source`` is the ``nfl-sos-ratings`` checkout (default: the one beside this repository's main
checkout), whose published Parquet files are read in place at its frozen commit and never
written; ``--seasons`` limits the comparison (default 1999-2026: every completed season and the
one in progress at the frozen commit; every season from 1999 through the last one compared is
still loaded and rated, as the ETL rates them); ``--out-dir`` writes the reports elsewhere
(default: beside this script). The script:

1. turns the snapshot of the nflreadpy cache that project read (``snapshot_cache.py``) into this
   repository's cache files in a temporary directory, the official weekly team stats included,
   with downloads blocked, so every source is read through this repository's loaders;
2. loads the sources as ``nfl-predictor data`` loads them (``data_collection._load_sources``): the
   play-by-play through ``loaders.load_pbp`` with the games credited to one team repaired
   (``nflverse_sources.repair_one_team_pbp_games``), the schedules, the ratings' sources
   (``ratings_build.load_ratings_sources``) and the season statistics' sources
   (``season_stats_build.load_season_stats_sources``);
3. builds every season through the ETL's ratings build with the season statistics
   (``ratings_build.RatingsBuild.build``), the last season as the season in progress, as the ETL
   builds the current one, and writes every table with the ETL's writer
   (``ratings_build.write_published_tables``);
4. reads every written season statistics table back and compares it with the published file cell
   by cell with ``season_stats_parity.py``'s comparison: the same columns in the same order, the
   same rows in the same order, integers, text and flags equal, floats within its
   ``FLOAT_TOLERANCE``, and Washington's published ``WAS`` read as ``WSH`` (counted). The rating
   columns of ``combined`` and ``qb_combined`` are counted apart: here they come from the ratings
   this run fitted, not from the published ratings.

The loaders' current season is 2027, so every snapshot season is read from the cache as a finished
one; the ratings build's current season is the last compared season, the one ``nfl-sos-ratings``
built as in progress. It writes ``SEASON_STATS_ETL_PARITY.md`` and
``season_stats_etl_parity.json`` and exits 1 if any cell is outside the tolerance or any table
differs in columns or rows.
"""

from __future__ import annotations

import os

# One BLAS and Polars thread unless the caller set the counts, as the other parity scripts run;
# the ratings fits pin one BLAS thread themselves. Both libraries read these when they load.
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import datetime as dt
import json
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path
from season_stats_parity import (  # a sibling module, importable once its dir is on the path
    FLOAT_TOLERANCE,
    KINDS,
    SOURCE_COMMIT,
    TableComparison,
    code_commit,
    compare_tables,
    default_source,
    sha256_file,
    source_head,
)

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import nflverse_sources, ratings_build, season_stats_build

HERE = Path(__file__).resolve().parent
OUT_MD = "SEASON_STATS_ETL_PARITY.md"
OUT_JSON = "season_stats_etl_parity.json"
FIRST_SEASON = 1999
LAST_SEASON = 2026
THREAD_VARIABLES = ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")


class TimedSeasonStatsBuild(season_stats_build.SeasonStatsBuild):
    """The ETL's season statistics build, adding up the seconds its tables take."""

    seconds = 0.0

    def tables(
        self,
        season: int,
        *,
        team_ratings: pl.DataFrame | None,
        qb_ratings: pl.DataFrame | None,
    ) -> dict[str, pl.DataFrame]:
        """Return the season's tables, as the ETL's build does, and time them."""
        start = time.perf_counter()
        try:
            return super().tables(season, team_ratings=team_ratings, qb_ratings=qb_ratings)
        finally:
            self.seconds += time.perf_counter() - start


def load_like_the_etl(
    cache: Path, last: int
) -> tuple[ratings_build.RatingsBuild, TimedSeasonStatsBuild]:
    """Return the ratings build with the season statistics, from the sources loaded as the ETL does.

    ``load_ratings_sources`` reads the default cache directory and the data directory's
    quarterback identity file, so both point into the temporary directory (the identity file is
    absent there; it names the starters the features read, not the tables). The season statistics
    build is returned too, for its time.
    """
    constants.NFLREADPY_CACHE_DIR = cache
    constants.DATA_PATH = cache.parent
    seasons = list(range(FIRST_SEASON, last + 1))
    options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
    pbp = nflverse_sources.repair_one_team_pbp_games(
        polars_utils.load_pbp(seasons, **options), **options
    )
    schedule = polars_utils.load_schedule(seasons, **options)
    loaded = ratings_build.load_ratings_sources(
        last,
        pbp_df=pbp,
        schedule_df=schedule,
        current_season=snapshot_cache.CURRENT_SEASON,
        force_refresh=False,
    )
    if loaded is None:
        message = "the snapshot cache has no play-by-play"
        raise SystemExit(message)
    season_stats = TimedSeasonStatsBuild(
        season_stats_build.load_season_stats_sources(
            seasons, loaded, current_season=snapshot_cache.CURRENT_SEASON, force_refresh=False
        )
    )
    build = ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(loaded), current_season=last, season_stats=season_stats
    )
    return build, season_stats


def compare_season(
    season: int, written: Path, source: Path, hashes: dict[str, str]
) -> list[TableComparison]:
    """Compare each season statistics table the ETL wrote for ``season`` with its published file."""
    comparisons: list[TableComparison] = []
    for kind in KINDS:
        mine = written / f"{season}_{kind}.parquet"
        theirs = source / "data" / f"{season}_{kind}.parquet"
        if not mine.exists() or not theirs.exists():
            comparison = TableComparison(season=season, kind=kind)
            comparison.structure.append(
                f"written {'present' if mine.exists() else 'absent'}, "
                f"published {'present' if theirs.exists() else 'absent'}"
            )
            comparisons.append(comparison)
            continue
        hashes[theirs.name] = sha256_file(theirs)
        comparisons.append(
            compare_tables(season, kind, pl.read_parquet(theirs), pl.read_parquet(mine))
        )
    return comparisons


def _summary(comparisons: list[TableComparison]) -> dict[str, Any]:
    """Return the totals of ``comparisons``, overall and by kind."""

    def totals(rows: list[TableComparison]) -> dict[str, Any]:
        return {
            "tables": len(rows),
            "rows": sum(c.rows for c in rows),
            "cells": sum(c.cells for c in rows),
            "identical": sum(c.identical for c in rows),
            "within_tolerance": sum(c.within_tolerance for c in rows),
            "outside": sum(c.outside for c in rows),
            "rating_cells": sum(c.supplied_cells for c in rows),
            "rating_identical": sum(c.supplied_identical for c in rows),
            "relabeled_cells": sum(c.relabeled_cells for c in rows),
            "max_abs_difference": max((c.max_abs_difference for c in rows), default=0.0),
            "tables_with_structure_differences": sum(1 for c in rows if c.structure),
            "tables_with_dtype_differences": sum(1 for c in rows if c.dtype_differences),
        }

    return {
        **totals(comparisons),
        "by_kind": {kind: totals([c for c in comparisons if c.kind == kind]) for kind in KINDS},
    }


def _wrap(text: str, prefix: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width=100,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrapped(lines: list[str]) -> list[str]:
    """Return ``lines`` with every bullet longer than 100 characters wrapped.

    Fenced code blocks and tables keep their lines as they are.
    """
    out: list[str] = []
    fenced = False
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("- ") and len(line) > 100:  # noqa: PLR2004 - the limit
            out += _wrap(line[2:], "- ")
        else:
            out.append(line)
    return out


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
    seasons = report["seasons"]
    computed = summary["cells"] - summary["rating_cells"]
    lines = [
        "# Season statistics through the ETL, parity with nfl-sos-ratings",
        "",
        *_wrap(
            "Generated by `.agents/sos_port/season_stats_etl_parity.py`; do not edit by hand. "
            "Every table below was built by the ETL's season statistics step "
            "(`utils/polars/season_stats_build.py`, inside `ratings_build.RatingsBuild.build`) "
            "from the nflreadpy disk-cache snapshot the published files were built from, read "
            "through this repository's loaders as `nfl-predictor data` reads them, written by "
            "the ETL's writer, read back, and compared cell by cell with the published file of "
            f"`nfl-sos-ratings` at `{report['source_commit']}`."
        ),
        "",
        "## Run",
        "",
        "Command, from the repository root:",
        "",
        "```bash",
        report["command"],
        "```",
        "",
        f"- Code: `{report['code_commit']}`",
        f"- Finished: {report['finished']}",
        (
            f"- Seasons compared: {_season_list(seasons)} ({len(seasons)}; the last one built as "
            "the season in progress)"
        ),
        (
            f"- Polars {report['versions']['polars']}, nflreadpy {report['versions']['nflreadpy']},"
            f" numpy {report['versions']['numpy']}; {report['threads']}"
        ),
        (
            f"- Snapshot files: {len(report['snapshot_files'])}, every one matching the manifest;"
            f" published files read: {len(report['published_files'])}"
        ),
        f"- Float tolerance: {report['float_tolerance']:g} absolute (`season_stats_parity.py`'s)",
        (f"- Machine load averages: before {report['load_before']}, after {report['load_after']}"),
        "",
        "## Headline counts",
        "",
        *_wrap(
            "The rating columns of `combined` and `qb_combined` (the team ratings, `sos`, `SRS`, "
            "and each passer's `adj_qb_epa_per_dropback` and `qb_faced_pass_defense`) come from "
            "the ratings this run fitted, so their cells are counted apart from the cells the "
            "season statistics compute."
        ),
        "",
        f"- Tables compared: {summary['tables']}",
        f"- Cells compared: {summary['cells']:,}; bit-identical {summary['identical']:,}",
        (
            f"- Of them, computed by the season statistics: {computed:,}; bit-identical"
            f" {summary['identical'] - summary['rating_identical']:,}"
        ),
        (
            f"- Of them, rating columns fitted by this run: {summary['rating_cells']:,};"
            f" bit-identical {summary['rating_identical']:,}"
        ),
        (
            f"- Within tolerance but not bit-identical: {summary['within_tolerance']:,}"
            f" (largest difference {summary['max_abs_difference']:.3g})"
        ),
        f"- Outside tolerance: {summary['outside']:,}",
        f"- Tables with column or row differences: {summary['tables_with_structure_differences']}",
        f"- Tables with a column type that differs: {summary['tables_with_dtype_differences']}",
        f"- Published team codes read as `WSH` instead of `WAS`: {summary['relabeled_cells']:,}",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## By table",
        "",
        (
            "| table | seasons | rows | cells | bit-identical | within tolerance | outside |"
            " rating cells | rating cells bit-identical |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, row in summary["by_kind"].items():
        lines.append(
            f"| `{kind}` | {row['tables']} | {row['rows']:,} | {row['cells']:,} |"
            f" {row['identical']:,} | {row['within_tolerance']:,} | {row['outside']:,} |"
            f" {row['rating_cells']:,} | {row['rating_identical']:,} |"
        )
    seasons_by_difference: dict[tuple[str, str], list[int]] = {}
    for comparison in report["comparisons"]:
        for difference in comparison["dtype_differences"]:
            seasons_by_difference.setdefault((comparison["kind"], difference), []).append(
                comparison["season"]
            )
    lines += [
        "",
        "## Column types that differ",
        "",
        "Written type first, published second; every value in these columns is compared.",
        "",
    ]
    lines += [
        f"- `{kind}` `{difference}` ({_season_list(found)})"
        for (kind, difference), found in sorted(seasons_by_difference.items())
    ] or ["None."]
    structure = [
        f"- {c['season']} `{c['kind']}`: {'; '.join(c['structure'])}"
        for c in report["comparisons"]
        if c["structure"]
    ]
    lines += ["", "## Column and row differences", ""]
    lines += structure or ["None."]
    differences = [
        f"- {c['season']} `{c['kind']}` `{d['column']}` ({d['row']}): published"
        f" {d['published']!r}, written {d['rebuilt']!r}"
        for c in report["comparisons"]
        for d in c["differences"]
    ]
    lines += ["", "## Cells outside tolerance (first ones per table)", ""]
    lines += differences or ["None."]
    timings = report["timings"]
    lines += [
        "",
        "## Timings",
        "",
        *_wrap(
            f"Wall-clock seconds with {report['threads']}, on a machine shared with other work "
            f"(load averages above): writing the snapshot's cache files {timings['cache']:.0f}, "
            f"loading the sources as the ETL does {timings['load']:.0f}, building every season "
            f"(ratings and season statistics) {timings['build']:.0f}, of which the season "
            f"statistics {timings['season_stats']:.0f}, writing the tables "
            f"{timings['write']:.1f}, comparing {timings['compare']:.0f}."
        ),
        "",
    ]
    return "\n".join(_wrapped(lines))


def build_and_compare(
    files: snapshot_cache.SnapshotFiles, seasons: list[int], source: Path
) -> tuple[list[TableComparison], dict[str, float], dict[str, str]]:
    """Build every season through the ETL's step in a temporary directory, and compare them.

    Returns:
        Each compared table's comparison, the seconds of each step, and the SHA-256 of every
        published file read.

    """
    last = seasons[-1]
    timings: dict[str, float] = {}
    hashes: dict[str, str] = {}
    comparisons: list[TableComparison] = []
    work = Path.home() / "scratch"
    with tempfile.TemporaryDirectory(prefix="season_stats_etl_parity_", dir=work) as directory:
        root = Path(directory)
        if "data" in root.resolve().parts:
            message = f"the work directory must not be under a data directory: {root}"
            raise SystemExit(message)
        cache = root / "cache" / "nflreadpy"
        start = time.perf_counter()
        snapshot_cache.write_cache(files, cache, list(range(FIRST_SEASON, last + 1)))
        timings["cache"] = time.perf_counter() - start
        start = time.perf_counter()
        build, season_stats = load_like_the_etl(cache, last)
        timings["load"] = time.perf_counter() - start
        start = time.perf_counter()
        for season in range(FIRST_SEASON, last + 1):
            build.build(season)
            sys.stderr.write(f"{season}: built\n")
        timings["build"] = time.perf_counter() - start
        timings["season_stats"] = season_stats.seconds
        written = root / constants.RATINGS_DIRNAME
        start = time.perf_counter()
        ratings_build.write_published_tables(build.published_tables(), written)
        timings["write"] = time.perf_counter() - start
        start = time.perf_counter()
        for season in seasons:
            found = compare_season(season, written, source, hashes)
            failed = [c.kind for c in found if not c.passed]
            sys.stderr.write(f"{season}: compared, failed: {failed or 'none'}\n")
            comparisons += found
        timings["compare"] = time.perf_counter() - start
    return comparisons, timings, hashes


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write the reports; return 1 when any table fails."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=list(range(FIRST_SEASON, LAST_SEASON + 1))
    )
    parser.add_argument("--out-dir", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    head = source_head(arguments.source)
    if head != SOURCE_COMMIT:
        message = f"{arguments.source} is at {head}, not the frozen commit {SOURCE_COMMIT}"
        raise SystemExit(message)
    seasons = sorted(arguments.seasons)
    load_before = ", ".join(f"{value:.2f}" for value in os.getloadavg())
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    comparisons, timings, hashes = build_and_compare(files, seasons, arguments.source)
    summary = _summary(comparisons)
    report: dict[str, Any] = {
        "command": " ".join(
            [".venv/bin/python", ".agents/sos_port/season_stats_etl_parity.py", *(argv or [])]
        ),
        "code_commit": code_commit(),
        "source": str(arguments.source),
        "source_commit": head,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seasons": seasons,
        "versions": {name: version(name) for name in ("polars", "nflreadpy", "numpy")},
        "threads": ", ".join(f"`{name}={os.environ.get(name)}`" for name in THREAD_VARIABLES),
        "float_tolerance": FLOAT_TOLERANCE,
        "load_before": load_before,
        "load_after": ", ".join(f"{value:.2f}" for value in os.getloadavg()),
        "timings": timings,
        "snapshot_files": files.hashes,
        "published_files": dict(sorted(hashes.items())),
        "summary": summary,
        "passed": summary["outside"] == 0 and summary["tables_with_structure_differences"] == 0,
        "comparisons": [asdict(c) for c in comparisons],
    }
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    (arguments.out_dir / OUT_JSON).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (arguments.out_dir / OUT_MD).write_text(_markdown(report), encoding="utf-8")
    sys.stderr.write(
        f"{summary['cells']:,} cells: {summary['identical']:,} identical,"
        f" {summary['within_tolerance']:,} within tolerance, {summary['outside']:,} outside;"
        f" {summary['tables_with_structure_differences']} tables with structure differences\n"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
