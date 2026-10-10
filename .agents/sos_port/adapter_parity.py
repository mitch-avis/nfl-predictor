"""Check the ratings adapter against ``nfl-sos-ratings``' game logs and published ratings files.

Run from the repository root (single-threaded, a few minutes):

    .venv/bin/python .agents/sos_port/adapter_parity.py [--source PATH] [--seasons S ...]
        [--out-dir DIR]

``--source`` is the ``nfl-sos-ratings`` checkout (default: the one beside this repository's main
checkout), read at its frozen commit only; ``--seasons`` limits the comparison (every season from
1999 is still built, for the penalties and the prior); ``--out-dir`` writes the reports elsewhere
(default: beside this script). The script runs with one BLAS and Polars thread unless the
environment sets the counts, and:

1. turns the snapshot of the nflreadpy cache that project read (``snapshot_cache.py``) into this
   repository's cache files in a temporary directory, with downloads blocked, and loads every
   season through this repository's own loaders (``loaders.load_pbp``, ``load_schedule``, and
   the player sources of ``nflverse_sources``);
2. builds each season's team-game and quarterback-game rows with the adapter
   (``nfl_predictor.ratings.pbp_inputs``) and compares them, row keys, order and every column the
   fits consume, with that project's published game logs read through
   ``nfl_predictor.ratings.published_logs``;
3. rebuilds every season's ratings tables from the adapter's rows and compares them with the
   published files cell by cell, with the ratings parity audit's comparison and tolerances
   (``ratings_parity.compare_table``);
4. counts, for every week, how many passer-games the quarterback identity known before the week
   treats differently from the season's identity (reported, not judged).

Team codes: the published files write Washington as ``WAS``; this repository's code is ``WSH``.
Published team columns are mapped through ``constants.ALIAS_TO_CANONICAL`` before comparing, which
moves no row (both codes sort last). It writes ``ADAPTER_PARITY.md`` and ``adapter_parity.json``
and exits 1 if any row or cell differs beyond the tolerance or any table differs in shape.
"""

from __future__ import annotations

import os

# One BLAS and Polars thread unless the caller set the counts, as nfl-sos-ratings built its
# files. Both libraries read these when they load, so they are set before anything imports them.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import datetime as dt
import json
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path
from ratings_parity import (  # a sibling module, importable once its dir is on the path
    COMBINED_RATING_COLUMNS,
    FLOAT_TOLERANCE,
    QB_TABLES,
    ROUNDED_TOLERANCE,
    TEAM_TABLES,
    THREAD_VARIABLES,
    TableResult,
    _environment,
    compare_table,
)
from ratings_source import (  # a sibling module, importable once its dir is on the path
    SOURCE_COMMIT,
    check_source,
    default_source,
    package_hashes,
    published_file,
    repo_root,
    sha256_file,
    source_head,
)

from nfl_predictor import constants
from nfl_predictor.ratings.pbp_inputs import (
    SeasonSources,
    passer_games_from_pbp,
    quarterback_rows,
    team_games_from_pbp,
)
from nfl_predictor.ratings.published_logs import (
    qb_games_from_published,
    team_games_from_published,
)
from nfl_predictor.ratings.season import build_qb_season, build_team_season
from nfl_predictor.ratings.team_prior import PriorHistory
from nfl_predictor.utils.polars import loaders, nflverse_sources

HERE = Path(__file__).resolve().parent
OUT_MD = "ADAPTER_PARITY.md"
OUT_JSON = "adapter_parity.json"
FIRST_SEASON = 1999
LAST_SEASON = 2025
TEAM_CODE_COLUMNS = ("team", "opponent_team", "other_team", "excluded_team")
TEAM_ROW_KEYS = ("game_id", "team")
QB_ROW_KEYS = ("game_id", "team", "qb_id")
MAX_IDENTITY_EXAMPLES = 12
# The season tables: every published table but the weekly ranges of a season in progress.
SEASON_TABLES = {
    name: keys
    for name, keys in {**TEAM_TABLES, **QB_TABLES}.items()
    if not name.endswith("_by_week") or name in {"ratings_by_week", "qb_ratings_by_week"}
}


def _canonical_teams(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` with every team-code column mapped to this repository's codes."""
    return frame.with_columns(
        pl.col(name).replace(constants.ALIAS_TO_CANONICAL)
        for name in TEAM_CODE_COLUMNS
        if name in frame.columns
    )


@dataclass
class SeasonRows:
    """One season's adapter rows and the seconds the adapter took."""

    team: pl.DataFrame
    passers: pl.DataFrame
    quarterbacks: pl.DataFrame
    seconds: float


@dataclass
class IdentityCounts:
    """Passer-games the identity known before a week treats unlike the season's identity."""

    weeks: int = 0
    passer_games_seen: int = 0
    kept_only_by_season: int = 0
    kept_only_before_week: int = 0
    official_values_differ: int = 0
    examples: list[str] = field(default_factory=list)


class Published:
    """The published files, read once each, with the SHA-256 of every file read."""

    def __init__(self, root: Path) -> None:
        """Remember the checkout root."""
        self.root = root
        self.hashes: dict[str, str] = {}

    def read(self, season: int, name: str) -> pl.DataFrame:
        """Return one published file with this repository's team codes."""
        path = published_file(self.root, season, name)
        self.hashes[f"data/{path.name}"] = sha256_file(path)
        return _canonical_teams(pl.read_parquet(path))


def load_rows(cache: Path, seasons: list[int]) -> dict[int, SeasonRows]:
    """Load every season through this repository's loaders and build the adapter's rows."""
    options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
    players = nflverse_sources.load_players(cache_dir=cache)
    rows: dict[int, SeasonRows] = {}
    for season in seasons:
        plays = loaders.load_pbp([season], **options)
        schedule = loaders.load_schedule([season], **options)
        sources = SeasonSources(
            season=season,
            plays=plays,
            player_stats=nflverse_sources.load_weekly_player_stats([season], **options),
            snap_counts=nflverse_sources.load_snap_counts([season], **options),
            players=players,
            weekly_rosters=nflverse_sources.load_weekly_rosters([season], **options),
        )
        start = time.perf_counter()
        team = team_games_from_pbp(plays, schedule).drop("season")
        passers = passer_games_from_pbp(sources, team)
        quarterbacks = quarterback_rows(passers)
        rows[season] = SeasonRows(team, passers, quarterbacks, time.perf_counter() - start)
    return rows


def compare_rows(
    rows: dict[int, SeasonRows], published: Published, seasons: list[int]
) -> list[TableResult]:
    """Compare each season's adapter rows with the published game logs."""
    results: list[TableResult] = []
    for season in seasons:
        team = team_games_from_published(published.read(season, "team_game_logs"))
        quarterbacks = qb_games_from_published(published.read(season, "qb_game_logs"))
        results.append(
            compare_table(season, "team-game rows", rows[season].team, team, TEAM_ROW_KEYS)
        )
        results.append(
            compare_table(
                season,
                "quarterback-game rows",
                rows[season].quarterbacks,
                quarterbacks,
                QB_ROW_KEYS,
            )
        )
    return results


def _season_tables(season: int, rows: SeasonRows, history: PriorHistory) -> dict[str, pl.DataFrame]:
    """Rebuild one season's tables from the adapter's rows."""
    team = build_team_season(season, rows.team, history)
    qb = build_qb_season(rows.quarterbacks, rows.team)
    return {
        "ratings": team.ratings,
        "team_prior": team.team_prior,
        "ratings_by_week": team.ratings_by_week,
        "rating_ranges": team.rating_ranges,
        "rating_pairs": team.rating_pairs,
        "combined": team.ratings.select(COMBINED_RATING_COLUMNS).sort("team"),
        "qb_ratings": qb.qb_ratings,
        "qb_ratings_by_week": qb.qb_ratings_by_week,
        "qb_rating_ranges": qb.qb_rating_ranges,
        "qb_rating_pairs": qb.qb_rating_pairs,
        "qb_combined": qb.qb_season,
    }


def compare_tables(
    rows: dict[int, SeasonRows], published: Published, seasons: list[int]
) -> tuple[list[TableResult], float]:
    """Rebuild every season's tables from the adapter's rows and compare them with the files."""
    history = PriorHistory(lambda season: rows[season].team)
    results: list[TableResult] = []
    start = time.perf_counter()
    for season in sorted(rows):
        if season not in seasons:
            continue
        ours = _season_tables(season, rows[season], history)
        for table, keys in SEASON_TABLES.items():
            expected = published.read(season, table)
            if table in {"combined", "qb_combined"}:
                expected = expected.select(ours[table].columns)
            results.append(compare_table(season, table, ours[table], expected, keys))
        status = all(r.passed for r in results if r.season == season)
        sys.stdout.write(f"{season}: tables {'ok' if status else 'DIFFER'}\n")
        sys.stdout.flush()
    return results, time.perf_counter() - start


def identity_counts(rows: dict[int, SeasonRows], seasons: list[int]) -> IdentityCounts:
    """Count the passer-games the identity known before each week treats unlike the season's."""
    counts = IdentityCounts()
    key = ["game_id", "team", "qb_id"]
    for season in seasons:
        passers = rows[season].passers
        season_rows = quarterback_rows(passers)
        last_week = int(passers.get_column("week").max() or 0)
        for week in range(2, last_week + 2):
            before = quarterback_rows(passers, before_week=week)
            whole = season_rows.filter(pl.col("week") < week)
            counts.weeks += 1
            counts.passer_games_seen += whole.height
            only_season = whole.join(before, on=key, how="anti")
            only_before = before.join(whole, on=key, how="anti")
            counts.kept_only_by_season += only_season.height
            counts.kept_only_before_week += only_before.height
            both = whole.join(before, on=key, suffix="_before")
            changed = both.filter(
                (pl.col("qb_dropback_epa") != pl.col("qb_dropback_epa_before"))
                | (pl.col("qb_attempts") != pl.col("qb_attempts_before"))
            )
            counts.official_values_differ += changed.height
            for frame, label in ((only_season, "season only"), (only_before, "before only")):
                for game_id, team, qb_id in frame.select(key).rows()[:2]:
                    if len(counts.examples) < MAX_IDENTITY_EXAMPLES:
                        counts.examples.append(
                            f"{season} before week {week}: {qb_id} ({team}, {game_id}), {label}"
                        )
    return counts


@dataclass
class Totals:
    """Headline counts over a set of compared tables."""

    tables: int
    rows: int
    cells: int
    identical: int
    within_tolerance: int
    outside_tolerance: int
    structural: int
    max_abs_difference: float
    columns_within_tolerance: list[str]

    @classmethod
    def of(cls, results: list[TableResult]) -> Totals:
        """Return the totals of ``results``."""
        return cls(
            tables=len(results),
            rows=sum(r.rows for r in results),
            cells=sum(r.cells for r in results),
            identical=sum(r.identical for r in results),
            within_tolerance=sum(r.within_tolerance for r in results),
            outside_tolerance=sum(r.outside_tolerance for r in results),
            structural=sum(1 for r in results if r.problems),
            max_abs_difference=max((r.max_abs_difference for r in results), default=0.0),
            columns_within_tolerance=sorted(
                {c for r in results for c in r.columns_within_tolerance}
            ),
        )


def _by_table(results: list[TableResult]) -> dict[str, Totals]:
    """Return the totals of each table name, in the order the names first appear."""
    names = list(dict.fromkeys(r.table for r in results))
    return {name: Totals.of([r for r in results if r.table == name]) for name in names}


def _span(results: list[TableResult], table: str) -> str:
    seasons = sorted({r.season for r in results if r.table == table})
    return f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])


@dataclass
class Report:
    """Everything one run found, for the JSON and Markdown reports."""

    generated: str
    source_path: str
    port_head: str
    port_files: dict[str, str]
    environment: str
    threads: str
    seasons: list[int]
    row_results: list[TableResult]
    table_results: list[TableResult]
    identity: IdentityCounts
    adapter_seconds: float
    rebuild_seconds: float
    snapshot_files: dict[str, str]
    files_read: dict[str, str]

    @property
    def passed(self) -> bool:
        """Return whether every row set and every table matched within the tolerance."""
        return all(r.passed for r in [*self.row_results, *self.table_results])

    def differences(self) -> list[str]:
        """Return every structural problem and every cell outside the tolerance."""
        found = []
        for result in [*self.row_results, *self.table_results]:
            found += [f"{result.season} {result.table}: {p}" for p in result.problems]
            found += [f"{result.season} {result.table}: {d}" for d in result.differences]
        return found

    def to_json(self) -> dict[str, object]:
        """Return the JSON record."""
        return {
            "generated": self.generated,
            "passed": self.passed,
            "source_path": self.source_path,
            "source_head": SOURCE_COMMIT,
            "port_head": self.port_head,
            "port_files": self.port_files,
            "environment": self.environment,
            "threads": self.threads,
            "seasons": self.seasons,
            "float_tolerance": FLOAT_TOLERANCE,
            "rounded_tolerance": ROUNDED_TOLERANCE,
            "rows": {
                "totals": asdict(Totals.of(self.row_results)),
                "by_table": {k: asdict(v) for k, v in _by_table(self.row_results).items()},
                "seasons": [asdict(r) for r in self.row_results],
            },
            "tables": {
                "totals": asdict(Totals.of(self.table_results)),
                "by_table": {k: asdict(v) for k, v in _by_table(self.table_results).items()},
                "seasons": [asdict(r) for r in self.table_results],
            },
            "differences": self.differences(),
            "identity": asdict(self.identity),
            "adapter_seconds": self.adapter_seconds,
            "rebuild_seconds": self.rebuild_seconds,
            "snapshot_files": self.snapshot_files,
            "files_read": self.files_read,
        }

    def markdown(self) -> str:
        """Return the Markdown report."""
        span = f"{self.seasons[0]}-{self.seasons[-1]}"
        rows, tables = Totals.of(self.row_results), Totals.of(self.table_results)
        lines = [
            "# Ratings adapter parity with `nfl-sos-ratings`",
            "",
            *_wrap(
                f"Generated by `.agents/sos_port/adapter_parity.py` on {self.generated}; do not "
                "edit by hand. `adapter_parity.json` beside it holds every number below, each "
                "table's result, and the SHA-256 of every file read."
            ),
            "",
            f"- **Result: {'PASSED' if self.passed else 'FAILED'}.**",
            *_wrap(
                f"Source: the `nfl-sos-ratings` checkout at `{self.source_path}`, HEAD "
                f"`{SOURCE_COMMIT}`, asserted to be the frozen commit; its published game logs "
                "and ratings files were read without writing there.",
                "- ",
            ),
            *_wrap(
                f"Inputs: the snapshot of the nflreadpy cache that project read "
                f"(`~/scratch/{snapshot_cache.SNAPSHOT.name}/`, every file checked against "
                f"`{snapshot_cache.SNAPSHOT_MANIFEST.name}`), prepared as this repository's "
                "loaders prepare a download, written to a temporary cache directory, and read "
                "back through `loaders.load_pbp`, `loaders.load_schedule` and `nflverse_sources` "
                "with downloads blocked.",
                "- ",
            ),
            *_wrap(
                f"Adapter: `nfl_predictor.ratings.pbp_inputs` at this repository's commit "
                f"`{self.port_head}`. Seasons compared: {span}.",
                "- ",
            ),
            *_wrap(
                "Tolerance: the ratings parity audit's. Row keys, their order, the columns and "
                "types must match; integers, text and flags must be equal; floats must agree "
                f"within {FLOAT_TOLERANCE:g} absolute (`SRS`, rounded to six decimals, within "
                f"{ROUNDED_TOLERANCE:g}). Bit-identical cells are counted separately.",
                "- ",
            ),
            *_wrap(
                "Team codes: the published files write Washington as `WAS`, this repository "
                "`WSH`; published team columns are mapped to this repository's codes before the "
                "comparison, which moves no row.",
                "- ",
            ),
            *_wrap(f"Environment: {self.environment}; {self.threads}.", "- "),
            "",
            "## Headline",
            "",
            "| measure | input rows | ratings tables |",
            "| --- | ---: | ---: |",
        ]
        for label, name in (
            ("tables compared", "tables"),
            ("rows compared", "rows"),
            ("cells compared", "cells"),
            ("cells bit-identical", "identical"),
            ("cells within tolerance, not bit-identical", "within_tolerance"),
            ("cells outside tolerance", "outside_tolerance"),
            ("tables with a structural difference", "structural"),
        ):
            lines.append(
                f"| {label} | {_fmt(getattr(rows, name))} | {_fmt(getattr(tables, name))} |"
            )
        lines += [
            "",
            "## By table",
            "",
            *_wrap(
                "The input rows are the columns the fits consume (`TEAM_GAME_SCHEMA` and "
                "`QB_GAME_SCHEMA` of `nfl_predictor.ratings.inputs`); the published side is each "
                "season's game logs read through `nfl_predictor.ratings.published_logs`. The "
                "ratings tables are every published season table rebuilt from the adapter's rows "
                "(`combined` on its rating columns, `qb_combined` on the passer columns the port "
                "makes)."
            ),
            "",
            (
                "| table | seasons | rows | cells | bit-identical | within tolerance | outside | "
                "max abs difference |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        results = [*self.row_results, *self.table_results]
        for name, totals in _by_table(results).items():
            lines.append(
                f"| `{name}` | {_span(results, name)} | {_fmt(totals.rows)} | "
                f"{_fmt(totals.cells)} | {_fmt(totals.identical)} | "
                f"{_fmt(totals.within_tolerance)} | {_fmt(totals.outside_tolerance)} | "
                f"{_fmt(totals.max_abs_difference)} |"
            )
        within = Totals.of(results).columns_within_tolerance
        lines += ["", "## Cells within tolerance but not bit-identical", ""]
        lines += _wrap(
            "Columns: " + (", ".join(f"`{c}`" for c in within) if within else "none") + "."
        )
        lines += ["", "## Differences", ""]
        differences = self.differences()
        lines += [f"- {d}" for d in differences] if differences else ["None."]
        lines += ["", "## The quarterback identity known before a week", ""]
        lines += _wrap(
            "Reported, not judged. For every season and every week from 2 to the week after the "
            "last, the passer-games before the week are built twice: with the season's identity "
            "(every roster week, as the published files) and with the identity known before the "
            "week (the players file and the roster weeks before it), which the pre-week features "
            "use."
        )
        identity = self.identity
        lines += [
            "",
            "| measure | count |",
            "| --- | ---: |",
            f"| season-weeks | {_fmt(identity.weeks)} |",
            (
                "| passer-games before those weeks, season identity (summed over the weeks) | "
                f"{_fmt(identity.passer_games_seen)} |"
            ),
            f"| kept by the season identity only | {_fmt(identity.kept_only_by_season)} |",
            (
                "| kept by the identity before the week only | "
                f"{_fmt(identity.kept_only_before_week)} |"
            ),
            (
                "| kept by both, with other attempts or dropback EPA | "
                f"{_fmt(identity.official_values_differ)} |"
            ),
            "",
        ]
        if identity.examples:
            lines += ["Examples:", "", *(f"- {example}" for example in identity.examples), ""]
        lines += ["## Timings", ""]
        lines += _wrap(
            f"Wall-clock seconds with {self.threads}, on a machine shared with other work: the "
            f"adapter's rows for every season built ({FIRST_SEASON}-{self.seasons[-1]}, after "
            f"loading) {self.adapter_seconds:.1f}; rebuilding every compared season's tables from "
            f"those rows, rank ranges and pairs included, {self.rebuild_seconds:.1f}."
        )
        lines += ["", "## Files read", ""]
        lines += _wrap(
            f"{len(self.files_read)} published files and {len(self.snapshot_files)} snapshot "
            "files; their SHA-256 are in `adapter_parity.json`, with those of the code that ran "
            f"(`{len(self.port_files)}` files: the ratings package and the loaders it read "
            "through)."
        )
        lines.append("")
        return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return "0" if value == 0.0 else f"{value:.2g}"
    return str(value)


def _wrap(text: str, prefix: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width=100,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the check and write the reports."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--seasons", type=int, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    check_source(arguments.source)
    compared = sorted(arguments.seasons or range(FIRST_SEASON, LAST_SEASON + 1))
    built = list(range(FIRST_SEASON, max(compared) + 1))
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    published = Published(arguments.source)
    with tempfile.TemporaryDirectory(prefix="adapter_parity_") as directory:
        cache = Path(directory)
        snapshot_cache.write_cache(files, cache, built)
        rows = load_rows(cache, built)
    row_results = compare_rows(rows, published, compared)
    table_results, rebuild_seconds = compare_tables(rows, published, compared)
    root = repo_root()
    report = Report(
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        source_path=str(arguments.source),
        port_head=source_head(root),
        port_files={
            **package_hashes(root, "src/nfl_predictor/ratings"),
            **{
                path: sha256_file(root / path)
                for path in (
                    "src/nfl_predictor/utils/polars/loaders.py",
                    "src/nfl_predictor/utils/polars/nflverse_sources.py",
                    "src/nfl_predictor/utils/polars/player_identity.py",
                    "src/nfl_predictor/utils/polars/player_team_repair.py",
                )
            },
        },
        environment=_environment(),
        threads=", ".join(f"`{name}={os.environ.get(name)}`" for name in THREAD_VARIABLES),
        seasons=compared,
        row_results=row_results,
        table_results=table_results,
        identity=identity_counts(rows, compared),
        adapter_seconds=sum(rows[season].seconds for season in built),
        rebuild_seconds=rebuild_seconds,
        snapshot_files=files.hashes,
        files_read=published.hashes,
    )
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    (arguments.out_dir / OUT_JSON).write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n"
    )
    (arguments.out_dir / OUT_MD).write_text(report.markdown())
    status = "PASSED" if report.passed else "FAILED"
    sys.stdout.write(f"{status}: {arguments.out_dir / OUT_MD}\n")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
