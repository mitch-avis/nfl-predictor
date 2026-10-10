"""Compare the ratings port with every ratings file ``nfl-sos-ratings`` published, cell by cell.

Run from the repository root (single-threaded, several minutes):

    .venv/bin/python .agents/sos_port/ratings_parity.py [--source PATH] [--seasons S ...]
        [--out-dir DIR]

``--source`` is the ``nfl-sos-ratings`` checkout (default: the one beside this repository's main
checkout); ``--seasons`` limits the comparison to some seasons (earlier ones are still read for
the penalties and the prior); ``--out-dir`` writes the reports elsewhere (default: beside this
script, where the checked-in report of the full run lives). The script runs with one BLAS and
Polars thread unless the environment sets the counts. It asserts the checkout is at the frozen
commit, reads its published Parquet files without writing there (recording each file's SHA-256),
and:

1. rebuilds every season's ratings tables from that project's published game logs with
   ``nfl_predictor.ratings``: the files its pipeline writes, rank ranges included (each season's
   ranges and pairs, and the weekly ranges of the season in progress), timing each season;
2. compares each rebuilt table with the published file cell by cell: the same columns, types and
   row keys in the same order, integers, text and flags equal, and floats equal within
   ``FLOAT_TOLERANCE``, counting the cells that are bit-identical;
3. runs the pre-week entry point for every week of every season and compares its team ratings with
   the published history of the week before (required to match), and its quarterback ratings with
   the published quarterback history, which uses the season's own penalty (reported, since the
   pre-week fit uses the previous season's), timing the pass.

It writes ``RATINGS_PARITY.md`` and ``ratings_parity.json`` beside itself and exits 1 if any
cell is outside the tolerance or any table differs in shape, type, or row keys.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ratings_source import (  # a sibling module, importable once its dir is on the path
    SOURCE_COMMIT,
    check_source,
    default_source,
    published_file,
    repo_root,
    sha256_file,
    source_head,
    uncommitted_changes,
)

from nfl_predictor.ratings.pre_week import RatingsHistory
from nfl_predictor.ratings.published_logs import (
    qb_games_from_published,
    team_games_from_published,
)
from nfl_predictor.ratings.qb_rating import fit_qb_ratings
from nfl_predictor.ratings.season import build_qb_season, build_team_season
from nfl_predictor.ratings.team_prior import PriorHistory

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "RATINGS_PARITY.md"
OUT_JSON = HERE / "ratings_parity.json"
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
FIRST_SEASON = 1999
# Floats agree within this absolute difference; the report also counts bit-identical cells.
FLOAT_TOLERANCE = 1e-12
MAX_LISTED_DIFFERENCES = 20

# Each published table and its row keys. The two wide tables are compared on the columns the
# ratings make: the team ratings in ``combined`` and the passer table in ``qb_combined``.
TEAM_TABLES: dict[str, tuple[str, ...]] = {
    "ratings": ("team",),
    "team_prior": ("excluded_team", "team"),
    "ratings_by_week": ("team", "week"),
    "rating_ranges": ("team",),
    "rating_pairs": ("team", "other_team"),
    "rating_ranges_by_week": ("team", "week"),
    "combined": ("team",),
}
QB_TABLES: dict[str, tuple[str, ...]] = {
    "qb_ratings": ("qb_id",),
    "qb_ratings_by_week": ("qb_id", "week"),
    "qb_rating_ranges": ("qb_id",),
    "qb_rating_pairs": ("qb_id", "other_qb_id"),
    "qb_rating_ranges_by_week": ("qb_id", "week"),
    "qb_combined": ("qb_id",),
}
BOOTSTRAP_TABLES = frozenset(
    {
        "rating_ranges",
        "rating_pairs",
        "rating_ranges_by_week",
        "qb_rating_ranges",
        "qb_rating_pairs",
        "qb_rating_ranges_by_week",
    }
)
IN_PROGRESS_ONLY = frozenset({"rating_ranges_by_week", "qb_rating_ranges_by_week"})
COMBINED_RATING_COLUMNS = (
    "team",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "team_rating",
    "sos",
    "SRS",
)


@dataclass
class TableResult:
    """How one rebuilt table compares with its published file."""

    season: int
    table: str
    rows: int
    columns: int
    cells: int = 0
    identical: int = 0
    within_tolerance: int = 0
    outside_tolerance: int = 0
    max_abs_difference: float = 0.0
    columns_within_tolerance: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether the table matches within the tolerance with no structural problem."""
        return not self.problems and self.outside_tolerance == 0


@dataclass
class SeasonTiming:
    """Seconds to rebuild one season's team and quarterback tables."""

    season: int
    team_seconds: float
    qb_seconds: float
    weekly_ranges: bool


@dataclass
class PreWeekSummary:
    """The pre-week entry point against the published histories, over every season and week."""

    seconds: float = 0.0
    calls: int = 0
    team_weeks_compared: int = 0
    team_cells: int = 0
    team_identical: int = 0
    team_within_tolerance: int = 0
    team_outside_tolerance: int = 0
    team_problems: list[str] = field(default_factory=list)
    qb_weeks_compared: int = 0
    qb_passer_weeks: int = 0
    qb_volume_mismatches: int = 0
    qb_set_mismatches: int = 0
    qb_identical_passer_weeks: int = 0
    qb_mean_abs_difference: float = 0.0
    qb_max_abs_difference: float = 0.0
    qb_max_difference_at: str = ""
    qb_same_penalty_seasons: list[int] = field(default_factory=list)
    qb_other_penalty_seasons: list[int] = field(default_factory=list)
    qb_mean_abs_difference_other_penalty: float = 0.0
    qb_penalties: dict[int, dict[str, float | None]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether the team part matched and the passer sets and volumes agreed."""
        return (
            self.team_outside_tolerance == 0
            and not self.team_problems
            and self.qb_set_mismatches == 0
            and self.qb_volume_mismatches == 0
        )


@dataclass
class InputChecks:
    """Facts about the published game logs that the port's input rows rely on (reported only)."""

    passer_games_with_dropbacks: int = 0
    rate_rebuilt_bit_for_bit: int = 0
    passer_games_without_dropbacks: int = 0
    dropback_epa_without_dropbacks: int = 0
    passer_ids_with_two_names: int = 0
    fit_order_ties: int = 0
    team_games: int = 0
    one_sided_games: int = 0


class Source:
    """The published files, read once each, with the SHA-256 of every file read."""

    def __init__(self, root: Path) -> None:
        """Remember the checkout root."""
        self.root = root
        self.hashes: dict[str, str] = {}
        self._frames: dict[tuple[int, str], pl.DataFrame] = {}

    def read(self, season: int, name: str) -> pl.DataFrame:
        """Return one published file, recording its hash the first time."""
        if (season, name) not in self._frames:
            path = published_file(self.root, season, name)
            self.hashes[f"data/{path.name}"] = sha256_file(path)
            self._frames[(season, name)] = pl.read_parquet(path)
        return self._frames[(season, name)]

    def exists(self, season: int, name: str) -> bool:
        """Return whether a published file exists."""
        return published_file(self.root, season, name).exists()

    def team_games(self, season: int) -> pl.DataFrame:
        """Return one season's team-game rows from its published game logs."""
        return team_games_from_published(self.read(season, "team_game_logs"))

    def qb_games(self, season: int) -> pl.DataFrame:
        """Return one season's quarterback-game rows from its published game logs."""
        return qb_games_from_published(self.read(season, "qb_game_logs"))


def _single_threaded() -> None:
    """Re-run this script with one BLAS and Polars thread unless the caller set the counts.

    ``nfl-sos-ratings`` builds its files single-threaded, and both libraries read the variables
    when they load, so the script restarts itself rather than setting them late.
    """
    missing = {name: "1" for name in THREAD_VARIABLES if name not in os.environ}
    if missing:
        os.execve(  # noqa: S606  # restarts this same interpreter and script, no shell
            sys.executable, [sys.executable, *sys.argv], {**os.environ, **missing}
        )


def _float_flags(ours: pl.Series, published: pl.Series) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Return per-cell flags (bit-identical, within tolerance) and the absolute differences."""
    both_null = ours.is_null() & published.is_null()
    both_nan = (ours.is_nan() & published.is_nan()).fill_null(value=False)
    identical = (ours == published).fill_null(value=False) | both_null | both_nan
    difference = (ours - published).abs()
    within = (difference <= FLOAT_TOLERANCE).fill_null(value=False) & ~identical
    return identical, within, difference.fill_null(0.0).fill_nan(0.0)


def _list_flags(ours: pl.Series, published: pl.Series) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Compare list cells element by element: a cell is as good as its worst element."""
    same_length = (ours.list.len() == published.list.len()).fill_null(value=False)
    pairs = (
        pl.DataFrame({"ours": ours, "published": published})
        .with_row_index("cell")
        .filter(same_length)
        .explode(["ours", "published"])
    )
    identical, within, difference = _float_flags(
        pairs.get_column("ours").cast(pl.Float64), pairs.get_column("published").cast(pl.Float64)
    )
    per_cell = (
        pairs.select("cell")
        .with_columns(identical=identical, within=within | identical, difference=difference)
        .group_by("cell")
        .agg(pl.col("identical").all(), pl.col("within").all(), pl.col("difference").max())
    )
    cells = pl.DataFrame({"cell": range(ours.len())}, schema={"cell": pl.UInt32}).join(
        per_cell, on="cell", how="left", maintain_order="left"
    )
    cell_identical = cells.get_column("identical").fill_null(value=False) & same_length
    cell_within = cells.get_column("within").fill_null(value=False) & same_length & ~cell_identical
    return cell_identical, cell_within, cells.get_column("difference").fill_null(0.0)


def _flags(ours: pl.Series, published: pl.Series) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Return one column's per-cell flags (bit-identical, within tolerance) and differences."""
    if ours.dtype.is_float():
        return _float_flags(ours, published)
    if isinstance(ours.dtype, pl.List):
        return _list_flags(ours, published)
    identical = (ours == published).fill_null(value=False) | (ours.is_null() & published.is_null())
    zeros = pl.Series([0.0] * ours.len(), dtype=pl.Float64)
    return identical, pl.Series([False] * ours.len(), dtype=pl.Boolean), zeros


def compare_table(
    season: int, table: str, ours: pl.DataFrame, published: pl.DataFrame, keys: tuple[str, ...]
) -> TableResult:
    """Compare one rebuilt table with its published file, cell by cell.

    Columns, row count, and row keys must match before cells are compared; a type difference is
    recorded as a problem and the cells are still compared.
    """
    result = TableResult(season, table, published.height, published.width)
    if ours.columns != published.columns:
        result.problems.append(
            f"columns differ: ours {ours.columns}, published {published.columns}"
        )
        return result
    if ours.height != published.height:
        result.problems.append(
            f"row counts differ: ours {ours.height}, published {published.height}"
        )
        return result
    if not ours.select(keys).equals(published.select(keys)):
        result.problems.append(f"row keys or their order differ ({', '.join(keys)})")
        return result
    result.problems.extend(
        f"{name} type differs: ours {ours.schema[name]}, published {published.schema[name]}"
        for name in ours.columns
        if ours.schema[name] != published.schema[name]
    )
    for name in ours.columns:
        column, published_column = ours.get_column(name), published.get_column(name)
        identical, within, difference = _flags(column, published_column)
        outside = ~identical & ~within
        result.cells += column.len()
        result.identical += int(identical.sum())
        result.within_tolerance += int(within.sum())
        result.outside_tolerance += int(outside.sum())
        result.max_abs_difference = max(result.max_abs_difference, float(difference.max() or 0.0))
        if within.any():
            result.columns_within_tolerance.append(name)
        for index in outside.arg_true().to_list()[:MAX_LISTED_DIFFERENCES]:
            key = ", ".join(f"{k}={published.get_column(k)[index]}" for k in keys)
            result.differences.append(
                f"{key}: {name} ours {column[index]!r}, published {published_column[index]!r}"
            )
    return result


def _seasons(source: Source) -> tuple[list[int], int | None]:
    """Return every published season in order and the season in progress (weekly ranges)."""
    seasons = []
    season = FIRST_SEASON
    while source.exists(season, "team_game_logs"):
        seasons.append(season)
        season += 1
    in_progress = [s for s in seasons if source.exists(s, "rating_ranges_by_week")]
    return seasons, (in_progress[-1] if in_progress else None)


def input_checks(source: Source, seasons: list[int]) -> InputChecks:
    """Count what the published game logs show about the port's input rows.

    - The per-game quarterback rate the fits read is the passing EPA less the spikes' EPA over
      the dropbacks, bit for bit, so the input can carry the EPA sum instead of the rate.
    - Passer-games without a dropback carry no dropback EPA, so the season rate is the same
      whether summed over every game or over the games with a dropback.
    - Each passer id has one name in a season, so the season summary may group by id alone.
    - No two passer-game rows tie on team, week, game, and name, the order the fits read them in.
    - Every game has both teams' rows, which the preseason prior's centering needs.
    """
    checks = InputChecks()
    dropback_epa = pl.col("qb_passing_epa") - pl.col("qb_spike_epa")
    for season in seasons:
        logs = source.read(season, "qb_game_logs")
        rated = logs.filter(pl.col("qb_dropbacks") > 0)
        rebuilt = dropback_epa / pl.col("qb_dropbacks")
        checks.passer_games_with_dropbacks += rated.height
        checks.rate_rebuilt_bit_for_bit += rated.filter(
            rebuilt == pl.col("qb_epa_per_dropback")
        ).height
        without = logs.filter(pl.col("qb_dropbacks") == 0)
        checks.passer_games_without_dropbacks += without.height
        checks.dropback_epa_without_dropbacks += without.filter(dropback_epa != 0.0).height
        checks.passer_ids_with_two_names += (
            logs.group_by("qb_id").agg(pl.col("qb_name").n_unique()).filter(pl.col("qb_name") > 1)
        ).height
        checks.fit_order_ties += (
            logs.group_by(["team", "week", "game_id", "qb_name"]).len().filter(pl.col("len") > 1)
        ).height
        games = source.read(season, "team_game_logs").group_by("game_id").len()
        checks.team_games += games.height
        checks.one_sided_games += games.filter(pl.col("len") != 2).height  # noqa: PLR2004  # a game's two teams
    return checks


def _rebuilt_tables(
    season: int, source: Source, history: PriorHistory, *, weekly: bool
) -> tuple[dict[str, pl.DataFrame | None], SeasonTiming]:
    """Rebuild one season's tables, timing the team and quarterback parts."""
    team_games, qb_games = source.team_games(season), source.qb_games(season)
    start = time.perf_counter()
    team = build_team_season(season, team_games, history, weekly_ranges=weekly)
    middle = time.perf_counter()
    qb = build_qb_season(qb_games, team_games, weekly_ranges=weekly)
    end = time.perf_counter()
    tables: dict[str, pl.DataFrame | None] = {
        "ratings": team.ratings,
        "team_prior": team.team_prior,
        "ratings_by_week": team.ratings_by_week,
        "rating_ranges": team.rating_ranges,
        "rating_pairs": team.rating_pairs,
        "rating_ranges_by_week": team.rating_ranges_by_week,
        "combined": team.ratings.select(COMBINED_RATING_COLUMNS).sort("team"),
        "qb_ratings": qb.qb_ratings,
        "qb_ratings_by_week": qb.qb_ratings_by_week,
        "qb_rating_ranges": qb.qb_rating_ranges,
        "qb_rating_pairs": qb.qb_rating_pairs,
        "qb_rating_ranges_by_week": qb.qb_rating_ranges_by_week,
        "qb_combined": qb.qb_season,
    }
    return tables, SeasonTiming(season, middle - start, end - middle, weekly)


def rebuild_and_compare(
    source: Source, seasons: list[int], in_progress: int | None
) -> tuple[list[TableResult], list[SeasonTiming]]:
    """Rebuild every season's tables in order, as that project's pipeline runs, and compare."""
    history = PriorHistory(source.team_games)
    results: list[TableResult] = []
    timings: list[SeasonTiming] = []
    for season in seasons:
        weekly = season == in_progress
        ours, timing = _rebuilt_tables(season, source, history, weekly=weekly)
        timings.append(timing)
        for table, keys in {**TEAM_TABLES, **QB_TABLES}.items():
            if table in IN_PROGRESS_ONLY and not weekly:
                continue
            published = source.read(season, table)
            frame = ours[table]
            if frame is None:
                result = TableResult(season, table, published.height, published.width)
                result.problems.append("not rebuilt")
            else:
                if table in {"combined", "qb_combined"}:
                    published = published.select(frame.columns)
                result = compare_table(season, table, frame, published, keys)
            results.append(result)
        season_ok = all(result.passed for result in results if result.season == season)
        seconds = timing.team_seconds + timing.qb_seconds
        status = "ok" if season_ok else "DIFFERS"
        sys.stdout.write(f"{season}: rebuilt in {seconds:4.1f} s, {status}\n")
        sys.stdout.flush()
    return results, timings


def _compare_pre_week_teams(
    summary: PreWeekSummary, season: int, week: int, teams: pl.DataFrame, published: pl.DataFrame
) -> None:
    """Add one week's team comparison: the pre-week rows against the history of the week before."""
    before = published.filter(pl.col("week") == week - 1).drop("week")
    ours = teams.join(before.select("team"), on="team", how="semi", maintain_order="left")
    result = compare_table(
        season, f"pre-week teams, week {week}", ours.select(before.columns), before, ("team",)
    )
    summary.team_weeks_compared += 1
    summary.team_cells += result.cells
    summary.team_identical += result.identical
    summary.team_within_tolerance += result.within_tolerance
    summary.team_outside_tolerance += result.outside_tolerance
    summary.team_problems.extend(f"{season} week {week}: {p}" for p in result.problems)
    summary.team_problems.extend(f"{season} week {week}: {d}" for d in result.differences[:3])


def _compare_pre_week_quarterbacks(
    summary: PreWeekSummary,
    label: str,
    quarterbacks: pl.DataFrame,
    published: pl.DataFrame,
    week: int,
) -> list[float]:
    """Add one week's passer comparison and return its absolute rating differences."""
    before = published.filter(pl.col("week") == week - 1).drop("week")
    joined = quarterbacks.join(before, on="qb_id", how="full", suffix="_published", coalesce=True)
    rating, published_rating = "adj_qb_epa_per_dropback", "adj_qb_epa_per_dropback_published"
    summary.qb_weeks_compared += 1
    summary.qb_set_mismatches += joined.filter(
        pl.col(rating).is_null() | pl.col(published_rating).is_null()
    ).height
    matched = joined.drop_nulls([rating, published_rating])
    summary.qb_passer_weeks += matched.height
    summary.qb_volume_mismatches += matched.filter(
        (pl.col("qb_games_played") != pl.col("qb_games_played_published"))
        | (pl.col("qb_dropbacks") != pl.col("qb_dropbacks_published"))
    ).height
    gaps = (matched.get_column(rating) - matched.get_column(published_rating)).abs()
    summary.qb_identical_passer_weeks += int((gaps == 0.0).sum())
    largest = float(gaps.max() or 0.0) if gaps.len() else 0.0
    if largest > summary.qb_max_abs_difference:
        summary.qb_max_abs_difference = largest
        passer = matched.get_column("qb_id")[int(gaps.arg_max() or 0)]
        summary.qb_max_difference_at = f"{label}, {passer}"
    return gaps.to_list()


def pre_week_pass(source: Source, seasons: list[int]) -> PreWeekSummary:
    """Run the pre-week entry point for every week and compare it with the published histories."""
    history = RatingsHistory(source.team_games, source.qb_games)
    summary = PreWeekSummary()
    gaps: list[float] = []
    other_penalty_gaps: list[float] = []
    start = time.perf_counter()
    for season in seasons:
        team_games = source.team_games(season)
        teams = sorted(set(team_games.get_column("team").to_list()))
        last_week = int(team_games.get_column("week").max() or 0)
        own = fit_qb_ratings(source.qb_games(season)).ridge_lambda
        previous = history.qb_penalty(season)
        summary.qb_penalties[season] = {"previous_season": previous, "own_season": own}
        if previous is not None and previous == own:
            summary.qb_same_penalty_seasons.append(season)
        elif previous is not None:
            summary.qb_other_penalty_seasons.append(season)
        for week in range(1, last_week + 2):
            ratings = history.pre_week(season, week, teams)
            summary.calls += 1
            if season == FIRST_SEASON or week == 1:
                continue
            _compare_pre_week_teams(
                summary, season, week, ratings.teams, source.read(season, "ratings_by_week")
            )
            week_gaps = _compare_pre_week_quarterbacks(
                summary,
                f"{season} week {week}",
                ratings.quarterbacks,
                source.read(season, "qb_ratings_by_week"),
                week,
            )
            gaps.extend(week_gaps)
            if previous is not None and previous != own:
                other_penalty_gaps.extend(week_gaps)
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"pre-week {season}: {elapsed:6.1f} s so far\n")
        sys.stdout.flush()
    summary.seconds = time.perf_counter() - start
    summary.qb_mean_abs_difference = statistics.fmean(gaps) if gaps else 0.0
    summary.qb_mean_abs_difference_other_penalty = (
        statistics.fmean(other_penalty_gaps) if other_penalty_gaps else 0.0
    )
    return summary


def _season_span(seasons: list[int]) -> str:
    """Return ``1999-2026`` for consecutive seasons, else the seasons listed."""
    if seasons == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}-{seasons[-1]}"
    return ", ".join(str(season) for season in seasons)


def _thread_settings() -> str:
    """Return the thread counts this run used, as the environment set them."""
    return ", ".join(f"`{name}={os.environ.get(name)}`" for name in THREAD_VARIABLES)


def _wrap(text: str) -> list[str]:
    """Return ``text`` as one Markdown paragraph wrapped at 100 characters."""
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def _item(text: str) -> list[str]:
    """Return ``text`` as one Markdown list item wrapped at 100 characters."""
    return textwrap.wrap(
        text,
        width=100,
        initial_indent="- ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _fmt(value: object) -> str:
    """Return a number for the Markdown tables: integers with separators, floats in short form."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return "0" if value == 0.0 else f"{value:.2g}"
    return str(value)


@dataclass
class Report:
    """Everything one run found, for the JSON and Markdown reports."""

    generated: str
    source_path: str
    source_head: str
    port_head: str
    port_uncommitted: list[str]
    seasons: list[int]
    in_progress: int | None
    in_progress_through_week: int | None
    results: list[TableResult]
    timings: list[SeasonTiming]
    pre_week: PreWeekSummary
    inputs: InputChecks
    load_start: tuple[float, ...]
    load_end: tuple[float, ...]
    files_read: dict[str, str]

    @property
    def passed(self) -> bool:
        """Return whether every table and the pre-week team comparison passed."""
        return all(result.passed for result in self.results) and self.pre_week.passed

    def totals(self) -> dict[str, int]:
        """Return the headline counts over every compared table."""
        return {
            "tables": len(self.results),
            "cells": sum(result.cells for result in self.results),
            "identical": sum(result.identical for result in self.results),
            "within_tolerance": sum(result.within_tolerance for result in self.results),
            "outside_tolerance": sum(result.outside_tolerance for result in self.results),
            "tables_with_structural_differences": sum(1 for r in self.results if r.problems),
        }

    def table_rows(self) -> list[dict[str, object]]:
        """Return one summary row per table, over every season compared."""
        rows: list[dict[str, object]] = []
        for table in (*TEAM_TABLES, *QB_TABLES):
            subset = [result for result in self.results if result.table == table]
            if not subset:
                continue
            rows.append(
                {
                    "table": table,
                    "seasons": _season_span([result.season for result in subset]),
                    "files": len(subset),
                    "rows": sum(result.rows for result in subset),
                    "cells": sum(result.cells for result in subset),
                    "identical": sum(result.identical for result in subset),
                    "within_tolerance": sum(result.within_tolerance for result in subset),
                    "outside_tolerance": sum(result.outside_tolerance for result in subset),
                    "structural_differences": sum(len(result.problems) for result in subset),
                    "max_abs_difference": max(result.max_abs_difference for result in subset),
                    "columns_within_tolerance": sorted(
                        {name for result in subset for name in result.columns_within_tolerance}
                    ),
                }
            )
        return rows

    def timing_totals(self) -> dict[str, float]:
        """Return the rebuild and pre-week seconds."""
        return {
            "rebuild_seconds": sum(t.team_seconds + t.qb_seconds for t in self.timings),
            "team_seconds": sum(t.team_seconds for t in self.timings),
            "qb_seconds": sum(t.qb_seconds for t in self.timings),
            "pre_week_seconds": self.pre_week.seconds,
        }

    def to_json(self) -> str:
        """Return the JSON report."""
        payload = {
            "result": "passed" if self.passed else "failed",
            "generated": self.generated,
            "source": {
                "repository": "nfl-sos-ratings",
                "path": self.source_path,
                "head": self.source_head,
                "frozen_commit": SOURCE_COMMIT,
            },
            "port": {"head": self.port_head, "uncommitted": self.port_uncommitted},
            "seasons": self.seasons,
            "season_in_progress": self.in_progress,
            "season_in_progress_through_week": self.in_progress_through_week,
            "float_tolerance": FLOAT_TOLERANCE,
            "totals": self.totals(),
            "tables": self.table_rows(),
            "timings": self.timing_totals()
            | {
                "load_average_start": [round(value, 2) for value in self.load_start],
                "load_average_end": [round(value, 2) for value in self.load_end],
                "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
                "seasons": [asdict(timing) for timing in self.timings],
            },
            "pre_week": asdict(self.pre_week) | {"passed": self.pre_week.passed},
            "input_checks": asdict(self.inputs),
            "results": [asdict(result) | {"passed": result.passed} for result in self.results],
            "files_read": self.files_read,
        }
        return json.dumps(payload, indent=2) + "\n"

    def to_markdown(self) -> str:
        """Return the Markdown report."""
        lines = [
            *self._header_lines(),
            *self._table_lines(),
            *self._input_lines(),
            *self._within_tolerance_lines(),
            *self._bootstrap_lines(),
            *self._pre_week_lines(),
            *self._timing_lines(),
            "",
            "## Files read",
            "",
            *_wrap(
                f"{len(self.files_read)} published files, each read once; their SHA-256 are in "
                "`ratings_parity.json` (`files_read`)."
            ),
            *self._difference_lines(),
        ]
        return "\n".join(lines) + "\n"

    def _header_lines(self) -> list[str]:
        """Return the title, provenance, tolerance, and headline counts."""
        totals = self.totals()
        uncommitted = (
            ", with uncommitted changes to the package's Python files: "
            f"{', '.join(f'`{line}`' for line in self.port_uncommitted)}."
            if self.port_uncommitted
            else ", with no uncommitted change to the package's Python files."
        )
        return [
            "# Ratings parity with `nfl-sos-ratings`",
            "",
            *_wrap(
                f"Generated by `.agents/sos_port/ratings_parity.py` on {self.generated}; do not "
                "edit by hand. `ratings_parity.json` beside it holds every number below, each "
                "table's result, and the SHA-256 of every file read."
            ),
            "",
            f"- **Result: {'PASSED' if self.passed else 'FAILED'}.**",
            *_item(
                f"Source: the `nfl-sos-ratings` checkout at `{self.source_path}`, HEAD "
                f"`{self.source_head}`, asserted to be the frozen commit before any file was "
                "read. Its published files were built from code identical to that commit for "
                "every output."
            ),
            *_item(
                f"Port: `nfl_predictor.ratings` at this repository's commit `{self.port_head}`"
                + uncommitted
            ),
            *_item(
                f"Seasons: {_season_span(self.seasons)}; {self.in_progress} is the season in "
                f"progress (games through week {self.in_progress_through_week}), the only one "
                "with weekly rank-range files."
            ),
            *_item(
                "Inputs: each season's published `team_game_logs` and `qb_game_logs`, through "
                "`nfl_predictor.ratings.published_logs`."
            ),
            *_item(
                "Tolerance: a table must have the published columns, types, and row keys in the "
                "published order; integers, ranks, text, and flags must be equal; floats must "
                f"agree within {FLOAT_TOLERANCE:g} absolute (lists element by element). The counts "
                "below separate the bit-identical cells from those only within the tolerance. Any "
                "cell outside it, or any structural difference, fails the run."
            ),
            "",
            "## Headline",
            "",
            "| measure | count |",
            "| --- | ---: |",
            f"| published tables compared | {_fmt(totals['tables'])} |",
            f"| cells compared | {_fmt(totals['cells'])} |",
            f"| cells bit-identical | {_fmt(totals['identical'])} |",
            f"| cells within tolerance, not bit-identical | {_fmt(totals['within_tolerance'])} |",
            f"| cells outside tolerance | {_fmt(totals['outside_tolerance'])} |",
            (
                "| tables with a structural difference | "
                f"{_fmt(totals['tables_with_structural_differences'])} |"
            ),
        ]

    def _table_lines(self) -> list[str]:
        """Return the per-table summary."""
        lines = [
            "",
            "## By table",
            "",
            *_wrap(
                "`combined` is compared on its rating columns (`offense_rating` through `SRS`) and "
                "`qb_combined` on the passer columns the port makes (volume, qualifier, faced pass "
                "defense, and rating); the rest of those two files are season statistics outside "
                "the ratings."
            ),
            "",
            (
                "| table | seasons | files | rows | cells | bit-identical | within tolerance |"
                " outside | max abs difference |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        numbers = (
            "files",
            "rows",
            "cells",
            "identical",
            "within_tolerance",
            "outside_tolerance",
            "max_abs_difference",
        )
        for row in self.table_rows():
            cells = [f"`{row['table']}`", str(row["seasons"]), *(_fmt(row[n]) for n in numbers)]
            lines.append(f"| {' | '.join(cells)} |")
        return lines

    def _input_lines(self) -> list[str]:
        """Return the section on what the published game logs show about the input rows."""
        checks = self.inputs
        return [
            "",
            "## Input checks",
            "",
            *_wrap(
                "Facts about the published game logs that the port's input rows rely on, over "
                "every season (reported, not pass or fail; the parity above is the test):"
            ),
            "",
            "| check | count |",
            "| --- | ---: |",
            f"| passer-games with a dropback | {_fmt(checks.passer_games_with_dropbacks)} |",
            (
                "| of which the published per-game EPA per dropback equals (passing EPA - spike"
                f" EPA) / dropbacks bit for bit | {_fmt(checks.rate_rebuilt_bit_for_bit)} |"
            ),
            f"| passer-games without a dropback | {_fmt(checks.passer_games_without_dropbacks)} |",
            (
                "| of which carry dropback EPA (passing EPA - spike EPA not zero) | "
                f"{_fmt(checks.dropback_epa_without_dropbacks)} |"
            ),
            (
                "| passer ids with two names in one season | "
                f"{_fmt(checks.passer_ids_with_two_names)} |"
            ),
            (
                "| passer-game rows tied on team, week, game, and name | "
                f"{_fmt(checks.fit_order_ties)} |"
            ),
            f"| games | {_fmt(checks.team_games)} |",
            f"| games without exactly two team rows | {_fmt(checks.one_sided_games)} |",
        ]

    def _within_tolerance_lines(self) -> list[str]:
        """Return which columns hold cells within tolerance but not bit-identical, and why."""
        rows = [row for row in self.table_rows() if row["columns_within_tolerance"]]
        lines = ["", "## Cells within tolerance but not bit-identical", ""]
        if not rows:
            return [*lines, "None: every compared cell is bit-identical."]
        columns: set[str] = set()
        for row in rows:
            names = row["columns_within_tolerance"]
            assert isinstance(names, list)  # noqa: S101  # built by table_rows
            columns.update(str(name) for name in names)
            lines.append(f"- `{row['table']}`: {', '.join(f'`{name}`' for name in names)}")
        lines.extend(
            [
                "",
                *_wrap(
                    "A passer's raw season EPA per dropback (`qb_epa_per_dropback`) is a season "
                    "statistic shown beside the rating, not a fit. There it is the season's "
                    "passing EPA total less the spikes' EPA total, over the dropback total, each "
                    "summed in that project's row order; here it is the per-game dropback EPA "
                    "(passing less spikes) summed over the passer's games, so the last bit can "
                    "differ."
                ),
            ]
        )
        if columns - {"qb_epa_per_dropback"}:
            lines.extend(
                [
                    "",
                    *_wrap(
                        f"This run used {_thread_settings()}. With more than one BLAS thread "
                        "the matrix products of the larger fits (the quarterbacks') sum in "
                        "another order, so fitted values can differ in their last bits too; "
                        "that project built its files with one thread."
                    ),
                ]
            )
        else:
            lines.extend(["", "No fitted value differs."])
        return lines

    def _bootstrap_lines(self) -> list[str]:
        """Return the section on the rank ranges and head-to-head pairs."""
        bootstrap = [result for result in self.results if result.table in BOOTSTRAP_TABLES]
        exact = all(result.identical == result.cells and result.passed for result in bootstrap)
        verdict = (
            "every cell of every one is bit-identical."
            if exact
            else "see the per-table counts and the differences below."
        )
        return [
            "",
            "## Rank ranges and head-to-head pairs",
            "",
            *_wrap(
                f"{len(bootstrap)} bootstrap tables were compared: {verdict} The port draws the "
                "same resamples as that project: `numpy.random.default_rng(0)`, one "
                "`integers(0,n,n)` call per resample over the sorted game ids, 1000 resamples, "
                "each refit on a design built once from the rows in the published order, so the "
                "rating percentiles, rank chances, and pair chances come out the same."
            ),
        ]

    def _pre_week_lines(self) -> list[str]:
        """Return the section on the pre-week entry point."""
        summary = self.pre_week
        team_verdict = "They match." if summary.passed else "See `ratings_parity.json`."
        same, other = summary.qb_same_penalty_seasons, summary.qb_other_penalty_seasons
        return [
            "",
            "## Pre-week entry point",
            "",
            *_wrap(
                "`RatingsHistory.pre_week` ran for every week of every season "
                f"({_fmt(summary.calls)} calls: week 1 through the week after the last "
                "regular-season week)."
            ),
            "",
            *_item(
                "Team ratings: for every season after 1999 and every week from 2 on, the rows of "
                "the teams that had played are compared with the published `ratings_by_week` rows "
                f"of the week before ({_fmt(summary.team_weeks_compared)} weeks, "
                f"{_fmt(summary.team_cells)} cells): {_fmt(summary.team_identical)} "
                f"bit-identical, {_fmt(summary.team_within_tolerance)} within tolerance, "
                f"{_fmt(summary.team_outside_tolerance)} outside. {team_verdict}"
            ),
            *_item(
                "Quarterback ratings: the pre-week fit takes the previous season's "
                "cross-validated penalty, the published `qb_ratings_by_week` the season's own. "
                f"Over {_fmt(summary.qb_weeks_compared)} weeks the passers and their games and "
                f"dropbacks agree ({_fmt(summary.qb_set_mismatches)} passer mismatches, "
                f"{_fmt(summary.qb_volume_mismatches)} volume mismatches in "
                f"{_fmt(summary.qb_passer_weeks)} passer-weeks). The two penalties are equal in "
                f"{len(same)} of {len(same) + len(other)} seasons, whose ratings are "
                "bit-identical. In the other seasons "
                f"({', '.join(str(season) for season in other)}) the absolute difference in "
                "adjusted EPA per dropback averages "
                f"{_fmt(summary.qb_mean_abs_difference_other_penalty)}; over every passer-week it "
                f"averages {_fmt(summary.qb_mean_abs_difference)}, with "
                f"{_fmt(summary.qb_identical_passer_weeks)} passer-weeks bit-identical. The "
                f"largest is {_fmt(summary.qb_max_abs_difference)} "
                f"({summary.qb_max_difference_at})."
            ),
            *_item(
                "1999, the first season, has no earlier season for its penalties, so it returns "
                "null team ratings and no quarterback rows; week 1 of every season returns each "
                "team's preseason prior (from 2003; null team ratings before) and no quarterback "
                "rows."
            ),
        ]

    def _timing_lines(self) -> list[str]:
        """Return the section on timings."""
        totals = self.timing_totals()
        weekly = [timing for timing in self.timings if timing.weekly_ranges]
        lines = [
            "",
            "## Timings",
            "",
            *_wrap(
                f"Wall-clock seconds with {_thread_settings()}, on a machine shared with other "
                f"work: load average {', '.join(f'{v:.2f}' for v in self.load_start)} at the "
                f"start and {', '.join(f'{v:.2f}' for v in self.load_end)} at the end."
            ),
            "",
            "| step | seconds |",
            "| --- | ---: |",
            (
                "| full rebuild of every season's ratings files, rank ranges and pairs included "
                f"| {totals['rebuild_seconds']:.1f} |"
            ),
            f"| of which the team tables | {totals['team_seconds']:.1f} |",
            f"| of which the quarterback tables | {totals['qb_seconds']:.1f} |",
        ]
        lines.extend(
            f"| of which {timing.season}, with its weekly rank ranges | "
            f"{timing.team_seconds + timing.qb_seconds:.1f} |"
            for timing in weekly
        )
        lines.append(
            "| pre-week entry point, every week of every season | "
            f"{totals['pre_week_seconds']:.1f} |"
        )
        return lines

    def _difference_lines(self) -> list[str]:
        """Return every failing table's problems and listed differences."""
        failures = [result for result in self.results if not result.passed]
        lines = ["", "## Differences", ""]
        if not failures and self.pre_week.passed:
            return [*lines, "None."]
        for result in failures:
            lines.append(f"- {result.season} `{result.table}`:")
            lines.extend(f"  - {problem}" for problem in result.problems)
            lines.extend(f"  - {difference}" for difference in result.differences)
        lines.extend(f"- pre-week: {problem}" for problem in self.pre_week.team_problems)
        return lines


def main(argv: list[str] | None = None) -> None:
    """Rebuild, compare, write the reports, and exit 1 on any difference."""
    _single_threaded()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=None, help="nfl-sos-ratings checkout")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="compare only these seasons (default: every published season)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE,
        help="where to write the reports (default: beside this script)",
    )
    args = parser.parse_args(argv)
    root = (args.source or default_source()).resolve()
    head = check_source(root)
    source = Source(root)
    load_start = os.getloadavg()
    seasons, in_progress = _seasons(source)
    if args.seasons is not None:
        seasons = [season for season in seasons if season in set(args.seasons)]
    inputs = input_checks(source, seasons)
    results, timings = rebuild_and_compare(source, seasons, in_progress)
    pre_week = pre_week_pass(source, seasons)
    report = Report(
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        source_path=str(root),
        source_head=head,
        port_head=source_head(repo_root()),
        port_uncommitted=uncommitted_changes(repo_root(), "src/nfl_predictor/ratings/*.py"),
        seasons=seasons,
        in_progress=in_progress,
        in_progress_through_week=(
            int(source.read(in_progress, "team_game_logs").get_column("week").max() or 0)
            if in_progress is not None
            else None
        ),
        results=results,
        timings=timings,
        pre_week=pre_week,
        inputs=inputs,
        load_start=load_start,
        load_end=os.getloadavg(),
        files_read=dict(sorted(source.hashes.items())),
    )
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_JSON.name).write_text(report.to_json())
    (out_dir / OUT_MD.name).write_text(report.to_markdown())
    sys.stdout.write(
        f"{'passed' if report.passed else 'FAILED'}: wrote {OUT_MD.name} and {OUT_JSON.name} "
        f"to {out_dir}\n"
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
