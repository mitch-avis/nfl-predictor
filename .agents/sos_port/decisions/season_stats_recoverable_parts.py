"""Find which team rates' hidden parts the published game logs determine, over 1999-2025.

Run from the repository root (about a minute):

    .venv/bin/python .agents/sos_port/decisions/season_stats_recoverable_parts.py

For every rate whose hidden numerator and denominator the play-by-play build carries but the
published-log adapter does not rebuild, it compares the true parts (the build's, through the
loaders from the snapshot, as the parity audit reads it) with the published team game logs of the
same season, row by row, and prints for each rate:

- the published columns equal to its denominator in every row, from the team's own row and from
  the opponent's row in the same game;
- the published columns equal to its numerator in every row, likewise;
- whether ``round(rate x denominator)`` gives the numerator in every row, for each denominator
  found (a whole-number numerator recovered from the published rate);

and a summary. Every equality is exact, with null equal to null.

The output beside this script (``season_stats_recoverable_parts_output.txt``) was produced at the
commit that adds the script, when the adapter rebuilt only the rates whose numerator, denominator
and rate are all published columns. The adapter now rebuilds the rates this output lists under
"both published" and "rounded", so a run on later code lists only the rest, under "neither".
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from season_stats_parity import (
    LAST_SEASON,
    SnapshotDownloads,
    default_source,
    read_manifest,
    relabel_washington,
)

from nfl_predictor import constants
from nfl_predictor.season_stats import published_logs, sources
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    numerator_column,
    rate_parts,
)

SEASONS = range(1999, LAST_SEASON + 1)
KEYS = ["game_id", "team"]


def _equal(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """Return whether two columns agree exactly, null with null."""
    return (left.is_null() & right.is_null()) | (left == right).fill_null(value=False)


def season_rows(season: int, cache_dir: Path) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Return one season's true parts joined to the published own and opponent rows."""
    inputs = sources.load_season_inputs(season, cache_dir=cache_dir, current_season=LAST_SEASON + 1)
    games = season_tables.build_season_games(inputs).team_games
    logs, _ = relabel_washington(
        pl.read_parquet(default_source() / "data" / f"{season}_team_game_logs.parquet")
    )
    unrecovered = sorted(
        set(rate_parts(games.columns)) - set(published_logs.rebuilt_team_rates(logs.columns))
    )
    parts = games.select(
        *KEYS,
        *(numerator_column(rate) for rate in unrecovered),
        *(denominator_column(rate) for rate in unrecovered),
    )
    numeric = [c for c in logs.columns if logs.schema[c].is_numeric() and c not in KEYS]
    opponent = logs.select(
        "game_id",
        pl.col("opponent_team").alias("team"),
        *(pl.col(c).alias(f"opp__{c}") for c in numeric),
    )
    joined = parts.join(logs, on=KEYS, how="left").join(opponent, on=KEYS, how="left")
    candidates = [*numeric, *(f"opp__{c}" for c in numeric)]
    return joined, unrecovered, candidates


def main() -> None:
    """Print, rate by rate, which published columns determine its hidden parts."""
    scratch = Path.home() / "scratch"
    SnapshotDownloads(
        scratch / "nflreadpy_cache_snapshot_20261008",
        read_manifest(scratch / "nflreadpy_cache_snapshot_20261008.sha256"),
    ).install()
    constants.PBP_COLUMNS = list(sources.SEASON_STATS_PBP_COLUMNS)
    cache_dir = Path(os.environ.get("SEASON_STATS_CACHE", scratch / "season_stats_parts_cache"))
    frames = []
    rates: set[str] = set()
    candidates: list[str] = []
    for season in SEASONS:
        joined, unrecovered, candidates = season_rows(season, cache_dir)
        frames.append(joined)
        rates |= set(unrecovered)
    rows = pl.concat(frames, how="diagonal_relaxed")
    print(f"rows: {rows.height:,} team-games of {SEASONS.start}-{SEASONS.stop - 1}")
    summary: dict[str, list[str]] = {"both published": [], "rounded": [], "neither": []}
    for rate in sorted(rates):
        num, den = pl.col(numerator_column(rate)), pl.col(denominator_column(rate))
        dens = [
            c
            for c in candidates
            if rows.select(_equal(den.cast(pl.Float64), pl.col(c).cast(pl.Float64)).all()).item()
        ]
        nums = [
            c
            for c in candidates
            if rows.select(_equal(num.cast(pl.Float64), pl.col(c).cast(pl.Float64)).all()).item()
        ]
        rounded = []
        published_rate = pl.col(rate) if rate in rows.columns else None
        if published_rate is not None:
            for column in dens:
                estimate = (published_rate * pl.col(column)).round(0)
                # Where the denominator is positive the rounded product must be the numerator.
                # Where it is zero the rate is null, so a zero numerator is assumed: it must be
                # zero or null, or pooling over games would count it. A null denominator leaves
                # the game out of every pooled rate.
                agrees = (
                    pl.when(pl.col(column) > 0)
                    .then(_equal(estimate, num.cast(pl.Float64)))
                    .when(pl.col(column) == 0)
                    .then(num.is_null() | (num == 0))
                    .otherwise(pl.lit(value=True))
                )
                if rows.select(agrees.all()).item():
                    rounded.append(column)
        print(f"{rate}")
        print(f"    denominator: {', '.join(dens) or 'none'}")
        print(f"    numerator: {', '.join(nums) or 'none'}")
        print(f"    numerator = round(rate x denominator) with: {', '.join(rounded) or 'none'}")
        if dens and nums:
            summary["both published"].append(rate)
        elif rounded:
            summary["rounded"].append(rate)
        else:
            summary["neither"].append(rate)
    for label, names in summary.items():
        print(f"{label}: {len(names)}")
        print("    " + ", ".join(names))


if __name__ == "__main__":
    main()
