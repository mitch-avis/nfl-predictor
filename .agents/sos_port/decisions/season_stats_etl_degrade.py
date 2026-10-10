"""What a season's statistics tables lose when the ETL step lacks one of its nflverse sources.

Run from the repository root (single-threaded, under a minute; nothing is written outside a
temporary directory under ``~/scratch``):

    .venv/bin/python .agents/sos_port/decisions/season_stats_etl_degrade.py [--season S]

It writes this repository's cache files from the snapshot of the nflreadpy cache
(``snapshot_cache.py``) for 1999 through the season, loads the sources as the ETL does, and builds
the season's tables through the ETL's step (``season_stats_build.SeasonStatsBuild``) five times:
with every source, then without the official weekly team stats, without the weekly player stats,
without the snap counts, and without the season's play-by-play (the current season before kickoff).
For each variant it prints every table's rows and columns, the columns it lacks against the full
build, and the warnings the step logged. The rating columns are left null in every variant.
"""

from __future__ import annotations

import os

for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import logging
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import snapshot_cache  # a sibling module of the parent directory, importable once it is on the path

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import nflverse_sources, ratings_build, season_stats_build

DEFAULT_SEASON = 2025


class _Warnings(logging.Handler):
    """Keep the step's warnings, for the report."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if "Season statistics" in record.getMessage():
            self.messages.append(record.getMessage())


def _sources(season: int, cache: Path) -> season_stats_build.SeasonStatsSources:
    """Load the season statistics' sources from the temporary cache, as the ETL does."""
    constants.NFLREADPY_CACHE_DIR = cache
    constants.DATA_PATH = cache.parent
    options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
    pbp = nflverse_sources.repair_one_team_pbp_games(
        polars_utils.load_pbp([season], **options), **options
    )
    loaded = ratings_build.load_ratings_sources(
        season,
        pbp_df=pbp,
        schedule_df=polars_utils.load_schedule([season], **options),
        current_season=snapshot_cache.CURRENT_SEASON,
        force_refresh=False,
    )
    if loaded is None:
        message = "the snapshot cache has no play-by-play"
        raise SystemExit(message)
    return season_stats_build.load_season_stats_sources(
        [season], loaded, current_season=snapshot_cache.CURRENT_SEASON, force_refresh=False
    )


def main(argv: list[str] | None = None) -> None:
    """Build the season's tables with each source left out and print what they lose."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    season = parser.parse_args(argv).season
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    warnings = _Warnings()
    log.addHandler(warnings)
    with tempfile.TemporaryDirectory(
        prefix="season_stats_degrade_", dir=Path.home() / "scratch"
    ) as directory:
        cache = Path(directory) / "nflreadpy"
        snapshot_cache.write_cache(files, cache, list(range(1999, season + 1)))
        sources = _sources(season, cache)

    def without(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("season") != season)

    variants = {
        "every source": sources,
        "no official weekly team stats": replace(
            sources, official_team_stats=without(sources.official_team_stats)
        ),
        "no weekly player stats": replace(sources, player_stats=without(sources.player_stats)),
        "no snap counts": replace(sources, snap_counts=without(sources.snap_counts)),
        "no play-by-play (before kickoff)": replace(sources, plays=without(sources.plays)),
    }
    full: dict[str, pl.DataFrame] = {}
    for label, variant in variants.items():
        warnings.messages.clear()
        tables = season_stats_build.SeasonStatsBuild(variant).tables(
            season, team_ratings=None, qb_ratings=None
        )
        if not full:
            full = tables
        sys.stdout.write(f"\n{season}, {label}: {len(tables)} tables\n")
        for kind, frame in tables.items():
            lacking = [column for column in full[kind].columns if column not in frame.columns]
            sys.stdout.write(
                f"  {kind}: {frame.height} rows, {frame.width} columns"
                f" (full build {full[kind].height} rows, {full[kind].width} columns)"
                + (f"; lacks {len(lacking)}: {', '.join(lacking)}" if lacking else "")
                + "\n"
            )
        for message in warnings.messages:
            sys.stdout.write(f"  warning: {message}\n")
    log.removeHandler(warnings)


if __name__ == "__main__":
    main()
