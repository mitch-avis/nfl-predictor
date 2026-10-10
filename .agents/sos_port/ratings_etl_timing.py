"""Time the ETL's ratings step on the snapshot of the nflreadpy cache, full and incremental.

Run from the repository root (a few minutes; nothing is written outside a temporary directory):

    .venv/bin/python .agents/sos_port/ratings_etl_timing.py [--seasons FIRST LAST]

It writes this repository's cache files for 1999 through the last season from the snapshot
(``snapshot_cache.py``) into a temporary directory, loads the play-by-play and schedules as the
ETL loads them before the ratings step (untimed: the ETL loads them anyway), and then times what
the ratings add to a run of ``nfl-predictor data``, with the thread counts the environment sets
(the ETL's own, unlike the single-threaded parity audits; the fits pin one BLAS thread
themselves):

1. loading the ratings' other sources (weekly player stats, snap counts, weekly rosters, the
   players file, from the cache) and the adapter's rows (``load_ratings_sources`` and
   ``ratings_inputs``);
2. a full rebuild: every season's tables and pre-week ratings (``RatingsBuild.build``), the last
   season as the season in progress with its weekly rank ranges;
3. an incremental run: every earlier season reused from its frames, as the season cache returns
   them, and the last season built;
4. a round trip of every season's frames through the season cache (``SeasonCache.store`` and
   ``load``), the extra cache I/O an incremental run does;
5. attaching the features to every scheduled game and writing every table.

It prints the seconds of each step and the machine's load averages before and after.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils, season_cache
from nfl_predictor.utils.polars import ratings_build

FIRST_SEASON = 1999
LAST_SEASON = 2026


def _seconds(start: float) -> float:
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> None:
    """Time the ratings step and print the report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs=2, default=(FIRST_SEASON, LAST_SEASON))
    arguments = parser.parse_args(argv)
    first, last = arguments.seasons
    seasons = list(range(first, last + 1))
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    load_before = os.getloadavg()
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="ratings_timing_") as directory:
        root = Path(directory)
        cache = root / "nflreadpy"
        snapshot_cache.write_cache(files, cache, seasons)
        options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
        pbp = polars_utils.load_pbp(seasons, **options)
        schedule = polars_utils.load_schedule(seasons, **options)

        start = time.perf_counter()
        sources = _load_sources(last, pbp, schedule, cache)
        inputs = ratings_build.ratings_inputs(sources)
        timings["load the other sources and build the adapter's rows"] = _seconds(start)

        start = time.perf_counter()
        full = ratings_build.RatingsBuild(inputs, current_season=last)
        for season in seasons:
            full.build(season)
        timings[f"full rebuild: build every season, {first}-{last}"] = _seconds(start)

        start = time.perf_counter()
        incremental = ratings_build.RatingsBuild(inputs, current_season=last)
        for season in seasons[:-1]:
            incremental.reuse(season, full.build_frames(season))
        incremental.build(last)
        timings[f"incremental: reuse {first}-{last - 1}, build {last}"] = _seconds(start)

        start = time.perf_counter()
        store = season_cache.SeasonCache(root / "seasons")
        empty = pl.DataFrame()
        for season in seasons[:-1]:
            store.store(
                season,
                "timing",
                season_cache.SeasonBuild(
                    games=empty, snapshots=empty, ratings=full.build_frames(season)
                ),
            )
            store.load(season, "timing")
        timings["season cache: store and load every finished season's frames"] = _seconds(start)

        games = schedule.select(
            "game_id",
            "season",
            "week",
            "away_abbr",
            "home_abbr",
            pl.lit(None, dtype=pl.String).alias("away_qb"),
            pl.lit(None, dtype=pl.String).alias("home_qb"),
        )
        start = time.perf_counter()
        attached = full.attach(games)
        timings[f"attach the features to {attached.height:,} game rows"] = _seconds(start)

        start = time.perf_counter()
        tables = full.published_tables()
        ratings_build.write_published_tables(tables, root / "ratings")
        timings[f"write {len(tables)} tables"] = _seconds(start)
    load_after = os.getloadavg()
    threads = {
        name: os.environ.get(name, "unset")
        for name in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    }
    sys.stdout.write(f"\nRatings ETL timing, seasons {first}-{last}; threads {threads}\n")
    sys.stdout.write(
        f"load averages before {load_before[0]:.2f}, {load_before[1]:.2f}, {load_before[2]:.2f}; "
        f"after {load_after[0]:.2f}, {load_after[1]:.2f}, {load_after[2]:.2f}\n"
    )
    for label, seconds in timings.items():
        sys.stdout.write(f"{seconds:8.1f} s  {label}\n")


def _load_sources(
    last: int, pbp: pl.DataFrame, schedule: pl.DataFrame, cache: Path
) -> ratings_build.LoadedSources:
    """Load the ratings' sources from the temporary cache, as the ETL's ratings step does.

    ``load_ratings_sources`` reads the default cache directory and the data directory's
    quarterback identity file, so both point into the temporary directory for the call (the
    identity file is absent there, and the identity then empty, which costs the same).
    """
    constants.NFLREADPY_CACHE_DIR = cache
    constants.DATA_PATH = cache.parent
    sources = ratings_build.load_ratings_sources(
        last,
        pbp_df=pbp,
        schedule_df=schedule,
        current_season=snapshot_cache.CURRENT_SEASON,
        force_refresh=False,
    )
    if sources is None:
        message = "the snapshot cache has no play-by-play"
        raise SystemExit(message)
    return sources


if __name__ == "__main__":
    main()
