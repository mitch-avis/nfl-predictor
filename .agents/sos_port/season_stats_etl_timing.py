"""Time what the season statistics add to an ETL run, full and incremental, on the snapshot.

Run from the repository root (a few minutes; nothing is written outside a temporary directory
under ``~/scratch``):

    .venv/bin/python .agents/sos_port/season_stats_etl_timing.py [--seasons FIRST LAST]

It writes this repository's cache files for the seasons from the snapshot of the nflreadpy cache
(``snapshot_cache.py``) into a temporary directory, loads the play-by-play, schedules and the
ratings' sources as the ETL loads them before the season statistics (untimed: the ETL loads them
anyway), and then times, with the thread counts the environment sets (unset: the ETL's own,
Polars sizing its pool to every core):

1. loading the season statistics' one source of their own, the official weekly team stats, and
   gathering the rest from the ratings' sources (``load_season_stats_sources``);
2. a full rebuild: every season's tables (``SeasonStatsBuild.tables``, its game rows included);
3. an incremental run: every finished season's part of the season cache key (its game rows,
   played schedule and play-by-play, ``SeasonStatsBuild.key_frames``, each digested by value as
   ``season_cache.SeasonKeys`` digests a season's frames), then the last season's tables, as the
   season in progress;
4. a round trip of every finished season's tables through the season cache (``SeasonCache.store``
   and ``load``), the extra cache input and output an incremental run does;
5. writing every table (``ratings_build.write_published_tables``).

The ratings are not fitted: each season's rating columns are left null, which costs the tables
the same join. It prints the seconds of each step, the machine's load averages before and after,
and the SHA-256 of every written table, so two runs (another thread count, say) can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path

from nfl_predictor import constants
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.utils import polars_utils, season_cache
from nfl_predictor.utils.polars import nflverse_sources, ratings_build, season_stats_build

FIRST_SEASON = 1999
LAST_SEASON = 2026
_NO_TEAM_RATINGS = pl.DataFrame(
    schema={
        "team": pl.String(),
        **dict.fromkeys(season_tables.TEAM_RATING_INPUT_COLUMNS[1:], pl.Float64()),
    }
)
_NO_QB_RATINGS = pl.DataFrame(
    schema={
        "qb_id": pl.String(),
        **dict.fromkeys(season_tables.QB_RATING_INPUT_COLUMNS[1:], pl.Float64()),
    }
)


def _seconds(start: float) -> float:
    return time.perf_counter() - start


def _load_ratings_sources(seasons: list[int], cache: Path) -> ratings_build.LoadedSources:
    """Load the ratings' sources from the temporary cache, as the ETL's ratings step does.

    The play-by-play and schedules of ``seasons`` are loaded as the ETL loads the run's, and the
    ratings' loader reads every earlier season from the cache, as in the ETL.
    """
    constants.NFLREADPY_CACHE_DIR = cache
    constants.DATA_PATH = cache.parent
    options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
    pbp = nflverse_sources.repair_one_team_pbp_games(
        polars_utils.load_pbp(seasons, **options), **options
    )
    loaded = ratings_build.load_ratings_sources(
        seasons[-1],
        pbp_df=pbp,
        schedule_df=polars_utils.load_schedule(seasons, **options),
        current_season=snapshot_cache.CURRENT_SEASON,
        force_refresh=False,
    )
    if loaded is None:
        message = "the snapshot cache has no play-by-play"
        raise SystemExit(message)
    return loaded


def _tables(build: season_stats_build.SeasonStatsBuild, season: int) -> dict[str, pl.DataFrame]:
    return build.tables(season, team_ratings=_NO_TEAM_RATINGS, qb_ratings=_NO_QB_RATINGS)


def main(argv: list[str] | None = None) -> None:
    """Time the season statistics step and print the report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs=2, default=(FIRST_SEASON, LAST_SEASON))
    arguments = parser.parse_args(argv)
    first, last = arguments.seasons
    seasons = list(range(first, last + 1))
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    load_before = os.getloadavg()
    timings: dict[str, float] = {}
    digests: dict[str, str] = {}
    # On disk, under ~/scratch, as the ETL's season cache is (the system temporary directory is
    # held in memory here).
    work = Path.home() / "scratch"
    with tempfile.TemporaryDirectory(prefix="season_stats_timing_", dir=work) as directory:
        root = Path(directory)
        cache = root / "nflreadpy"
        snapshot_cache.write_cache(files, cache, list(range(FIRST_SEASON, last + 1)))
        loaded = _load_ratings_sources(seasons, cache)

        start = time.perf_counter()
        sources = season_stats_build.load_season_stats_sources(
            seasons, loaded, current_season=snapshot_cache.CURRENT_SEASON, force_refresh=False
        )
        timings["load the official team stats and gather the other sources"] = _seconds(start)

        start = time.perf_counter()
        full = season_stats_build.SeasonStatsBuild(sources)
        tables = {season: _tables(full, season) for season in seasons}
        timings[f"full rebuild: every season's tables, {first}-{last}"] = _seconds(start)

        start = time.perf_counter()
        incremental = season_stats_build.SeasonStatsBuild(sources)
        for season in seasons[:-1]:
            for frame in incremental.key_frames(season).values():
                season_cache.frame_digest(frame)
        timings[f"incremental: the key of {first}-{last - 1}"] = _seconds(start)
        start = time.perf_counter()
        _tables(incremental, last)
        timings[f"incremental: {last}'s tables"] = _seconds(start)

        start = time.perf_counter()
        store = season_cache.SeasonCache(root / "seasons")
        empty = pl.DataFrame()
        for season in seasons[:-1]:
            store.store(
                season,
                "timing",
                season_cache.SeasonBuild(games=empty, snapshots=empty, ratings=tables[season]),
            )
            store.load(season, "timing")
        timings["season cache: store and load every finished season's tables"] = _seconds(start)

        published = {
            f"{season}_{kind}": frame
            for season, by_kind in tables.items()
            for kind, frame in by_kind.items()
        }
        start = time.perf_counter()
        ratings_build.write_published_tables(published, root / "ratings")
        timings[f"write {len(published)} tables"] = _seconds(start)
        for path in sorted((root / "ratings").iterdir()):
            digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    load_after = os.getloadavg()
    threads = {
        name: os.environ.get(name, "unset")
        for name in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    }
    sys.stdout.write(
        f"\nSeason statistics ETL timing, seasons {first}-{last}; threads {threads};"
        f" Polars pool {pl.thread_pool_size()}\n"
    )
    sys.stdout.write(
        f"load averages before {load_before[0]:.2f}, {load_before[1]:.2f}, {load_before[2]:.2f};"
        f" after {load_after[0]:.2f}, {load_after[1]:.2f}, {load_after[2]:.2f}\n"
    )
    for label, seconds in timings.items():
        sys.stdout.write(f"{seconds:8.1f} s  {label}\n")
    every = hashlib.sha256("".join(f"{n}:{d}\n" for n, d in digests.items()).encode())
    sys.stdout.write(
        f"SHA-256 of the {len(digests)} written tables' digests: {every.hexdigest()}\n"
    )


if __name__ == "__main__":
    main()
