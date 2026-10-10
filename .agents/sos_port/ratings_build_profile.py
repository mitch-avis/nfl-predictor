"""Profile one season's ratings build on the snapshot of the nflreadpy cache.

Run from the repository root (a minute or two; nothing is written outside a temporary directory):

    .venv/bin/python .agents/sos_port/ratings_build_profile.py [--season S] [--top N]

It writes this repository's cache files for 1999 through ``--season`` (default 2001) from the
snapshot (``snapshot_cache.py``) into a temporary directory, builds every earlier season's
ratings (``RatingsBuild.build``), then profiles the build of ``--season`` with ``cProfile`` and
prints the ``--top`` functions by cumulative and by own time. It runs with the thread counts the
environment sets, the ETL's own unless set; the fits pin one BLAS thread themselves.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import ratings_build


def main(argv: list[str] | None = None) -> None:
    """Profile the season's build and print the report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--season", type=int, default=2001)
    parser.add_argument("--top", type=int, default=30)
    arguments = parser.parse_args(argv)
    seasons = list(range(constants.NFLREADPY_MIN_SEASON, arguments.season + 1))
    snapshot_cache.block_downloads()
    files = snapshot_cache.find_snapshot_files()
    with tempfile.TemporaryDirectory(prefix="ratings_profile_") as directory:
        root = Path(directory)
        cache = root / "nflreadpy"
        snapshot_cache.write_cache(files, cache, seasons)
        options = {"cache_dir": cache, "current_season": snapshot_cache.CURRENT_SEASON}
        pbp = polars_utils.load_pbp(seasons, **options)
        schedule = polars_utils.load_schedule(seasons, **options)
        constants.NFLREADPY_CACHE_DIR = cache
        constants.DATA_PATH = root
        sources = ratings_build.load_ratings_sources(
            arguments.season,
            pbp_df=pbp,
            schedule_df=schedule,
            current_season=snapshot_cache.CURRENT_SEASON,
            force_refresh=False,
        )
        if sources is None:
            message = "the snapshot cache has no play-by-play"
            raise SystemExit(message)
        build = ratings_build.RatingsBuild(
            ratings_build.ratings_inputs(sources), current_season=snapshot_cache.CURRENT_SEASON
        )
        for season in seasons[:-1]:
            build.build(season)
        profile = cProfile.Profile()
        profile.enable()
        build.build(arguments.season)
        profile.disable()
    stats = pstats.Stats(profile, stream=sys.stdout)
    stats.sort_stats("cumulative").print_stats(arguments.top)
    stats.sort_stats("tottime").print_stats(arguments.top)


if __name__ == "__main__":
    main()
