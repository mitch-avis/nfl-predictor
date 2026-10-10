"""Compute the play-by-play team-game counts and the published game columns at a checkout.

The integration's play-by-play corrections, quarterback starter and own Elo change feature
values on purpose. To measure by how much, run this script at the commit before a change and at
the commit that makes it, then compare the two outputs with ``integration_feature_diff.py``. It
builds, from the measurement cache ``integration_cache.py`` writes (read-only), with the code it
imports:

- ``team_games.parquet``: ``pbp.aggregate_pbp_team_game_stats`` over 1999-2025;
- ``box_scores.parquet``: ``pbp.aggregate_pbp_team_box_score_stats`` over the same plays;
- ``qb_games.parquet``: ``qb_stats.aggregate_qb_game_stats`` over the same plays;
- ``games.parquet``: every 1999-2025 game row as ``nfl-predictor data`` publishes it: each
  season from ``data_collection.process_season`` (the play-by-play box scores and situational
  percentages, ``--team-stats-source pbp`` and ``--tr-stats-source pbp``, the stat and strength
  prior blends on, the ``qb_elos.csv`` columns), then ``data_collection._finish_games`` (the
  quarterback family and, from the commit that wires it, the own Elo, then the published
  schema). The TeamRankings ratings stay null: they come from a scrape no correction touches.

The current season is set after the last season, so every season reads the cache; the data
directory is a scratch directory holding copies of the main checkout's ``qb_elos.csv`` and
``meta_data.csv`` and, for the own Elo's win-total preset, a copy of nfelosrs's
``wt_ratings.csv`` given with ``--win-totals`` (its download is turned off, so the copy is read
as the cached file). Nothing is written into a checkout and nothing is downloaded. Run from the
repository root, with a fixed Polars thread count so two runs compare bit for bit:

    POLARS_MAX_THREADS=8 .venv/bin/python -I .agents/sos_port/decisions/integration_feature_values.py \
        --cache-dir /tmp/step6_int/cache --main-data /home/mitch/workspace/nfl-predictor/data \
        --win-totals /home/mitch/scratch/elo_research/wt_ratings_nfelosrs.csv \
        --out-dir /tmp/step6_int/values/<label>

An exported tree (``git archive``) runs the same way with ``PYTHONPATH=<export>/src`` instead of
``-I`` and ``--commit <commit>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import elo_win_totals, pbp, qb_stats


def _git_head() -> str:
    """Return the commit this script runs at, with a mark when the tree has changes."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}{' (uncommitted changes)' if dirty else ''}"


def _no_download(url: str) -> bytes:
    """Refuse the win-total download, so the copy in the scratch cache is read."""
    msg = f"downloads are off in this measurement ({url})"
    raise OSError(msg)


def _scratch_data(scratch: Path, main_data: Path, win_totals: Path | None) -> None:
    """Point the package's data paths at ``scratch`` and copy the inputs it reads there."""
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("qb_elos.csv", "meta_data.csv"):
        shutil.copyfile(main_data / name, scratch / name)
    constants.DATA_PATH = scratch
    constants.NFELO_CACHE_DIR = scratch / "cache" / "nfelo"
    if win_totals is not None:
        cached = constants.NFELO_CACHE_DIR / "win_totals" / "wt_ratings.csv"
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(win_totals, cached)
    elo_win_totals._download = _no_download


def _sources(fields: dict[str, object]) -> object:
    """Build ``data_collection._EtlSources`` from the fields this checkout's version has."""
    known = {field.name for field in dataclasses.fields(data_collection._EtlSources)}
    return data_collection._EtlSources(**{k: v for k, v in fields.items() if k in known})


def main() -> None:
    """Build and write the four frames."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--main-data", type=Path, required=True)
    parser.add_argument("--win-totals", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument(
        "--commit",
        default=None,
        help=(
            "The commit of an exported tree (git archive) run through PYTHONPATH; by default "
            "the commit of the checkout this runs in."
        ),
    )
    args = parser.parse_args()
    commit = args.commit or _git_head()
    sys.stdout.write(f"package {Path(data_collection.__file__).parent}\n")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _scratch_data(out_dir / "scratch_data", args.main_data, args.win_totals)
    constants.NFLREADPY_CACHE_DIR = args.cache_dir
    seasons = list(range(args.first_season, args.last_season + 1))
    current = args.last_season + 1
    config = data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=args.first_season,
        max_season=args.last_season,
        line_source=constants.LINE_SOURCE_STORED,
    )
    started = time.perf_counter()
    schedule = polars_utils.load_schedule(seasons, force_refresh=False, current_season=current)
    stats_seasons, stats_schedule = data_collection._stats_window(
        seasons, schedule, config, current
    )
    team_stats, plays = data_collection._load_team_game_frame(
        stats_seasons, stats_schedule, config, current
    )
    pbp.aggregate_pbp_team_game_stats(plays).write_parquet(out_dir / "team_games.parquet")
    pbp.aggregate_pbp_team_box_score_stats(plays).write_parquet(out_dir / "box_scores.parquet")
    qb_stats.aggregate_qb_game_stats(plays).write_parquet(out_dir / "qb_games.parquet")
    elo = polars_utils.load_elo_ratings(seasons)
    frames: list[pl.DataFrame] = []
    for season in seasons:
        inputs = data_collection.SeasonInputs(
            min_season=args.first_season,
            elo_df=elo,
            tr_df=pl.DataFrame(),
            prev_tr_df=pl.DataFrame() if season > args.first_season else None,
            tr_stats_source=config.tr_stats_source,
        )
        frames.append(data_collection.process_season(season, schedule, team_stats, inputs))
    sources = _sources(
        {
            "current_season": current,
            "current_week": 1,
            "schedule_df": schedule,
            "team_stats_df": team_stats,
            "pbp_df": plays,
            "elo_df": elo,
            "raw_elo_df": polars_utils.load_raw_elo_data(),
            "pbp_seasons": tuple(stats_seasons),
        }
    )
    finish_options: dict[str, object] = {"enable_timing": False}
    record: dict[str, object] = {}
    if "own_elo_metadata" in inspect.signature(data_collection._finish_games).parameters:
        finish_options["own_elo_metadata"] = record
    games = data_collection._finish_games(
        data_collection._combine_seasons(frames), sources, args.last_season, **finish_options
    ).sort("game_id")
    games.write_parquet(out_dir / "games.parquet")
    (out_dir / "own_elo_metadata.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    elapsed = time.perf_counter() - started
    sys.stdout.write(
        f"commit {commit}\nplays {plays.height}, team-games {team_stats.height}, "
        f"games {games.height}, columns {games.width}, {elapsed:.0f} s\n"
    )


if __name__ == "__main__":
    main()
