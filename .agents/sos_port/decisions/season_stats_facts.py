"""Print the facts the season statistics port's decisions cite, from the snapshot and the code.

Run from the repository root:

    .venv/bin/python .agents/sos_port/decisions/season_stats_facts.py

It reads nflverse through the snapshot of nflreadpy's disk cache, as the parity audit does
(``season_stats_parity.SnapshotDownloads``, every file checked against the manifest), and prints:

1. for each season 1999-2025, the regular-season games whose ``play_id`` does not increase in the
   order nflverse publishes the plays (why plays are not re-sorted);
2. the regular-season games with a final score and no play-by-play (why the published-log check
   reads the schedule, not the logs, for each team's opponents);
3. which official weekly team stats the override reads that this repository's team-stat loader
   sums into one column (why the official stats have their own cached source);
4. whether the package imports numpy or any linear algebra (whether a BLAS thread count matters).
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import ast
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from season_stats_parity import REPO_ROOT, SnapshotDownloads, read_manifest

from nfl_predictor.season_stats.game_logs import OFFICIAL_TEAM_STAT_COLUMNS
from nfl_predictor.utils.polars import loaders

SEASONS = range(1999, 2026)


def play_order() -> None:
    """Print each season's games whose play ids do not increase in publication order."""
    print("1. Games whose play_id does not increase in publication order")
    total = 0
    for season in SEASONS:
        plays = (
            nfl.load_pbp(seasons=[season])
            .filter(pl.col("season_type") == "REG")
            .select("game_id", "play_id")
        )
        games = (
            plays.group_by("game_id", maintain_order=True)
            .agg((pl.col("play_id").diff().drop_nulls() <= 0).any().alias("unordered"))
            .filter(pl.col("unordered"))
        )
        total += games.height
        print(f"   {season}: {games.height} of {plays.get_column('game_id').n_unique()} games")
    print(f"   total: {total}")


def games_without_plays() -> None:
    """Print the regular-season games with a final score and no play-by-play."""
    print("2. Regular-season games with a final score and no play-by-play")
    schedule = nfl.load_schedules(seasons=list(SEASONS)).filter(
        (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )
    for season in SEASONS:
        played = set(
            schedule.filter(pl.col("season") == season).get_column("game_id").to_list()
        )
        with_plays = set(
            nfl.load_pbp(seasons=[season])
            .filter(pl.col("season_type") == "REG")
            .get_column("game_id")
            .unique()
            .to_list()
        )
        missing = sorted(played - with_plays)
        if missing:
            print(f"   {season}: {', '.join(missing)}")


def summed_team_stats() -> None:
    """Print the official stats the override reads that the team-stat loader sums away."""
    print("3. Official team stats the override reads that loaders.load_team_stats sums away")
    for name, parts in loaders._SUMMED_STATS:  # noqa: SLF001 - reading the loader's own table
        read = [part for part in parts if part in OFFICIAL_TEAM_STAT_COLUMNS]
        if read:
            print(f"   {name} = {' + '.join(parts)}; the override reads {', '.join(read)}")


def linear_algebra() -> None:
    """Print the package's imports of numpy, scipy, or sklearn."""
    print("4. numpy, scipy, or sklearn imports in src/nfl_predictor/season_stats")
    found = []
    for path in sorted((REPO_ROOT / "src/nfl_predictor/season_stats").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            found += [
                f"{path.name}: {name}"
                for name in names
                if name.split(".")[0] in {"numpy", "scipy", "sklearn"}
            ]
    print("   " + ("; ".join(found) if found else "none"))


def main() -> None:
    """Install the snapshot downloads and print every fact."""
    scratch = Path.home() / "scratch"
    SnapshotDownloads(
        scratch / "nflreadpy_cache_snapshot_20261008",
        read_manifest(scratch / "nflreadpy_cache_snapshot_20261008.sha256"),
    ).install()
    play_order()
    games_without_plays()
    summed_team_stats()
    linear_algebra()


if __name__ == "__main__":
    main()
