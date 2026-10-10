"""Measure the facts behind the ratings integration's decisions (integration_ratings.md).

Run from the repository root (read-only; nothing is written):

    .venv/bin/python .agents/sos_port/decisions/integration_ratings_facts.py [--source PATH]

It reads the snapshot of the nflreadpy cache ``nfl-sos-ratings`` read (its schedules and
play-by-play, through ``snapshot_cache.find_snapshot_files``) and that project's published files
at the frozen commit, and prints, for the regular seasons 1999-2025:

- the games played at a neutral site, and the published team-game rows with a null home flag;
- the plays whose ``posteam`` or ``defteam`` is an empty string;
- the passers each published ``qb_ratings`` file ranks (qualified passers), per season.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import snapshot_cache  # a sibling module of the parent directory, importable once on the path
from ratings_source import check_source, default_source, published_file

SEASONS = range(1999, 2026)


def main(argv: list[str] | None = None) -> None:
    """Print the facts."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    arguments = parser.parse_args(argv)
    check_source(arguments.source)
    files = snapshot_cache.find_snapshot_files()
    if files.schedules is None:
        message = "the snapshot has no schedules file"
        raise SystemExit(message)
    schedule = pl.read_parquet(files.schedules).filter(
        pl.col("season").is_in(list(SEASONS)) & (pl.col("game_type") == "REG")
    )
    neutral = schedule.filter(pl.col("location") == "Neutral")
    by_season = neutral.group_by("season", maintain_order=True).len().sort("season")
    sys.stdout.write(
        f"Neutral-site regular-season games 1999-2025: {neutral.height} "
        f"(of {schedule.height}); by season: {by_season.rows()}\n"
    )
    null_home = 0
    qualified: list[tuple[int, int]] = []
    for season in SEASONS:
        logs = pl.read_parquet(published_file(arguments.source, season, "team_game_logs"))
        null_home += logs.get_column("is_home").null_count()
        ratings = pl.read_parquet(published_file(arguments.source, season, "qb_ratings"))
        qualified.append((season, ratings.height))
    sys.stdout.write(f"Published team-game rows with a null home flag: {null_home}\n")
    empty: list[tuple[int, int, int]] = []
    for season in SEASONS:
        plays = pl.read_parquet(
            files.pbp[season], columns=["season_type", "posteam", "defteam"]
        ).filter(pl.col("season_type") == "REG")
        empty.append(
            (
                season,
                plays.filter(pl.col("posteam") == "").height,
                plays.filter(pl.col("defteam") == "").height,
            )
        )
    nonzero = [row for row in empty if row[1] or row[2]]
    sys.stdout.write(
        "Regular-season plays with an empty posteam or defteam (season, posteam, defteam): "
        f"{nonzero}\n"
    )
    sys.stdout.write(
        f"Qualified passers in each published qb_ratings file: {qualified}; "
        f"fewest {min(count for _, count in qualified)}\n"
    )


if __name__ == "__main__":
    main()
