"""List every venue id the cached schedules use and check it against ``constants.STADIUMS``.

Run from the root of an ``nfl-predictor`` checkout, naming a schedule cache (read only):

    .venv/bin/python .agents/sos_port/decisions/stadium_venues.py \
        --cache-dir /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy \
        --first-season 1999 --last-season 2026

It reads ``schedule_<season>.parquet`` for each season and prints every ``stadium_id`` with the
names, seasons, games, home teams, roof and surface the schedules give it, then the ids that
``constants.STADIUMS`` lacks and, per season, the games played at them: the games whose stadium
features a new table entry changes. With ``--fixture PATH`` it also writes the venue ids with
their first and last season and game count as JSON, the list the stadium test checks. It exits 1
when an id lacks an entry, so the run after the table change is the check that every id
resolves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants

COLUMNS = (
    "season",
    "week",
    "game_id",
    "stadium_id",
    "stadium_name",
    "stadium_roof",
    "stadium_surface",
    "home_abbr",
)


def _schedules(cache_dir: Path, first: int, last: int) -> pl.DataFrame:
    """Return the cached schedules' venue columns for the seasons, in season order."""
    frames = []
    for season in range(first, last + 1):
        path = cache_dir / f"schedule_{season}.parquet"
        frames.append(pl.read_parquet(path, columns=list(COLUMNS)))
    return pl.concat(frames, how="vertical_relaxed")


def _venues(games: pl.DataFrame) -> pl.DataFrame:
    """Return one row per venue id with what the schedules say about it."""
    return (
        games.group_by("stadium_id")
        .agg(
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
            pl.len().alias("games"),
            pl.col("stadium_name").unique().sort().str.join(" | ").alias("names"),
            pl.col("home_abbr").unique().sort().str.join(" ").alias("home_teams"),
            pl.col("stadium_roof").fill_null("<null>").unique().sort().str.join(" ").alias("roof"),
            pl.col("stadium_surface")
            .fill_null("<null>")
            .unique()
            .sort()
            .str.join(" ")
            .alias("surface"),
        )
        .with_columns(pl.col("stadium_id").is_in(list(constants.STADIUMS)).alias("in_table"))
        .sort("stadium_id")
    )


def main() -> int:
    """Print the venue report and return 1 when a venue id lacks a table entry."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True, help="Schedule cache (read).")
    parser.add_argument("--first-season", type=int, default=constants.NFLREADPY_MIN_SEASON)
    parser.add_argument("--last-season", type=int, required=True)
    parser.add_argument("--fixture", type=Path, help="Write the venue ids as JSON here.")
    arguments = parser.parse_args()
    games = _schedules(arguments.cache_dir, arguments.first_season, arguments.last_season)
    venues = _venues(games)
    missing = venues.filter(~pl.col("in_table"))
    affected = games.filter(pl.col("stadium_id").is_in(missing["stadium_id"].to_list()))
    out = sys.stdout
    out.write(
        f"Schedules {arguments.first_season}-{arguments.last_season}: {games.height} games, "
        f"{games.get_column('stadium_id').null_count()} without a venue id, "
        f"{venues.height} venue ids, {missing.height} without a constants.STADIUMS entry, "
        f"{affected.height} games at them\n"
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=120):
        out.write(f"\n## Every venue id\n\n{venues}\n")
        out.write(f"\n## Venue ids without an entry\n\n{missing}\n")
    out.write("\n## Games at those venues, per season\n\n")
    per_season = (
        affected.group_by("season", "stadium_id")
        .agg(pl.col("game_id").sort())
        .sort("season", "stadium_id")
    )
    out.writelines(
        f"{season} {stadium_id} ({len(game_ids)} games): {' '.join(game_ids)}\n"
        for season, stadium_id, game_ids in per_season.iter_rows()
    )
    totals = affected.group_by("season").len().sort("season")
    out.write("\nGames per season: ")
    out.write(", ".join(f"{season} {count}" for season, count in totals.iter_rows()) + "\n")
    if arguments.fixture is not None:
        rows = venues.select("stadium_id", "first_season", "last_season", "games").to_dicts()
        payload = {
            "source": (
                "Every stadium_id of the cached nflverse schedules for these seasons, written by "
                ".agents/sos_port/decisions/stadium_venues.py"
            ),
            "seasons": [arguments.first_season, arguments.last_season],
            "venues": rows,
        }
        arguments.fixture.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if missing.height else 0


if __name__ == "__main__":
    sys.exit(main())
