"""Count the players weekly player stats list at QB whom the season quarterback identity misses.

Run from the repository root (live nflverse downloads; the loaders' cache goes to a temporary
directory that is removed at the end, so nothing under ``data/`` is read or written):

    .venv/bin/python .agents/sos_port/decisions/sources_qb_listing_check.py 2001 2012 2022

``nfl-sos-ratings`` counts a weekly player-stats row as a quarterback's when the row itself lists
QB or the season identity does (``data_loader._load_official_weekly_qb_stats``, ``listed_qb``):
a third listing beside the players file and the weekly roster. For each season this prints how
many players the regular-season stats list at QB, and how many of them the identity
(``nflverse_sources.load_season_qb_identity``) does not mark as quarterbacks, with their ids.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import logging
import sys
import tempfile
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import nflverse_sources, player_identity

CURRENT = 2026


def main() -> None:
    """Print one line per season named on the command line."""
    logging.getLogger("nfl_predictor").setLevel(logging.WARNING)
    print("date:", datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"))
    for package in ("nflreadpy", "polars"):
        print(package, importlib.metadata.version(package))
    with tempfile.TemporaryDirectory(prefix="sources-qb-listing-") as directory:
        cache = Path(directory)
        for season in (int(value) for value in sys.argv[1:]):
            stats = nflverse_sources.load_weekly_player_stats(
                [season], cache_dir=cache, current_season=CURRENT
            )
            identity = nflverse_sources.load_season_qb_identity(
                season, cache_dir=cache, current_season=CURRENT
            )
            quarterbacks = set(player_identity.season_quarterback_ids(identity))
            listed = set(
                stats.filter(pl.col("position") == constants.QB_POSITION)
                .get_column("player_id")
                .drop_nulls()
                .to_list()
            )
            extra = sorted(listed - quarterbacks)
            print(
                f"{season}: stats list {len(listed)} players at QB; the identity misses "
                f"{len(extra)}: {extra}"
            )


if __name__ == "__main__":
    main()
