"""Compare every season's quarterback identity read from shared sources with its own load.

The ETL loads the players file and the weekly rosters once, for the ratings, and reads each
season's quarterback identity from those frames (``nflverse_sources.QbIdentitySources``);
without them it loads each season's identity on its own
(``nflverse_sources.load_season_qb_identity``). This script builds both for every season from
1999 to 2025 on the measurement cache (``integration_cache.py``) and prints, per season, each
identity's row count and whether the two frames are equal. Every season it reads is a past
season, so the loaders read the cache and download nothing; it writes nothing.

Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_identity_check.py \
        --cache-dir /tmp/step6_int/cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nfl_predictor.utils.polars import nflverse_sources

FIRST_SEASON = 1999
LAST_SEASON = 2025
# nflverse publishes weekly rosters from this season on.
FIRST_WEEKLY_ROSTER_SEASON = 2002
# A season after the last one, so every season read is a past season the cache serves.
CURRENT_SEASON = LAST_SEASON + 1


def main() -> None:
    """Print each season's comparison and the seasons that differ."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    shared = nflverse_sources.QbIdentitySources(
        nflverse_sources.load_players(cache_dir=args.cache_dir),
        nflverse_sources.load_weekly_rosters(
            list(range(FIRST_WEEKLY_ROSTER_SEASON, LAST_SEASON + 1)),
            cache_dir=args.cache_dir,
            current_season=CURRENT_SEASON,
        ),
    )
    differing: list[int] = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        own = nflverse_sources.load_season_qb_identity(
            season, cache_dir=args.cache_dir, current_season=CURRENT_SEASON
        )
        from_shared = shared.season_identity(season)
        equal = own.equals(from_shared)
        if not equal:
            differing.append(season)
        sys.stdout.write(
            f"{season}: {own.height} rows loaded on its own, {from_shared.height} from the shared "
            f"sources, equal: {equal}\n"
        )
    sys.stdout.write(
        f"seasons compared: {LAST_SEASON - FIRST_SEASON + 1}; "
        f"differing: {differing or 'none'}\n"
    )


if __name__ == "__main__":
    main()
