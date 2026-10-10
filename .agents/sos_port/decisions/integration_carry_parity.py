"""Compare this repository's carries and rush successes with nfl-sos-ratings' designed runs.

The acceptance check of the carry definition: after the change, every 1999-2025 team-game's
carries and rush successes from this repository's play-by-play (``pbp.aggregate_pbp_team_game_
stats``, as ``integration_feature_values.py`` writes them to ``team_games.parquet``) must equal
``nfl-sos-ratings``' ``designed_carries`` and its rush successes in its published
``{season}_team_game_logs.parquet``, apart from causes that can be named. The source publishes
the success rate over designed runs, not the count, so its successes are
``rush_success_rate * designed_carries``, rounded (no designed run gives 0).

Team-games are matched on season, week and team, with the source's team codes mapped through
``constants.ALIAS_TO_CANONICAL``. Both inputs are read only. Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_carry_parity.py \
        --team-games /tmp/step6_int/values/<label>/team_games.parquet \
        --sos-data /home/mitch/workspace/nfl-sos-ratings/data
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants


def _source_logs(data_dir: Path, seasons: range) -> tuple[pl.DataFrame, str]:
    """Return the source's designed carries and rush successes, and one hash over the files."""
    digest = hashlib.sha256()
    frames: list[pl.DataFrame] = []
    for season in seasons:
        path = data_dir / f"{season}_team_game_logs.parquet"
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        frames.append(
            pl.read_parquet(path).select(
                pl.lit(season, dtype=pl.Int64).alias("season"),
                pl.col("week").cast(pl.Int64),
                pl.col("team").replace(constants.ALIAS_TO_CANONICAL).alias("team_abbr"),
                pl.col("designed_carries").cast(pl.Int64).alias("source_carries"),
                (pl.col("rush_success_rate").fill_null(0.0) * pl.col("designed_carries"))
                .round(0)
                .cast(pl.Int64)
                .alias("source_rush_successes"),
            )
        )
    return pl.concat(frames), digest.hexdigest()


def main() -> None:
    """Print the comparison."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--team-games", type=Path, required=True)
    parser.add_argument("--sos-data", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    seasons = range(args.first_season, args.last_season + 1)
    ours = (
        pl.read_parquet(args.team_games)
        .filter(pl.col("season").is_between(args.first_season, args.last_season))
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            "team_abbr",
            pl.col("carries").cast(pl.Int64),
            pl.col("rush_success_count").cast(pl.Int64).alias("rush_successes"),
        )
    )
    source, digest = _source_logs(args.sos_data, seasons)
    joined = ours.join(source, on=["season", "week", "team_abbr"], how="full", coalesce=True)
    both = joined.filter(pl.col("carries").is_not_null() & pl.col("source_carries").is_not_null())
    carries_differ = both.filter(pl.col("carries") != pl.col("source_carries"))
    successes_differ = both.filter(pl.col("rush_successes") != pl.col("source_rush_successes"))
    by_season = (
        both.group_by("season", maintain_order=True)
        .agg(
            pl.len().alias("team_games"),
            (pl.col("carries") != pl.col("source_carries")).sum().alias("carries_differ"),
            (pl.col("rush_successes") != pl.col("source_rush_successes"))
            .sum()
            .alias("successes_differ"),
            pl.col("carries").sum().alias("carries"),
            pl.col("source_carries").sum().alias("source_carries"),
        )
        .sort("season")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
        sys.stdout.write(
            f"source game logs {args.first_season}-{args.last_season} SHA-256 {digest}\n"
            f"team-games: ours {ours.height}, source {source.height}, both {both.height}, "
            f"only ours {joined.filter(pl.col('source_carries').is_null()).height}, "
            f"only the source {joined.filter(pl.col('carries').is_null()).height}\n"
            f"carries differ in {carries_differ.height}, rush successes differ in "
            f"{successes_differ.height}\n{by_season}\n"
            f"carries that differ:\n{carries_differ.sort('season', 'week', 'team_abbr')}\n"
            f"rush successes that differ:\n"
            f"{successes_differ.sort('season', 'week', 'team_abbr')}\n"
        )


if __name__ == "__main__":
    main()
