"""Compare four starting-quarterback sources for every 1999-2025 regular-season team-game.

Measured before the quarterback family switches to the play-by-play starter. For each team-game
of a regular-season game with play-by-play it names the starter four ways:

- ``rule``: this repository's port of the starter rule (``qb_starters.play_by_play_starters``),
  on the measurement cache's play-by-play, with this checkout's quarterback games
  (``qb_stats.aggregate_qb_game_stats``) and the season identity
  (``nflverse_sources.load_season_qb_identity``: the players file and the weekly rosters);
- ``source``: the same rule computed the way ``nfl-sos-ratings`` computes it
  (``first_qb_snaps`` in its ``qb_protocol_checks.py`` at ``9081bce``), with its published
  quarterback rows (``{season}_qb_game_logs.parquet``, a row with at least one dropback) in place
  of this checkout's, on the same play-by-play: a check of the port;
- ``chain``: the identity chain the quarterback family uses today, the ``qb_elos.csv`` starter
  name mapped to a GSIS id through ``meta_data.csv`` (``qb_stats.build_qb_identity``), then the
  play-by-play passer name;
- ``schedule``: nflverse's schedule columns ``home_qb_id``/``away_qb_id``.

It prints, per season and in total (1999-2025 and 2006-2025): the team-games, how many each
source names, and for each pair how many team-games both name and of those how many differ; the
schedule ids naming a player without a quarterback-game row in the game (``nfl-sos-ratings``
counted 42 in 2006-2025 against its own rows); and every team-game where the rule and the source
differ. Every input is read only. Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_starters.py \
        --cache-dir /tmp/step6_int/cache \
        --schedule-snapshot /home/mitch/scratch/nflreadpy_cache_snapshot_20261008 \
        --sos-data /home/mitch/workspace/nfl-sos-ratings/data \
        --main-data /home/mitch/workspace/nfl-predictor/data --scratch "$(mktemp -d)"
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import loaders, nflverse_sources, pbp, qb_starters, qb_stats

KEYS = ["season", "game_id", "team_abbr"]
SOURCES = ("rule", "source", "chain", "schedule")


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    """Return the commit this script runs at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def source_rule(plays: pl.DataFrame, logs: pl.DataFrame) -> pl.DataFrame:
    """Return ``first_qb_snaps`` of ``nfl-sos-ratings``: its rule with its quarterback rows."""
    rows = (
        logs.filter(pl.col("qb_dropbacks").fill_null(0) > 0)
        .select("game_id", pl.col("team_abbr"), "qb_id")
        .unique(maintain_order=True)
    )
    # The source keeps every regular-season play with a possession team (its loader turns an
    # empty team code into null) in the first quarter.
    first_quarter = plays.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & (pl.col("posteam") != "")
        & (pl.col("qtr") == 1)
    )
    candidates = pl.concat(
        [
            first_quarter.select(
                "game_id",
                pl.col("posteam").alias("team_abbr"),
                "play_id",
                pl.lit(order).alias("_order"),
                pl.col(column).alias("qb_id"),
            )
            for order, column in enumerate(("passer_player_id", "rusher_player_id"))
        ]
    ).drop_nulls("qb_id")
    return (
        candidates.join(rows, on=["game_id", "team_abbr", "qb_id"], how="semi")
        .sort("game_id", "team_abbr", "play_id", "_order")
        .unique(subset=["game_id", "team_abbr"], keep="first", maintain_order=True)
        .select("game_id", "team_abbr", pl.col("qb_id").alias("source"))
    )


def chain_starters(
    schedule: pl.DataFrame, qb_games: pl.DataFrame, seasons: list[int]
) -> pl.DataFrame:
    """Return the identity chain's starters: ``qb_elos.csv`` names mapped to GSIS ids."""
    elo = polars_utils.load_elo_ratings(seasons).select(
        "season", "week", "home_abbr", "away_abbr", "home_qb", "away_qb"
    )
    games = schedule.join(elo, on=["season", "week", "home_abbr", "away_abbr"], how="left")
    identity = qb_stats.load_qb_identity(constants.DATA_PATH / "meta_data.csv")
    passer_names = qb_stats._passer_name_map(qb_games)
    sides = []
    for side in ("home", "away"):
        targets = games.with_row_index(qb_stats._ROW).select(
            qb_stats._ROW,
            "season",
            "game_id",
            pl.col(f"{side}_abbr").alias("team_abbr"),
            pl.col(f"{side}_qb").alias("qb_name"),
        )
        resolved = qb_stats._resolve_ids(targets, identity, passer_names)
        sides.append(
            resolved.select("season", "game_id", "team_abbr", pl.col("qb_id").alias("chain"))
        )
    return pl.concat(sides)


def schedule_ids(snapshot: Path) -> tuple[pl.DataFrame, Path]:
    """Return nflverse's schedule quarterback ids from the saved schedule download."""
    for path in sorted(snapshot.glob("*.parquet")):
        if {"away_qb_id", "home_qb_id", "game_id"} <= set(pl.read_parquet_schema(path)):
            raw = pl.read_parquet(path)
            return pl.concat(
                [
                    raw.select(
                        "game_id",
                        pl.col(f"{side}_team")
                        .replace(constants.ALIAS_TO_CANONICAL)
                        .alias("team_abbr"),
                        pl.col(f"{side}_qb_id").alias("schedule"),
                    )
                    for side in ("home", "away")
                ]
            ), path
    msg = f"no saved schedule with quarterback ids in {snapshot}"
    raise FileNotFoundError(msg)


def main() -> None:
    """Print the agreement tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--schedule-snapshot", type=Path, required=True)
    parser.add_argument("--sos-data", type=Path, required=True)
    parser.add_argument("--main-data", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    seasons = list(range(args.first_season, args.last_season + 1))
    current = args.last_season + 1
    for name in ("qb_elos.csv", "meta_data.csv"):
        shutil.copyfile(args.main_data / name, args.scratch / name)
    constants.DATA_PATH = args.scratch
    constants.NFLREADPY_CACHE_DIR = args.cache_dir

    plays = loaders.load_pbp(seasons, cache_dir=args.cache_dir, current_season=current)
    qb_games = qb_stats.aggregate_qb_game_stats(plays)
    positions = qb_starters.season_positions(
        {
            season: nflverse_sources.load_season_qb_identity(
                season, cache_dir=args.cache_dir, current_season=current
            )
            for season in seasons
        }
    )
    rule = qb_starters.play_by_play_starters(plays, qb_games=qb_games, positions=positions)
    logs = pl.concat(
        [
            pl.read_parquet(args.sos_data / f"{season}_qb_game_logs.parquet").select(
                "game_id",
                pl.col("team").replace(constants.ALIAS_TO_CANONICAL).alias("team_abbr"),
                "qb_id",
                "qb_dropbacks",
            )
            for season in seasons
        ]
    )
    source = source_rule(plays, logs)
    schedule = (
        loaders.load_schedule(seasons, cache_dir=args.cache_dir, current_season=current)
        .filter(pl.col("game_type") == "REG")
        .select(
            "game_id",
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            "home_abbr",
            "away_abbr",
        )
    )
    with_plays = pbp.regular_season_plays(plays).select("game_id").unique(maintain_order=True)
    schedule = schedule.join(with_plays, on="game_id", how="semi")
    team_games = pl.concat(
        [
            schedule.select("season", "game_id", pl.col(f"{side}_abbr").alias("team_abbr"))
            for side in ("home", "away")
        ]
    )
    chain = chain_starters(schedule, qb_games, seasons)
    listed, schedule_path = schedule_ids(args.schedule_snapshot)
    named = (
        team_games.join(
            rule.select("game_id", "team_abbr", pl.col("qb_id").alias("rule")),
            on=["game_id", "team_abbr"],
            how="left",
        )
        .join(source, on=["game_id", "team_abbr"], how="left")
        .join(chain.drop("season"), on=["game_id", "team_abbr"], how="left")
        .join(listed, on=["game_id", "team_abbr"], how="left")
    )
    quarterback_rows = (
        qb_games.join(
            schedule.select("game_id", "season", "week", "home_abbr", "away_abbr"),
            on=["season", "week"],
        )
        .filter(
            (pl.col("team_abbr") == pl.col("home_abbr"))
            | (pl.col("team_abbr") == pl.col("away_abbr"))
        )
        .select("game_id", "team_abbr", pl.col("qb_id").alias("schedule"))
    )
    named = named.with_columns(
        (
            pl.col("schedule").is_not_null()
            & ~pl.struct("game_id", "team_abbr", "schedule").is_in(
                quarterback_rows.select(pl.struct("game_id", "team_abbr", "schedule"))
                .to_series()
                .implode()
            )
        ).alias("schedule_without_row")
    )
    pairs = [(a, b) for index, a in enumerate(SOURCES) for b in SOURCES[index + 1 :]]

    def summary(frame: pl.DataFrame) -> pl.DataFrame:
        aggregations = [pl.len().alias("team_games")]
        aggregations += [
            pl.col(name).is_not_null().sum().alias(f"{name}_named") for name in SOURCES
        ]
        for a, b in pairs:
            both = pl.col(a).is_not_null() & pl.col(b).is_not_null()
            aggregations += [
                both.sum().alias(f"{a}_{b}_both"),
                (both & (pl.col(a) != pl.col(b))).sum().alias(f"{a}_{b}_differ"),
            ]
        aggregations.append(pl.col("schedule_without_row").sum())
        return frame.select(aggregations)

    by_season = pl.concat(
        [
            summary(named.filter(pl.col("season") == season)).select(
                pl.lit(season).alias("season"), pl.all()
            )
            for season in seasons
        ]
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=400, tbl_hide_dataframe_shape=True):
        sys.stdout.write(
            f"commit {_git_head()}\nschedule ids from {schedule_path} "
            f"(sha256 {_sha256(schedule_path)})\n"
            f"qb_elos.csv sha256 {_sha256(args.main_data / 'qb_elos.csv')}, "
            f"meta_data.csv sha256 {_sha256(args.main_data / 'meta_data.csv')}\n"
        )
        sys.stdout.write(f"\n## Per season\n{by_season}\n")
        for first in (1999, 2006):
            total = summary(named.filter(pl.col("season") >= first))
            sys.stdout.write(f"\n## Total {first}-{args.last_season}\n{total}\n")
        rule_vs_source = named.filter(pl.col("rule").ne_missing(pl.col("source"))).sort(
            "game_id", "team_abbr"
        )
        sys.stdout.write(
            f"\n## Team-games where the rule and the source differ ({rule_vs_source.height})\n"
            f"{rule_vs_source}\n"
        )
        rule_vs_chain = named.filter(
            pl.col("rule").is_not_null()
            & pl.col("chain").is_not_null()
            & (pl.col("rule") != pl.col("chain"))
        ).sort("game_id", "team_abbr")
        sys.stdout.write(
            f"\n## Team-games where the rule and the chain both name a starter and differ "
            f"({rule_vs_chain.height})\n{rule_vs_chain}\n"
        )
        unnamed = named.filter(pl.col("rule").is_null()).sort("game_id", "team_abbr")
        sys.stdout.write(
            f"\n## Team-games the rule names no starter for ({unnamed.height})\n{unnamed}\n"
        )


if __name__ == "__main__":
    main()
