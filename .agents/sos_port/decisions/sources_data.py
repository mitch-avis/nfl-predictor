"""Reproduce the numbers behind the nflverse-source decisions in ``sources.md`` (live download).

Run from the repository root, with nflreadpy's on-disk cache so a rerun the same day reuses the
files:

    NFLREADPY_CACHE=filesystem NFLREADPY_CACHE_DIR=/tmp/nflreadpy_cache \
        .venv/bin/python .agents/sos_port/decisions/sources_data.py \
        --pbp-cache /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy

It downloads the nflverse files through nflreadpy and the two ESPN QBR release assets, reads the
regular-season play-by-play passers from an existing nflreadpy cache directory (read only), and
prints every number the decision log quotes. It writes nothing. nflverse revises its files, so
the output can drift from the copy in ``sources.md``, which records the date and versions.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import io
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

from nfl_predictor import constants

FIRST_SEASON = 1999
LAST_SEASON = 2025
WEEKLY_ROSTER_FIRST_SEASON = 2002
SNAP_COUNT_FIRST_SEASON = 2012
QBR_URL = "https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_{level}_level.parquet"
STAT_GAP_COLUMNS = ("def_tackles_for_loss", "def_qb_hits")
LEAGUE_ROSTER_CODES = ("ARZ", "BLT", "CLV", "HST", "SL")

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(80)


def heading(text: str) -> None:
    """Print a section heading."""
    print(f"\n== {text}")


def seasons(first: int) -> list[int]:
    """Return every season from ``first`` through ``LAST_SEASON``."""
    return list(range(first, LAST_SEASON + 1))


def code_table(frame: pl.DataFrame, source: str, columns: list[str]) -> pl.DataFrame:
    """Return each raw team code in ``columns`` with its canonical form and row count."""
    parts = [
        frame.select(pl.col(column).cast(pl.String).alias("code")).with_columns(
            pl.lit(column).alias("column")
        )
        for column in columns
        if column in frame.columns
    ]
    codes = pl.concat(parts).group_by("column", "code", maintain_order=True).agg(pl.len())
    return codes.with_columns(
        pl.lit(source).alias("source"),
        pl.col("code")
        .replace_strict(constants.ALIAS_TO_CANONICAL, default=None)
        .alias("canonical"),
    ).select("source", "column", "code", "canonical", "len")


def report_codes(tables: list[pl.DataFrame]) -> None:
    """Print the codes that the canonical mapping renames or does not know."""
    codes = pl.concat(tables)
    heading("Team codes the canonical mapping renames (code != canonical) or does not know")
    print(
        codes.filter(pl.col("canonical").is_null() | (pl.col("code") != pl.col("canonical")))
        .group_by("source", "code", "canonical", maintain_order=True)
        .agg(pl.col("len").sum().alias("rows"))
        .sort("source", "code", nulls_last=True)
    )
    heading("Empty-string and null team codes by source and column")
    print(
        codes.filter(pl.col("code").is_null() | (pl.col("code") == ""))
        .group_by("source", "column", "code", maintain_order=True)
        .agg(pl.col("len").sum().alias("rows"))
        .sort("source", "column")
    )


def main() -> None:
    """Print every section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp-cache", type=Path, required=True)
    args = parser.parse_args()

    heading("Environment")
    print("date:", datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"))
    for package in ("nflreadpy", "polars"):
        print(package, importlib.metadata.version(package))

    player_stats = pl.concat(
        [nfl.load_player_stats(seasons=season, summary_level="week") for season in seasons(1999)],
        how="diagonal_relaxed",
    )
    regular_stats = player_stats.filter(pl.col("season_type") == "REG")
    rosters = pl.concat(
        [nfl.load_rosters(seasons=season) for season in seasons(FIRST_SEASON)],
        how="diagonal_relaxed",
    )
    weekly_rosters = pl.concat(
        [nfl.load_rosters_weekly(seasons=season) for season in seasons(2002)],
        how="diagonal_relaxed",
    )
    snaps = pl.concat(
        [nfl.load_snap_counts(seasons=season) for season in seasons(SNAP_COUNT_FIRST_SEASON)],
        how="diagonal_relaxed",
    )
    qbr = {
        level: pl.read_parquet(
            io.BytesIO(requests.get(QBR_URL.format(level=level), timeout=60).content)
        )
        for level in ("season", "week")
    }
    players = nfl.load_players()

    report_codes(
        [
            code_table(player_stats, "weekly player stats", ["team", "opponent_team"]),
            code_table(snaps, "snap counts", ["team", "opponent"]),
            code_table(rosters, "season rosters", ["team"]),
            code_table(weekly_rosters, "weekly rosters", ["team"]),
            code_table(qbr["season"], "QBR season", ["team_abb"]),
            code_table(qbr["week"], "QBR week", ["team_abb", "opp_abb"]),
        ]
    )

    heading("Seasons whose season or weekly rosters use league (GSIS) team codes")
    for name, frame in (("season rosters", rosters), ("weekly rosters", weekly_rosters)):
        print(
            name,
            frame.filter(pl.col("team").is_in(LEAGUE_ROSTER_CODES))
            .group_by("season", maintain_order=True)
            .agg(pl.col("team").unique(maintain_order=True).sort(), pl.len())
            .sort("season"),
        )

    heading("Games whose regular-season weekly player stats name one team (20+ rows)")
    one_team = (
        regular_stats.group_by("game_id", maintain_order=True)
        .agg(
            pl.col("team").n_unique().alias("teams"),
            pl.len().alias("rows"),
            pl.col("team").first().alias("credited_team"),
        )
        .filter((pl.col("teams") == 1) & (pl.col("rows") >= 20))
        .sort("game_id")
        .with_columns(
            pl.col("game_id").str.split("_").list.get(2).alias("away_code"),
            pl.col("game_id").str.split("_").list.get(3).alias("home_code"),
        )
    )
    print(one_team)
    print("games:", one_team.height, "rows:", one_team["rows"].sum())

    heading("Repair coverage: rows whose player the season roster puts on exactly one side")
    league_codes = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "STL"}
    roster_teams = (
        rosters.select(
            "season",
            pl.col("gsis_id").alias("player_id"),
            pl.col("team").replace(league_codes).replace(constants.ALIAS_TO_CANONICAL),
        )
        .drop_nulls()
        .unique(maintain_order=True)
    )
    affected = regular_stats.join(
        one_team.select("game_id", "away_code", "home_code"), on="game_id"
    )
    for label, side in (
        ("sides normalized", lambda column: pl.col(column).replace(constants.ALIAS_TO_CANONICAL)),
        ("sides as written in game_id", pl.col),
    ):
        sides = affected.with_columns(
            side("away_code").alias("away"), side("home_code").alias("home")
        )
        matched = (
            sides.select("season", "game_id", "player_id", "away", "home")
            .join(roster_teams, on=["season", "player_id"], how="left")
            .filter((pl.col("team") == pl.col("away")) | (pl.col("team") == pl.col("home")))
            .group_by("game_id", "player_id", maintain_order=True)
            .agg(pl.col("team").n_unique().alias("sides"))
        )
        placed = matched.filter(pl.col("sides") == 1).height
        print(
            f"{label}: {affected.height} rows, placed on one side {placed}, "
            f"on both sides {matched.filter(pl.col('sides') > 1).height}, "
            f"left unknown {affected.height - placed}"
        )
        print(
            "  by game:",
            sides.select("game_id", "player_id")
            .join(matched.filter(pl.col("sides") == 1), on=["game_id", "player_id"], how="left")
            .group_by("game_id", maintain_order=True)
            .agg(pl.len().alias("rows"), pl.col("sides").is_not_null().sum().alias("placed"))
            .sort("game_id")
            .rows(),
        )

    heading("Stat gaps: regular-season totals of the gap columns by season")
    print(
        regular_stats.filter(pl.col("season") <= 2013)
        .group_by("season", maintain_order=True)
        .agg(
            *(pl.col(column).sum() for column in STAT_GAP_COLUMNS),
            *(
                (pl.col(column) > 0).sum().alias(f"{column}_rows_above_zero")
                for column in STAT_GAP_COLUMNS
            ),
        )
        .sort("season")
    )

    heading("Snap counts by season and game type (rows)")
    print(
        snaps.group_by("season", maintain_order=True)
        .agg(pl.len().alias("rows"), (pl.col("game_type") == "REG").sum().alias("regular_rows"))
        .sort("season")
        .head(3)
    )

    heading("Quarterback identity: regular-season passers by where a source lists them at QB")
    players_qb = set(players.filter(pl.col("position") == "QB")["gsis_id"].to_list())
    rows = []
    for season in seasons(FIRST_SEASON):
        pbp = pl.read_parquet(args.pbp_cache / f"pbp_{season}_reg.parquet")
        passers = (
            pbp.filter(pl.col("passer_player_id").is_not_null())
            .group_by("passer_player_id", maintain_order=True)
            .agg(pl.len().alias("plays"))
        )
        weekly_qb = set(
            weekly_rosters.filter(
                (pl.col("season") == season)
                & (pl.col("game_type") == "REG")
                & (pl.col("position") == "QB")
            )["gsis_id"].to_list()
        )
        season_qb = set(
            rosters.filter((pl.col("season") == season) & (pl.col("position") == "QB"))[
                "gsis_id"
            ].to_list()
        )
        ids = passers["passer_player_id"].to_list()
        plays = dict(passers.iter_rows())
        by_rule = [player for player in ids if player in players_qb or player in weekly_qb]
        season_only = [player for player in ids if player in season_qb and player not in by_rule]
        rows.append(
            {
                "season": season,
                "passers": len(ids),
                "qb_by_rule": len(by_rule),
                "qb_by_rule_plays": sum(plays[player] for player in by_rule),
                "weekly_roster_only": sum(
                    1 for player in ids if player in weekly_qb and player not in players_qb
                ),
                "season_roster_only": len(season_only),
                "season_roster_only_plays": sum(plays[player] for player in season_only),
                "not_qb": len(ids) - len(by_rule) - len(season_only),
                "passer_plays": sum(plays.values()),
            }
        )
    print(pl.DataFrame(rows))

    heading("Season-roster-only passers before weekly rosters exist (1999-2001)")
    names = players.select("gsis_id", "display_name", "position")
    for season in range(FIRST_SEASON, WEEKLY_ROSTER_FIRST_SEASON):
        pbp = pl.read_parquet(args.pbp_cache / f"pbp_{season}_reg.parquet")
        season_qb = rosters.filter((pl.col("season") == season) & (pl.col("position") == "QB"))
        passers = (
            pbp.filter(pl.col("passer_player_id").is_not_null())
            .group_by("passer_player_id", maintain_order=True)
            .agg(pl.len().alias("plays"))
            .rename({"passer_player_id": "gsis_id"})
        )
        missed = (
            passers.join(season_qb.select("gsis_id").unique(maintain_order=True), on="gsis_id")
            .filter(~pl.col("gsis_id").is_in(list(players_qb)))
            .join(names, on="gsis_id", how="left")
            .sort("plays", descending=True)
        )
        print(season, missed.rows())

    heading("Errors for a season after the current one")
    future = LAST_SEASON + 2
    for name, call in (
        ("load_player_stats", lambda: nfl.load_player_stats(seasons=future, summary_level="week")),
        ("load_snap_counts", lambda: nfl.load_snap_counts(seasons=future)),
        ("load_rosters_weekly", lambda: nfl.load_rosters_weekly(seasons=future)),
        ("load_rosters", lambda: nfl.load_rosters(seasons=future)),
    ):
        try:
            frame = call()
        except (ConnectionError, ValueError) as error:
            print(name, future, type(error).__name__, str(error)[:120])
        else:
            print(name, future, "rows", frame.height)


if __name__ == "__main__":
    main()
