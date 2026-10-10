"""Head-to-head-excluded opponent profiles: what a team's opponents did against everyone else.

For each team, every unique opponent is profiled from its games that did not involve that team
(counts per game, rates pooled over those games), and the opponent profiles are averaged with equal
weight per opponent. A division rival played twice is profiled once, because removing both
head-to-head games makes the two profiles identical. The averaged profile is descriptive context
for the web pages; the ratings come from the simultaneous solve in ``nfl_predictor.ratings``.
"""

from typing import TypedDict

import polars as pl

from nfl_predictor.constants import TEAM_TO_DIVISION
from nfl_predictor.season_stats.team_stats import compute_team_stats_excluding_opponents

type OpponentDetail = dict[str, str | bool | int]


class OpponentProfile(TypedDict):
    """One team's averaged opponent profile and the per-opponent game counts behind it."""

    team_stats: pl.DataFrame | None
    opponents: list[str]
    opponent_details: list[OpponentDetail]


def get_opponents(schedule_df: pl.DataFrame, team: str) -> list[str]:
    """Return the sorted unique regular-season opponents of ``team``."""
    home_opps = schedule_df.filter(pl.col("home_team") == team).get_column("away_team").to_list()
    away_opps = schedule_df.filter(pl.col("away_team") == team).get_column("home_team").to_list()
    return sorted(set(home_opps + away_opps))


def is_division_opponent(team: str, opponent: str) -> bool:
    """Return whether two teams share a division."""
    return TEAM_TO_DIVISION.get(team) == TEAM_TO_DIVISION.get(opponent)


def _opponent_pairs(team: str, opponents: list[str]) -> pl.DataFrame:
    """Return ``team``'s opponents, each paired with ``team`` as the opponent to leave out."""
    return pl.DataFrame(
        {"team": opponents, "excluded_opponent": [team] * len(opponents)},
        schema={"team": pl.String, "excluded_opponent": pl.String},
    )


def _profile_from_rows(
    team: str, opponents: list[str], opponent_rows: pl.DataFrame
) -> OpponentProfile:
    """Return ``team``'s averaged profile from its opponents' rows without head-to-head games.

    ``opponent_rows`` holds one row per opponent with games left
    (``compute_team_stats_excluding_opponents``); each counts once in the average.
    """
    games = dict(opponent_rows.select("team", "games_included").iter_rows())
    details: list[OpponentDetail] = [
        {
            "opponent": opponent,
            "division": is_division_opponent(team, opponent),
            "games_included": int(games.get(opponent, 0)),
        }
        for opponent in opponents
    ]

    averaged = None
    if not opponent_rows.is_empty():
        combined = opponent_rows.drop("excluded_opponent").sort("team")
        numeric = [
            column
            for column, dtype in combined.schema.items()
            if dtype.is_numeric() and column != "games_included"
        ]
        averaged = combined.select(
            pl.lit(team).alias("team"), *[pl.col(column).mean() for column in numeric]
        )
    return {"team_stats": averaged, "opponents": opponents, "opponent_details": details}


def compute_all_opponent_profiles(
    weekly_df: pl.DataFrame, schedule_df: pl.DataFrame
) -> tuple[pl.DataFrame | None, dict[str, list[OpponentDetail]]]:
    """Return every team's averaged opponent profile plus per-team opponent details.

    Every opponent is profiled without its games against each team it played in one aggregation.
    """
    teams = sorted(weekly_df.get_column("team").unique().cast(pl.String).to_list())
    if not teams:
        return None, {}
    opponents = {team: get_opponents(schedule_df, team) for team in teams}
    pairs = pl.concat(
        [_opponent_pairs(team, opponents[team]) for team in teams],
        how="vertical",
    )
    rows = compute_team_stats_excluding_opponents(weekly_df, pairs)
    profiles: list[pl.DataFrame] = []
    details: dict[str, list[OpponentDetail]] = {}
    for team in teams:
        profile = _profile_from_rows(
            team, opponents[team], rows.filter(pl.col("excluded_opponent") == team)
        )
        if profile["team_stats"] is not None:
            profiles.append(profile["team_stats"])
        details[team] = profile["opponent_details"]
    return (pl.concat(profiles).sort("team") if profiles else None), details
