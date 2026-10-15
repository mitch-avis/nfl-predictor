"""Parity: a game's features do not depend on whether the game has been played.

The model trains on completed rows and predicts upcoming ones, so every pre-game feature of a
game must be the same in both. A small season (one division's four teams with their coaches,
seven regular-season weeks and one playoff round, with team stats and ELO rows) runs through
``data_collection.process_season`` twice: with every game played, and with one week and every
week after it unplayed (no scores and no team-stat rows; the ELO rows stay, since an upcoming
week has its pre-game ratings), the way they look before that week's kickoff. Every column of
that week's rows but the two scores must be the same in the two builds, for the last
regular-season week and for the playoff round.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor import constants, data_collection

# One division, so the division standings carry values too.
_TEAMS = ("BUF", "MIA", "NE", "NYJ")
_SEASON = 2021
# Enough weeks for a team's last four games to differ from its season-to-date mean.
_REGULAR_WEEKS = 7
_PLAYOFF_WEEK = constants.get_regular_season_weeks(_SEASON) + 1
_SCORES = ("away_score", "home_score")
# A column of each feature family the world exercises, so the comparison cannot pass empty.
_FAMILY_COLUMNS = {
    "away_scoring_margin_4wk_trend",
    "home_turnover_margin_4wk_trend",
    "elo_4wk_trend_diff",
    "away_qb_elo_4wk_trend",
    "home_qb_value_4wk_trend",
    "away_elo_pre",
    "away_points_scored",
    "home_wins",
    "away_division_rank",
    "away_adj_srs",
    "home_sos_played_adj",
    "away_coach_win_pct_prior",
}


@dataclass(frozen=True)
class _World:
    """A season's schedule, per-team-game stats (regular season only) and ELO rows."""

    schedule: pl.DataFrame
    team_stats: pl.DataFrame
    elo: pl.DataFrame


def _slate(week: int) -> list[tuple[str, str]]:
    """Return the week's two games as (away, home): a round robin, one round per week."""
    first, *rest = _TEAMS
    shift = week % len(rest)
    order = [first, *rest[shift:], *rest[:shift]]
    pairs = [(order[0], order[3]), (order[1], order[2])]
    return pairs if week % 2 else [(home, away) for away, home in pairs]


def _team_game(
    rng: np.random.Generator, week: int, sides: tuple[str, str], points: tuple[int, int]
) -> dict[str, object]:
    """Return one team's per-game stat row."""
    (team, opponent), (scored, allowed) = sides, points
    return {
        "season": _SEASON,
        "week": week,
        "team_abbr": team,
        "opponent_abbr": opponent,
        "offensive_snaps": float(rng.integers(55, 75)),
        "defensive_snaps": float(rng.integers(55, 75)),
        "pass_epa_sum": float(rng.normal(0.0, 6.0)),
        "rush_epa_sum": float(rng.normal(0.0, 4.0)),
        "pass_epa_allowed_sum": float(rng.normal(0.0, 6.0)),
        "rush_epa_allowed_sum": float(rng.normal(0.0, 4.0)),
        "points_scored": float(scored),
        "points_allowed": float(allowed),
        "pass_yards": float(rng.integers(120, 360)),
        "rush_yards": float(rng.integers(40, 180)),
        "scoring_margin": float(scored - allowed),
        "turnover_margin": float(rng.integers(-3, 4)),
    }


def _elo_row(rng: np.random.Generator, week: int, away: str, home: str) -> dict[str, object]:
    """Return one game's pre-game ELO ratings and starting quarterbacks."""
    return {
        "season": _SEASON,
        "week": week,
        "away_abbr": away,
        "home_abbr": home,
        "away_elo_pre": float(rng.normal(1500.0, 60.0)),
        "home_elo_pre": float(rng.normal(1500.0, 60.0)),
        "away_qb": f"QB {away}",
        "home_qb": f"QB {home}",
        "away_qb_elo_pre": float(rng.normal(0.0, 30.0)),
        "home_qb_elo_pre": float(rng.normal(0.0, 30.0)),
        "away_qb_value_pre": float(rng.normal(100.0, 20.0)),
        "home_qb_value_pre": float(rng.normal(100.0, 20.0)),
    }


def _world() -> _World:
    """Build the season with every game played."""
    rng = np.random.default_rng(20261010)
    weeks = [(week, "REG") for week in range(1, _REGULAR_WEEKS + 1)]
    games: list[dict[str, object]] = []
    team_games: list[dict[str, object]] = []
    elo_rows: list[dict[str, object]] = []
    for week, game_type in [*weeks, (_PLAYOFF_WEEK, "WC")]:
        for index, (away, home) in enumerate(_slate(week)):
            away_points, home_points = (int(points) for points in rng.integers(3, 38, size=2))
            games.append(
                {
                    "game_id": f"{_SEASON}_{week:02d}_{away}_{home}",
                    "season": _SEASON,
                    "week": week,
                    "game_type": game_type,
                    "date": date(_SEASON, 9, 9) + timedelta(days=7 * (week - 1) + index),
                    "away_abbr": away,
                    "home_abbr": home,
                    "away_score": away_points,
                    "home_score": home_points,
                    "away_coach": f"Coach {away}",
                    "home_coach": f"Coach {home}",
                }
            )
            elo_rows.append(_elo_row(rng, week, away, home))
            if game_type == "REG":
                team_games.append(_team_game(rng, week, (away, home), (away_points, home_points)))
                team_games.append(_team_game(rng, week, (home, away), (home_points, away_points)))
    return _World(
        schedule=pl.DataFrame(games),
        team_stats=pl.DataFrame(team_games),
        elo=pl.DataFrame(elo_rows),
    )


def _without_results(world: _World, from_week: int) -> _World:
    """Return the world unplayed from ``from_week`` on: no scores and no team-stat rows."""
    unplayed = pl.col("week") >= from_week
    return _World(
        schedule=world.schedule.with_columns(
            pl.when(unplayed).then(None).otherwise(pl.col(column)).alias(column)
            for column in ("away_score", "home_score")
        ),
        team_stats=world.team_stats.filter(~unplayed),
        elo=world.elo,
    )


def _week_rows(world: _World, week: int) -> pl.DataFrame:
    """Build the season and return the week's rows without the scores, one row per game."""
    rows = data_collection.process_season(
        _SEASON,
        world.schedule,
        world.team_stats,
        data_collection.SeasonInputs(min_season=_SEASON, elo_df=world.elo),
    )
    return rows.filter(pl.col("week") == week).drop(_SCORES).sort("game_id")


@pytest.mark.parametrize(
    "week", [_REGULAR_WEEKS, _PLAYOFF_WEEK], ids=["last-regular-week", "playoff-round"]
)
def test_unplayed_games_match_their_completed_rows_but_for_the_scores(week: int) -> None:
    """An unplayed week's rows equal its completed rows in every column but the two scores.

    The guards keep the comparison from passing empty: the week's games are all there, every
    feature family the world exercises is among the compared columns, and the scoring-margin
    trend, which once differed, is non-zero on every compared row.
    """
    # Arrange
    world = _world()
    completed = _week_rows(world, week)
    unplayed = _without_results(world, from_week=week)

    # Act
    upcoming = _week_rows(unplayed, week)

    # Assert
    assert completed.height == len(_slate(week))
    assert set(completed.columns) >= _FAMILY_COLUMNS
    assert completed.select(pl.col("away_scoring_margin_4wk_trend").abs().min()).item() > 0
    assert_frame_equal(upcoming, completed)
