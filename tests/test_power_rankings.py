"""Unit tests for power rankings utilities."""

from __future__ import annotations

import pandas as pd

from nfl_predictor.reporting.power_rankings import (
    build_power_rankings_and_standings,
    compute_projected_standings,
    fit_bradley_terry_ratings,
    outcome_to_home_prob,
    scale_ratings_1_to_10,
)


def test_outcome_to_home_prob_encodes_results() -> None:
    """Encodes win/loss/tie into probability targets."""

    home_score = pd.Series([21, 10, 14])
    away_score = pd.Series([17, 13, 14])
    p = outcome_to_home_prob(home_score, away_score, eps=0.1)
    assert p.tolist() == [0.9, 0.1, 0.5]


def test_fit_bradley_terry_ratings_orders_strength() -> None:
    """A consistently winning team should rank above a losing team."""

    games = pd.DataFrame(
        {
            "home_abbr": ["A", "A", "B", "C"],
            "away_abbr": ["B", "C", "C", "A"],
            "p_home": [0.8, 0.75, 0.6, 0.2],
        }
    )
    ratings, _home_adv = fit_bradley_terry_ratings(games, ridge_alpha=1.0)
    assert ratings["A"] > ratings["C"]


def test_scale_ratings_1_to_10_bounds() -> None:
    """Scaled ratings stay in [1, 10]."""

    raw = pd.Series({"A": -2.0, "B": 0.0, "C": 2.0})
    scaled = scale_ratings_1_to_10(raw)
    assert scaled.min() >= 1.0
    assert scaled.max() <= 10.0


def test_build_power_rankings_and_standings_smoke() -> None:
    """End-to-end smoke test on a tiny synthetic league."""

    current_records = pd.DataFrame(
        {
            "team_abbr": ["A", "B", "C", "D"],
            "wins": [2, 1, 0, 1],
            "losses": [0, 1, 2, 1],
            "ties": [0, 0, 0, 0],
            "games_played": [2, 2, 2, 2],
        }
    )

    games_for_ratings = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 3],
            "away_abbr": ["B", "C", "D", "B"],
            "home_abbr": ["A", "D", "A", "C"],
            "p_home": [0.8, 0.55, 0.7, 0.6],
        }
    )

    future_games = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [4, 4],
            "away_abbr": ["C", "D"],
            "home_abbr": ["A", "B"],
            "home_win_prob": [0.75, 0.6],
        }
    )

    result = build_power_rankings_and_standings(
        season=2025,
        through_week=3,
        current_records=current_records,
        games_for_ratings=games_for_ratings,
        future_games_with_probs=future_games,
    )

    assert not result.power_rankings.empty
    assert set(result.power_rankings.columns).issuperset({"team_abbr", "power_rating_1_10", "rank"})
    assert not result.projected_standings.empty
    assert "projected_wins" in result.projected_standings.columns
    assert not result.projected_division_standings.empty


def test_compute_projected_standings_no_future_games_ok() -> None:
    """Handles postseason weeks (no remaining REG games) without error."""

    current_records = pd.DataFrame(
        {
            "team_abbr": ["A", "B"],
            "wins": [10, 7],
            "losses": [7, 10],
            "ties": [0, 0],
            "games_played": [17, 17],
        }
    )

    # Intentionally missing away/home columns; should be treated as empty schedule.
    future_games = pd.DataFrame(columns=["home_win_prob"])

    out = compute_projected_standings(
        current_records=current_records,
        future_games=future_games,
        season=2025,
    )
    assert set(out["team_abbr"]) == {"A", "B"}
    assert (out["games_remaining"] == 0).all()
