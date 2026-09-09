"""Unit tests for power rankings utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.reporting.power_rankings import (
    build_power_rankings_and_standings,
    compute_projected_standings,
    fit_bradley_terry_ratings,
    outcome_to_home_prob,
    ratings_to_power_0_to_10,
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
    """Scaled ratings stay in [1, 10] with a stable mapping."""
    raw = pd.Series({"A": -2.0, "B": 0.0, "C": 2.0})
    scaled = scale_ratings_1_to_10(raw)
    assert scaled.min() >= 1.0
    assert scaled.max() <= 10.0
    # Average team (~0) maps to midscale.
    assert scaled["B"] == 5.5


def test_ratings_to_power_0_to_10_bounds() -> None:
    """0-10 scale stays in [0, 10] with a stable mapping."""
    raw = pd.Series({"A": -10.0, "B": 0.0, "C": 10.0})
    scaled = ratings_to_power_0_to_10(raw)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 10.0
    assert scaled["B"] == 5.0


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
    assert set(result.power_rankings.columns).issuperset(
        {"team_abbr", "power_rating_1_10", "power_rating_0_10", "rank"}
    )
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


def test_outcome_to_home_prob_margin_mode_separates_blowouts_from_squeakers() -> None:
    """Margin-based targets carry how decisively a game was won, not just who won."""
    home = pd.Series([31.0, 20.0, 17.0])
    away = pd.Series([3.0, 19.0, 24.0])

    binary = outcome_to_home_prob(home, away)
    margin = outcome_to_home_prob(home, away, target="margin")

    # The binary mapping cannot tell a 28-point win from a 1-point win.
    assert binary[0] == binary[1]
    # The margin mapping can, and still agrees on who won.
    assert margin[0] > margin[1] > 0.5
    assert margin[2] < 0.5


def test_outcome_to_home_prob_margin_mode_keeps_ties_at_even() -> None:
    """A tie is an even result under either target."""
    home = pd.Series([20.0])
    away = pd.Series([20.0])

    assert outcome_to_home_prob(home, away, target="margin")[0] == pytest.approx(0.5)


def test_fit_bradley_terry_ratings_honors_sample_weights() -> None:
    """Down-weighting a season pulls the fit toward the weighted-up games."""
    # AAA beats BBB in the old season; BBB beats AAA in the recent one.
    games = pd.DataFrame(
        {
            "home_abbr": ["AAA", "BBB"],
            "away_abbr": ["BBB", "AAA"],
            "p_home": [0.97, 0.97],
        }
    )

    old_heavy, _ = fit_bradley_terry_ratings(
        games, sample_weights=pd.Series([1.0, 0.05]), include_home_advantage=False
    )
    recent_heavy, _ = fit_bradley_terry_ratings(
        games, sample_weights=pd.Series([0.05, 1.0]), include_home_advantage=False
    )

    assert old_heavy["AAA"] > old_heavy["BBB"]
    assert recent_heavy["BBB"] > recent_heavy["AAA"]


def test_fit_bradley_terry_ratings_unweighted_matches_uniform_weights() -> None:
    """Passing uniform weights is the same as passing none."""
    games = pd.DataFrame(
        {
            "home_abbr": ["AAA", "BBB", "CCC"],
            "away_abbr": ["BBB", "CCC", "AAA"],
            "p_home": [0.9, 0.8, 0.4],
        }
    )

    plain, plain_hfa = fit_bradley_terry_ratings(games)
    uniform, uniform_hfa = fit_bradley_terry_ratings(games, sample_weights=pd.Series([1.0] * 3))

    pd.testing.assert_series_equal(plain, uniform)
    assert plain_hfa == pytest.approx(uniform_hfa)
