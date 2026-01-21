"""Tests for coach feature engineering."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils import polars_utils


def _sample_schedule() -> pl.DataFrame:
    """Return a minimal schedule fixture with coaches."""

    return pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["BBB", "AAA"],
            "away_score": [10, 14],
            "home_score": [20, 7],
            "away_coach": ["Coach A", "Coach B"],
            "home_coach": ["Coach B", "Coach A"],
        }
    )


def test_build_coach_features_no_leakage() -> None:
    """Coach features use only prior games for win-rate calculations."""

    out = polars_utils.build_coach_features(_sample_schedule(), season=2024)

    week1 = out.filter((pl.col("week") == 1) & (pl.col("team_abbr") == "AAA")).to_dicts()
    assert week1[0]["coach_games_prior"] == 0
    assert week1[0]["coach_win_pct_prior"] == 0.0

    week2 = out.filter((pl.col("week") == 2) & (pl.col("team_abbr") == "AAA")).to_dicts()
    assert week2[0]["coach_games_prior"] == 1
    assert week2[0]["coach_win_pct_prior"] == 0.0
    assert week2[0]["coach_team_games_prior"] == 1
    assert week2[0]["coach_team_win_pct_prior"] == 0.0

    week2_bbb = out.filter((pl.col("week") == 2) & (pl.col("team_abbr") == "BBB")).to_dicts()
    assert week2_bbb[0]["coach_games_prior"] == 1
    assert week2_bbb[0]["coach_win_pct_prior"] == 1.0
    assert week2_bbb[0]["coach_team_games_prior"] == 1
    assert week2_bbb[0]["coach_team_win_pct_prior"] == 1.0

