"""Tests for walk_forward_backtest script helpers."""

from __future__ import annotations

import pandas as pd

from scripts import walk_forward_backtest


def test_trend_feature_columns_collects_trend_and_phase_fields() -> None:
    """Trend ablation drops trend and season-phase columns only."""

    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "week_in_season_norm": [0.1],
            "season_phase_early": [1],
            "season_phase_mid": [0],
            "season_phase_late": [0],
            "away_elo_4wk_trend": [0.0],
            "home_qb_value_4wk_trend": [0.0],
            "last_5_games_rating_trend_diff": [0.0],
            "away_total_yards": [300.0],
            "home_scoring_margin": [7.0],
        }
    )

    dropped = walk_forward_backtest._trend_feature_columns(df)

    assert set(dropped) == {
        "week_in_season_norm",
        "season_phase_early",
        "season_phase_mid",
        "season_phase_late",
        "away_elo_4wk_trend",
        "home_qb_value_4wk_trend",
        "last_5_games_rating_trend_diff",
    }
