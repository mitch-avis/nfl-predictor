"""Tests for market anchoring and the market-implied probability yardstick."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor import ml_model
from nfl_predictor.ml import walk_forward


def test_market_anchor_targets_residualize_baseline() -> None:
    """Market anchoring should train on residuals vs market baseline."""
    df = pd.DataFrame(
        {
            "away_score": [20, 10],
            "home_score": [27, 17],
            "market_home_margin": [3.0, -1.5],
            "market_total_line": [44.0, 39.0],
        }
    )
    target_cols = ("away_score", "home_score")

    margin, total, base_margin, base_total = ml_model._prepare_margin_total_targets_with_anchor(
        df,
        target_cols,
        market_anchor=True,
    )

    actual_margin = df["home_score"].to_numpy(dtype=float) - df["away_score"].to_numpy(dtype=float)
    actual_total = df["home_score"].to_numpy(dtype=float) + df["away_score"].to_numpy(dtype=float)

    assert base_margin is not None
    assert base_total is not None
    assert np.allclose(base_margin, df["market_home_margin"].to_numpy(dtype=float))
    assert np.allclose(base_total, df["market_total_line"].to_numpy(dtype=float))
    assert np.allclose(margin, actual_margin - base_margin)
    assert np.allclose(total, actual_total - base_total)


def test_the_market_yardstick_uses_no_vig_moneyline_probabilities() -> None:
    """The scored market probability normalizes home/away implied probs to sum to one."""
    games_df = pd.DataFrame({"home_moneyline": [-150], "away_moneyline": [130]})

    home_raw = 150.0 / (150.0 + 100.0)
    away_raw = 100.0 / (130.0 + 100.0)
    expected = home_raw / (home_raw + away_raw)

    assert np.allclose(walk_forward._resolve_market_home_win_prob(games_df), expected)
