"""Tests for market anchoring and market probability adjustments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor import ml_model

# pylint: disable=protected-access


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


def test_market_prob_adjust_blend_and_clamp() -> None:
    """Blend/clamp should move probabilities toward market implied odds."""
    games_df = pd.DataFrame(
        {
            "home_moneyline": [-110, -110],
        }
    )
    base = np.array([0.7, 0.3], dtype=float)

    # Implied probability for -110 is 110/(110+100) ~= 0.523809...
    market_prob = 110.0 / (110.0 + 100.0)

    blended = ml_model.adjust_home_win_prob(
        games_df,
        base,
        ml_model.MarketProbConfig(blend_weight=1.0, clamp_delta=0.0),
    )
    assert np.allclose(blended, market_prob)

    clamped = ml_model.adjust_home_win_prob(
        games_df,
        base,
        ml_model.MarketProbConfig(blend_weight=0.0, clamp_delta=0.05),
    )
    lower = market_prob - 0.05
    upper = market_prob + 0.05
    assert np.all((clamped >= lower) & (clamped <= upper))
