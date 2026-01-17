"""Tests for uncertainty-aware win probability helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nfl_predictor.ml import ml_model_core as core


def test_estimate_sigma_from_quantiles_falls_back() -> None:
    """Fallback sigma should be used when quantiles are invalid."""

    p10 = np.array([np.nan, 10.0])
    p90 = np.array([np.nan, 10.0])
    sigma = core._estimate_sigma_from_quantiles(p10, p90, fallback=12.0)
    assert np.allclose(sigma, np.array([12.0, 12.0]))


def test_resolve_margin_sigma_uses_quantiles() -> None:
    """Sigma should reflect p10/p90 width when provided."""

    margin = np.array([0.0, 0.0])
    quantiles = {0.1: np.array([0.0, -2.0]), 0.9: np.array([4.0, 2.0])}
    sigma = core._resolve_margin_sigma(margin, quantiles, fallback=10.0)
    expected = (quantiles[0.9] - quantiles[0.1]) / core.P10_P90_TO_SIGMA_DENOM
    assert np.allclose(sigma, expected)


def test_predict_home_win_prob_uses_sigma() -> None:
    """Uncertainty-aware probabilities should use margin/sigma."""

    margin = np.array([7.0])
    sigma = np.array([7.0])
    prob = core.predict_home_win_prob(margin, None, sigma=sigma, use_uncertainty=True)
    expected = norm.cdf(margin / sigma)
    assert np.allclose(prob, expected)
