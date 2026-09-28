"""Tests for the deterministic floor, the one win-probability mapping."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core as core


def test_the_floor_maps_margin_through_the_fixed_normal_curve() -> None:
    """Home win probability is Phi(margin / SCORE_DIFF_STD_DEV); a zero margin is a coin flip."""
    margin = np.array([-7.0, 0.0, 3.0, 14.0])

    prob = core.margin_to_home_win_prob(margin)

    np.testing.assert_allclose(prob, norm.cdf(margin / constants.SCORE_DIFF_STD_DEV))
    assert prob[1] == 0.5
