"""Tests for confidence pool ranking tie-breaking."""

from __future__ import annotations

import numpy as np

from nfl_predictor.ml.metrics import confidence_pool_columns


def test_confidence_pool_columns_tiebreaker_orders_ranks() -> None:
    """Tied confidence strengths are deterministically ordered by the tiebreaker."""

    home_win_prob = np.array([0.6, 0.4], dtype=float)
    # Scores are only used for correctness; set to non-ties.
    home_score = np.array([21.0, 14.0], dtype=float)
    away_score = np.array([17.0, 20.0], dtype=float)

    # Both games have identical confidence strength; tiebreaker should decide ordering.
    tiebreaker = np.array(["b", "a"], dtype=object)
    cols = confidence_pool_columns(
        home_win_prob=home_win_prob,
        home_score=home_score,
        away_score=away_score,
        tiebreaker=tiebreaker,
    )

    # With equal strength, lexsort puts "a" first -> lower rank.
    assert cols["confidence_rank"].tolist() == [2, 1]
