"""Tests for walk-forward metric helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfl_predictor.ml.metrics import (
    confidence_pool_columns,
    confidence_pool_summary,
    margin_total_metrics,
    probability_metrics,
    reliability_ece,
    reliability_table,
)


def test_margin_total_metrics_reports_mae() -> None:
    """Margin/total metrics should report simple mean absolute errors."""
    metrics = margin_total_metrics(
        actual_margin=np.array([3.0, -1.0]),
        actual_total=np.array([44.0, 38.0]),
        pred_margin=np.array([1.0, -2.0]),
        pred_total=np.array([46.0, 36.0]),
    )

    assert metrics == {"margin_mae": pytest.approx(1.5), "total_mae": pytest.approx(2.0)}


def test_probability_metrics_clip_extreme_inputs() -> None:
    """Probability metrics should clip impossible values before scoring."""
    metrics = probability_metrics(
        actual_home_win=np.array([0, 1]),
        home_win_prob=np.array([-0.25, 1.25]),
    )

    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["log_loss"] == pytest.approx(0.0)


def test_confidence_pool_columns_without_tiebreaker_and_summary() -> None:
    """Default ranking should use confidence strength order and summary should aggregate points."""
    columns = confidence_pool_columns(
        home_win_prob=np.array([0.9, 0.55, 0.2]),
        home_score=np.array([21.0, 17.0, 10.0]),
        away_score=np.array([14.0, 17.0, 14.0]),
        tiebreaker=None,
    )

    assert columns["confidence_rank"].tolist() == [3, 1, 2]
    assert columns["pick_correct"].tolist() == [True, False, True]

    summary = confidence_pool_summary(columns)

    assert summary == {
        "expected_points": pytest.approx(4.85),
        "actual_points": 5.0,
        "picks_correct": 2,
        "games": 3,
    }


def test_reliability_helpers_cover_empty_bins_and_nan_ece() -> None:
    """Reliability helpers should preserve empty bins and return NaN when no counts exist."""
    bins = reliability_table(
        home_win_prob=np.array([0.05, 0.95]),
        actual_home_win=np.array([0, 1]),
        bins=4,
    )

    assert bins[1]["count"] == 0
    assert bins[1]["avg_pred"] is None
    assert bins[1]["avg_actual"] is None
    assert reliability_ece(bins) == pytest.approx(0.05)

    empty_ece = reliability_ece([{"count": 0, "avg_pred": None, "avg_actual": None}])
    assert math.isnan(empty_ece)
