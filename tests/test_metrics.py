"""Tests for walk-forward metric helpers."""

from __future__ import annotations

import math
from typing import Any

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
    # Act
    metrics = margin_total_metrics(
        actual_margin=np.array([3.0, -1.0]),
        actual_total=np.array([44.0, 38.0]),
        pred_margin=np.array([1.0, -2.0]),
        pred_total=np.array([46.0, 36.0]),
    )

    # Assert
    assert metrics == {"margin_mae": pytest.approx(1.5), "total_mae": pytest.approx(2.0)}


def test_probability_metrics_clip_extreme_inputs() -> None:
    """Probability metrics should clip impossible values before scoring."""
    # Act
    metrics = probability_metrics(
        actual_home_win=np.array([0, 1]),
        home_win_prob=np.array([-0.25, 1.25]),
    )

    # Assert
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["log_loss"] == pytest.approx(0.0)


def _three_game_columns() -> dict[str, np.ndarray]:
    """Return confidence-pool columns for three games ranked without a tiebreaker."""
    return confidence_pool_columns(
        home_win_prob=np.array([0.9, 0.55, 0.2]),
        home_score=np.array([21.0, 17.0, 10.0]),
        away_score=np.array([14.0, 17.0, 14.0]),
        tiebreaker=None,
    )


def test_confidence_pool_columns_without_tiebreaker_and_summary() -> None:
    """Default ranking should use confidence strength order and summary should aggregate points."""
    # Act
    columns = _three_game_columns()

    # Assert
    assert columns["confidence_rank"].tolist() == [3, 1, 2]
    assert columns["pick_correct"].tolist() == [True, False, True]


def test_confidence_pool_summary_aggregates_points() -> None:
    """The summary aggregates expected and actual points over the ranked games."""
    # Arrange
    columns = _three_game_columns()

    # Act
    summary = confidence_pool_summary(columns)

    # Assert
    assert summary == {
        "expected_points": pytest.approx(4.85),
        "actual_points": 5.0,
        "picks_correct": 2,
        "games": 3,
    }


def _two_game_bins() -> list[dict[str, Any]]:
    """Return a four-bin reliability table over two confident games."""
    return reliability_table(
        home_win_prob=np.array([0.05, 0.95]),
        actual_home_win=np.array([0, 1]),
        bins=4,
    )


def test_reliability_helpers_cover_empty_bins_and_nan_ece() -> None:
    """Reliability helpers should preserve empty bins and return NaN when no counts exist."""
    # Act
    bins = _two_game_bins()

    # Assert
    assert bins[1]["count"] == 0
    assert bins[1]["avg_pred"] is None
    assert bins[1]["avg_actual"] is None


def test_reliability_ece_skips_empty_bins() -> None:
    """The expected calibration error weights only the populated bins."""
    # Arrange
    bins = _two_game_bins()

    # Act
    ece = reliability_ece(bins)

    # Assert
    assert ece == pytest.approx(0.05)


def test_reliability_ece_is_nan_without_counts() -> None:
    """The expected calibration error is NaN when no bin holds a game."""
    # Act
    empty_ece = reliability_ece([{"count": 0, "avg_pred": None, "avg_actual": None}])

    # Assert
    assert math.isnan(empty_ece)
