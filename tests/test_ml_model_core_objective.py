"""Unit tests for small objective/scoring helpers in ml_model_core.

These tests focus on deterministic, dependency-light logic.
"""

from __future__ import annotations

import math

import pytest

from nfl_predictor.ml import ml_model_core as core


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("expected_points", "maximize"),
        ("winner_accuracy", "maximize"),
        ("margin_mae", "minimize"),
    ],
)
def test_optuna_direction_maximize_and_minimize(objective: str, expected: str) -> None:
    """Selects maximize for accuracy/points objectives; minimize otherwise."""
    # Act
    direction = core._optuna_direction(objective)

    # Assert
    assert direction == expected


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("margin_mae", 3.0),
        ("total_mae", 7.0),
        ("combined_mae", 5.0),
        ("winner_accuracy", 0.55),
        ("brier", 0.21),
        ("expected_points", 12.5),
    ],
)
def test_select_objective_score_supported_metrics(objective: str, expected: float) -> None:
    """Selects the correct scalar objective for supported names."""
    # Arrange
    metrics = {
        "margin_mae": 3.0,
        "total_mae": 7.0,
        "winner_accuracy": 0.55,
        "brier": 0.21,
    }
    pool = {"weekly_expected_points_avg": 12.5}

    # Act
    score = core._select_objective_score(metrics, pool, objective)

    # Assert
    assert score == expected


def test_select_objective_score_missing_metrics_defaults() -> None:
    """Uses an infinite Brier when the Brier metric is missing."""
    # Arrange
    metrics = {"margin_mae": 3.0, "total_mae": 7.0, "winner_accuracy": 0.55}
    pool = {}

    # Act
    score = core._select_objective_score(metrics, pool, "brier")

    # Assert
    assert math.isinf(score)


def test_select_objective_score_missing_pool_points_defaults_to_zero() -> None:
    """Uses zero expected points when the pool summary is missing them."""
    # Arrange
    metrics = {"margin_mae": 3.0, "total_mae": 7.0, "winner_accuracy": 0.55}
    pool = {}

    # Act
    score = core._select_objective_score(metrics, pool, "expected_points")

    # Assert
    assert score == 0.0


def test_select_objective_score_unknown_raises() -> None:
    """Raises ValueError for unknown objective names."""
    # Act & Assert
    with pytest.raises(ValueError, match=r"Unknown objective metric"):
        core._select_objective_score({"margin_mae": 1.0, "total_mae": 1.0}, {}, "nope")
