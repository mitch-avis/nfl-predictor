"""Unit tests for small objective/scoring helpers in ml_model_core.

These tests focus on deterministic, dependency-light logic.
"""

from __future__ import annotations

import math

import pytest

from nfl_predictor.ml import ml_model_core as core


def test_optuna_direction_maximize_and_minimize() -> None:
    """Selects maximize for accuracy/points objectives; minimize otherwise."""
    assert core._optuna_direction("expected_points") == "maximize"
    assert core._optuna_direction("winner_accuracy") == "maximize"
    assert core._optuna_direction("margin_mae") == "minimize"


def test_select_objective_score_supported_metrics() -> None:
    """Selects the correct scalar objective for supported names."""
    metrics = {
        "margin_mae": 3.0,
        "total_mae": 7.0,
        "winner_accuracy": 0.55,
        "brier": 0.21,
    }
    pool = {"weekly_expected_points_avg": 12.5}

    assert core._select_objective_score(metrics, pool, "margin_mae") == 3.0
    assert core._select_objective_score(metrics, pool, "total_mae") == 7.0
    assert core._select_objective_score(metrics, pool, "combined_mae") == 5.0
    assert core._select_objective_score(metrics, pool, "winner_accuracy") == 0.55
    assert core._select_objective_score(metrics, pool, "brier") == 0.21
    assert core._select_objective_score(metrics, pool, "expected_points") == 12.5


def test_select_objective_score_missing_metrics_defaults() -> None:
    """Uses safe defaults for optional objective inputs."""
    metrics = {"margin_mae": 3.0, "total_mae": 7.0, "winner_accuracy": 0.55}
    pool = {}

    assert math.isinf(core._select_objective_score(metrics, pool, "brier"))
    assert core._select_objective_score(metrics, pool, "expected_points") == 0.0


def test_select_objective_score_unknown_raises() -> None:
    """Raises ValueError for unknown objective names."""
    with pytest.raises(ValueError, match=r"Unknown objective metric"):
        core._select_objective_score({"margin_mae": 1.0, "total_mae": 1.0}, {}, "nope")
