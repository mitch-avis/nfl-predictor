"""Tests for win-prob calibration helpers."""

from __future__ import annotations

from nfl_predictor.ml import ml_model_core as core


def test_normalize_win_prob_calibration_method_alias() -> None:
    """Normalize maps logistic to platt and lowercases values."""

    assert core.normalize_win_prob_calibration_method("LOGISTIC") == "platt"
    assert core.normalize_win_prob_calibration_method("Platt") == "platt"


def test_resolve_win_prob_calibration_method_auto_thresholds() -> None:
    """Auto calibration selects isotonic only when sample size is large."""

    min_samples = core.AUTO_CALIBRATION_ISOTONIC_MIN_SAMPLES
    assert core.resolve_win_prob_calibration_method("auto", 0) == "none"
    assert core.resolve_win_prob_calibration_method("auto", min_samples - 1) == "platt"
    assert core.resolve_win_prob_calibration_method("auto", min_samples) == "isotonic"
