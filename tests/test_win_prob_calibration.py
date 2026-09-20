"""Tests for win-prob calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nfl_predictor.ml import ml_model_core as core


def test_normalize_win_prob_calibration_method_alias() -> None:
    """Normalize maps logistic to platt and lowercases values."""
    assert core.normalize_win_prob_calibration_method("LOGISTIC") == "platt"
    assert core.normalize_win_prob_calibration_method("Platt") == "platt"


def test_resolve_win_prob_calibration_method_auto_thresholds() -> None:
    """Auto calibration should resolve to the deterministic floor until a fitter beats it."""
    min_samples = core.AUTO_CALIBRATION_ISOTONIC_MIN_SAMPLES
    assert core.resolve_win_prob_calibration_method("auto", 0) == "none"
    assert core.resolve_win_prob_calibration_method("auto", min_samples - 1) == "none"
    assert core.resolve_win_prob_calibration_method("auto", min_samples) == "none"


def test_resolve_win_prob_calibration_method_passes_sigma_through() -> None:
    """An explicit sigma request is a first-class method, not an alias."""
    assert core.resolve_win_prob_calibration_method("sigma", 5) == "sigma"
    assert core.normalize_win_prob_calibration_method("deterministic") == "deterministic"


def test_resolve_win_prob_calibration_method_downgrades_small_isotonic() -> None:
    """Explicit isotonic should downgrade below the minimum pooled sample size."""
    min_samples = core.AUTO_CALIBRATION_ISOTONIC_MIN_SAMPLES

    assert core.resolve_win_prob_calibration_method("isotonic", min_samples - 1) == "sigma"
    assert core.resolve_win_prob_calibration_method("isotonic", min_samples) == "isotonic"


def test_fit_win_prob_calibrator_sigma_uses_margin_residual_scale() -> None:
    """Sigma calibration should fit a single residual scale from actual versus predicted margins."""
    pred_margin = np.array([3.0, -4.0, 1.0])
    actual_margin = np.array([9.0, -8.0, 4.0])
    actual_home_win = (actual_margin > 0).astype(int)

    calibrator = core._fit_win_prob_calibrator(
        pred_margin,
        actual_home_win,
        "sigma",
        actual_margin=actual_margin,
    )

    assert calibrator is not None
    assert calibrator.method == "sigma"
    residual = actual_margin - pred_margin
    residual_centered = residual - np.mean(residual)
    assert float(calibrator.model) == np.sqrt(np.mean(residual_centered**2))


def test_predict_home_win_prob_uses_sigma_calibrator() -> None:
    """A sigma calibrator should map margins through Phi(margin / fitted_sigma)."""
    margin = np.array([7.0])
    calibrator = core.WinProbCalibrator(method="sigma", model=7.0)

    prob = core.predict_home_win_prob(margin, calibrator)

    assert np.allclose(prob, norm.cdf(margin / 7.0))
