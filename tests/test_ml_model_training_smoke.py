"""Smoke tests for margin/total model training wiring."""

from __future__ import annotations

import numpy as np

from nfl_predictor import ml_model


def test_margin_total_early_stopping_wired() -> None:
    """Early stopping should be wired through to XGBoost training."""
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(20, 3))
    x_eval = rng.normal(size=(6, 3))
    y_margin = rng.normal(size=20)
    y_total = rng.normal(size=20)
    y_margin_eval = rng.normal(size=6)
    y_total_eval = rng.normal(size=6)

    params = ml_model._resolve_xgb_params(
        ml_model.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 25,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
        },
    )

    margin_model, total_model = ml_model._fit_margin_total_models(
        x_train,
        y_margin,
        y_total,
        params,
        x_eval=x_eval,
        y_margin_eval=y_margin_eval,
        y_total_eval=y_total_eval,
        early_stopping_rounds=5,
    )

    assert hasattr(margin_model, "evals_result_")
    assert hasattr(total_model, "evals_result_")


def test_blend_layer_coefficients_constrained() -> None:
    """Blend layer coefficients are non-negative and sum to 1."""
    rng = np.random.default_rng(7)
    team = rng.normal(size=200)
    market = rng.normal(size=200)
    x = np.column_stack([team, market])
    y = 0.8 * team + 0.2 * market + rng.normal(scale=0.05, size=200)

    model = ml_model._fit_blend_ridge_constrained(x, y, alpha=1.0)

    assert model.coef_.shape == (2,)
    assert float(model.coef_[0]) >= 0
    assert float(model.coef_[1]) >= 0
    assert abs(float(model.coef_.sum()) - 1.0) < 1e-6
    preds = model.predict(x)
    assert preds.shape == (200,)
