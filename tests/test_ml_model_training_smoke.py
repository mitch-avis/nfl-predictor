from __future__ import annotations

import numpy as np

from nfl_predictor import ml_model


def test_margin_total_early_stopping_wired() -> None:
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
