"""Regression tests for early stopping in the paired margin/total XGBoost fit.

The margin and total heads are fit one after the other from the same parameters. Each
head must stop on its own validation curve: a total head that inherits the margin head's
early-stopping state stops after one round and predicts a near-constant total.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

from nfl_predictor.ml import ml_model_core as core
from nfl_predictor.ml import ml_model_xgb_utils as xgb_utils

xgb.set_config(verbosity=0)

_EARLY_STOPPING_ROUNDS = 50


def _synthetic_margin_total(
    n_rows: int = 3000, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build features plus a margin and a total target on NFL-like noise scales.

    ``margin = 4 * x0 + e_m`` with ``e_m ~ N(0, 9)`` and
    ``total = 44 + 4 * x1 + 3 * x2 + e_t`` with ``e_t ~ N(0, 13)``, so the total head faces
    a harder target than the margin head and never reaches the margin's validation error.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 5))
    margin = 4.0 * x[:, 0] + rng.normal(scale=9.0, size=n_rows)
    total = 44.0 + 4.0 * x[:, 1] + 3.0 * x[:, 2] + rng.normal(scale=13.0, size=n_rows)
    return x, margin, total


def _params() -> dict[str, Any]:
    """Return small, deterministic XGBoost params resolved the way training resolves them."""
    return xgb_utils._resolve_xgb_params(
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 7,
            "n_jobs": 1,
            "tree_method": "hist",
        }
    )


def test_total_head_stops_on_its_own_validation_curve() -> None:
    """The paired total head keeps the rounds it would keep alone and predicts a spread."""
    x, margin, total = _synthetic_margin_total()
    split = 2400
    x_train, x_eval = x[:split], x[split:]

    _margin_model, total_model = core._fit_margin_total_models(
        x_train,
        margin[:split],
        total[:split],
        _params(),
        x_eval=x_eval,
        y_margin_eval=margin[split:],
        y_total_eval=total[split:],
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )

    solo_params = xgb_utils._with_xgb_early_stopping_params(_params(), _EARLY_STOPPING_ROUNDS)
    solo_total = xgb.XGBRegressor(**solo_params)
    solo_total.fit(
        x_train,
        total[:split],
        **xgb_utils._build_xgb_fit_kwargs(x_eval, total[split:], _EARLY_STOPPING_ROUNDS),
    )

    paired_rounds = total_model.get_booster().num_boosted_rounds()
    assert paired_rounds > 1
    assert paired_rounds == solo_total.get_booster().num_boosted_rounds()
    assert float(np.std(xgb_utils._predict_xgb(total_model, x_eval))) > 1.0
