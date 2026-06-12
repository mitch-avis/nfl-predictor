"""Tests for XGBoost utility helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import xgboost as xgb

from nfl_predictor.ml import ml_model_xgb_utils as xgb_utils

xgb.set_config(verbosity=0)


def _reset_runtime_state() -> None:
    xgb_utils._RUNTIME_STATE.early_stopping_fallback_logged = False
    xgb_utils._RUNTIME_STATE.gpu_fallback_logged = False
    xgb_utils._RUNTIME_STATE.gpu_tree_method_disabled = False


def test_resolve_xgb_params_gpu_tree_method(monkeypatch) -> None:
    """GPU tree_method and device are coerced properly."""
    _reset_runtime_state()

    def fake_supported(param: str) -> bool:
        return param in {"device", "predictor"}

    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", fake_supported)

    params = xgb_utils._resolve_xgb_params(
        {"max_depth": 3},
        tree_method="gpu",
        device="auto",
    )

    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"
    assert params["eval_metric"] == "mae"


def test_build_xgb_fit_kwargs_with_early_stopping(monkeypatch) -> None:
    """XGB fit kwargs include early stopping when supported."""
    monkeypatch.setattr(xgb_utils, "_xgb_fit_supports", lambda _p: True)

    kwargs = xgb_utils._build_xgb_fit_kwargs(
        np.zeros((2, 1)),
        np.array([1.0, 2.0]),
        early_stopping_rounds=5,
    )

    assert "eval_set" in kwargs
    assert kwargs["verbose"] is False
    assert kwargs["early_stopping_rounds"] == 5


def test_build_xgb_fit_kwargs_with_callbacks(monkeypatch) -> None:
    """XGB fit kwargs include callbacks when supported."""
    monkeypatch.setattr(xgb_utils, "_xgb_fit_supports", lambda _p: True)

    sentinel = object()
    kwargs = xgb_utils._build_xgb_fit_kwargs(
        np.zeros((2, 1)),
        np.array([1.0, 2.0]),
        early_stopping_rounds=None,
        callbacks=[sentinel],
    )

    assert kwargs["callbacks"] == [sentinel]


def test_with_xgb_early_stopping_params_with_callbacks(monkeypatch) -> None:
    """XGB params are updated with early stopping using callbacks when needed."""
    _reset_runtime_state()

    monkeypatch.setattr(xgb_utils, "_xgb_fit_supports", lambda _p: False)

    def fake_param_supported(param: str) -> bool:
        return param in {"early_stopping_rounds", "callbacks"}

    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", fake_param_supported)

    updated = xgb_utils._with_xgb_early_stopping_params({"max_depth": 2}, 7)

    assert updated["early_stopping_rounds"] == 7
    assert "callbacks" in updated

    callbacks = cast(list[Any], updated["callbacks"])
    assert callbacks
    callback = callbacks[0]
    if hasattr(callback, "save_best"):
        assert callback.save_best is False


def test_with_xgb_early_stopping_params_fallback(monkeypatch) -> None:
    """XGB params early stopping fallback logs when needed."""
    _reset_runtime_state()

    monkeypatch.setattr(xgb_utils, "_xgb_fit_supports", lambda _p: False)
    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", lambda _p: False)

    messages: list[str] = []
    monkeypatch.setattr(xgb_utils, "_log_early_stopping_fallback", messages.append)

    params = {"max_depth": 2}
    updated = xgb_utils._with_xgb_early_stopping_params(params, 3)

    assert updated == params
    assert messages


def test_coerce_tree_method_on_error_device_cuda() -> None:
    """tree_method and device are coerced on RuntimeError with device=cuda."""
    _reset_runtime_state()

    params = {"device": "cuda", "tree_method": "hist", "predictor": "gpu_predictor"}
    updated = xgb_utils._coerce_tree_method_on_error(params, RuntimeError("No visible GPU"))

    assert updated is not None
    assert updated["device"] == "cpu"
    assert updated["tree_method"] == "hist"
    assert "predictor" not in updated


def test_coerce_tree_method_on_error_gpu_tree_method() -> None:
    """tree_method and device are coerced on ValueError with GPU tree_method."""
    _reset_runtime_state()

    params = {"tree_method": "gpu", "predictor": "gpu_predictor"}
    updated = xgb_utils._coerce_tree_method_on_error(params, ValueError("tree_method"))

    assert updated is not None
    assert updated["tree_method"] == "hist"
    assert "device" not in updated
    assert "predictor" not in updated


def test_predict_xgb_uses_best_iteration() -> None:
    """XGB predictions use best_iteration when available."""

    class DummyBooster:
        """Dummy booster to capture predict calls."""

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def predict(self, dmatrix, iteration_range=None):
            """Capture iteration_range calls for inspection."""
            self.calls.append(iteration_range)
            return np.zeros(dmatrix.num_row())

    class DummyModel:
        """Dummy model to simulate XGB model behavior."""

        def __init__(self, best_iteration=None) -> None:
            self.best_iteration = best_iteration
            self._booster = DummyBooster()

        def get_booster(self):
            """Return the dummy booster."""
            return self._booster

    data = np.zeros((2, 1))
    model = DummyModel(best_iteration=3)
    preds = xgb_utils._predict_xgb(cast(xgb.XGBRegressor, model), data)

    assert preds.shape == (2,)
    assert model.get_booster().calls == [(0, 4)]

    model = DummyModel(best_iteration=None)
    preds = xgb_utils._predict_xgb(cast(xgb.XGBRegressor, model), data)
    assert preds.shape == (2,)
    assert model.get_booster().calls == [None]
