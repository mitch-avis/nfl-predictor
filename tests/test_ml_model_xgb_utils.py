"""Tests for XGBoost utility helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import ml_model_xgb_utils as xgb_utils

# The suite-wide fixture replaces the module attribute; keep the real cached probe for its test.
real_xgb_cuda_usable = xgb_utils.xgb_cuda_usable

xgb.set_config(verbosity=0)


def _reset_runtime_state() -> None:
    xgb_utils._RUNTIME_STATE.early_stopping_fallback_logged = False
    xgb_utils._RUNTIME_STATE.gpu_fallback_logged = False
    xgb_utils._RUNTIME_STATE.gpu_tree_method_disabled = False


class _DummyMatrixPreprocessor:
    """Minimal preprocessor that records fit/transform calls for wrapper tests."""

    def __init__(self) -> None:
        self.fit_inputs: list[Any] = []
        self.transform_inputs: list[Any] = []

    def fit_transform(self, x: Any) -> np.ndarray:
        """Capture the fit-transform input and return a stable matrix."""
        self.fit_inputs.append(x)
        return np.ones((2, 1))

    def transform(self, x: Any) -> np.ndarray:
        """Capture the transform input and return a stable matrix."""
        self.transform_inputs.append(x)
        return np.zeros((3, 1))


def test_fit_transform_matrix_delegates_to_preprocessor() -> None:
    """fit-transform wrapper preserves the preprocessor result and input."""
    preprocessor = _DummyMatrixPreprocessor()
    payload = object()

    result = xgb_utils._fit_transform_matrix(cast(ColumnTransformer, preprocessor), payload)

    assert result.shape == (2, 1)
    assert preprocessor.fit_inputs == [payload]


def test_transform_matrix_delegates_to_preprocessor() -> None:
    """Transform wrapper preserves the preprocessor result and input."""
    preprocessor = _DummyMatrixPreprocessor()
    payload = object()

    result = xgb_utils._transform_matrix(cast(ColumnTransformer, preprocessor), payload)

    assert result.shape == (3, 1)
    assert preprocessor.transform_inputs == [payload]


def test_resolve_xgb_params_gpu_tree_method(monkeypatch) -> None:
    """GPU tree_method and device are coerced properly."""
    _reset_runtime_state()

    def fake_supported(param: str) -> bool:
        return param in {"device", "predictor"}

    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", fake_supported)
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)

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


def test_with_xgb_early_stopping_params_uses_the_init_param_only(monkeypatch) -> None:
    """Early stopping goes in the init param and never in a callback object.

    A callback object in the params dict is shared by every estimator built from that dict
    and carries its best score and patience from one fit into the next.
    """
    _reset_runtime_state()

    monkeypatch.setattr(xgb_utils, "_xgb_fit_supports", lambda _p: False)

    def fake_param_supported(param: str) -> bool:
        return param in {"early_stopping_rounds", "callbacks"}

    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", fake_param_supported)

    params = {"max_depth": 2}
    updated = xgb_utils._with_xgb_early_stopping_params(params, 7)

    assert updated == {"max_depth": 2, "early_stopping_rounds": 7}
    assert params == {"max_depth": 2}


def test_with_xgb_early_stopping_params_never_shares_a_callback() -> None:
    """On the installed XGBoost, two estimators' params carry no callback object at all."""
    first = xgb_utils._with_xgb_early_stopping_params({"max_depth": 2}, 5)
    second = xgb_utils._with_xgb_early_stopping_params({"max_depth": 2}, 5)

    assert first["early_stopping_rounds"] == 5
    assert "callbacks" not in first
    assert "callbacks" not in second

    first_model = xgb.XGBRegressor(**first)
    second_model = xgb.XGBRegressor(**second)
    assert first_model.get_params()["callbacks"] is None
    assert second_model.get_params()["callbacks"] is None


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


class _ProbeBooster:
    """Stands in for the booster a probe fit returns, reporting the device it trained on."""

    def __init__(self, device: str) -> None:
        self.device = device

    def save_config(self) -> str:
        """Return the part of XGBoost's JSON config that names the device."""
        return json.dumps({"learner": {"generic_param": {"device": self.device}}})


def _fake_build_info(use_cuda: bool) -> Callable[[], dict[str, Any]]:
    """Return a stand-in for ``xgb.build_info`` reporting whether the build has CUDA."""
    return lambda: {"USE_CUDA": use_cuda}


def _record_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the module's warning messages, formatted."""
    messages: list[str] = []

    def _capture(msg: str, *args: object) -> None:
        messages.append(msg % args if args else msg)

    monkeypatch.setattr(xgb_utils.log, "warning", _capture)
    return messages


def test_detect_xgb_cuda_skips_the_probe_when_the_build_lacks_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU-only XGBoost build reports no GPU without attempting a fit."""
    monkeypatch.setattr(xgb_utils.xgb, "build_info", _fake_build_info(False))

    def _no_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the probe fit must not run on a CPU-only build")

    monkeypatch.setattr(xgb_utils.xgb, "train", _no_fit)

    assert xgb_utils.detect_xgb_cuda() is False


@pytest.mark.parametrize(("probe_device", "expected"), [("cuda:0", True), ("cpu", False)])
def test_detect_xgb_cuda_reads_the_device_the_probe_fit_used(
    monkeypatch: pytest.MonkeyPatch, probe_device: str, expected: bool
) -> None:
    """With no visible GPU, XGBoost moves a CUDA fit to the CPU instead of raising."""
    monkeypatch.setattr(xgb_utils.xgb, "build_info", _fake_build_info(True))
    requested: list[dict[str, Any]] = []

    def _fake_train(params: dict[str, Any], *_args: object, **_kwargs: object) -> _ProbeBooster:
        requested.append(params)
        return _ProbeBooster(probe_device)

    monkeypatch.setattr(xgb_utils.xgb, "train", _fake_train)

    assert xgb_utils.detect_xgb_cuda() is expected
    assert requested[0]["device"] == "cuda"


def test_detect_xgb_cuda_reports_no_gpu_when_the_probe_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An XGBoost error from the probe fit means the GPU is not usable."""
    monkeypatch.setattr(xgb_utils.xgb, "build_info", _fake_build_info(True))

    def _failing_train(*_args: object, **_kwargs: object) -> object:
        raise xgb.core.XGBoostError("CUDA driver version is insufficient")

    monkeypatch.setattr(xgb_utils.xgb, "train", _failing_train)

    assert xgb_utils.detect_xgb_cuda() is False


def test_xgb_cuda_usable_probes_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe runs on the first call only; later calls reuse its answer."""
    calls: list[int] = []

    def _counting_detect() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(xgb_utils, "detect_xgb_cuda", _counting_detect)
    real_xgb_cuda_usable.cache_clear()
    try:
        assert real_xgb_cuda_usable() is True
        assert real_xgb_cuda_usable() is True
    finally:
        real_xgb_cuda_usable.cache_clear()

    assert len(calls) == 1


@pytest.mark.parametrize(("usable", "expected"), [(True, "cuda"), (False, "cpu")])
@pytest.mark.parametrize("requested", ["auto", None])
def test_resolve_xgb_device_auto_prefers_a_usable_gpu(
    monkeypatch: pytest.MonkeyPatch, requested: str | None, usable: bool, expected: str
) -> None:
    """``auto`` (and no device at all) resolves to CUDA when a GPU is usable, else the CPU."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: usable)

    assert xgb_utils.resolve_xgb_device(requested) == expected


def test_resolve_xgb_device_keeps_an_explicit_cpu_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CPU request never looks for a GPU."""

    def _no_probe() -> bool:
        raise AssertionError("an explicit cpu request must not probe for a GPU")

    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", _no_probe)

    assert xgb_utils.resolve_xgb_device("cpu") == "cpu"


def test_resolve_xgb_device_keeps_an_explicit_cuda_when_a_gpu_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CUDA request stands when the GPU can train."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)

    assert xgb_utils.resolve_xgb_device("cuda") == "cuda"


def test_resolve_xgb_device_moves_an_explicit_cuda_to_the_cpu_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a usable GPU an explicit CUDA request trains, and is recorded, on the CPU."""
    _reset_runtime_state()
    warnings_logged = _record_warnings(monkeypatch)

    assert xgb_utils.resolve_xgb_device("cuda") == "cpu"
    assert len(warnings_logged) == 1
    assert "cuda" in warnings_logged[0]


@pytest.mark.parametrize(("usable", "expected"), [(True, "cuda"), (False, "cpu")])
def test_resolve_xgb_params_defaults_to_the_auto_device(
    monkeypatch: pytest.MonkeyPatch, usable: bool, expected: str
) -> None:
    """Params resolved without a device name the concrete device ``auto`` picked."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: usable)

    params = xgb_utils._resolve_xgb_params({"max_depth": 3})

    assert params["device"] == expected
    if usable:
        assert params["tree_method"] == "hist"


def test_resolve_xgb_params_resolves_an_auto_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device passed through the overrides is resolved like the device argument."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)

    params = xgb_utils._resolve_xgb_params({"max_depth": 3}, overrides={"device": "auto"})

    assert params["device"] == "cuda"


def test_resolve_xgb_params_gpu_tree_method_without_a_gpu_uses_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU tree method asks for CUDA, which falls back to the CPU when no GPU is usable."""
    _reset_runtime_state()
    monkeypatch.setattr(xgb_utils, "_xgb_param_supported", lambda p: p in {"device", "predictor"})

    params = xgb_utils._resolve_xgb_params({"max_depth": 3}, tree_method="gpu", device="auto")

    assert params["device"] == "cpu"
    assert params["tree_method"] == "hist"


def test_coerce_tree_method_on_error_build_without_gpu_support() -> None:
    """A CUDA request on a build compiled without GPU support falls back to the CPU."""
    _reset_runtime_state()

    params = {"device": "cuda", "tree_method": "hist"}
    updated = xgb_utils._coerce_tree_method_on_error(
        params, xgb.core.XGBoostError("XGBoost version not compiled with GPU support.")
    )

    assert updated is not None
    assert updated["device"] == "cpu"


def test_gpu_fallback_is_logged_as_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training on the CPU after a CUDA failure is a warning, not routine information."""
    _reset_runtime_state()
    warnings_logged = _record_warnings(monkeypatch)

    xgb_utils._coerce_tree_method_on_error(
        {"device": "cuda", "tree_method": "hist"}, xgb.core.XGBoostError("CUDA error")
    )

    assert len(warnings_logged) == 1


def test_every_fit_time_fallback_warns_even_after_a_resolution_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit that falls back is never silent, whatever was warned about before it."""
    _reset_runtime_state()
    warnings_logged = _record_warnings(monkeypatch)
    cuda_params = {"device": "cuda", "tree_method": "hist"}

    xgb_utils.resolve_xgb_device("cuda")
    xgb_utils._coerce_tree_method_on_error(cuda_params, xgb.core.XGBoostError("CUDA error"))
    xgb_utils._coerce_tree_method_on_error(cuda_params, xgb.core.XGBoostError("CUDA error"))

    assert len(warnings_logged) == 3


def test_coerce_tree_method_on_error_replaces_a_gpu_tree_method_on_cuda() -> None:
    """The CPU retry never keeps a GPU tree method, which XGBoost would reject again."""
    _reset_runtime_state()

    updated = xgb_utils._coerce_tree_method_on_error(
        {"device": "cuda", "tree_method": "gpu_hist"},
        xgb.core.XGBoostError("Invalid Input: 'gpu_hist', valid values are: approx, exact, hist"),
    )

    assert updated is not None
    assert updated["device"] == "cpu"
    assert updated["tree_method"] == "hist"


@pytest.mark.parametrize(("usable", "expected"), [(True, "cuda"), (False, "cpu")])
def test_resolve_xgb_params_maps_a_gpu_tree_method_override_to_hist(
    monkeypatch: pytest.MonkeyPatch, usable: bool, expected: str
) -> None:
    """A ``gpu_hist`` override asks for CUDA with ``hist``, like the ``tree_method`` argument."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: usable)

    params = xgb_utils._resolve_xgb_params(
        {"max_depth": 3}, overrides={"tree_method": "gpu_hist", "device": "auto"}
    )

    assert params["tree_method"] == "hist"
    assert params["device"] == expected


def test_a_gpu_hist_override_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--xgb-tree-method gpu_hist`` trains instead of failing on XGBoost's rejection."""
    from nfl_predictor.ml import ml_model_core

    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)
    params = xgb_utils._resolve_xgb_params(
        {"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
        overrides={"tree_method": "gpu_hist", "device": "auto"},
    )
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 2))

    margin_model, _total_model = ml_model_core._fit_margin_total_models(
        x, x[:, 0], 40.0 + x[:, 1], params
    )

    assert margin_model.get_params()["tree_method"] == "hist"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("gpu", "cuda"), ("GPU", "cuda"), ("cuda:1", "cuda:1"), (" CPU ", "cpu")],
)
def test_resolve_xgb_device_normalizes_accepted_spellings(
    monkeypatch: pytest.MonkeyPatch, requested: str, expected: str
) -> None:
    """``gpu`` means ``cuda``, so the two never fingerprint differently."""
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)

    assert xgb_utils.resolve_xgb_device(requested) == expected


@pytest.mark.parametrize("requested", ["cdua", "gpu:0", "cuda:", "cuda:x", "tpu"])
def test_resolve_xgb_device_rejects_unknown_devices(requested: str) -> None:
    """A typo fails here instead of reaching XGBoost."""
    with pytest.raises(ValueError, match="XGBoost device"):
        xgb_utils.resolve_xgb_device(requested)


def test_xgb_device_arg_rejects_a_typo_as_an_argparse_error() -> None:
    """The command-line type reports a bad device as a usage error and keeps good ones."""
    import argparse

    assert xgb_utils.xgb_device_arg("auto") == "auto"
    assert xgb_utils.xgb_device_arg("gpu") == "cuda"
    with pytest.raises(argparse.ArgumentTypeError):
        xgb_utils.xgb_device_arg("cdua")


def test_fitted_xgb_device_reads_every_head() -> None:
    """Every head is read: one device when they agree, ``mixed`` when a head fell back."""
    from types import SimpleNamespace

    def heads(margin: str, total: str, quantile: str) -> SimpleNamespace:
        return SimpleNamespace(
            margin_model=xgb.XGBRegressor(device=margin),
            total_model=xgb.XGBRegressor(device=total),
            margin_quantile_models={0.1: xgb.XGBRegressor(device=quantile)},
            total_quantile_models=None,
        )

    assert xgb_utils.fitted_xgb_device(heads("cuda", "cuda", "cuda")) == "cuda"
    assert xgb_utils.fitted_xgb_device(heads("cuda", "cpu", "cuda")) == "mixed"
    assert xgb_utils.fitted_xgb_device(heads("cuda", "cuda", "cpu")) == "mixed"


def test_fitted_xgb_device_reads_the_fitted_estimator() -> None:
    """The device comes from the fitted estimator; a model without one records none."""
    from types import SimpleNamespace

    plain = SimpleNamespace(margin_model=xgb.XGBRegressor(device="cpu"))

    assert xgb_utils.fitted_xgb_device(plain) == "cpu"
    assert xgb_utils.fitted_xgb_device({"model": "stub"}) is None
