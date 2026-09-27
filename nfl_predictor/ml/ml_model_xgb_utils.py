"""XGBoost utility helpers for model training and inference.

This module centralizes version-/build-specific handling for XGBoost parameters,
GPU fallbacks, and early-stopping support. It is intentionally dependency-light
and is imported by higher-level training code.
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer

from nfl_predictor.utils.logger import log

xgb.set_config(verbosity=0)

XGB_DEVICE_AUTO = "auto"
XGB_DEVICE_HELP = (
    "XGBoost device: auto (the GPU when this XGBoost build has CUDA and a usable GPU is "
    "present, else the CPU), cpu, or cuda (default: auto)."
)


@dataclass
class _RuntimeState:
    """Tracks one-time runtime fallbacks to keep logs concise."""

    early_stopping_fallback_logged: bool = False
    gpu_fallback_logged: bool = False
    gpu_tree_method_disabled: bool = False
    auto_device_logged: bool = False


_RUNTIME_STATE = _RuntimeState()


def _predict_xgb(model: xgb.XGBRegressor, data: np.ndarray | spmatrix) -> np.ndarray:
    """Predict using an XGBoost sklearn model via Booster.predict.

    Uses `best_iteration` when present (early stopping) to match sklearn wrapper behavior
    across versions.
    """
    dmatrix = xgb.DMatrix(data)
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = None
    if best_iteration is not None:
        iteration_range = (0, best_iteration + 1)
    booster = model.get_booster()
    if iteration_range is not None:
        return booster.predict(dmatrix, iteration_range=iteration_range)
    return booster.predict(dmatrix)


def _fit_transform_matrix(
    preprocessor: ColumnTransformer,
    x: Any,
) -> np.ndarray | spmatrix:
    """Fit-transform ``x`` and narrow the sklearn output type for static analysis.

    ``ColumnTransformer.fit_transform()`` is inferred by pyright with a broad union
    type that does not directly satisfy ``np.ndarray | spmatrix``. This wrapper
    narrows the result so downstream functions receive the expected type.
    """
    return preprocessor.fit_transform(x)  # type: ignore[return-value]


def _transform_matrix(
    preprocessor: ColumnTransformer,
    x: Any,
) -> np.ndarray | spmatrix:
    """Transform ``x`` and narrow the sklearn output type for static analysis.

    ``ColumnTransformer.transform()`` is inferred by pyright with a broad union type
    that does not directly satisfy ``np.ndarray | spmatrix``. This wrapper narrows
    the result so downstream functions receive the expected type.
    """
    return preprocessor.transform(x)  # type: ignore[return-value]


def _xgb_fit_supports(param: str) -> bool:
    """Return True if `xgb.XGBRegressor.fit` accepts a given kwarg."""
    try:
        return param in inspect.signature(xgb.XGBRegressor.fit).parameters
    except TypeError, ValueError:
        return False


@lru_cache(maxsize=1)
def _xgb_supported_params() -> set[str]:
    """Return the supported sklearn init params for the installed XGBoost."""
    return set(xgb.XGBRegressor().get_params().keys())


def _xgb_param_supported(param: str) -> bool:
    """Return True if `xgb.XGBRegressor` supports a given init param."""
    return param in _xgb_supported_params()


def _log_early_stopping_fallback(message: str) -> None:
    """Log the early-stopping fallback message at most once."""
    if _RUNTIME_STATE.early_stopping_fallback_logged:
        return
    log.info(message)
    _RUNTIME_STATE.early_stopping_fallback_logged = True


def _log_gpu_fallback(message: str) -> None:
    """Warn about the GPU fallback at most once."""
    if _RUNTIME_STATE.gpu_fallback_logged:
        return
    log.warning(message)
    _RUNTIME_STATE.gpu_fallback_logged = True


def detect_xgb_cuda() -> bool:
    """Return True when this XGBoost build has CUDA and a GPU it can train on.

    A build without CUDA answers without touching a device. Otherwise a one-round fit on two
    rows asks for ``cuda`` and reads back the device XGBoost used: with no visible GPU it
    moves the fit to the CPU with a warning instead of raising, so only the reported device
    tells the two apart.
    """
    try:
        if not xgb.build_info().get("USE_CUDA", False):
            return False
    except xgb.core.XGBoostError:
        return False
    data = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0.0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            booster = xgb.train(
                {"device": "cuda", "tree_method": "hist", "verbosity": 0},
                data,
                num_boost_round=1,
            )
        except xgb.core.XGBoostError:
            return False
    config = json.loads(booster.save_config())
    device = config.get("learner", {}).get("generic_param", {}).get("device", "")
    return str(device).startswith("cuda")


@lru_cache(maxsize=1)
def xgb_cuda_usable() -> bool:
    """Return ``detect_xgb_cuda()``, probed once per process."""
    return detect_xgb_cuda()


def resolve_xgb_device(requested: str | None) -> str:
    """Return the concrete XGBoost device for a requested one.

    ``auto`` (or no device) becomes ``cuda`` when a usable GPU is present and ``cpu``
    otherwise. An explicit GPU request without a usable GPU becomes ``cpu`` with a warning,
    so the recorded device is the one the run trains on. Any other value is kept.
    """
    choice = (requested or XGB_DEVICE_AUTO).strip().lower()
    if choice == XGB_DEVICE_AUTO:
        device = "cuda" if xgb_cuda_usable() else "cpu"
        if not _RUNTIME_STATE.auto_device_logged:
            log.info("XGBoost device auto resolved to %s.", device)
            _RUNTIME_STATE.auto_device_logged = True
        return device
    if choice.startswith(("cuda", "gpu")) and not xgb_cuda_usable():
        _log_gpu_fallback(
            f"XGBoost device {choice} requested but no usable CUDA GPU was found; using the CPU."
        )
        return "cpu"
    return choice


def _resolve_xgb_params(
    base_params: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    tree_method: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Resolve XGBoost params with build-aware device/tree_method handling.

    The device always ends concrete: ``auto`` or no device goes through
    ``resolve_xgb_device``, and a legacy GPU tree method with no device asks for ``cuda``.
    """
    params = base_params.copy()
    supports_device = _xgb_param_supported("device")
    supports_predictor = _xgb_param_supported("predictor")

    requested_device = device if device and device != XGB_DEVICE_AUTO else None
    gpu_tree_method = bool(tree_method) and "gpu" in str(tree_method)
    if supports_device:
        if requested_device is None and gpu_tree_method:
            requested_device = "cuda"
        if requested_device:
            params["device"] = requested_device

    if tree_method and tree_method != "auto":
        resolved_tree_method = tree_method
        if gpu_tree_method and (supports_device or _RUNTIME_STATE.gpu_tree_method_disabled):
            resolved_tree_method = "hist"
        params["tree_method"] = resolved_tree_method
        if "gpu" in resolved_tree_method and supports_predictor:
            params["predictor"] = "gpu_predictor"

    if overrides:
        params.update(overrides)

    if supports_device:
        params["device"] = resolve_xgb_device(params.get("device"))
        if params["device"].startswith("cuda"):
            params.setdefault("tree_method", "hist")

    params.setdefault("eval_metric", "mae")
    return params


def _build_xgb_fit_kwargs(
    x_eval: np.ndarray | spmatrix | None,
    y_eval: np.ndarray | None,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    """Build kwargs for `XGBRegressor.fit` across XGBoost versions."""
    fit_kwargs: dict[str, Any] = {}
    if x_eval is None or y_eval is None:
        return fit_kwargs

    if _xgb_fit_supports("eval_set"):
        fit_kwargs["eval_set"] = [(x_eval, y_eval)]
    if _xgb_fit_supports("verbose"):
        fit_kwargs["verbose"] = False

    if not early_stopping_rounds:
        return fit_kwargs

    if _xgb_fit_supports("early_stopping_rounds"):
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    # Newer XGBoost sklearn wrappers moved early-stopping to model params (init kwargs).
    # We handle that in `_with_xgb_early_stopping_params`.
    return fit_kwargs


def _with_xgb_early_stopping_params(
    params: dict[str, Any],
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    """Attach early-stopping parameters for XGBoost versions that require init kwargs.

    Only the ``early_stopping_rounds`` init parameter is set. XGBoost builds a fresh
    ``EarlyStopping`` callback from it on every ``fit``, so each estimator stops on its own
    validation curve. An explicit callback object must not go into ``params``: every
    estimator built from the dict would share it, and its best score and patience counter
    would carry from one fit into the next (a total head fit after the margin head would
    start against the margin's best error and stop after one round).
    """
    if not early_stopping_rounds:
        return params
    if _xgb_fit_supports("early_stopping_rounds"):
        return params
    if not _xgb_param_supported("early_stopping_rounds"):
        _log_early_stopping_fallback(
            "XGBoost early stopping not supported by this version; continuing without it."
        )
        return params

    updated = params.copy()
    updated.setdefault("early_stopping_rounds", int(early_stopping_rounds))
    return updated


def _coerce_tree_method_on_error(
    params: dict[str, Any],
    exc: Exception,
) -> dict[str, Any] | None:
    """Coerce tree_method/device when XGBoost errors imply unsupported GPU settings."""
    tree_method = params.get("tree_method")
    message = str(exc)
    lower_message = message.lower()

    if str(params.get("device", "")).startswith("cuda") and (
        "gpu" in lower_message or "cuda" in lower_message
    ):
        new_params = params.copy()
        new_params["device"] = "cpu"
        new_params.pop("predictor", None)
        new_params.setdefault("tree_method", "hist")
        _RUNTIME_STATE.gpu_tree_method_disabled = True
        _log_gpu_fallback("CUDA device not available for XGBoost; using CPU hist instead.")
        return new_params

    if not tree_method or "gpu" not in str(tree_method):
        return None
    if "tree_method" not in lower_message and "invalid input" not in lower_message:
        return None

    new_params = params.copy()
    new_params["tree_method"] = "hist"
    new_params.pop("predictor", None)
    new_params.pop("device", None)
    _RUNTIME_STATE.gpu_tree_method_disabled = True
    _log_gpu_fallback(
        f"tree_method={tree_method} is not supported by this XGBoost build; using hist instead."
    )
    return new_params
