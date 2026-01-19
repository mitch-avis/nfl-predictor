"""XGBoost utility helpers for model training and inference.

This module centralizes version-/build-specific handling for XGBoost parameters,
GPU fallbacks, and early-stopping support. It is intentionally dependency-light
and is imported by higher-level training code.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from scipy.sparse import spmatrix

from nfl_predictor.utils.logger import log

xgb.set_config(verbosity=0)


@dataclass
class _RuntimeState:
    """Tracks one-time runtime fallbacks to keep logs concise."""

    early_stopping_fallback_logged: bool = False
    gpu_fallback_logged: bool = False
    gpu_tree_method_disabled: bool = False


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


def _xgb_fit_supports(param: str) -> bool:
    """Return True if `xgb.XGBRegressor.fit` accepts a given kwarg."""

    try:
        return param in inspect.signature(xgb.XGBRegressor.fit).parameters
    except (TypeError, ValueError):
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
    """Log the GPU fallback message at most once."""

    if _RUNTIME_STATE.gpu_fallback_logged:
        return
    log.info(message)
    _RUNTIME_STATE.gpu_fallback_logged = True


def _resolve_xgb_params(
    base_params: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
    tree_method: Optional[str] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve XGBoost params with build-aware device/tree_method handling."""

    params = base_params.copy()
    supports_device = _xgb_param_supported("device")
    supports_predictor = _xgb_param_supported("predictor")

    requested_device = device if device and device != "auto" else None
    if supports_device and requested_device:
        params["device"] = requested_device

    if tree_method and tree_method != "auto":
        resolved_tree_method = tree_method
        if "gpu" in tree_method:
            if supports_device:
                resolved_tree_method = "hist"
                if not requested_device:
                    params["device"] = "cuda"
            elif _RUNTIME_STATE.gpu_tree_method_disabled:
                resolved_tree_method = "hist"
        params["tree_method"] = resolved_tree_method
        if "gpu" in resolved_tree_method and supports_predictor:
            params["predictor"] = "gpu_predictor"

    if overrides:
        params.update(overrides)

    if supports_device and params.get("device") == "cuda":
        params.setdefault("tree_method", "hist")

    params.setdefault("eval_metric", "mae")
    return params


def _build_xgb_fit_kwargs(
    x_eval: Optional[np.ndarray | spmatrix],
    y_eval: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
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
    early_stopping_rounds: Optional[int],
) -> dict[str, Any]:
    """Attach early-stopping parameters for XGBoost versions that require init kwargs."""

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

    if _xgb_param_supported("callbacks"):
        early_stop_cls = getattr(getattr(xgb, "callback", None), "EarlyStopping", None)
        if early_stop_cls is not None:
            updated.setdefault(
                "callbacks",
                [early_stop_cls(rounds=int(early_stopping_rounds))],
            )
    return updated


def _coerce_tree_method_on_error(
    params: dict[str, Any],
    exc: Exception,
) -> Optional[dict[str, Any]]:
    """Coerce tree_method/device when XGBoost errors imply unsupported GPU settings."""

    tree_method = params.get("tree_method")
    message = str(exc)
    lower_message = message.lower()

    if params.get("device") == "cuda" and (
        "no visible gpu" in lower_message or "cuda" in lower_message
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
