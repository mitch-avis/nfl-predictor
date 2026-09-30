"""XGBoost utility helpers for model training and inference.

This module centralizes version-/build-specific handling for XGBoost parameters,
GPU fallbacks, and early-stopping support. It is intentionally dependency-light
and is imported by higher-level training code.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb
from scipy.sparse import spmatrix

from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.compose import ColumnTransformer

xgb.set_config(verbosity=0)

XGB_DEVICE_AUTO = "auto"
# Recorded when a model's XGBoost heads trained on different devices.
XGB_DEVICE_MIXED = "mixed"
XGB_DEVICE_HELP = (
    "XGBoost device: auto (the GPU when this XGBoost build has CUDA and a usable GPU is "
    "present, else the CPU), cpu, cuda or cuda:N; gpu means cuda (default: auto)."
)
_XGB_DEVICE_PATTERN = re.compile(r"auto|cpu|cuda(:\d+)?")


def normalize_xgb_device(value: str | None) -> str:
    """Return the canonical spelling of a requested XGBoost device.

    Accepts ``auto``, ``cpu``, ``cuda``, ``cuda:N`` and ``gpu`` (which means ``cuda``), in any
    case; no value means ``auto``. Anything else raises ``ValueError``, so a typo never reaches
    XGBoost and ``gpu`` never fingerprints differently from ``cuda``.
    """
    choice = (value or XGB_DEVICE_AUTO).strip().lower()
    if choice == "gpu":
        choice = "cuda"
    if not _XGB_DEVICE_PATTERN.fullmatch(choice):
        msg = f"Unknown XGBoost device {value!r}; use auto, cpu, cuda or cuda:N (gpu means cuda)."
        raise ValueError(msg)
    return choice


def xgb_device_arg(value: str) -> str:
    """Parse an ``--xgb-device`` value, reporting an unknown device as a usage error."""
    try:
        return normalize_xgb_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass
class _RuntimeState:
    """Tracks one-time runtime fallbacks to keep logs concise."""

    early_stopping_fallback_logged: bool = False
    gpu_fallback_logged: bool = False
    gpu_tree_method_disabled: bool = False
    auto_device_logged: bool = False


_RUNTIME_STATE = _RuntimeState()


def predict_xgb(model: xgb.XGBRegressor, data: np.ndarray | spmatrix) -> np.ndarray:
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


def _as_matrix(result: object) -> np.ndarray | spmatrix:
    """Narrow a ``ColumnTransformer`` output to the dense array or sparse matrix it is.

    sklearn types the output as a broad union; the preprocessor this project builds returns
    one of these two, and anything else is a bug worth failing on.
    """
    if isinstance(result, np.ndarray | spmatrix):
        return result
    msg = f"Expected an array or sparse matrix from the preprocessor, got {type(result).__name__}"
    raise TypeError(msg)


def fit_transform_matrix(preprocessor: ColumnTransformer, x: pd.DataFrame) -> np.ndarray | spmatrix:
    """Fit-transform ``x`` and return the matrix the model trains on."""
    return _as_matrix(preprocessor.fit_transform(x))


def transform_matrix(preprocessor: ColumnTransformer, x: pd.DataFrame) -> np.ndarray | spmatrix:
    """Transform ``x`` into the matrix the model predicts from."""
    return _as_matrix(preprocessor.transform(x))


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
    so the recorded device is the one the run trains on. The spelling is normalized first
    (``normalize_xgb_device``), so an unknown device raises ``ValueError``.
    """
    choice = normalize_xgb_device(requested)
    if choice == XGB_DEVICE_AUTO:
        device = "cuda" if xgb_cuda_usable() else "cpu"
        if not _RUNTIME_STATE.auto_device_logged:
            log.info("XGBoost device auto resolved to %s.", device)
            _RUNTIME_STATE.auto_device_logged = True
        return device
    if choice.startswith("cuda") and not xgb_cuda_usable():
        _log_gpu_fallback(
            f"XGBoost device {choice} requested but no usable CUDA GPU was found; using the CPU."
        )
        return "cpu"
    return choice


def _requested_device(device: str | None, *, gpu_tree_method: bool) -> str | None:
    """Return the device a caller named, or ``cuda`` for a legacy GPU tree method."""
    requested = device if device and device != XGB_DEVICE_AUTO else None
    if requested is None and gpu_tree_method:
        return "cuda"
    return requested


def _apply_tree_method(
    params: dict[str, Any], tree_method: str, *, supports_device: bool, supports_predictor: bool
) -> None:
    """Set the tree method, mapping a GPU one to ``hist`` where XGBoost takes a device."""
    resolved_tree_method = tree_method
    if "gpu" in tree_method and (supports_device or _RUNTIME_STATE.gpu_tree_method_disabled):
        resolved_tree_method = "hist"
    params["tree_method"] = resolved_tree_method
    if "gpu" in resolved_tree_method and supports_predictor:
        params["predictor"] = "gpu_predictor"


def _finalize_device(params: dict[str, Any]) -> None:
    """Map a GPU tree method in the final params to ``hist`` on CUDA and resolve the device."""
    if "gpu" in str(params.get("tree_method", "")):
        params["tree_method"] = "hist"
        params.pop("predictor", None)
        if normalize_xgb_device(params.get("device")) == XGB_DEVICE_AUTO:
            params["device"] = "cuda"
    params["device"] = resolve_xgb_device(params.get("device"))
    if params["device"].startswith("cuda"):
        params.setdefault("tree_method", "hist")


def resolve_xgb_params(
    base_params: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    tree_method: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Resolve XGBoost params with build-aware device/tree_method handling.

    The device always ends concrete: ``auto`` or no device goes through
    ``resolve_xgb_device``. A legacy GPU tree method (``gpu_hist``), whether passed as
    ``tree_method`` or in ``overrides``, becomes ``hist`` and asks for ``cuda`` unless a
    device is named, because current XGBoost rejects the ``gpu_*`` names.
    """
    params = base_params.copy()
    supports_device = _xgb_param_supported("device")
    supports_predictor = _xgb_param_supported("predictor")

    if supports_device:
        gpu_tree_method = bool(tree_method) and "gpu" in str(tree_method)
        requested_device = _requested_device(device, gpu_tree_method=gpu_tree_method)
        if requested_device:
            params["device"] = requested_device

    if tree_method and tree_method != "auto":
        _apply_tree_method(
            params,
            tree_method,
            supports_device=supports_device,
            supports_predictor=supports_predictor,
        )

    if overrides:
        params.update(overrides)

    if supports_device:
        _finalize_device(params)

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
    """Coerce tree_method/device when XGBoost errors imply unsupported GPU settings.

    Every fallback is logged as a warning: each one retrains a fit on the CPU, which a run on
    CUDA must not pass over silently.
    """
    tree_method = params.get("tree_method")
    message = str(exc)
    lower_message = message.lower()

    if str(params.get("device", "")).startswith("cuda") and (
        "gpu" in lower_message or "cuda" in lower_message
    ):
        new_params = params.copy()
        new_params["device"] = "cpu"
        new_params.pop("predictor", None)
        if not tree_method or "gpu" in str(tree_method):
            new_params["tree_method"] = "hist"
        _RUNTIME_STATE.gpu_tree_method_disabled = True
        log.warning("XGBoost fit failed on CUDA (%s); retrying on the CPU.", message)
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
    log.warning(
        "tree_method=%s is not supported by this XGBoost build; using hist instead.", tree_method
    )
    return new_params


def fitted_xgb_device(model: object) -> str | None:
    """Return the device a fitted model's XGBoost heads were trained on.

    Reads every head (margin, total and any quantile models) from the estimators themselves,
    so a fit that fell back to the CPU is recorded as ``cpu``, and heads that trained on
    different devices are recorded as ``mixed``. Returns None when the model holds no XGBoost
    estimator.
    """
    heads: list[object] = [
        getattr(model, "margin_model", None),
        getattr(model, "total_model", None),
    ]
    for attribute in ("margin_quantile_models", "total_quantile_models"):
        heads.extend((getattr(model, attribute, None) or {}).values())
    devices = {
        estimator.get_params().get("device")
        for estimator in heads
        if isinstance(estimator, xgb.XGBModel)
    }
    if not devices:
        return None
    if len(devices) > 1:
        return XGB_DEVICE_MIXED
    (device,) = devices
    return None if device is None else str(device)
