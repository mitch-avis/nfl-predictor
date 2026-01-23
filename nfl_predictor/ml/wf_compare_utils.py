"""Helpers for walk-forward comparison candidate keys."""

from __future__ import annotations

from typing import Any

from nfl_predictor.ml import artifacts


def build_candidate_key(
    *,
    model_kind: str,
    feature_start: str,
    feature_end: str,
    calibration: str,
    market_mode: str,
    market_prob_source: str,
    market_prob_blend_method: str,
    win_prob_use_uncertainty: bool,
    market_prob_weight: float,
    market_prob_clamp: float,
    include_quantiles: bool,
    xgb_params_overrides: dict[str, Any] | None,
) -> str:
    """Return a stable, human-readable candidate key."""

    payload = {
        "model_kind": model_kind,
        "feature_start": feature_start,
        "feature_end": feature_end,
        "calibration": calibration,
        "market_mode": market_mode,
        "market_prob_source": market_prob_source,
        "market_prob_blend_method": market_prob_blend_method,
        "win_prob_use_uncertainty": bool(win_prob_use_uncertainty),
        "market_prob_weight": float(market_prob_weight),
        "market_prob_clamp": float(market_prob_clamp),
        "include_quantiles": bool(include_quantiles),
        "xgb_params_overrides": xgb_params_overrides or {},
    }
    short_hash = artifacts.stable_short_hash(payload)

    weight = _format_float(market_prob_weight)
    clamp = _format_float(market_prob_clamp)
    uncertainty = "on" if win_prob_use_uncertainty else "off"
    quantiles = "on" if include_quantiles else "off"
    xgb_tag = _format_xgb_overrides(xgb_params_overrides or {})

    return (
        f"{model_kind}_fs-{feature_start}_fe-{feature_end}_calib-{calibration}"
        f"_mkt-{market_mode}_src-{market_prob_source}_blend-{market_prob_blend_method}"
        f"_w{weight}_c{clamp}_uncert-{uncertainty}_quant-{quantiles}_xgb-{xgb_tag}_{short_hash}"
    )


def _format_float(value: float, *, precision: int = 3) -> str:
    """Format floats consistently for candidate keys."""

    return f"{float(value):.{precision}f}"


def _format_xgb_overrides(overrides: dict[str, Any]) -> str:
    """Format XGBoost override settings for candidate keys."""

    if not overrides:
        return "default"

    parts: list[str] = []
    mapping = {
        "n_estimators": "n",
        "max_depth": "md",
        "learning_rate": "lr",
        "subsample": "sub",
        "colsample_bytree": "col",
        "tree_method": "tm",
        "device": "dev",
        "n_jobs": "nj",
    }
    for key, label in mapping.items():
        if key not in overrides:
            continue
        value = overrides[key]
        if isinstance(value, float):
            value_str = _format_float(value)
        else:
            value_str = str(value)
        parts.append(f"{label}{value_str}")
    if not parts:
        return "default"
    return "-".join(parts)
