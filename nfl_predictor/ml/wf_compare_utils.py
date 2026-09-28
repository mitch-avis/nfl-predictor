"""The key that names a walk-forward configuration in the weekly run's artifacts."""

from __future__ import annotations

from typing import Any

from nfl_predictor.ml import artifacts


def build_candidate_key(
    *,
    model_kind: str,
    feature_start: str,
    feature_end: str,
    market_mode: str,
    include_quantiles: bool,
    xgb_params_overrides: dict[str, Any] | None,
) -> str:
    """Return a stable, human-readable candidate key."""
    payload = {
        "model_kind": model_kind,
        "feature_start": feature_start,
        "feature_end": feature_end,
        "market_mode": market_mode,
        "include_quantiles": bool(include_quantiles),
        "xgb_params_overrides": xgb_params_overrides or {},
    }
    short_hash = artifacts.stable_short_hash(payload)

    quantiles = "on" if include_quantiles else "off"
    xgb_tag = _format_xgb_overrides(xgb_params_overrides or {})

    return (
        f"{model_kind}_fs-{feature_start}_fe-{feature_end}_mkt-{market_mode}"
        f"_quant-{quantiles}_xgb-{xgb_tag}_{short_hash}"
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
        value_str = _format_float(value) if isinstance(value, float) else str(value)
        parts.append(f"{label}{value_str}")
    if not parts:
        return "default"
    return "-".join(parts)
