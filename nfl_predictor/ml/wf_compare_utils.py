"""The key that names a walk-forward configuration in the weekly run's artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nfl_predictor.ml import artifacts


@dataclass(frozen=True, kw_only=True)
class CandidateSpec:
    """The settings a walk-forward configuration's candidate key is built from."""

    model_kind: str
    feature_start: str
    feature_end: str
    market_mode: str
    include_quantiles: bool
    xgb_params_overrides: dict[str, Any] | None


def build_candidate_key(spec: CandidateSpec) -> str:
    """Return a stable, human-readable candidate key."""
    overrides = spec.xgb_params_overrides or {}
    payload = {
        "model_kind": spec.model_kind,
        "feature_start": spec.feature_start,
        "feature_end": spec.feature_end,
        "market_mode": spec.market_mode,
        "include_quantiles": bool(spec.include_quantiles),
        "xgb_params_overrides": overrides,
    }
    short_hash = artifacts.stable_short_hash(payload)

    quantiles = "on" if spec.include_quantiles else "off"
    xgb_tag = _format_xgb_overrides(overrides)

    return (
        f"{spec.model_kind}_fs-{spec.feature_start}_fe-{spec.feature_end}"
        f"_mkt-{spec.market_mode}_quant-{quantiles}_xgb-{xgb_tag}_{short_hash}"
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
