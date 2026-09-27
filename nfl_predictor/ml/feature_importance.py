"""Feature importance helpers for XGBoost-based models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    MarginTotalModel,
)
from nfl_predictor.utils.logger import log

xgb.set_config(verbosity=0)

SCHEMA_VERSION = 2
"""Version of the report layout; version 1 files (no ``schema_version`` key) summed ``gain``."""

COLUMN_MEASURES: tuple[str, ...] = ("gain", "total_gain", "weight")
"""XGBoost importance types recorded for every encoded column of each model head."""

BASE_MEASURES: tuple[str, ...] = ("total_gain", "weight")
"""Measures that stay meaningful when summed over encoded columns and over heads."""

MEASURES: dict[str, str] = {
    "gain": (
        "XGBoost importance_type='gain': average loss reduction per split on the encoded "
        "column. Recorded per encoded column only; a sum of averages is not a total."
    ),
    "total_gain": (
        "XGBoost importance_type='total_gain': loss reduction summed over every split "
        "(average gain times splits). base_features sums it over a base feature's encoded "
        "columns, and 'combined' over the margin and total heads; base features rank by it."
    ),
    "weight": (
        "XGBoost importance_type='weight': number of splits. base_features sums it like total_gain."
    ),
}
"""What each number in the report measures; written into every report."""


def resolve_feature_names(
    preprocessor: ColumnTransformer,
    model: xgb.XGBRegressor,
) -> list[str]:
    """Resolve output feature names for a fitted preprocessor/model pair."""
    names: list[str] | None = None
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            names = None

    n_features = getattr(model, "n_features_in_", None)
    if names and n_features is not None and len(names) != int(n_features):
        log.debug(
            "Feature name count mismatch (names=%d, model=%s); falling back to f0..",
            len(names),
            n_features,
        )
        names = None

    if names:
        return names

    if n_features is None:
        return []
    return [f"f{i}" for i in range(int(n_features))]


def build_feature_importance_report(model: Any) -> dict[str, Any] | None:
    """Build a feature-importance report for supported model types."""
    try:
        if isinstance(model, BlendedMarginTotalModel):
            team_report = _build_margin_total_report(model.team_model)
            if team_report is None:
                return None
            return {"model_kind": "blend", "components": {"team": team_report}}
        if isinstance(model, MarginTotalModel):
            report = _build_margin_total_report(model)
            if report is None:
                return None
            report["model_kind"] = "margin_total"
            return report
    except AttributeError as exc:
        log.debug("Skipping feature importance: %s", exc)
        return None
    return None


def _build_margin_total_report(model: MarginTotalModel) -> dict[str, Any] | None:
    """Build a feature-importance report for margin/total models."""
    return _build_report_from_models(
        model.preprocessor,
        {
            "margin": model.margin_model,
            "total": model.total_model,
        },
    )


def _build_report_from_models(
    preprocessor: ColumnTransformer,
    models: dict[str, xgb.XGBRegressor],
) -> dict[str, Any] | None:
    """Build a feature-importance report for labeled XGBoost models."""
    if not models:
        return None

    if not _models_support_importance(models.values()):
        log.debug("Skipping feature importance; model lacks XGBoost boosters.")
        return None

    first_model = next(iter(models.values()))
    feature_names = resolve_feature_names(preprocessor, first_model)

    model_importance: dict[str, dict[str, list[float]]] = {}
    for label, model in models.items():
        model_importance[label] = _build_model_importance(model, feature_names)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measures": dict(MEASURES),
        "feature_names": feature_names,
        "models": model_importance,
    }

    base_features = _build_base_features(preprocessor, feature_names, model_importance)
    if base_features is not None:
        report["base_features"] = base_features
    return report


def _build_model_importance(
    model: xgb.XGBRegressor,
    feature_names: Sequence[str],
) -> dict[str, list[float]]:
    """Compute one importance vector per XGBoost importance type for a model."""
    booster = model.get_booster()
    return {
        measure: _score_dict_to_list(booster.get_score(importance_type=measure), feature_names)
        for measure in COLUMN_MEASURES
    }


def _score_dict_to_list(
    scores: Mapping[str, float | Sequence[float]],
    feature_names: Sequence[str],
) -> list[float]:
    """Map an XGBoost importance dict into a list aligned to feature names."""
    values = [0.0] * len(feature_names)
    index_map = {name: idx for idx, name in enumerate(feature_names)}
    for key, value in scores.items():
        idx = index_map.get(key)
        if idx is None and key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
        if idx is None or idx >= len(values):
            continue
        values[idx] = _coerce_score_value(value)
    return values


def _coerce_score_value(value: float | Sequence[float]) -> float:
    """Convert score values to a float, summing sequences when needed."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(np.asarray(value, dtype=float).sum())
    return float(value)


def _models_support_importance(models: Iterable[object]) -> bool:
    """Return True when all models provide XGBoost booster access."""
    return all(hasattr(model, "get_booster") for model in models)


def _build_base_features(
    preprocessor: ColumnTransformer,
    feature_names: Sequence[str],
    model_importance: dict[str, dict[str, list[float]]],
) -> dict[str, Any] | None:
    """Aggregate total gain and split counts by base (pre-encoded) feature."""
    base_map = _build_base_feature_map(preprocessor, feature_names)
    if not base_map:
        return None

    base_keys: set[str] = set()
    base_values: dict[str, dict[str, dict[str, float]]] = {}
    for label, importance in model_importance.items():
        base_values[label] = {
            measure: _aggregate_by_base(feature_names, importance[measure], base_map)
            for measure in BASE_MEASURES
        }
        for aggregated in base_values[label].values():
            base_keys.update(aggregated.keys())

    base_names = sorted(base_keys)
    base_features: dict[str, Any] = {"feature_names": base_names}
    for label, values in base_values.items():
        base_features[label] = {
            measure: [values[measure].get(name, 0.0) for name in base_names]
            for measure in BASE_MEASURES
        }

    if len(base_values) > 1:
        base_features["combined"] = {
            measure: [
                float(sum(base_features[label][measure][idx] for label in base_values))
                for idx in range(len(base_names))
            ]
            for measure in BASE_MEASURES
        }

    return base_features


def _aggregate_by_base(
    feature_names: Sequence[str],
    values: Sequence[float],
    base_map: dict[str, str],
) -> dict[str, float]:
    """Aggregate aligned values by base feature name."""
    aggregated: dict[str, float] = {}
    for name, value in zip(feature_names, values, strict=False):
        base = base_map.get(name, name)
        aggregated[base] = aggregated.get(base, 0.0) + float(value)
    return aggregated


def _build_base_feature_map(
    preprocessor: ColumnTransformer,
    feature_names: Sequence[str],
) -> dict[str, str] | None:
    """Map transformed feature names to their base column names."""
    if not hasattr(preprocessor, "transformers_"):
        return None
    if not hasattr(preprocessor, "get_feature_names_out"):
        return None

    try:
        output_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None

    if list(feature_names) != output_names:
        return None

    mapping: dict[str, str] = {}
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer == "drop":
            continue

        cols_list = _normalize_cols(cols, preprocessor)
        if not cols_list:
            continue

        if transformer == "passthrough":
            for col in cols_list:
                mapping[col] = col
            continue

        onehot = getattr(transformer, "named_steps", {}).get("onehot")
        if onehot is not None and getattr(onehot, "categories_", None) is not None:
            for col, categories in zip(cols_list, onehot.categories_, strict=False):
                for category in categories:
                    mapping[f"{col}_{category}"] = col
            continue

        for col in cols_list:
            mapping[col] = col

    base_map: dict[str, str] = {}
    for output_name in output_names:
        rest = output_name.split("__", 1)[-1]
        base_map[output_name] = mapping.get(rest, rest)
    return base_map


def _normalize_cols(
    cols: Any,
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Normalize column selectors to a list of string names."""
    if isinstance(cols, slice):
        names_in = getattr(preprocessor, "feature_names_in_", None)
        if names_in is None:
            return []
        return [str(col) for col in names_in[cols]]
    if isinstance(cols, (list, tuple, np.ndarray)):
        return [str(col) for col in cols]
    return [str(cols)]
