"""Feature importance helpers for XGBoost-based models."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml.ml_model_core import BlendedMarginTotalModel, MarginTotalModel, ScoreModel
from nfl_predictor.utils.logger import log

xgb.set_config(verbosity=0)


def resolve_feature_names(
    preprocessor: ColumnTransformer,
    model: xgb.XGBRegressor,
) -> list[str]:
    """Resolve output feature names for a fitted preprocessor/model pair."""

    names: Optional[list[str]] = None
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


def build_feature_importance_report(model: Any) -> Optional[dict[str, Any]]:
    """Build a feature-importance report for supported model types."""

    try:
        if isinstance(model, BlendedMarginTotalModel):
            team_report = _build_margin_total_report(model.team_model)
            if team_report is None:
                return None
            market_report = None
            if model.market_model is not None:
                market_report = _build_margin_total_report(model.market_model)
            return {
                "model_kind": "blend",
                "components": {
                    "team": team_report,
                    "market": market_report,
                },
            }
        if isinstance(model, MarginTotalModel):
            report = _build_margin_total_report(model)
            if report is None:
                return None
            report["model_kind"] = "margin_total"
            return report
        if isinstance(model, ScoreModel):
            report = _build_score_report(model)
            if report is None:
                return None
            report["model_kind"] = "score"
            return report
    except AttributeError as exc:
        log.debug("Skipping feature importance: %s", exc)
        return None
    return None


def _build_margin_total_report(model: MarginTotalModel) -> Optional[dict[str, Any]]:
    """Build a feature-importance report for margin/total models."""
    return _build_report_from_models(
        model.preprocessor,
        {
            "margin": model.margin_model,
            "total": model.total_model,
        },
    )


def _build_score_report(model: ScoreModel) -> Optional[dict[str, Any]]:
    """Build a feature-importance report for score models."""
    return _build_report_from_models(
        model.preprocessor,
        {
            "away": model.away_model,
            "home": model.home_model,
        },
    )


def _build_report_from_models(
    preprocessor: ColumnTransformer,
    models: dict[str, xgb.XGBRegressor],
) -> Optional[dict[str, Any]]:
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
    """Compute gain/weight importance vectors for a model."""
    booster = model.get_booster()
    return {
        "gain": _score_dict_to_list(booster.get_score(importance_type="gain"), feature_names),
        "weight": _score_dict_to_list(booster.get_score(importance_type="weight"), feature_names),
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
        return float(sum(float(item) for item in value))
    return float(value)


def _models_support_importance(models: Iterable[object]) -> bool:
    """Return True when all models provide XGBoost booster access."""
    return all(hasattr(model, "get_booster") for model in models)


def _build_base_features(
    preprocessor: ColumnTransformer,
    feature_names: Sequence[str],
    model_importance: dict[str, dict[str, list[float]]],
) -> Optional[dict[str, Any]]:
    """Aggregate importance values by base (pre-encoded) feature."""
    base_map = _build_base_feature_map(preprocessor, feature_names)
    if not base_map:
        return None

    base_keys: set[str] = set()
    base_values: dict[str, dict[str, dict[str, float]]] = {}
    for label, importance in model_importance.items():
        gain_map = _aggregate_by_base(feature_names, importance["gain"], base_map)
        weight_map = _aggregate_by_base(feature_names, importance["weight"], base_map)
        base_keys.update(gain_map.keys())
        base_keys.update(weight_map.keys())
        base_values[label] = {"gain": gain_map, "weight": weight_map}

    base_names = sorted(base_keys)
    base_features: dict[str, Any] = {"feature_names": base_names}
    for label, values in base_values.items():
        base_features[label] = {
            "gain": [values["gain"].get(name, 0.0) for name in base_names],
            "weight": [values["weight"].get(name, 0.0) for name in base_names],
        }

    if len(base_values) > 1:
        combined_gain = [
            float(sum(base_features[label]["gain"][idx] for label in base_values))
            for idx in range(len(base_names))
        ]
        combined_weight = [
            float(sum(base_features[label]["weight"][idx] for label in base_values))
            for idx in range(len(base_names))
        ]
        base_features["combined"] = {"gain": combined_gain, "weight": combined_weight}

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
) -> Optional[dict[str, str]]:
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
