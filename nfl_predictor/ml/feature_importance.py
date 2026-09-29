"""Feature importance helpers for XGBoost-based models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb

from nfl_predictor.ml.feature_spec import apply_feature_spec
from nfl_predictor.ml.ml_model_core import (
    MarginTotalModel,
)
from nfl_predictor.ml.ml_model_xgb_utils import transform_matrix
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import spmatrix
    from sklearn.compose import ColumnTransformer

xgb.set_config(verbosity=0)

SCHEMA_VERSION = 3
"""Version of the report layout.

Version 3 adds ``mean_abs_shap`` and the ``shap`` block when the report explains rows; version 2
recorded only the XGBoost measures; version 1 files (no ``schema_version`` key) summed ``gain``.
"""

SHAP_MEASURE = "mean_abs_shap"
"""Key of the SHAP measure under ``models``, ``base_features`` and ``measures``."""

SHAP_ROWS = "train"
"""Which rows the SHAP values explain: the final model's own tree-training rows."""

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
"""What each XGBoost number in the report measures; written into every report."""

SHAP_DESCRIPTION = (
    "Mean absolute SHAP value in points (XGBoost pred_contribs, exact TreeSHAP) over the rows "
    "the 'shap' block names: the final model's own tree-training rows, which exclude the "
    "calibration and holdout rows. A base feature's value sums each row's signed contributions "
    "over its encoded columns before the absolute value, and 'combined' adds the margin and "
    "total heads' values. Each head is explained on its raw output, which is the residual over "
    "the market line when the model is market-anchored. Under 'models', the mean absolute value "
    "of each encoded column alone. Base features rank by it when it is present."
)
"""What the SHAP number measures; written into reports that explain rows."""


def resolve_feature_names(
    preprocessor: ColumnTransformer,
    model: xgb.XGBRegressor,
) -> list[str]:
    """Resolve output feature names for a fitted preprocessor/model pair."""
    names: list[str] | None = None
    if hasattr(preprocessor, "get_feature_names_out"):
        # sklearn raises NotFittedError (a ValueError) before fitting and AttributeError when a
        # transformer cannot name its outputs.
        try:
            names = [str(name) for name in preprocessor.get_feature_names_out()]
        except AttributeError, ValueError:
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


def shap_values(
    model: xgb.XGBRegressor,
    data: np.ndarray | spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one head's SHAP values per row and encoded column, and each row's base value.

    Uses XGBoost's exact TreeSHAP (``pred_contribs``) over the trees ``predict_xgb`` uses, so
    each row's values plus its base value reproduce the head's prediction.
    """
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)
    contributions = np.asarray(
        model.get_booster().predict(
            xgb.DMatrix(data), pred_contribs=True, iteration_range=iteration_range
        ),
        dtype=float,
    )
    return contributions[:, :-1], contributions[:, -1]


def mean_abs_shap_by_base(
    values: np.ndarray,
    feature_names: Sequence[str],
    base_map: Mapping[str, str],
) -> dict[str, float]:
    """Return mean |SHAP| per base feature, summing each row's signed column values first.

    A one-hot feature's columns can push one row in opposite directions; summing them before
    the absolute value measures what the base feature did to the prediction.
    """
    bases = [base_map.get(name, name) for name in feature_names]
    base_names = list(dict.fromkeys(bases))
    position = {name: idx for idx, name in enumerate(base_names)}
    indicator = np.zeros((len(bases), len(base_names)))
    indicator[np.arange(len(bases)), [position[base] for base in bases]] = 1.0
    per_base = np.abs(values @ indicator).mean(axis=0)
    return {name: float(per_base[idx]) for idx, name in enumerate(base_names)}


def build_feature_importance_report(
    model: object,
    shap_rows: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Build a feature-importance report for supported model types.

    ``shap_rows`` are the games whose predictions SHAP explains (the final model's training
    rows); without them, or when they are empty, the report records only the XGBoost measures.
    """
    try:
        if isinstance(model, MarginTotalModel):
            report = _build_margin_total_report(model, shap_rows)
            if report is None:
                return None
            report["model_kind"] = "margin_total"
            return report
    except AttributeError as exc:
        log.debug("Skipping feature importance: %s", exc)
        return None
    return None


def _build_margin_total_report(
    model: MarginTotalModel,
    shap_rows: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Build a feature-importance report for margin/total models."""
    if shap_rows is not None and shap_rows.empty:
        shap_rows = None
    shap_data = None
    if shap_rows is not None:
        shap_data = transform_matrix(
            model.preprocessor, apply_feature_spec(shap_rows, model.feature_spec)
        )
    report = _build_report_from_models(
        model.preprocessor,
        {
            "margin": model.margin_model,
            "total": model.total_model,
        },
        shap_data=shap_data,
    )
    if report is not None and shap_rows is not None:
        report["shap"] = _shap_rows_record(shap_rows)
    return report


def _shap_rows_record(rows: pd.DataFrame) -> dict[str, Any]:
    """Describe the rows SHAP explained: which set, how many and their season range."""
    record: dict[str, Any] = {"rows": SHAP_ROWS, "row_count": len(rows)}
    if "season" in rows.columns:
        record["seasons"] = [int(rows["season"].min()), int(rows["season"].max())]
    return record


def _build_report_from_models(
    preprocessor: ColumnTransformer,
    models: dict[str, xgb.XGBRegressor],
    shap_data: np.ndarray | spmatrix | None = None,
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
    shap_by_head: dict[str, np.ndarray] = {}
    for label, model in models.items():
        model_importance[label] = _build_model_importance(model, feature_names)
        if shap_data is not None:
            values, _ = shap_values(model, shap_data)
            shap_by_head[label] = values
            model_importance[label][SHAP_MEASURE] = np.abs(values).mean(axis=0).tolist()

    measures = dict(MEASURES)
    if shap_by_head:
        measures[SHAP_MEASURE] = SHAP_DESCRIPTION
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measures": measures,
        "feature_names": feature_names,
        "models": model_importance,
    }

    base_features = _build_base_features(
        preprocessor, feature_names, model_importance, shap_by_head
    )
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
    shap_by_head: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    """Aggregate total gain, split counts and (when given) mean |SHAP| by base feature."""
    base_map = _build_base_feature_map(preprocessor, feature_names)
    if not base_map:
        return None

    measures = BASE_MEASURES + ((SHAP_MEASURE,) if shap_by_head else ())
    base_keys: set[str] = set()
    base_values: dict[str, dict[str, dict[str, float]]] = {}
    for label, importance in model_importance.items():
        base_values[label] = {
            measure: _aggregate_by_base(feature_names, importance[measure], base_map)
            for measure in BASE_MEASURES
        }
        if shap_by_head:
            base_values[label][SHAP_MEASURE] = mean_abs_shap_by_base(
                shap_by_head[label], feature_names, base_map
            )
        for aggregated in base_values[label].values():
            base_keys.update(aggregated.keys())

    base_names = sorted(base_keys)
    base_features: dict[str, Any] = {"feature_names": base_names}
    for label, values in base_values.items():
        base_features[label] = {
            measure: [values[measure].get(name, 0.0) for name in base_names] for measure in measures
        }

    if len(base_values) > 1:
        base_features["combined"] = {
            measure: [
                float(sum(base_features[label][measure][idx] for label in base_values))
                for idx in range(len(base_names))
            ]
            for measure in measures
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


def _output_names(preprocessor: ColumnTransformer) -> list[str] | None:
    """Return a fitted preprocessor's output feature names, or None when it cannot name them."""
    if not hasattr(preprocessor, "transformers_"):
        return None
    if not hasattr(preprocessor, "get_feature_names_out"):
        return None
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except AttributeError, ValueError:
        return None


def _transformer_base_columns(transformer: object, columns: list[str]) -> dict[str, str]:
    """Map one transformer's output columns to its input columns.

    A one-hot step names each category's column ``<column>_<category>``; every other
    transformer keeps its input column names.
    """
    if transformer != "passthrough":
        onehot = getattr(transformer, "named_steps", {}).get("onehot")
        if onehot is not None and getattr(onehot, "categories_", None) is not None:
            return {
                f"{col}_{category}": col
                for col, categories in zip(columns, onehot.categories_, strict=False)
                for category in categories
            }
    return {col: col for col in columns}


def _build_base_feature_map(
    preprocessor: ColumnTransformer,
    feature_names: Sequence[str],
) -> dict[str, str] | None:
    """Map transformed feature names to their base column names."""
    output_names = _output_names(preprocessor)
    if output_names is None or list(feature_names) != output_names:
        return None

    mapping: dict[str, str] = {}
    for _name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue
        cols_list = _normalize_cols(cols, preprocessor)
        if cols_list:
            mapping.update(_transformer_base_columns(transformer, cols_list))

    base_map: dict[str, str] = {}
    for output_name in output_names:
        rest = output_name.split("__", 1)[-1]
        base_map[output_name] = mapping.get(rest, rest)
    return base_map


def _normalize_cols(
    cols: object,
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
