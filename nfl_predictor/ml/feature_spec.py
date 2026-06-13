"""Feature specification helpers for ML preprocessing."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

DEFAULT_FEATURE_START_COLUMN = "away_rest"
DEFAULT_FEATURE_END_COLUMN = "home_moneyline"

MARKET_DERIVED_COLUMNS = (
    "market_home_margin",
    "market_total_line",
    "home_market_prob",
    "away_market_prob",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Feature metadata for model training and inference."""

    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    dropped_columns: list[str]
    id_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    feature_start: str
    feature_end: str
    metadata_columns: list[str]
    post_feature_columns: list[str]
    market_columns: list[str]


def _get_feature_range_columns(
    df: pd.DataFrame, feature_start: str, feature_end: str
) -> tuple[list[str], list[str], list[str]]:
    """Return feature range columns plus leading/trailing metadata groups."""
    columns = df.columns.tolist()
    if feature_start not in columns or feature_end not in columns:
        raise ValueError(
            f"Expected feature range columns '{feature_start}'..'{feature_end}' in dataset."
        )
    start_idx = columns.index(feature_start)
    end_idx = columns.index(feature_end)
    if start_idx > end_idx:
        raise ValueError(f"Feature start column '{feature_start}' occurs after '{feature_end}'.")
    feature_range = columns[start_idx : end_idx + 1]
    metadata_columns = columns[:start_idx]
    post_feature_columns = columns[end_idx + 1 :]
    return feature_range, metadata_columns, post_feature_columns


def _implied_prob_from_moneyline(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert moneyline values into implied win probabilities."""
    if isinstance(values, pd.Series):
        moneyline_series = pd.to_numeric(values, errors="coerce")
    else:
        moneyline_series = pd.to_numeric(pd.Series(values), errors="coerce")
    moneyline = moneyline_series.to_numpy(dtype=float)
    probs = np.full_like(moneyline, np.nan, dtype=float)
    neg_mask = moneyline < 0
    pos_mask = moneyline > 0
    probs[neg_mask] = -moneyline[neg_mask] / (-moneyline[neg_mask] + 100)
    probs[pos_mask] = 100 / (moneyline[pos_mask] + 100)
    return probs


def _add_market_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived market columns exist on the input DataFrame."""
    df = df.copy()
    if "market_home_margin" not in df.columns:
        if "home_spread" in df.columns:
            df["market_home_margin"] = -pd.to_numeric(df["home_spread"], errors="coerce")
        elif "away_spread" in df.columns:
            df["market_home_margin"] = pd.to_numeric(df["away_spread"], errors="coerce")
    if "market_total_line" not in df.columns and "total_line" in df.columns:
        df["market_total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    if "home_market_prob" not in df.columns and "home_moneyline" in df.columns:
        df["home_market_prob"] = _implied_prob_from_moneyline(df["home_moneyline"])
    if "away_market_prob" not in df.columns and "away_moneyline" in df.columns:
        df["away_market_prob"] = _implied_prob_from_moneyline(df["away_moneyline"])
    return df


def get_market_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return market baseline margin and total arrays from input DataFrame."""
    df = _add_market_transforms(df)
    if "market_home_margin" not in df.columns or "market_total_line" not in df.columns:
        raise ValueError("Market anchor requested but spread/total columns are missing.")
    baseline_margin = pd.to_numeric(df["market_home_margin"], errors="coerce").to_numpy(dtype=float)
    baseline_total = pd.to_numeric(df["market_total_line"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(baseline_margin).any() or np.isnan(baseline_total).any():
        raise ValueError("Market anchor requested but spread/total contains missing values.")
    return baseline_margin, baseline_total


def _drop_identifier_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove identifier columns (suffix _id) from a DataFrame."""
    id_columns = [col for col in df.columns if col.endswith("_id")]
    if not id_columns:
        return df, []
    return df.drop(columns=id_columns), id_columns


def _drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with zero variance."""
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if not constant_cols:
        return df, []
    return df.drop(columns=constant_cols), constant_cols


def _drop_high_cardinality_columns(
    df: pd.DataFrame, max_cardinality_ratio: float
) -> tuple[pd.DataFrame, list[str]]:
    """Drop categorical columns with high cardinality ratios."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    dropped = []
    row_count = max(len(df), 1)
    for col in cat_cols:
        unique_ratio = df[col].nunique(dropna=True) / row_count
        if unique_ratio >= max_cardinality_ratio:
            dropped.append(col)
    if not dropped:
        return df, []
    return df.drop(columns=dropped), dropped


def _build_feature_spec(
    df: pd.DataFrame,
    include_market: bool,
    max_cardinality_ratio: float,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
    market_only: bool = False,
    market_transform: bool = False,
    disable_pruning: bool = False,
) -> FeatureSpec:
    """Build a FeatureSpec describing selected and dropped columns."""
    if market_transform:
        df = _add_market_transforms(df)

    feature_range, metadata_columns, post_feature_columns = _get_feature_range_columns(
        df, feature_start, feature_end
    )

    result_columns = [col for col in constants.RESULT_COLUMNS if col in df.columns]
    drop_columns = set(result_columns)

    derived_market_columns = [col for col in MARKET_DERIVED_COLUMNS if col in df.columns]
    if market_transform:
        feature_range = feature_range + [
            col for col in derived_market_columns if col not in feature_range
        ]

    raw_market_columns = [col for col in feature_range if col in constants.LINES_COLUMNS]
    market_feature_columns = derived_market_columns if market_transform else raw_market_columns
    excluded_market_columns: list[str] = []
    dropped_raw_market_columns: list[str] = []
    if market_only:
        selected_columns = market_feature_columns
    else:
        if market_transform and raw_market_columns:
            drop_columns.update(raw_market_columns)
            dropped_raw_market_columns = raw_market_columns
        if not include_market:
            excluded_market_columns = market_feature_columns
            drop_columns.update(excluded_market_columns)
        selected_columns = [col for col in feature_range if col not in drop_columns]

    pruned_columns: list[str] = []
    if not disable_pruning:
        pruned_columns = [
            col for col in constants.PRUNED_FEATURE_COLUMNS if col in selected_columns
        ]
        if pruned_columns:
            drop_columns.update(pruned_columns)
            selected_columns = [col for col in selected_columns if col not in pruned_columns]

    if market_only and not selected_columns:
        raise ValueError("Market-only model requested but no market columns were found.")
    feature_df = df[selected_columns].copy()

    feature_df, id_columns = _drop_identifier_columns(feature_df)
    feature_df, constant_columns = _drop_constant_columns(feature_df)
    feature_df, high_cardinality_columns = _drop_high_cardinality_columns(
        feature_df, max_cardinality_ratio
    )

    categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [col for col in feature_df.columns if col not in categorical_columns]

    if market_only:
        log.debug("Market-only feature selection enabled.")
    if market_transform:
        log.debug("Market feature transforms enabled: %s", derived_market_columns)
    log.debug("Dropped metadata columns (%d): %s", len(metadata_columns), metadata_columns)
    log.debug(
        "Dropped post-feature columns (%d): %s",
        len(post_feature_columns),
        post_feature_columns,
    )
    if result_columns:
        log.debug(
            "Target/result columns present (%d): %s",
            len(result_columns),
            result_columns,
        )
    dropped_market_columns = sorted(set(excluded_market_columns + dropped_raw_market_columns))
    if dropped_market_columns:
        log.debug(
            "Dropped market columns (%d): %s",
            len(dropped_market_columns),
            dropped_market_columns,
        )
    if disable_pruning:
        log.debug("Feature pruning disabled.")
    elif pruned_columns:
        log.debug("Dropped pruned columns (%d): %s", len(pruned_columns), pruned_columns)
    if id_columns:
        log.debug("Dropped identifier columns (%d): %s", len(id_columns), id_columns)
    if constant_columns:
        log.debug("Dropped constant columns (%d): %s", len(constant_columns), constant_columns)
    if high_cardinality_columns:
        log.debug(
            "Dropped high-cardinality categoricals (%d, threshold=%.2f): %s",
            len(high_cardinality_columns),
            max_cardinality_ratio,
            high_cardinality_columns,
        )

    return FeatureSpec(
        feature_columns=feature_df.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        dropped_columns=sorted(drop_columns),
        id_columns=id_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        feature_start=feature_start,
        feature_end=feature_end,
        metadata_columns=metadata_columns,
        post_feature_columns=post_feature_columns,
        market_columns=market_feature_columns,
    )


def _apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Apply a FeatureSpec to reorder/select model features."""
    df = df.copy()
    if any(col in spec.feature_columns for col in MARKET_DERIVED_COLUMNS):
        df = _add_market_transforms(df)
    missing = [col for col in spec.feature_columns if col not in df.columns]
    if missing:
        log.debug(
            "Missing %d feature columns in input data; filling with NaN: %s",
            len(missing),
            missing,
        )
    return df.reindex(columns=spec.feature_columns)


def _build_preprocessor(spec: FeatureSpec, *, for_tree: bool = True) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric/categorical features."""
    encoder_params: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_params["sparse_output"] = for_tree
    else:
        encoder_params["sparse"] = for_tree

    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        )
    ]
    if not for_tree:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(**encoder_params)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_columns:
        transformers.append(("num", numeric_transformer, spec.numeric_columns))
    if spec.categorical_columns:
        transformers.append(("cat", categorical_transformer, spec.categorical_columns))
    if not transformers:
        raise ValueError("No feature columns available after preprocessing.")

    if for_tree:
        return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    return ColumnTransformer(transformers=transformers, remainder="drop")
