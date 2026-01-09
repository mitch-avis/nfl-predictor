"""
Train and evaluate score prediction models for NFL games.

This module uses time-aware splits by season, trains separate models for away/home scores,
reports score-focused metrics, and can generate weekly predictions with confidence ranks.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import __main__
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.sparse import spmatrix
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nfl_predictor import constants
from nfl_predictor.ml.ml_model_xgb_utils import (
    _build_xgb_fit_kwargs,
    _coerce_tree_method_on_error,
    _predict_xgb,
    _resolve_xgb_params,
    _with_xgb_early_stopping_params,
)
from nfl_predictor.utils.logger import log

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": os.cpu_count() or 1,
    "verbosity": 2,
}

DEFAULT_FEATURE_START_COLUMN = "away_rest"
DEFAULT_FEATURE_END_COLUMN = "home_moneyline"

DEFAULT_OPTUNA_TIMEOUT_SECONDS = 600
DEFAULT_OPTUNA_CV_SPLITS = 3
DEFAULT_EARLY_STOPPING_ROUNDS = 50

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)

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


@dataclass(frozen=True)
class ScoreModel:
    """Trained score models and preprocessing state."""

    preprocessor: ColumnTransformer
    feature_spec: FeatureSpec
    away_model: xgb.XGBRegressor
    home_model: xgb.XGBRegressor
    target_columns: tuple[str, str]
    market_prob_config: Optional["MarketProbConfig"] = None
    xgb_params: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class WinProbCalibrator:
    """Calibration model for mapping margin predictions to win probabilities."""

    method: str
    model: Any


@dataclass(frozen=True)
class MarginTotalModel:
    """Trained models for margin/total prediction plus preprocessing state."""

    preprocessor: ColumnTransformer
    feature_spec: FeatureSpec
    margin_model: xgb.XGBRegressor
    total_model: xgb.XGBRegressor
    target_columns: tuple[str, str]
    calibrator: Optional[WinProbCalibrator]
    margin_quantile_models: Optional[dict[float, xgb.XGBRegressor]] = None
    total_quantile_models: Optional[dict[float, xgb.XGBRegressor]] = None
    quantiles: Optional[tuple[float, ...]] = None
    market_anchor: bool = False
    market_prob_config: Optional["MarketProbConfig"] = None
    xgb_params: Optional[dict[str, Any]] = None
    tuned_params: Optional[dict[str, Any]] = None
    tuned_cv_summary: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class BlendLayer:
    """Linear blend layer for margin/total predictions."""

    margin_model: Ridge
    total_model: Ridge


@dataclass(frozen=True)
class BlendedMarginTotalModel:
    """Blended margin/total model that combines team and market signals."""

    team_model: MarginTotalModel
    market_model: Optional[MarginTotalModel]
    blend_layer: BlendLayer
    calibrator: Optional[WinProbCalibrator]
    target_columns: tuple[str, str]
    market_prob_config: Optional["MarketProbConfig"] = None
    xgb_params: Optional[dict[str, Any]] = None
    tuned_params: Optional[dict[str, Any]] = None
    tuned_cv_summary: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class MarketProbConfig:
    """Configuration for blending/clamping win probabilities vs market implied odds."""

    blend_weight: float
    clamp_delta: float


@dataclass(frozen=True)
class TrainingResult:
    """Training output bundle used for artifact writing."""

    model: Any
    metrics_report: dict[str, Any]
    splits: dict[str, Any]
    params: dict[str, Any]
    tuned_params: Optional[dict[str, Any]]
    feature_list: list[str]
    early_stopping: dict[str, Any]


@dataclass(frozen=True)
class OptunaConfig:
    """Configuration for Optuna hyperparameter tuning."""

    enabled: bool
    timeout_seconds: int
    n_trials: Optional[int]
    cv_splits: int
    objective: str
    early_stopping_rounds: int
    tree_method: Optional[str]
    device: Optional[str]
    tune_scope: str
    storage: Optional[str]
    study_name: Optional[str]
    best_params_out: Optional[Path]
    xgb_n_jobs: Optional[int] = None


def _available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def _get_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    result_columns = _available_columns(df, constants.RESULT_COLUMNS)
    score_columns = [col for col in result_columns if col.endswith("_score")]
    away_col = next((col for col in score_columns if col.startswith("away_")), None)
    home_col = next((col for col in score_columns if col.startswith("home_")), None)
    if not away_col or not home_col:
        raise ValueError("Expected away/home score columns in training data.")
    return away_col, home_col


def _get_feature_range_columns(
    df: pd.DataFrame, feature_start: str, feature_end: str
) -> tuple[list[str], list[str], list[str]]:
    columns = df.columns.tolist()
    if feature_start not in columns or feature_end not in columns:
        raise ValueError(
            f"Expected feature range columns '{feature_start}'..'{feature_end}' in dataset."
        )
    start_idx = columns.index(feature_start)
    end_idx = columns.index(feature_end)
    if start_idx > end_idx:
        raise ValueError(f"Feature start column '{feature_start}' occurs after '{feature_end}'.")
    feature_range = columns[start_idx : end_idx + 1]  # noqa: E203
    metadata_columns = columns[:start_idx]
    post_feature_columns = columns[end_idx + 1 :]  # noqa: E203
    return feature_range, metadata_columns, post_feature_columns


def _implied_prob_from_moneyline(values: pd.Series | np.ndarray) -> np.ndarray:
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
    id_columns = [col for col in df.columns if col.endswith("_id")]
    if not id_columns:
        return df, []
    return df.drop(columns=id_columns), id_columns


def _drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if not constant_cols:
        return df, []
    return df.drop(columns=constant_cols), constant_cols


def _drop_high_cardinality_columns(
    df: pd.DataFrame, max_cardinality_ratio: float
) -> tuple[pd.DataFrame, list[str]]:
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
) -> FeatureSpec:
    if market_transform:
        df = _add_market_transforms(df)

    feature_range, metadata_columns, post_feature_columns = _get_feature_range_columns(
        df, feature_start, feature_end
    )

    result_columns = _available_columns(df, constants.RESULT_COLUMNS)
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
        "Dropped post-feature columns (%d): %s", len(post_feature_columns), post_feature_columns
    )
    if result_columns:
        log.debug("Target/result columns present (%d): %s", len(result_columns), result_columns)
    dropped_market_columns = sorted(set(excluded_market_columns + dropped_raw_market_columns))
    if dropped_market_columns:
        log.debug(
            "Dropped market columns (%d): %s",
            len(dropped_market_columns),
            dropped_market_columns,
        )
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


def _load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def _summarize_missing_data(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize missing-data prevalence for key feature groups.

    When some sources are missing historically, the schema remains invariant and these counters make
    the impact visible in metrics.
    """

    def _group_summary(columns: list[str]) -> dict[str, Any]:
        present = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if not present:
            return {
                "present_columns": 0,
                "missing_columns": len(missing),
                "null_cells": None,
                "rows_with_any_null": None,
                "columns_all_null": None,
            }
        view = df[present]
        is_null = view.isna()
        return {
            "present_columns": len(present),
            "missing_columns": len(missing),
            "null_cells": int(is_null.to_numpy().sum()),
            "rows_with_any_null": int(is_null.any(axis=1).sum()),
            "columns_all_null": int(is_null.all(axis=0).sum()),
        }

    groups: dict[str, list[str]] = {
        "lines": list(constants.LINES_COLUMNS),
        "injuries": list(constants.INJURY_FEATURE_COLUMNS),
        "records": list(constants.RECORD_FEATURE_COLUMNS),
        "lookahead": list(constants.LOOKAHEAD_FEATURE_COLUMNS),
        "motivation": list(constants.MOTIVATION_FEATURE_COLUMNS),
    }
    return {
        "total_rows": int(len(df)),
        "groups": {name: _group_summary(cols) for name, cols in groups.items()},
    }


def _should_enable_injury_features(
    injury_features: Optional[bool],
    predict_df: Optional[pd.DataFrame],
    *,
    current_season: int,
) -> bool:
    """Resolve whether injury burden features should be used.

    When predicting an in-progress season, NFLverse participation/injury data does not update
    during the season, so the prediction dataset will typically have all-null injury columns.
    In that case, we default to disabling injury features to avoid training a model that depends
    on unavailable in-season inputs.

    Args:
        injury_features: Explicit user override (True/False) or None for auto.
        predict_df: Optional prediction dataset DataFrame.
        current_season: Current NFL season inferred from today's date.

    Returns:
        True if injury features should be enabled.
    """

    if injury_features is not None:
        return bool(injury_features)
    if predict_df is None:
        return True

    cols = [c for c in constants.INJURY_FEATURE_COLUMNS if c in predict_df.columns]
    if cols and predict_df[cols].isna().to_numpy().all():
        return False

    if "season" in predict_df.columns:
        season_values = pd.to_numeric(predict_df["season"], errors="coerce")
        max_season = season_values.max() if not season_values.empty else None
        if pd.notna(max_season) and int(max_season) >= int(current_season):
            return False

    return True


def _drop_injury_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove injury burden feature columns from a dataset.

    This is used to hard-disable the injury feature group for training/tuning so the resulting
    preprocessor and model never rely on them.
    """

    present = [c for c in constants.INJURY_FEATURE_COLUMNS if c in df.columns]
    if not present:
        return df, []
    return df.drop(columns=present), present


def _filter_season_bounds(
    df: pd.DataFrame,
    min_season: Optional[int],
    max_season: Optional[int],
) -> pd.DataFrame:
    if "season" not in df.columns:
        return df

    original_rows = len(df)
    if min_season is not None:
        df = df[df["season"] >= min_season]
    if max_season is not None:
        df = df[df["season"] <= max_season]
    if len(df) != original_rows:
        log.info(
            "Filtered seasons to range %s-%s: %d -> %d rows",
            min_season,
            max_season,
            original_rows,
            len(df),
        )
    return df


def _split_by_season(
    df: pd.DataFrame, holdout_seasons: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    if "season" not in df.columns:
        raise ValueError("Expected a season column for time-aware splits.")
    seasons = sorted(df["season"].dropna().unique())
    if holdout_seasons <= 0:
        return df.copy(), df.iloc[0:0].copy(), []
    if len(seasons) <= holdout_seasons:
        raise ValueError("Not enough seasons to create a holdout split.")
    holdout = seasons[-holdout_seasons:]
    train_df = df[~df["season"].isin(holdout)].copy()
    holdout_df = df[df["season"].isin(holdout)].copy()
    return train_df, holdout_df, holdout


def _split_train_calibration_holdout(
    df: pd.DataFrame,
    holdout_seasons: int,
    calibration_seasons: int,
    calibration_weeks: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[int],
    list[int],
    list[int],
    Optional[int],
    list[int],
]:
    if "season" not in df.columns:
        raise ValueError("Expected a season column for time-aware splits.")
    seasons = sorted(df["season"].dropna().unique())
    if holdout_seasons < 0 or calibration_seasons < 0 or calibration_weeks < 0:
        raise ValueError("Holdout and calibration values must be non-negative.")
    if len(seasons) <= holdout_seasons:
        raise ValueError("Not enough seasons to create a holdout split.")

    holdout = seasons[-holdout_seasons:] if holdout_seasons else []
    base_pool = seasons[:-holdout_seasons] if holdout_seasons else seasons
    if not base_pool:
        raise ValueError("Not enough seasons to create a training split.")

    inseason_calibration_season: Optional[int] = None
    inseason_calibration_weeks: list[int] = []
    inseason_calibration_df = df.iloc[0:0].copy()
    if calibration_weeks:
        if "week" not in df.columns:
            raise ValueError("Expected a week column for in-season calibration.")
        inseason_calibration_season = base_pool[-1]
        season_weeks = (
            df.loc[df["season"] == inseason_calibration_season, "week"].dropna().unique().tolist()
        )
        season_weeks = sorted(int(week) for week in season_weeks)
        if len(season_weeks) < calibration_weeks:
            raise ValueError(
                f"Not enough weeks in season {inseason_calibration_season} for calibration."
            )
        inseason_calibration_weeks = season_weeks[-calibration_weeks:]
        inseason_calibration_df = df[
            (df["season"] == inseason_calibration_season)
            & (df["week"].isin(inseason_calibration_weeks))
        ].copy()

    calibration_candidates = base_pool
    if inseason_calibration_season is not None:
        calibration_candidates = [s for s in base_pool if s != inseason_calibration_season]

    if calibration_seasons and len(calibration_candidates) <= calibration_seasons:
        raise ValueError("Not enough seasons to create train/calibration/holdout splits.")
    calibration = calibration_candidates[-calibration_seasons:] if calibration_seasons else []
    train = [season for season in base_pool if season not in calibration]

    train_df = df[df["season"].isin(train)].copy()
    if inseason_calibration_season is not None and inseason_calibration_weeks:
        train_df = train_df[
            ~(
                (train_df["season"] == inseason_calibration_season)
                & (train_df["week"].isin(inseason_calibration_weeks))
            )
        ]

    calibration_df = df[df["season"].isin(calibration)].copy()
    if not inseason_calibration_df.empty:
        calibration_df = pd.concat([calibration_df, inseason_calibration_df], axis=0)

    holdout_df = df[df["season"].isin(holdout)].copy()

    return (
        train_df,
        calibration_df,
        holdout_df,
        train,
        calibration,
        holdout,
        inseason_calibration_season,
        inseason_calibration_weeks,
    )


def _build_time_series_folds(
    seasons: Sequence[int],
    n_splits: int,
    val_window: int = 1,
    min_train_seasons: int = 3,
) -> list[tuple[list[int], list[int]]]:
    seasons = list(sorted(seasons))
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if len(seasons) < (min_train_seasons + n_splits * val_window):
        raise ValueError("Not enough seasons to create requested CV folds.")

    folds: list[tuple[list[int], list[int]]] = []
    for split_idx in range(n_splits):
        val_start = len(seasons) - n_splits * val_window + split_idx * val_window
        val_end = val_start + val_window
        val_seasons = seasons[val_start:val_end]
        train_seasons = seasons[:val_start]
        if len(train_seasons) < min_train_seasons:
            continue
        folds.append((train_seasons, val_seasons))
    if not folds:
        raise ValueError("Unable to create valid CV folds with the provided settings.")
    return folds


def _build_season_week_timepoints(df: pd.DataFrame) -> list[int]:
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("Expected 'season' and 'week' columns for time-series CV.")
    season = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    week = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    timepoint = (season * 100 + week).dropna().astype(int)
    unique = sorted(timepoint.unique().tolist())
    return unique


def _build_blocked_timepoint_folds(
    timepoints: Sequence[int],
    n_splits: int,
    *,
    min_train_timepoints: int = 10,
) -> list[tuple[list[int], list[int]]]:
    """Blocked time-series CV over ordered season-week timepoints.

    Each fold validates on a contiguous block of timepoints; training uses all
    strictly earlier timepoints.
    """
    points = list(timepoints)
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if len(points) < (min_train_timepoints + n_splits):
        raise ValueError("Not enough timepoints to create requested CV folds.")

    test_size = max(1, len(points) // (n_splits + 1))
    folds: list[tuple[list[int], list[int]]] = []
    for split_idx in range(n_splits):
        train_end = test_size * (split_idx + 1)
        val_start = train_end
        val_end = min(val_start + test_size, len(points))
        train_pts = points[:train_end]
        val_pts = points[val_start:val_end]
        if len(train_pts) < min_train_timepoints or not val_pts:
            continue
        folds.append((train_pts, val_pts))

    if not folds:
        raise ValueError("Unable to create valid time-series CV folds with the provided settings.")
    return folds


def _fit_models(
    x_train: np.ndarray | spmatrix,
    y_train: pd.DataFrame,
    target_columns: tuple[str, str],
    params: Optional[dict[str, Any]] = None,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    away_col, home_col = target_columns
    resolved_params = params or _resolve_xgb_params(DEFAULT_XGB_PARAMS)
    away_model = xgb.XGBRegressor(**resolved_params)
    home_model = xgb.XGBRegressor(**resolved_params)

    away_model.fit(x_train, y_train[away_col])
    home_model.fit(x_train, y_train[home_col])

    return away_model, home_model


def _evaluate_predictions(
    y_true: pd.DataFrame,
    pred_away: np.ndarray,
    pred_home: np.ndarray,
    target_columns: tuple[str, str],
) -> dict[str, float]:
    away_col, home_col = target_columns
    away_true = y_true[away_col].to_numpy()
    home_true = y_true[home_col].to_numpy()

    metrics = {
        "away_mae": mean_absolute_error(away_true, pred_away),
        "home_mae": mean_absolute_error(home_true, pred_home),
        "away_rmse": _rmse(away_true, pred_away),
        "home_rmse": _rmse(home_true, pred_home),
    }

    actual_margin = home_true - away_true
    predicted_margin = pred_home - pred_away
    metrics["margin_mae"] = mean_absolute_error(actual_margin, predicted_margin)

    actual_total = home_true + away_true
    predicted_total = pred_home + pred_away
    metrics["total_mae"] = mean_absolute_error(actual_total, predicted_total)

    actual_winner = np.where(home_true > away_true, "home", "away")
    pred_winner = np.where(pred_home > pred_away, "home", "away")
    is_tie = home_true == away_true
    metrics["winner_accuracy"] = float(np.mean((pred_winner == actual_winner) & ~is_tie))

    return metrics


def _prepare_margin_total_targets(
    df: pd.DataFrame, target_columns: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    away_col, home_col = target_columns
    margin = df[home_col].to_numpy() - df[away_col].to_numpy()
    total = df[home_col].to_numpy() + df[away_col].to_numpy()
    return margin, total


def _prepare_margin_total_targets_with_anchor(
    df: pd.DataFrame,
    target_columns: tuple[str, str],
    market_anchor: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    margin, total = _prepare_margin_total_targets(df, target_columns)
    if not market_anchor:
        return margin, total, None, None
    baseline_margin, baseline_total = get_market_baseline(df)
    return margin - baseline_margin, total - baseline_total, baseline_margin, baseline_total


def _derive_scores_from_margin_total(
    pred_margin: np.ndarray, pred_total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred_home = (pred_total + pred_margin) / 2
    pred_away = (pred_total - pred_margin) / 2
    return pred_away, pred_home


def _evaluate_margin_total_predictions(
    y_true: pd.DataFrame,
    pred_margin: np.ndarray,
    pred_total: np.ndarray,
    target_columns: tuple[str, str],
    home_win_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    away_col, home_col = target_columns
    away_true = y_true[away_col].to_numpy()
    home_true = y_true[home_col].to_numpy()

    actual_margin = home_true - away_true
    actual_total = home_true + away_true

    pred_away, pred_home = _derive_scores_from_margin_total(pred_margin, pred_total)

    metrics = {
        "margin_mae": mean_absolute_error(actual_margin, pred_margin),
        "total_mae": mean_absolute_error(actual_total, pred_total),
        "away_mae": mean_absolute_error(away_true, pred_away),
        "home_mae": mean_absolute_error(home_true, pred_home),
    }

    actual_winner = np.where(home_true > away_true, "home", "away")
    pred_winner = np.where(pred_margin > 0, "home", "away")
    is_tie = home_true == away_true
    metrics["winner_accuracy"] = float(np.mean((pred_winner == actual_winner) & ~is_tie))

    if home_win_prob is not None:
        actual_home_win = (home_true > away_true).astype(int)
        metrics["brier"] = float(brier_score_loss(actual_home_win, home_win_prob))

    return metrics


def _fit_margin_total_models(
    x_train: np.ndarray | spmatrix,
    y_margin: np.ndarray,
    y_total: np.ndarray,
    params: dict[str, Any],
    x_eval: Optional[np.ndarray | spmatrix] = None,
    y_margin_eval: Optional[np.ndarray] = None,
    y_total_eval: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    def _train_with_params(
        active_params: dict[str, Any],
    ) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
        resolved_early_stopping = early_stopping_rounds
        if x_eval is None or y_margin_eval is None or y_total_eval is None:
            resolved_early_stopping = None

        active_params = _with_xgb_early_stopping_params(active_params, resolved_early_stopping)
        margin_model = xgb.XGBRegressor(**active_params)
        total_model = xgb.XGBRegressor(**active_params)

        fit_kwargs = _build_xgb_fit_kwargs(x_eval, y_margin_eval, resolved_early_stopping)
        margin_model.fit(x_train, y_margin, **fit_kwargs)

        fit_kwargs = _build_xgb_fit_kwargs(x_eval, y_total_eval, resolved_early_stopping)
        total_model.fit(x_train, y_total, **fit_kwargs)

        return margin_model, total_model

    try:
        return _train_with_params(params)
    except xgb.core.XGBoostError as exc:
        fallback_params = _coerce_tree_method_on_error(params, exc)
        if fallback_params is None:
            raise
        return _train_with_params(fallback_params)


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    if not quantiles:
        raise ValueError("Quantiles must be non-empty.")
    normalized = tuple(float(q) for q in quantiles)
    for q in normalized:
        if not 0.0 < q < 1.0:
            raise ValueError(f"Quantile must be in (0, 1): {q}")
    return normalized


def _fit_quantile_models(
    x_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    params: dict[str, Any],
    quantiles: Sequence[float],
    x_eval: Optional[np.ndarray | spmatrix] = None,
    y_eval: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
) -> dict[float, xgb.XGBRegressor]:
    """Fit one XGBoost quantile regressor per requested quantile.

    Uses `objective='reg:quantileerror'` and passes `quantile_alpha` via model params.
    """

    resolved = _validate_quantiles(quantiles)
    models: dict[float, xgb.XGBRegressor] = {}

    resolved_early_stopping = early_stopping_rounds
    if x_eval is None or y_eval is None:
        resolved_early_stopping = None

    def _train_with_params(active_params: dict[str, Any]) -> dict[float, xgb.XGBRegressor]:
        fitted: dict[float, xgb.XGBRegressor] = {}
        for quantile in resolved:
            q_params = active_params.copy()
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = quantile
            q_params = _with_xgb_early_stopping_params(q_params, resolved_early_stopping)
            model = xgb.XGBRegressor(**q_params)
            fit_kwargs = _build_xgb_fit_kwargs(x_eval, y_eval, resolved_early_stopping)
            model.fit(x_train, y_train, **fit_kwargs)
            fitted[quantile] = model
        return fitted

    try:
        models = _train_with_params(params)
    except xgb.core.XGBoostError as exc:
        fallback_params = _coerce_tree_method_on_error(params, exc)
        if fallback_params is None:
            raise
        models = _train_with_params(fallback_params)

    return models


def _fit_blend_ridge_constrained(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, y)

    coef = np.asarray(model.coef_, dtype=float)
    if coef.shape != (2,):
        raise ValueError(f"Expected blend coefficients shape (2,), got {coef.shape}")

    coef = np.maximum(coef, 0.0)
    coef_sum = float(coef.sum())
    if coef_sum <= 0:
        coef = np.array([0.5, 0.5], dtype=float)
    else:
        coef = coef / coef_sum

    # Keep an intercept term but recompute it after constraining weights.
    intercept = float(np.mean(y - x @ coef))

    model.coef_ = coef
    model.intercept_ = intercept
    return model


def _fit_win_prob_calibrator(
    pred_margin: np.ndarray,
    actual_home_win: np.ndarray,
    method: str,
) -> Optional[WinProbCalibrator]:
    method = method.lower()
    if method == "none":
        return None
    if method == "elo":
        # Deterministic mapping; no fitting.
        return WinProbCalibrator(method=method, model=None)
    if method == "platt":
        model = LogisticRegression(solver="lbfgs")
        model.fit(pred_margin.reshape(-1, 1), actual_home_win)
        return WinProbCalibrator(method=method, model=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(pred_margin, actual_home_win)
        return WinProbCalibrator(method=method, model=model)
    raise ValueError(f"Unknown win probability calibration method: {method}")


def _predict_home_win_prob(
    pred_margin: np.ndarray,
    calibrator: Optional[WinProbCalibrator],
) -> np.ndarray:
    if calibrator is None:
        return _margin_to_home_win_prob(pred_margin)
    if calibrator.method == "elo":
        return np.clip(_margin_to_home_win_prob_elo_style(pred_margin), 0.0, 1.0)
    if calibrator.method == "isotonic":
        probs = calibrator.model.predict(pred_margin)
    else:
        probs = calibrator.model.predict_proba(pred_margin.reshape(-1, 1))[:, 1]
    return np.clip(probs, 0.0, 1.0)


def _adjust_home_win_prob(
    games_df: pd.DataFrame,
    home_win_prob: np.ndarray,
    market_prob_config: Optional[MarketProbConfig],
) -> np.ndarray:
    if market_prob_config is None:
        return home_win_prob

    blend_weight = market_prob_config.blend_weight
    clamp_delta = market_prob_config.clamp_delta
    if blend_weight < 0 or blend_weight > 1:
        raise ValueError("Market blend weight must be between 0 and 1.")
    if clamp_delta < 0 or clamp_delta > 0.5:
        raise ValueError("Market clamp delta must be between 0 and 0.5.")
    if blend_weight == 0 and clamp_delta == 0:
        return home_win_prob

    df = _add_market_transforms(games_df)
    if "home_market_prob" not in df.columns:
        log.debug("Market probabilities missing; skipping win-prob adjustments.")
        return home_win_prob

    market_prob = pd.to_numeric(df["home_market_prob"], errors="coerce").to_numpy(dtype=float)
    adjusted = home_win_prob.astype(float, copy=True)
    valid_mask = ~np.isnan(market_prob)
    if not valid_mask.any():
        return adjusted

    if blend_weight:
        adjusted[valid_mask] = (
            blend_weight * market_prob[valid_mask] + (1 - blend_weight) * adjusted[valid_mask]
        )

    if clamp_delta:
        lower = market_prob[valid_mask] - clamp_delta
        upper = market_prob[valid_mask] + clamp_delta
        adjusted[valid_mask] = np.clip(adjusted[valid_mask], lower, upper)

    return np.clip(adjusted, 0.0, 1.0)


def _predict_margin_total_from_model(
    model: MarginTotalModel, games_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    feature_df = _apply_feature_spec(games_df, model.feature_spec)
    log.debug("Prediction feature matrix: %d rows x %d columns", *feature_df.shape)
    x_games = model.preprocessor.transform(feature_df)
    pred_margin = _predict_xgb(model.margin_model, x_games)
    pred_total = _predict_xgb(model.total_model, x_games)
    if getattr(model, "market_anchor", False):
        baseline_margin, baseline_total = get_market_baseline(games_df)
        pred_margin = pred_margin + baseline_margin
        pred_total = pred_total + baseline_total
    return pred_margin, pred_total


def _predict_margin_total_quantiles_from_model(
    model: MarginTotalModel,
    games_df: pd.DataFrame,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    margin_models = getattr(model, "margin_quantile_models", None)
    total_models = getattr(model, "total_quantile_models", None)
    if not margin_models or not total_models:
        return {}, {}

    feature_df = _apply_feature_spec(games_df, model.feature_spec)
    x_games = model.preprocessor.transform(feature_df)

    margin_preds: dict[float, np.ndarray] = {
        q: _predict_xgb(q_model, x_games) for q, q_model in margin_models.items()
    }
    total_preds: dict[float, np.ndarray] = {
        q: _predict_xgb(q_model, x_games) for q, q_model in total_models.items()
    }

    if getattr(model, "market_anchor", False):
        baseline_margin, baseline_total = get_market_baseline(games_df)
        for q in list(margin_preds.keys()):
            margin_preds[q] = margin_preds[q] + baseline_margin
        for q in list(total_preds.keys()):
            total_preds[q] = total_preds[q] + baseline_total

    return margin_preds, total_preds


def _summarize_confidence_pool(
    df: pd.DataFrame,
    home_win_prob: np.ndarray,
    target_columns: tuple[str, str],
) -> dict[str, float]:
    if "season" not in df.columns or "week" not in df.columns:
        return {}

    away_col, home_col = target_columns
    required_cols = {away_col, home_col}
    if not required_cols.issubset(df.columns):
        return {}

    summary_df = df[["season", "week", away_col, home_col]].copy()
    summary_df["home_win_prob"] = home_win_prob
    summary_df["away_win_prob"] = 1.0 - home_win_prob
    summary_df["predicted_winner"] = np.where(home_win_prob >= 0.5, "home", "away")
    summary_df["actual_winner"] = np.where(
        summary_df[home_col] > summary_df[away_col],
        "home",
        np.where(summary_df[home_col] < summary_df[away_col], "away", "tie"),
    )
    summary_df["confidence_strength"] = np.abs(summary_df["home_win_prob"] - 0.5)
    summary_df["confidence_rank"] = (
        summary_df.groupby(["season", "week"])["confidence_strength"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    summary_df["pick_correct"] = (summary_df["predicted_winner"] == summary_df["actual_winner"]) & (
        summary_df["actual_winner"] != "tie"
    )
    summary_df["pick_win_prob"] = np.where(
        summary_df["predicted_winner"] == "home",
        summary_df["home_win_prob"],
        summary_df["away_win_prob"],
    )
    summary_df["expected_points"] = summary_df["confidence_rank"] * summary_df["pick_win_prob"]
    summary_df["actual_points"] = summary_df["confidence_rank"] * summary_df["pick_correct"].astype(
        int
    )

    weekly = (
        summary_df.groupby(["season", "week"], as_index=False)
        .agg(
            expected_points=("expected_points", "sum"),
            actual_points=("actual_points", "sum"),
            picks_correct=("pick_correct", "sum"),
            games=("confidence_rank", "size"),
        )
        .copy()
    )

    return {
        "weekly_expected_points_avg": float(weekly["expected_points"].mean()),
        "weekly_actual_points_avg": float(weekly["actual_points"].mean()),
        "weekly_picks_correct_avg": float(weekly["picks_correct"].mean()),
        "weeks": float(len(weekly)),
    }


def _margin_to_home_win_prob(margin: np.ndarray) -> np.ndarray:
    if constants.SCORE_DIFF_STD_DEV <= 0:
        raise ValueError("SCORE_DIFF_STD_DEV must be positive.")
    return norm.cdf(margin / constants.SCORE_DIFF_STD_DEV)


def _margin_to_home_win_prob_elo_style(
    margin: np.ndarray,
    *,
    points_per_400_elo: float = 16.0,
) -> np.ndarray:
    """Map predicted margin to win probability via an Elo-style logistic.

    This is a deterministic mapping:

        p(home win) = 1 / (1 + 10 ** (-margin / points_per_400_elo))

    where `margin` is in points (home_score - away_score).

    Compared to Platt scaling, this tends to produce less extreme probabilities for
    large-but-plausible margins and can be useful as an alternative for pool display.
    """

    if points_per_400_elo <= 0:
        raise ValueError("points_per_400_elo must be positive.")
    margin = np.asarray(margin, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -margin / points_per_400_elo))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rank_confidence(strength: np.ndarray, tiebreaker: Optional[np.ndarray] = None) -> np.ndarray:
    strength = np.asarray(strength)
    if tiebreaker is None:
        order = np.argsort(strength, kind="mergesort")
    else:
        order = np.lexsort((tiebreaker, strength))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(strength) + 1)
    return ranks


def _build_prediction_output(
    games_df: pd.DataFrame,
    pred_away: np.ndarray,
    pred_home: np.ndarray,
    home_win_prob: np.ndarray,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Build a prediction output table with optional score post-processing.

    Score post-processing is applied only to emitted score columns (and their derived
    total/margin) and never affects training targets or the underlying model outputs.
    """

    def _apply_score_rounding(values: np.ndarray, mode: str) -> np.ndarray:
        mode_norm = (mode or "none").strip().lower()
        if mode_norm == "none":
            return values
        if mode_norm in {"int", "integer"}:
            return np.round(values, 0)
        if mode_norm in {"half", "0.5", "nearest_half"}:
            return np.round(values * 2.0, 0) / 2.0
        if mode_norm in {"nfl", "football"}:
            # Snap to plausible NFL team scores (for display only).
            #
            # We model score plausibility with a lightweight cost function:
            # - TD (7) and FG (3) have cost 0 (common)
            # - TD-without-XP (6) and TD+2 (8) have cost 1
            # - safety (2) has cost 2 (rarer)
            #
            # Then we choose the candidate with minimal:
            #   abs(candidate - value) + penalty * cost
            max_score = 70
            # Penalize both rarity (2pt/safety) and also the number of scoring events.
            event_penalty = 0.35
            increments = [
                (3, 0.0 + event_penalty),
                (7, 0.0 + event_penalty),
                (6, 1.0 + event_penalty),
                (8, 1.0 + event_penalty),
                (2, 2.0 + event_penalty),
            ]

            # Compute minimal "rarity" cost for each reachable score.
            # Use Dijkstra to guarantee correctness regardless of increment/cost structure.
            import heapq

            best_cost = np.full(max_score + 1, np.inf, dtype=float)
            best_cost[0] = 0.0
            heap: list[tuple[float, int]] = [(0.0, 0)]
            while heap:
                cost, score = heapq.heappop(heap)
                if cost != best_cost[score]:
                    continue
                for inc, inc_cost in increments:
                    nxt = score + inc
                    if nxt > max_score:
                        continue
                    new_cost = cost + inc_cost
                    if new_cost < best_cost[nxt]:
                        best_cost[nxt] = new_cost
                        heapq.heappush(heap, (new_cost, nxt))

            candidates = np.where(np.isfinite(best_cost))[0]
            penalty = 1.0
            snapped: list[float] = []
            for v in np.asarray(values, dtype=float):
                if not np.isfinite(v):
                    snapped.append(float(v))
                    continue
                v_clip = float(np.clip(v, 0.0, float(max_score)))
                diffs = np.abs(candidates.astype(float) - v_clip)
                scores = diffs + penalty * best_cost[candidates]
                snapped.append(float(candidates[int(np.argmin(scores))]))
            return np.asarray(snapped, dtype=float)
        raise ValueError(f"Unknown score rounding mode: {mode}")

    output_df = games_df.copy()
    away_scores = _apply_score_rounding(np.asarray(pred_away, dtype=float), score_rounding)
    home_scores = _apply_score_rounding(np.asarray(pred_home, dtype=float), score_rounding)

    output_df["predicted_away_score"] = np.round(away_scores, 1)
    output_df["predicted_home_score"] = np.round(home_scores, 1)
    output_df["predicted_total"] = np.round(away_scores + home_scores, 1)
    output_df["predicted_margin"] = np.round(home_scores - away_scores, 1)

    # Round first (for stable output), then clip so values don't collapse to 0.0/1.0
    # at 4-decimal precision (which can distort pool rankings and log-loss stability).
    home_win_prob_out = np.clip(np.round(home_win_prob, 4), 0.0001, 0.9999)
    output_df["home_win_prob"] = home_win_prob_out
    output_df["away_win_prob"] = np.round(1.0 - home_win_prob_out, 4)

    team_cols = [
        col
        for col in constants.METADATA_COLUMNS
        if col.endswith("_abbr") and col in output_df.columns
    ]
    away_team_col = next((col for col in team_cols if col.startswith("away_")), None)
    home_team_col = next((col for col in team_cols if col.startswith("home_")), None)
    if away_team_col and home_team_col:
        output_df["predicted_winner"] = np.where(
            home_win_prob_out >= 0.5,
            output_df[home_team_col],
            output_df[away_team_col],
        )

    confidence_strength = np.abs(home_win_prob_out - 0.5)
    output_df["confidence_strength"] = confidence_strength
    tiebreaker = output_df["game_id"].to_numpy() if "game_id" in output_df.columns else None
    output_df["confidence_rank"] = _rank_confidence(confidence_strength, tiebreaker)

    return output_df


def _save_model_checkpoint(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model checkpoint to %s", path)


def _early_stopping_info(model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}

    def _capture(prefix: str, estimator: Any) -> None:
        for key in ("best_iteration", "best_score", "best_ntree_limit"):
            if hasattr(estimator, key):
                info[f"{prefix}.{key}"] = getattr(estimator, key)

    if isinstance(model, ScoreModel):
        _capture("away_model", model.away_model)
        _capture("home_model", model.home_model)
    elif isinstance(model, MarginTotalModel):
        _capture("margin_model", model.margin_model)
        _capture("total_model", model.total_model)
        if model.margin_quantile_models:
            for q, est in model.margin_quantile_models.items():
                _capture(f"margin_q{q}", est)
        if model.total_quantile_models:
            for q, est in model.total_quantile_models.items():
                _capture(f"total_q{q}", est)
    elif isinstance(model, BlendedMarginTotalModel):
        _capture("team.margin_model", model.team_model.margin_model)
        _capture("team.total_model", model.team_model.total_model)
        if model.market_model is not None:
            _capture("market.margin_model", model.market_model.margin_model)
            _capture("market.total_model", model.market_model.total_model)
    return info


def _load_model_checkpoint(path: Path, model_kind: str) -> Any:
    for cls in (
        FeatureSpec,
        ScoreModel,
        MarginTotalModel,
        BlendedMarginTotalModel,
        MarketProbConfig,
        WinProbCalibrator,
        BlendLayer,
    ):
        setattr(__main__, cls.__name__, cls)

    model = joblib.load(path)
    model = _ensure_backward_compatible_model(model)
    expected_types = {
        "score": ScoreModel,
        "margin_total": MarginTotalModel,
        "blend": BlendedMarginTotalModel,
    }
    expected_type = expected_types.get(model_kind)
    if expected_type is None:
        raise ValueError(f"Unknown model kind: {model_kind}")
    if not isinstance(model, expected_type):
        raise ValueError(f"Model checkpoint type mismatch; expected {expected_type.__name__}.")
    log.info("Loaded model checkpoint from %s", path)
    return model


def _ensure_backward_compatible_model(model: Any) -> Any:
    """Patch older pickled models missing newer fields."""

    def _ensure_margin_total(instance: Any) -> None:
        if not hasattr(instance, "margin_quantile_models"):
            setattr(instance, "margin_quantile_models", None)
        if not hasattr(instance, "total_quantile_models"):
            setattr(instance, "total_quantile_models", None)
        if not hasattr(instance, "quantiles"):
            setattr(instance, "quantiles", None)

    if isinstance(model, MarginTotalModel):
        _ensure_margin_total(model)
        return model
    if isinstance(model, BlendedMarginTotalModel):
        _ensure_margin_total(model.team_model)
        if model.market_model is not None:
            _ensure_margin_total(model.market_model)
        return model
    return model


def _with_market_prob_config(model: Any, config: Optional[MarketProbConfig]) -> Any:
    if config is None:
        return model
    if isinstance(model, ScoreModel):
        return replace(model, market_prob_config=config)
    if isinstance(model, MarginTotalModel):
        return replace(model, market_prob_config=config)
    if isinstance(model, BlendedMarginTotalModel):
        market_model = (
            replace(model.market_model, market_prob_config=config)
            if model.market_model is not None
            else None
        )
        return replace(
            model,
            market_prob_config=config,
            team_model=replace(model.team_model, market_prob_config=config),
            market_model=market_model,
        )
    return model


def load_model_checkpoint(path: Path, model_kind: str) -> Any:
    """Load a saved model checkpoint with type validation."""
    return _load_model_checkpoint(path, model_kind)


def get_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return away/home score column names."""
    return _get_target_columns(df)


def apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Apply a feature spec to an input DataFrame."""
    return _apply_feature_spec(df, spec)


def margin_to_home_win_prob(margin: np.ndarray) -> np.ndarray:
    """Convert predicted margin to home win probability."""
    return _margin_to_home_win_prob(margin)


def predict_margin_total_from_model(
    model: MarginTotalModel, games_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Predict margin and total from a margin/total model."""
    return _predict_margin_total_from_model(model, games_df)


def derive_scores_from_margin_total(
    pred_margin: np.ndarray, pred_total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive away/home scores from margin and total."""
    return _derive_scores_from_margin_total(pred_margin, pred_total)


def predict_home_win_prob(
    pred_margin: np.ndarray, calibrator: Optional[WinProbCalibrator]
) -> np.ndarray:
    """Predict home win probability from margin predictions."""
    return _predict_home_win_prob(pred_margin, calibrator)


def adjust_home_win_prob(
    games_df: pd.DataFrame,
    home_win_prob: np.ndarray,
    market_prob_config: Optional[MarketProbConfig],
) -> np.ndarray:
    """Adjust win probabilities using market blend/clamp settings."""
    return _adjust_home_win_prob(games_df, home_win_prob, market_prob_config)


def predict_xgb(model: xgb.XGBRegressor, data: np.ndarray | spmatrix) -> np.ndarray:
    """Predict using an XGBoost model with DMatrix inputs."""
    return _predict_xgb(model, data)


def build_prediction_output(
    games_df: pd.DataFrame,
    pred_away: np.ndarray,
    pred_home: np.ndarray,
    home_win_prob: np.ndarray,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Build the prediction output DataFrame."""
    return _build_prediction_output(
        games_df,
        pred_away,
        pred_home,
        home_win_prob,
        score_rounding=score_rounding,
    )


def _optuna_direction(objective: str) -> str:
    if objective in {"expected_points", "winner_accuracy"}:
        return "maximize"
    return "minimize"


def _select_objective_score(
    metrics: dict[str, float],
    pool_summary: dict[str, float],
    objective: str,
) -> float:
    if objective == "margin_mae":
        return metrics["margin_mae"]
    if objective == "total_mae":
        return metrics["total_mae"]
    if objective == "combined_mae":
        return (metrics["margin_mae"] + metrics["total_mae"]) / 2
    if objective == "winner_accuracy":
        return metrics["winner_accuracy"]
    if objective == "brier":
        return metrics.get("brier", float("inf"))
    if objective == "expected_points":
        return pool_summary.get("weekly_expected_points_avg", 0.0)
    raise ValueError(f"Unknown objective metric: {objective}")


def _score_margin_total_fold(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_columns: tuple[str, str],
    include_market: bool,
    max_cardinality_ratio: float,
    feature_start: str,
    feature_end: str,
    params: dict[str, Any],
    early_stopping_rounds: int,
    objective: str,
    market_only: bool = False,
    market_transform: bool = False,
    market_anchor: bool = False,
    market_prob_config: Optional[MarketProbConfig] = None,
) -> float:
    feature_spec = _build_feature_spec(
        train_df,
        include_market=include_market,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=feature_start,
        feature_end=feature_end,
        market_only=market_only,
        market_transform=market_transform,
    )
    preprocessor = _build_preprocessor(feature_spec, for_tree=True)

    x_train = preprocessor.fit_transform(_apply_feature_spec(train_df, feature_spec))
    x_val = preprocessor.transform(_apply_feature_spec(val_df, feature_spec))

    (
        y_margin_train,
        y_total_train,
        _,
        _,
    ) = _prepare_margin_total_targets_with_anchor(train_df, target_columns, market_anchor)
    (
        y_margin_val,
        y_total_val,
        baseline_margin_val,
        baseline_total_val,
    ) = _prepare_margin_total_targets_with_anchor(val_df, target_columns, market_anchor)

    margin_model, total_model = _fit_margin_total_models(
        x_train,
        y_margin_train,
        y_total_train,
        params,
        x_eval=x_val,
        y_margin_eval=y_margin_val,
        y_total_eval=y_total_val,
        early_stopping_rounds=early_stopping_rounds,
    )

    pred_margin = _predict_xgb(margin_model, x_val)
    pred_total = _predict_xgb(total_model, x_val)
    if market_anchor:
        pred_margin = pred_margin + baseline_margin_val
        pred_total = pred_total + baseline_total_val
    home_win_prob = _margin_to_home_win_prob(pred_margin)
    home_win_prob = _adjust_home_win_prob(val_df, home_win_prob, market_prob_config)

    metrics = _evaluate_margin_total_predictions(
        val_df, pred_margin, pred_total, target_columns, home_win_prob
    )
    pool_summary = _summarize_confidence_pool(val_df, home_win_prob, target_columns)
    return _select_objective_score(metrics, pool_summary, objective)


def _evaluate_margin_total_cv(
    df: pd.DataFrame,
    target_columns: tuple[str, str],
    include_market: bool,
    max_cardinality_ratio: float,
    feature_start: str,
    feature_end: str,
    params: dict[str, Any],
    cv_splits: int,
    early_stopping_rounds: int,
    objective: str,
    market_only: bool = False,
    market_transform: bool = False,
    market_anchor: bool = False,
    market_prob_config: Optional[MarketProbConfig] = None,
) -> float:
    timepoints = _build_season_week_timepoints(df)
    folds = _build_blocked_timepoint_folds(timepoints, n_splits=cv_splits)
    fold_scores: list[float] = []

    for train_points, val_points in folds:
        point_series = df["season"].astype(int) * 100 + df["week"].astype(int)
        train_mask = point_series.isin(train_points)
        val_mask = point_series.isin(val_points)
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        fold_scores.append(
            float(
                _score_margin_total_fold(
                    train_df=train_df,
                    val_df=val_df,
                    target_columns=target_columns,
                    include_market=include_market,
                    max_cardinality_ratio=max_cardinality_ratio,
                    feature_start=feature_start,
                    feature_end=feature_end,
                    params=params,
                    early_stopping_rounds=early_stopping_rounds,
                    objective=objective,
                    market_only=market_only,
                    market_transform=market_transform,
                    market_anchor=market_anchor,
                    market_prob_config=market_prob_config,
                )
            )
        )

    return float(np.mean(fold_scores))


def _evaluate_margin_total_cv_summary(
    df: pd.DataFrame,
    target_columns: tuple[str, str],
    include_market: bool,
    max_cardinality_ratio: float,
    feature_start: str,
    feature_end: str,
    params: dict[str, Any],
    cv_splits: int,
    early_stopping_rounds: int,
    objective: str,
    market_only: bool = False,
    market_transform: bool = False,
    market_anchor: bool = False,
    market_prob_config: Optional[MarketProbConfig] = None,
) -> dict[str, Any]:
    timepoints = _build_season_week_timepoints(df)
    folds = _build_blocked_timepoint_folds(timepoints, n_splits=cv_splits)
    fold_scores: list[float] = []
    for train_points, val_points in folds:
        point_series = df["season"].astype(int) * 100 + df["week"].astype(int)
        train_mask = point_series.isin(train_points)
        val_mask = point_series.isin(val_points)
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        fold_scores.append(
            float(
                _score_margin_total_fold(
                    train_df=train_df,
                    val_df=val_df,
                    target_columns=target_columns,
                    include_market=include_market,
                    max_cardinality_ratio=max_cardinality_ratio,
                    feature_start=feature_start,
                    feature_end=feature_end,
                    params=params,
                    early_stopping_rounds=early_stopping_rounds,
                    objective=objective,
                    market_only=market_only,
                    market_transform=market_transform,
                    market_anchor=market_anchor,
                    market_prob_config=market_prob_config,
                )
            )
        )

    return {
        "cv_splits": int(len(fold_scores)),
        "objective": objective,
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)) if fold_scores else None,
        "std": float(np.std(fold_scores)) if fold_scores else None,
    }


def _run_optuna_search(
    df: pd.DataFrame,
    target_columns: tuple[str, str],
    include_market: bool,
    max_cardinality_ratio: float,
    feature_start: str,
    feature_end: str,
    optuna_config: OptunaConfig,
    market_only: bool = False,
    market_transform: bool = False,
    market_anchor: bool = False,
    market_prob_config: Optional[MarketProbConfig] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None:  # pragma: no cover
        raise ImportError("Optuna is required for hyperparameter tuning.")
    optuna_module = optuna
    assert optuna_module is not None

    sampler = optuna_module.samplers.TPESampler(seed=42)
    study_kwargs: dict[str, Any] = {
        "direction": _optuna_direction(optuna_config.objective),
        "sampler": sampler,
    }
    if optuna_config.storage:
        study_kwargs["storage"] = optuna_config.storage
        study_kwargs["study_name"] = optuna_config.study_name
        study_kwargs["load_if_exists"] = True
    study = optuna_module.create_study(**study_kwargs)

    def objective_fn(trial: Any) -> float:
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        }
        if optuna_config.xgb_n_jobs is not None:
            trial_params["n_jobs"] = optuna_config.xgb_n_jobs
        params = _resolve_xgb_params(
            DEFAULT_XGB_PARAMS,
            overrides=trial_params,
            tree_method=optuna_config.tree_method,
            device=optuna_config.device,
        )
        return _evaluate_margin_total_cv(
            df,
            target_columns=target_columns,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            params=params,
            cv_splits=optuna_config.cv_splits,
            early_stopping_rounds=optuna_config.early_stopping_rounds,
            objective=optuna_config.objective,
            market_only=market_only,
            market_transform=market_transform,
            market_anchor=market_anchor,
            market_prob_config=market_prob_config,
        )

    def _persist_best_params(study: Any, trial: Any) -> None:
        if optuna_config.best_params_out is None:
            return
        if trial.state != optuna_module.trial.TrialState.COMPLETE:
            return
        best_params_out = optuna_config.best_params_out
        if best_params_out is None:
            return
        best_params_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": optuna_config.objective,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        }
        best_params_out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _trial_logger(study: Any, trial: Any) -> None:
        if trial.state != optuna_module.trial.TrialState.COMPLETE:
            return
        log.info(
            "Optuna trial %d complete: value=%.4f | best=%.4f",
            trial.number,
            trial.value,
            study.best_value,
        )
        _persist_best_params(study, trial)

    study.optimize(
        objective_fn,
        timeout=optuna_config.timeout_seconds,
        n_trials=optuna_config.n_trials,
        callbacks=[_trial_logger],
    )

    log.info("Optuna best %s: %.4f", optuna_config.objective, study.best_value)
    log.info("Optuna best params: %s", study.best_params)

    raw_best_params: Any = getattr(study, "best_params", {})
    best_params: dict[str, Any] = {}
    try:
        items = raw_best_params.items()  # type: ignore[union-attr]
    except AttributeError:
        items = raw_best_params
    for key, value in items:
        if isinstance(key, bytes):
            key_str = key.decode("utf-8", errors="replace")
        else:
            key_str = str(key)
        best_params[key_str] = value

    resolved_best = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=best_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )
    cv_summary = _evaluate_margin_total_cv_summary(
        df,
        target_columns=target_columns,
        include_market=include_market,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=feature_start,
        feature_end=feature_end,
        params=resolved_best,
        cv_splits=optuna_config.cv_splits,
        early_stopping_rounds=optuna_config.early_stopping_rounds,
        objective=optuna_config.objective,
        market_only=market_only,
        market_transform=market_transform,
        market_anchor=market_anchor,
        market_prob_config=market_prob_config,
    )
    return best_params, cv_summary
