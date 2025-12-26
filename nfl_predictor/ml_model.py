"""
Train and evaluate score prediction models for NFL games.

This module uses time-aware splits by season, trains separate models for away/home scores,
reports score-focused metrics, and can generate weekly predictions with confidence ranks.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import spmatrix
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error

try:
    # sklearn>=1.4
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover
    root_mean_squared_error = None
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

from nfl_predictor import constants
from nfl_predictor.utils import ml_utils
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
    "n_jobs": 24,
    "device": "cuda",
    "verbosity": 2,
}

DEFAULT_FEATURE_START_COLUMN = "away_rest"
DEFAULT_FEATURE_END_COLUMN = "home_moneyline"

DEFAULT_OPTUNA_TIMEOUT_SECONDS = 600
DEFAULT_OPTUNA_CV_SPLITS = 3
DEFAULT_EARLY_STOPPING_ROUNDS = 50

MARKET_DERIVED_COLUMNS = (
    "market_home_margin",
    "market_total_line",
    "home_market_prob",
    "away_market_prob",
)


@dataclass
class _RuntimeState:
    early_stopping_fallback_logged: bool = False
    gpu_fallback_logged: bool = False
    gpu_tree_method_disabled: bool = False


_RUNTIME_STATE = _RuntimeState()


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
    market_anchor: bool = False


@dataclass(frozen=True)
class BlendLayer:
    """Linear blend layer for margin/total predictions."""

    margin_model: Ridge
    total_model: Ridge


@dataclass(frozen=True)
class BlendedMarginTotalModel:
    """Blended margin/total model that combines team and market signals."""

    team_model: MarginTotalModel
    market_model: MarginTotalModel
    blend_layer: BlendLayer
    calibrator: Optional[WinProbCalibrator]
    target_columns: tuple[str, str]


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


def _get_market_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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


def _build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    encoder_params: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_params["sparse_output"] = False
    else:
        encoder_params["sparse"] = False

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


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


def _fit_models(
    x_train: np.ndarray | spmatrix,
    y_train: pd.DataFrame,
    target_columns: tuple[str, str],
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    away_col, home_col = target_columns
    away_model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
    home_model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)

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
    baseline_margin, baseline_total = _get_market_baseline(df)
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
        margin_model = xgb.XGBRegressor(**active_params)
        total_model = xgb.XGBRegressor(**active_params)

        fit_kwargs = _build_xgb_fit_kwargs(x_eval, y_margin_eval, early_stopping_rounds)
        margin_model.fit(x_train, y_margin, **fit_kwargs)

        fit_kwargs = _build_xgb_fit_kwargs(x_eval, y_total_eval, early_stopping_rounds)
        total_model.fit(x_train, y_total, **fit_kwargs)

        return margin_model, total_model

    try:
        return _train_with_params(params)
    except xgb.core.XGBoostError as exc:
        fallback_params = _coerce_tree_method_on_error(params, exc)
        if fallback_params is None:
            raise
        return _train_with_params(fallback_params)


def _fit_win_prob_calibrator(
    pred_margin: np.ndarray,
    actual_home_win: np.ndarray,
    method: str,
) -> Optional[WinProbCalibrator]:
    method = method.lower()
    if method == "none":
        return None
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
    if calibrator.method == "isotonic":
        probs = calibrator.model.predict(pred_margin)
    else:
        probs = calibrator.model.predict_proba(pred_margin.reshape(-1, 1))[:, 1]
    return np.clip(probs, 0.0, 1.0)


def _predict_margin_total_from_model(
    model: MarginTotalModel, games_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    feature_df = _apply_feature_spec(games_df, model.feature_spec)
    log.debug("Prediction feature matrix: %d rows x %d columns", *feature_df.shape)
    x_games = model.preprocessor.transform(feature_df)
    pred_margin = model.margin_model.predict(x_games)
    pred_total = model.total_model.predict(x_games)
    if getattr(model, "market_anchor", False):
        baseline_margin, baseline_total = _get_market_baseline(games_df)
        pred_margin = pred_margin + baseline_margin
        pred_total = pred_total + baseline_total
    return pred_margin, pred_total


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
    summary_df["away_win_prob"] = 1 - home_win_prob
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
) -> pd.DataFrame:
    output_df = games_df.copy()
    output_df["predicted_away_score"] = np.round(pred_away, 1)
    output_df["predicted_home_score"] = np.round(pred_home, 1)
    output_df["predicted_total"] = np.round(pred_away + pred_home, 1)
    output_df["predicted_margin"] = np.round(pred_home - pred_away, 1)
    output_df["home_win_prob"] = np.round(home_win_prob, 4)
    output_df["away_win_prob"] = np.round(1 - home_win_prob, 4)

    team_cols = [
        col
        for col in constants.POLARS_METADATA_COLUMNS
        if col.endswith("_abbr") and col in output_df.columns
    ]
    away_team_col = next((col for col in team_cols if col.startswith("away_")), None)
    home_team_col = next((col for col in team_cols if col.startswith("home_")), None)
    if away_team_col and home_team_col:
        output_df["predicted_winner"] = np.where(
            pred_home >= pred_away, output_df[home_team_col], output_df[away_team_col]
        )

    confidence_strength = np.abs(home_win_prob - 0.5)
    tiebreaker = output_df["game_id"].to_numpy() if "game_id" in output_df.columns else None
    output_df["confidence_rank"] = _rank_confidence(confidence_strength, tiebreaker)

    return output_df


def _save_model_checkpoint(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model checkpoint to %s", path)


def _load_model_checkpoint(path: Path, model_kind: str) -> Any:
    model = joblib.load(path)
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


def _resolve_xgb_params(
    base_params: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
    tree_method: Optional[str] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
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


def _xgb_fit_supports(param: str) -> bool:
    try:
        return param in inspect.signature(xgb.XGBRegressor.fit).parameters
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _xgb_supported_params() -> set[str]:
    return set(xgb.XGBRegressor().get_params().keys())


def _xgb_param_supported(param: str) -> bool:
    return param in _xgb_supported_params()


def _log_early_stopping_fallback(message: str) -> None:
    if _RUNTIME_STATE.early_stopping_fallback_logged:
        return
    log.info(message)
    _RUNTIME_STATE.early_stopping_fallback_logged = True


def _log_gpu_fallback(message: str) -> None:
    if _RUNTIME_STATE.gpu_fallback_logged:
        return
    log.info(message)
    _RUNTIME_STATE.gpu_fallback_logged = True


def _build_xgb_fit_kwargs(
    x_eval: Optional[np.ndarray | spmatrix],
    y_eval: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> dict[str, Any]:
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
        return fit_kwargs

    if _xgb_fit_supports("callbacks"):
        early_stop = None
        callback_module = getattr(xgb, "callback", None)
        if callback_module is not None:
            early_stop_cls = getattr(callback_module, "EarlyStopping", None)
            if early_stop_cls is not None:
                early_stop = early_stop_cls(rounds=early_stopping_rounds, save_best=True)
        if early_stop is not None:
            fit_kwargs["callbacks"] = [early_stop]
            return fit_kwargs

    _log_early_stopping_fallback(
        "XGBoost early stopping not supported by this version; continuing without it."
    )
    return fit_kwargs


def _coerce_tree_method_on_error(
    params: dict[str, Any],
    exc: Exception,
) -> Optional[dict[str, Any]]:
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
    if (
        "gpu_hist" not in message
        and "tree_method" not in message
        and "invalid input" not in lower_message
    ):
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
) -> float:
    seasons = sorted(df["season"].dropna().unique())
    folds = _build_time_series_folds(seasons, n_splits=cv_splits)
    fold_scores: list[float] = []

    for train_seasons, val_seasons in folds:
        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"].isin(val_seasons)].copy()

        feature_spec = _build_feature_spec(
            train_df,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            market_only=market_only,
            market_transform=market_transform,
        )
        preprocessor = _build_preprocessor(feature_spec)

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

        pred_margin = margin_model.predict(x_val)
        pred_total = total_model.predict(x_val)
        if market_anchor:
            pred_margin = pred_margin + baseline_margin_val
            pred_total = pred_total + baseline_total_val
        home_win_prob = _margin_to_home_win_prob(pred_margin)

        metrics = _evaluate_margin_total_predictions(
            val_df, pred_margin, pred_total, target_columns, home_win_prob
        )
        pool_summary = _summarize_confidence_pool(val_df, home_win_prob, target_columns)
        fold_scores.append(_select_objective_score(metrics, pool_summary, objective))

    return float(np.mean(fold_scores))


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
) -> dict[str, Any]:
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
    return study.best_params


def train_score_model(
    data_path: Path,
    holdout_seasons: int,
    include_market: bool,
    max_cardinality_ratio: float,
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
) -> ScoreModel:
    """Train score models using time-aware season splits."""
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    df = _filter_season_bounds(df, min_season, max_season)
    train_df, holdout_df, holdout = _split_by_season(df, holdout_seasons)
    log.info("Training seasons: %s", sorted(train_df["season"].unique()))
    log.info("Holdout seasons: %s", holdout)
    log.debug("Training rows: %d | Holdout rows: %d", len(train_df), len(holdout_df))

    feature_spec = _build_feature_spec(
        train_df,
        include_market=include_market,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=feature_start,
        feature_end=feature_end,
    )
    log.info(
        "Feature columns: %d (numeric=%d, categorical=%d)",
        len(feature_spec.feature_columns),
        len(feature_spec.numeric_columns),
        len(feature_spec.categorical_columns),
    )
    if feature_spec.high_cardinality_columns:
        log.info("Dropped high-cardinality columns: %s", feature_spec.high_cardinality_columns)

    x_train_df = _apply_feature_spec(train_df, feature_spec)
    x_holdout_df = _apply_feature_spec(holdout_df, feature_spec)

    preprocessor = _build_preprocessor(feature_spec)
    x_train = preprocessor.fit_transform(x_train_df)
    away_model, home_model = _fit_models(x_train, train_df, target_columns)

    if not holdout_df.empty:
        x_holdout = preprocessor.transform(x_holdout_df)
        pred_away = away_model.predict(x_holdout)
        pred_home = home_model.predict(x_holdout)

        metrics = _evaluate_predictions(holdout_df, pred_away, pred_home, target_columns)
        log.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    else:
        log.info("No holdout seasons configured; skipping holdout evaluation.")

    return ScoreModel(
        preprocessor=preprocessor,
        feature_spec=feature_spec,
        away_model=away_model,
        home_model=home_model,
        target_columns=target_columns,
    )


def train_margin_total_model(
    data_path: Path,
    holdout_seasons: int,
    calibration_seasons: int,
    calibration_weeks: int,
    include_market: bool,
    max_cardinality_ratio: float,
    win_prob_calibration: str,
    optuna_config: OptunaConfig,
    market_transform: bool,
    market_anchor: bool,
    refit_after_calibration: bool,
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
) -> MarginTotalModel:
    """Train margin/total models with optional calibration."""
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    df = _filter_season_bounds(df, min_season, max_season)
    (
        train_df,
        calibration_df,
        holdout_df,
        train_seasons,
        calibration,
        holdout,
        calibration_season_inseason,
        calibration_weeks_inseason,
    ) = _split_train_calibration_holdout(
        df, holdout_seasons, calibration_seasons, calibration_weeks
    )

    log.info("Training seasons: %s", train_seasons)
    log.info("Calibration seasons: %s", calibration)
    if calibration_season_inseason is not None and calibration_weeks_inseason:
        log.info(
            "Calibration weeks: season %s weeks %s",
            calibration_season_inseason,
            calibration_weeks_inseason,
        )
    log.info("Holdout seasons: %s", holdout)
    log.debug(
        "Training rows: %d | Calibration rows: %d | Holdout rows: %d",
        len(train_df),
        len(calibration_df),
        len(holdout_df),
    )
    if market_transform:
        log.info("Market feature transforms enabled.")
    if market_anchor:
        _get_market_baseline(train_df)
        log.info("Market anchor enabled: training residuals vs spread/total.")
    if refit_after_calibration:
        if calibration_df.empty:
            raise ValueError("Refit-after-calibration requires calibration seasons or weeks.")
        log.info("Refit-after-calibration enabled: training on calibration data before final fit.")

    tuned_params: dict[str, Any] = {}
    if optuna_config.enabled:
        tuned_params = _run_optuna_search(
            train_df,
            target_columns=target_columns,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            optuna_config=optuna_config,
            market_transform=market_transform,
            market_anchor=market_anchor,
        )

    params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=tuned_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )

    def _train_models(
        train_frame: pd.DataFrame,
        calib_frame: pd.DataFrame,
    ) -> tuple[
        ColumnTransformer,
        FeatureSpec,
        xgb.XGBRegressor,
        xgb.XGBRegressor,
        Optional[np.ndarray | spmatrix],
        Optional[np.ndarray],
    ]:
        local_spec = _build_feature_spec(
            train_frame,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            market_transform=market_transform,
        )
        log.info(
            "Feature columns: %d (numeric=%d, categorical=%d)",
            len(local_spec.feature_columns),
            len(local_spec.numeric_columns),
            len(local_spec.categorical_columns),
        )

        local_preprocessor = _build_preprocessor(local_spec)
        x_train = local_preprocessor.fit_transform(_apply_feature_spec(train_frame, local_spec))
        (
            y_margin_train,
            y_total_train,
            _,
            _,
        ) = _prepare_margin_total_targets_with_anchor(train_frame, target_columns, market_anchor)

        x_calibration = None
        baseline_margin_calibration = None
        y_margin_calibration = None
        y_total_calibration = None
        if not calib_frame.empty:
            x_calibration = local_preprocessor.transform(
                _apply_feature_spec(calib_frame, local_spec)
            )
            (
                y_margin_calibration,
                y_total_calibration,
                baseline_margin_calibration,
                _,
            ) = _prepare_margin_total_targets_with_anchor(
                calib_frame, target_columns, market_anchor
            )

        margin_model, total_model = _fit_margin_total_models(
            x_train,
            y_margin_train,
            y_total_train,
            params,
            x_eval=x_calibration,
            y_margin_eval=y_margin_calibration,
            y_total_eval=y_total_calibration,
            early_stopping_rounds=optuna_config.early_stopping_rounds,
        )

        return (
            local_preprocessor,
            local_spec,
            margin_model,
            total_model,
            x_calibration,
            baseline_margin_calibration,
        )

    (
        preprocessor,
        feature_spec,
        margin_model,
        total_model,
        x_calibration,
        baseline_margin_calibration,
    ) = _train_models(train_df, calibration_df)

    calibrator = None
    if win_prob_calibration != "none" and not refit_after_calibration:
        if calibration_df.empty:
            raise ValueError("Calibration requested but no calibration seasons configured.")
        if x_calibration is None:
            raise ValueError("Calibration features are unavailable.")
        pred_margin_calib = margin_model.predict(x_calibration)
        if market_anchor:
            if baseline_margin_calibration is None:
                raise ValueError("Market anchor baseline missing for calibration data.")
            pred_margin_calib = pred_margin_calib + baseline_margin_calibration
        away_col, home_col = target_columns
        actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
        calibrator = _fit_win_prob_calibrator(
            pred_margin_calib, actual_home_win.to_numpy(), win_prob_calibration
        )

    if refit_after_calibration:
        full_train_df = pd.concat([train_df, calibration_df], axis=0)
        (
            preprocessor,
            feature_spec,
            margin_model,
            total_model,
            x_calibration,
            baseline_margin_calibration,
        ) = _train_models(full_train_df, calibration_df)

        if win_prob_calibration != "none":
            if x_calibration is None:
                raise ValueError("Calibration features are unavailable after refit.")
            pred_margin_calib = margin_model.predict(x_calibration)
            if market_anchor:
                if baseline_margin_calibration is None:
                    raise ValueError("Market anchor baseline missing for calibration data.")
                pred_margin_calib = pred_margin_calib + baseline_margin_calibration
            away_col, home_col = target_columns
            actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
            calibrator = _fit_win_prob_calibrator(
                pred_margin_calib, actual_home_win.to_numpy(), win_prob_calibration
            )

    if not holdout_df.empty:
        x_holdout = preprocessor.transform(_apply_feature_spec(holdout_df, feature_spec))
        pred_margin = margin_model.predict(x_holdout)
        pred_total = total_model.predict(x_holdout)
        if market_anchor:
            baseline_margin_holdout, baseline_total_holdout = _get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _predict_home_win_prob(pred_margin, calibrator)

        metrics = _evaluate_margin_total_predictions(
            holdout_df, pred_margin, pred_total, target_columns, home_win_prob
        )
        log.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        pool_summary = _summarize_confidence_pool(holdout_df, home_win_prob, target_columns)
        if pool_summary:
            log.info(
                "Confidence pool (avg weekly): expected=%.1f actual=%.1f picks=%.2f",
                pool_summary["weekly_expected_points_avg"],
                pool_summary["weekly_actual_points_avg"],
                pool_summary["weekly_picks_correct_avg"],
            )
    else:
        log.info("No holdout seasons configured; skipping holdout evaluation.")

    return MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=feature_spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=target_columns,
        calibrator=calibrator,
        market_anchor=market_anchor,
    )


def train_blended_margin_total_model(
    data_path: Path,
    holdout_seasons: int,
    calibration_seasons: int,
    calibration_weeks: int,
    max_cardinality_ratio: float,
    win_prob_calibration: str,
    optuna_config: OptunaConfig,
    market_transform: bool,
    market_anchor: bool,
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
) -> BlendedMarginTotalModel:
    """Train blended margin/total models using team vs market signals."""
    if calibration_seasons <= 0 and calibration_weeks <= 0:
        raise ValueError("Blended models require calibration seasons or calibration weeks.")
    if market_anchor:
        raise ValueError("Market anchoring is only supported for margin_total models.")

    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    df = _filter_season_bounds(df, min_season, max_season)
    (
        train_df,
        calibration_df,
        holdout_df,
        train_seasons,
        calibration,
        holdout,
        calibration_season_inseason,
        calibration_weeks_inseason,
    ) = _split_train_calibration_holdout(
        df, holdout_seasons, calibration_seasons, calibration_weeks
    )

    log.info("Training seasons: %s", train_seasons)
    log.info("Calibration seasons: %s", calibration)
    if calibration_season_inseason is not None and calibration_weeks_inseason:
        log.info(
            "Calibration weeks: season %s weeks %s",
            calibration_season_inseason,
            calibration_weeks_inseason,
        )
    log.info("Holdout seasons: %s", holdout)
    log.debug(
        "Training rows: %d | Calibration rows: %d | Holdout rows: %d",
        len(train_df),
        len(calibration_df),
        len(holdout_df),
    )

    team_params: dict[str, Any] = {}
    market_params: dict[str, Any] = {}
    if optuna_config.enabled:
        tune_scope = optuna_config.tune_scope
        timeout = optuna_config.timeout_seconds
        team_optuna = optuna_config
        market_optuna = optuna_config
        if tune_scope == "both":
            split_timeout = max(timeout // 2, 1)
            team_optuna = OptunaConfig(
                enabled=True,
                timeout_seconds=split_timeout,
                n_trials=optuna_config.n_trials,
                cv_splits=optuna_config.cv_splits,
                objective=optuna_config.objective,
                early_stopping_rounds=optuna_config.early_stopping_rounds,
                tree_method=optuna_config.tree_method,
                device=optuna_config.device,
                tune_scope=optuna_config.tune_scope,
                storage=optuna_config.storage,
                study_name=optuna_config.study_name,
                best_params_out=optuna_config.best_params_out,
            )
            market_optuna = OptunaConfig(
                enabled=True,
                timeout_seconds=split_timeout,
                n_trials=optuna_config.n_trials,
                cv_splits=optuna_config.cv_splits,
                objective=optuna_config.objective,
                early_stopping_rounds=optuna_config.early_stopping_rounds,
                tree_method=optuna_config.tree_method,
                device=optuna_config.device,
                tune_scope=optuna_config.tune_scope,
                storage=optuna_config.storage,
                study_name=optuna_config.study_name,
                best_params_out=optuna_config.best_params_out,
            )

        if optuna_config.storage:
            suffix = "_team" if tune_scope in {"team", "both"} else ""
            team_optuna = OptunaConfig(
                enabled=team_optuna.enabled,
                timeout_seconds=team_optuna.timeout_seconds,
                n_trials=team_optuna.n_trials,
                cv_splits=team_optuna.cv_splits,
                objective=team_optuna.objective,
                early_stopping_rounds=team_optuna.early_stopping_rounds,
                tree_method=team_optuna.tree_method,
                device=team_optuna.device,
                tune_scope=team_optuna.tune_scope,
                storage=team_optuna.storage,
                study_name=f"{team_optuna.study_name}{suffix}" if team_optuna.study_name else None,
                best_params_out=team_optuna.best_params_out,
            )
            suffix = "_market" if tune_scope in {"market", "both"} else ""
            market_optuna = OptunaConfig(
                enabled=market_optuna.enabled,
                timeout_seconds=market_optuna.timeout_seconds,
                n_trials=market_optuna.n_trials,
                cv_splits=market_optuna.cv_splits,
                objective=market_optuna.objective,
                early_stopping_rounds=market_optuna.early_stopping_rounds,
                tree_method=market_optuna.tree_method,
                device=market_optuna.device,
                tune_scope=market_optuna.tune_scope,
                storage=market_optuna.storage,
                study_name=(
                    f"{market_optuna.study_name}{suffix}" if market_optuna.study_name else None
                ),
                best_params_out=market_optuna.best_params_out,
            )

        if tune_scope in {"team", "both"}:
            log.info("Tuning team-feature model hyperparameters...")
            team_params = _run_optuna_search(
                train_df,
                target_columns=target_columns,
                include_market=False,
                max_cardinality_ratio=max_cardinality_ratio,
                feature_start=feature_start,
                feature_end=feature_end,
                optuna_config=team_optuna,
                market_only=False,
                market_transform=market_transform,
            )
        if tune_scope in {"market", "both"}:
            log.info("Tuning market-only model hyperparameters...")
            market_params = _run_optuna_search(
                train_df,
                target_columns=target_columns,
                include_market=True,
                max_cardinality_ratio=max_cardinality_ratio,
                feature_start=feature_start,
                feature_end=feature_end,
                optuna_config=market_optuna,
                market_only=True,
                market_transform=market_transform,
            )

    team_xgb_params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=team_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )
    market_xgb_params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=market_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )

    team_spec = _build_feature_spec(
        train_df,
        include_market=False,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=feature_start,
        feature_end=feature_end,
        market_transform=market_transform,
    )
    market_spec = _build_feature_spec(
        train_df,
        include_market=True,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=feature_start,
        feature_end=feature_end,
        market_only=True,
        market_transform=market_transform,
    )

    team_preprocessor = _build_preprocessor(team_spec)
    market_preprocessor = _build_preprocessor(market_spec)

    team_train = team_preprocessor.fit_transform(_apply_feature_spec(train_df, team_spec))
    market_train = market_preprocessor.fit_transform(_apply_feature_spec(train_df, market_spec))
    y_margin_train, y_total_train = _prepare_margin_total_targets(train_df, target_columns)

    team_calib = team_preprocessor.transform(_apply_feature_spec(calibration_df, team_spec))
    market_calib = market_preprocessor.transform(_apply_feature_spec(calibration_df, market_spec))
    y_margin_calib, y_total_calib = _prepare_margin_total_targets(calibration_df, target_columns)

    team_margin_model, team_total_model = _fit_margin_total_models(
        team_train,
        y_margin_train,
        y_total_train,
        team_xgb_params,
        x_eval=team_calib,
        y_margin_eval=y_margin_calib,
        y_total_eval=y_total_calib,
        early_stopping_rounds=optuna_config.early_stopping_rounds,
    )
    market_margin_model, market_total_model = _fit_margin_total_models(
        market_train,
        y_margin_train,
        y_total_train,
        market_xgb_params,
        x_eval=market_calib,
        y_margin_eval=y_margin_calib,
        y_total_eval=y_total_calib,
        early_stopping_rounds=optuna_config.early_stopping_rounds,
    )

    team_margin_calib = team_margin_model.predict(team_calib)
    team_total_calib = team_total_model.predict(team_calib)
    market_margin_calib = market_margin_model.predict(market_calib)
    market_total_calib = market_total_model.predict(market_calib)

    margin_blender = Ridge(alpha=1.0)
    total_blender = Ridge(alpha=1.0)
    margin_blender.fit(np.column_stack([team_margin_calib, market_margin_calib]), y_margin_calib)
    total_blender.fit(np.column_stack([team_total_calib, market_total_calib]), y_total_calib)

    blended_margin_calib = margin_blender.predict(
        np.column_stack([team_margin_calib, market_margin_calib])
    )
    away_col, home_col = target_columns
    actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
    calibrator = None
    if win_prob_calibration != "none":
        calibrator = _fit_win_prob_calibrator(
            blended_margin_calib, actual_home_win.to_numpy(), win_prob_calibration
        )

    if not holdout_df.empty:
        team_holdout = team_preprocessor.transform(_apply_feature_spec(holdout_df, team_spec))
        market_holdout = market_preprocessor.transform(_apply_feature_spec(holdout_df, market_spec))

        team_margin_holdout = team_margin_model.predict(team_holdout)
        team_total_holdout = team_total_model.predict(team_holdout)
        market_margin_holdout = market_margin_model.predict(market_holdout)
        market_total_holdout = market_total_model.predict(market_holdout)

        blended_margin = margin_blender.predict(
            np.column_stack([team_margin_holdout, market_margin_holdout])
        )
        blended_total = total_blender.predict(
            np.column_stack([team_total_holdout, market_total_holdout])
        )
        home_win_prob = _predict_home_win_prob(blended_margin, calibrator)

        metrics = _evaluate_margin_total_predictions(
            holdout_df, blended_margin, blended_total, target_columns, home_win_prob
        )
        log.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        pool_summary = _summarize_confidence_pool(holdout_df, home_win_prob, target_columns)
        if pool_summary:
            log.info(
                "Confidence pool (avg weekly): expected=%.1f actual=%.1f picks=%.2f",
                pool_summary["weekly_expected_points_avg"],
                pool_summary["weekly_actual_points_avg"],
                pool_summary["weekly_picks_correct_avg"],
            )
    else:
        log.info("No holdout seasons configured; skipping holdout evaluation.")

    team_model = MarginTotalModel(
        preprocessor=team_preprocessor,
        feature_spec=team_spec,
        margin_model=team_margin_model,
        total_model=team_total_model,
        target_columns=target_columns,
        calibrator=None,
    )
    market_model = MarginTotalModel(
        preprocessor=market_preprocessor,
        feature_spec=market_spec,
        margin_model=market_margin_model,
        total_model=market_total_model,
        target_columns=target_columns,
        calibrator=None,
    )

    return BlendedMarginTotalModel(
        team_model=team_model,
        market_model=market_model,
        blend_layer=BlendLayer(margin_model=margin_blender, total_model=total_blender),
        calibrator=calibrator,
        target_columns=target_columns,
    )


def predict_week(
    model: ScoreModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
) -> pd.DataFrame:
    """Generate weekly predictions and optional confidence ranks."""
    games_df = _load_games(games_path)
    feature_df = _apply_feature_spec(games_df, model.feature_spec)
    log.debug("Prediction feature matrix: %d rows x %d columns", *feature_df.shape)
    x_games = model.preprocessor.transform(feature_df)

    pred_away = model.away_model.predict(x_games)
    pred_home = model.home_model.predict(x_games)

    home_win_prob = _margin_to_home_win_prob(pred_home - pred_away)
    output_df = _build_prediction_output(games_df, pred_away, pred_home, home_win_prob)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df


def predict_week_margin_total(
    model: MarginTotalModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
) -> pd.DataFrame:
    """Generate weekly predictions from a margin/total model."""
    games_df = _load_games(games_path)
    pred_margin, pred_total = _predict_margin_total_from_model(model, games_df)
    pred_away, pred_home = _derive_scores_from_margin_total(pred_margin, pred_total)
    home_win_prob = _predict_home_win_prob(pred_margin, model.calibrator)
    output_df = _build_prediction_output(games_df, pred_away, pred_home, home_win_prob)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df


def predict_week_blended(
    model: BlendedMarginTotalModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
) -> pd.DataFrame:
    """Generate weekly predictions from a blended margin/total model."""
    games_df = _load_games(games_path)

    team_margin, team_total = _predict_margin_total_from_model(model.team_model, games_df)
    market_margin, market_total = _predict_margin_total_from_model(model.market_model, games_df)

    blended_margin = model.blend_layer.margin_model.predict(
        np.column_stack([team_margin, market_margin])
    )
    blended_total = model.blend_layer.total_model.predict(
        np.column_stack([team_total, market_total])
    )
    pred_away, pred_home = _derive_scores_from_margin_total(blended_margin, blended_total)
    home_win_prob = _predict_home_win_prob(blended_margin, model.calibrator)

    output_df = _build_prediction_output(games_df, pred_away, pred_home, home_win_prob)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFL score prediction models.")
    parser.add_argument(
        "--model-kind",
        choices=["score", "margin_total", "blend"],
        default="margin_total",
        help="Model pipeline to use (score, margin_total, or blend).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "completed_games_ml.csv",
        help="Path to completed games dataset.",
    )
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=2,
        help="Number of most recent seasons to hold out for evaluation.",
    )
    parser.add_argument(
        "--exclude-market",
        action="store_true",
        help="Exclude market features like spreads/totals/moneylines.",
    )
    parser.add_argument(
        "--market-transform",
        action="store_true",
        help=("Use transformed market features (implied probs, home margin) instead of raw lines."),
    )
    parser.add_argument(
        "--market-anchor",
        action="store_true",
        help=(
            "Train on residuals vs market spread/total and add market baseline at prediction time."
        ),
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Optional minimum season to include.",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Optional maximum season to include.",
    )
    parser.add_argument(
        "--feature-start",
        type=str,
        default=DEFAULT_FEATURE_START_COLUMN,
        help="First feature column (inclusive).",
    )
    parser.add_argument(
        "--feature-end",
        type=str,
        default=DEFAULT_FEATURE_END_COLUMN,
        help="Last feature column (inclusive).",
    )
    parser.add_argument(
        "--max-cardinality-ratio",
        type=float,
        default=0.5,
        help="Drop categorical columns with unique ratio above this threshold.",
    )
    parser.add_argument(
        "--calibration-seasons",
        type=int,
        default=1,
        help="Number of seasons reserved for calibration/blending.",
    )
    parser.add_argument(
        "--calibration-weeks",
        type=int,
        default=0,
        help="Number of weeks from the latest season reserved for calibration.",
    )
    parser.add_argument(
        "--refit-after-calibration",
        action="store_true",
        help="Refit the model on train+calibration data before calibrating probabilities.",
    )
    parser.add_argument(
        "--win-prob-calibration",
        choices=["none", "platt", "isotonic"],
        default="isotonic",
        help="Calibration method for win probabilities.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning.",
    )
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=DEFAULT_OPTUNA_TIMEOUT_SECONDS,
        help="Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        help="Optional max number of Optuna trials.",
    )
    parser.add_argument(
        "--tune-metric",
        choices=[
            "margin_mae",
            "total_mae",
            "combined_mae",
            "winner_accuracy",
            "brier",
            "expected_points",
        ],
        default="combined_mae",
        help="Objective metric for Optuna tuning.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=DEFAULT_OPTUNA_CV_SPLITS,
        help="Number of time-series CV folds for tuning.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds for XGBoost.",
    )
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default="auto",
        help="XGBoost tree_method (e.g., hist, gpu_hist, auto).",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default="auto",
        help="XGBoost device (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--tune-scope",
        choices=["team", "market", "both"],
        default="both",
        help="Which models to tune when blending.",
    )
    parser.add_argument(
        "--tune-storage",
        type=str,
        default=None,
        help="Optuna storage URL for persistent studies (e.g., sqlite:///optuna.db).",
    )
    parser.add_argument(
        "--tune-study-name",
        type=str,
        default=None,
        help="Optuna study name for persistent storage.",
    )
    parser.add_argument(
        "--tune-best-params-out",
        type=Path,
        default=None,
        help="Optional path to save the best Optuna params JSON after each trial.",
    )
    parser.add_argument(
        "--model-in",
        type=Path,
        default=None,
        help="Optional path to load a saved model checkpoint instead of training.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--predict-path",
        type=Path,
        default=None,
        help="Optional path to upcoming games for prediction.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for predictions CSV.",
    )
    parser.add_argument(
        "--pretty-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log a formatted weekly summary when predicting.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training and prediction."""
    args = _parse_args()

    study_name = args.tune_study_name
    if args.tune_storage and study_name is None:
        study_name = f"nfl_predictor_{args.model_kind}_{args.tune_metric}"
        log.info("Using default Optuna study name: %s", study_name)

    optuna_config = OptunaConfig(
        enabled=args.tune,
        timeout_seconds=args.tune_timeout,
        n_trials=args.tune_trials,
        cv_splits=args.cv_splits,
        objective=args.tune_metric,
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method=args.xgb_tree_method,
        device=args.xgb_device,
        tune_scope=args.tune_scope,
        storage=args.tune_storage,
        study_name=study_name,
        best_params_out=args.tune_best_params_out,
    )

    output_path = args.output_path
    if args.predict_path and output_path is None:
        output_path = args.predict_path.with_name(f"{args.predict_path.stem}_predictions.csv")

    if args.model_in is not None:
        if args.tune:
            log.info("Model checkpoint provided; ignoring training and Optuna tuning.")
        model = _load_model_checkpoint(args.model_in, args.model_kind)
        if not args.predict_path:
            log.info("No --predict-path provided; exiting after loading model.")
            return
        if args.model_kind == "score":
            predict_week(model, args.predict_path, output_path, pretty_output=args.pretty_output)
        elif args.model_kind == "margin_total":
            predict_week_margin_total(
                model, args.predict_path, output_path, pretty_output=args.pretty_output
            )
        elif args.model_kind == "blend":
            predict_week_blended(
                model, args.predict_path, output_path, pretty_output=args.pretty_output
            )
        else:
            raise ValueError(f"Unknown model kind: {args.model_kind}")
        return

    if args.model_kind == "score":
        model = train_score_model(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            include_market=not args.exclude_market,
            max_cardinality_ratio=args.max_cardinality_ratio,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
        )

        if args.model_out is not None:
            _save_model_checkpoint(model, args.model_out)
        if args.predict_path:
            predict_week(model, args.predict_path, output_path, pretty_output=args.pretty_output)
        return

    if args.model_kind == "margin_total":
        model = train_margin_total_model(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            calibration_seasons=args.calibration_seasons,
            calibration_weeks=args.calibration_weeks,
            include_market=not args.exclude_market,
            max_cardinality_ratio=args.max_cardinality_ratio,
            win_prob_calibration=args.win_prob_calibration,
            optuna_config=optuna_config,
            market_transform=args.market_transform,
            market_anchor=args.market_anchor,
            refit_after_calibration=args.refit_after_calibration,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
        )

        if args.model_out is not None:
            _save_model_checkpoint(model, args.model_out)
        if args.predict_path:
            predict_week_margin_total(
                model, args.predict_path, output_path, pretty_output=args.pretty_output
            )
        return

    if args.model_kind == "blend":
        model = train_blended_margin_total_model(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            calibration_seasons=args.calibration_seasons,
            calibration_weeks=args.calibration_weeks,
            max_cardinality_ratio=args.max_cardinality_ratio,
            win_prob_calibration=args.win_prob_calibration,
            optuna_config=optuna_config,
            market_transform=args.market_transform,
            market_anchor=args.market_anchor,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
        )

        if args.model_out is not None:
            _save_model_checkpoint(model, args.model_out)
        if args.predict_path:
            predict_week_blended(
                model, args.predict_path, output_path, pretty_output=args.pretty_output
            )
        return

    raise ValueError(f"Unknown model kind: {args.model_kind}")


if __name__ == "__main__":
    main()
