"""Train and evaluate score prediction models for NFL games.

This module uses time-aware splits by season, trains separate models for away/home scores,
reports score-focused metrics, and can generate weekly predictions with confidence ranks.
"""

from __future__ import annotations

import functools
import heapq
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
)

import __main__
from nfl_predictor import constants
from nfl_predictor.ml import feature_spec as _feature_spec
from nfl_predictor.ml.floor_sigma import FloorSigma, home_win_prob
from nfl_predictor.ml.metrics import confidence_ranks, confidence_strength, picks_home
from nfl_predictor.ml.ml_model_xgb_utils import (
    _build_xgb_fit_kwargs,
    _coerce_tree_method_on_error,
    _with_xgb_early_stopping_params,
    fit_transform_matrix,
    predict_xgb,
    resolve_xgb_params,
    transform_matrix,
)
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from scipy.sparse import spmatrix
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge

xgb.set_config(verbosity=0)

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "learning_rate": 0.0165,
    "max_depth": 5,
    "min_child_weight": 2.1878,
    "subsample": 0.6354,
    "colsample_bytree": 0.6098,
    "gamma": 2.3651,
    "reg_alpha": 1.9871,
    "reg_lambda": 1.6467,
    "random_state": 42,
    "n_jobs": os.cpu_count() or 1,
    "verbosity": 2,
}

DEFAULT_FEATURE_START_COLUMN = _feature_spec.DEFAULT_FEATURE_START_COLUMN
DEFAULT_FEATURE_END_COLUMN = _feature_spec.DEFAULT_FEATURE_END_COLUMN
MARKET_DERIVED_COLUMNS = _feature_spec.MARKET_DERIVED_COLUMNS
FeatureSpec = _feature_spec.FeatureSpec
FeatureSelection = _feature_spec.FeatureSelection
add_market_transforms = _feature_spec.add_market_transforms
apply_feature_spec = _feature_spec.apply_feature_spec
build_feature_spec = _feature_spec.build_feature_spec
build_preprocessor = _feature_spec.build_preprocessor
get_market_baseline = _feature_spec.get_market_baseline

DEFAULT_OPTUNA_TIMEOUT_SECONDS = 600
DEFAULT_OPTUNA_CV_SPLITS = 3
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
# The one win-probability calibration: the deterministic floor, the predicted margin through
# a normal curve whose spread is estimated from earlier out-of-fold errors (``floor_sigma``).
# ``none`` is an accepted second spelling of ``auto``.
CALIBRATION_FLOOR = "auto"
_CALIBRATION_ALIASES = frozenset({"auto", "none"})
RETIRED_CALIBRATIONS = frozenset({"platt", "isotonic", "sigma", "logistic", "elo"})


@dataclass(frozen=True)
class WinProbCalibrator:
    """A saved model's fitted or Elo calibrator; those calibrators were retired.

    Kept only so checkpoints saved before the retirement still unpickle. Loading one logs the
    calibrator and drops it; predictions are the deterministic floor.
    """

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
    margin_quantile_models: dict[float, xgb.XGBRegressor] | None = None
    total_quantile_models: dict[float, xgb.XGBRegressor] | None = None
    quantiles: tuple[float, ...] | None = None
    market_anchor: bool = False
    xgb_params: dict[str, Any] | None = None
    tuned_params: dict[str, Any] | None = None
    tuned_cv_summary: dict[str, Any] | None = None
    optuna_summary: dict[str, Any] | None = None
    # The floor's sigma for the week this model predicts, estimated at the final fit from
    # earlier out-of-fold errors. None (models saved before it was recorded) means the constant.
    floor_sigma: FloorSigma | None = None


@dataclass(frozen=True)
class BlendLayer:
    """Ridge layer of a saved blend model; the blend model kind was retired.

    Kept only so blend checkpoints saved before the retirement still unpickle for display.
    """

    margin_model: Ridge
    total_model: Ridge


@dataclass(frozen=True)
class BlendedMarginTotalModel:
    """A saved blend model (the team model and the market line through a ridge layer).

    The blend model kind was retired: nothing trains or predicts with one. The class stays
    so checkpoints saved before the retirement still unpickle, and loading one for prediction
    fails with the reason (``load_model_checkpoint``).
    """

    team_model: MarginTotalModel
    blend_layer: BlendLayer
    calibrator: WinProbCalibrator | None
    target_columns: tuple[str, str]
    market_prob_config: MarketProbConfig | None = None
    xgb_params: dict[str, Any] | None = None
    tuned_params: dict[str, Any] | None = None
    tuned_cv_summary: dict[str, Any] | None = None
    optuna_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class MarketProbConfig:
    """A saved model's market blend and clamp; market probability blending was retired.

    Kept only so checkpoints saved before the retirement still unpickle. Loading one logs the
    setting and drops it; predictions are the deterministic floor.
    """

    blend_weight: float
    clamp_delta: float
    prob_source: str = "raw"
    blend_method: str = "prob"


@dataclass(frozen=True)
class TrainingResult:
    """Training output bundle used for artifact writing."""

    model: Any
    metrics_report: dict[str, Any]
    splits: dict[str, Any]
    params: dict[str, Any]
    tuned_params: dict[str, Any] | None
    feature_list: list[str]
    early_stopping: dict[str, Any]
    feature_importance: dict[str, Any] | None = None


@dataclass(frozen=True)
class OptunaConfig:
    """Configuration for Optuna hyperparameter tuning."""

    enabled: bool
    timeout_seconds: int
    n_trials: int | None
    cv_splits: int
    objective: str
    early_stopping_rounds: int
    tree_method: str | None
    device: str | None
    storage: str | None
    study_name: str | None
    best_params_out: Path | None
    xgb_n_jobs: int | None = None


def _available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def _get_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    result_columns = _available_columns(df, constants.RESULT_COLUMNS)
    score_columns = [col for col in result_columns if col.endswith("_score")]
    away_col = next((col for col in score_columns if col.startswith("away_")), None)
    home_col = next((col for col in score_columns if col.startswith("home_")), None)
    if not away_col or not home_col:
        msg = "Expected away/home score columns in training data."
        raise ValueError(msg)
    return away_col, home_col


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
        "records": list(constants.RECORD_FEATURE_COLUMNS),
        "lookahead": list(constants.LOOKAHEAD_FEATURE_COLUMNS),
        "motivation": list(constants.MOTIVATION_FEATURE_COLUMNS),
        "qb_def_adj": list(constants.QB_DEF_ADJ_FEATURE_COLUMNS),
    }
    return {
        "total_rows": len(df),
        "groups": {name: _group_summary(cols) for name, cols in groups.items()},
    }


def _filter_season_bounds(
    df: pd.DataFrame,
    min_season: int | None,
    max_season: int | None,
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
        msg = "Expected a season column for time-aware splits."
        raise ValueError(msg)
    seasons = sorted(df["season"].dropna().unique())
    if holdout_seasons <= 0:
        return df.copy(), df.iloc[0:0].copy(), []
    if len(seasons) <= holdout_seasons:
        msg = "Not enough seasons to create a holdout split."
        raise ValueError(msg)
    holdout = seasons[-holdout_seasons:]
    train_df = df[~df["season"].isin(holdout)].copy()
    holdout_df = df[df["season"].isin(holdout)].copy()
    return train_df, holdout_df, holdout


def _build_time_series_folds(
    seasons: Sequence[int],
    n_splits: int,
    val_window: int = 1,
    min_train_seasons: int = 3,
) -> list[tuple[list[int], list[int]]]:
    seasons = sorted(seasons)
    if n_splits <= 0:
        msg = "n_splits must be positive."
        raise ValueError(msg)
    if len(seasons) < (min_train_seasons + n_splits * val_window):
        msg = "Not enough seasons to create requested CV folds."
        raise ValueError(msg)

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
        msg = "Unable to create valid CV folds with the provided settings."
        raise ValueError(msg)
    return folds


def _build_season_week_timepoints(df: pd.DataFrame) -> list[int]:
    if "season" not in df.columns or "week" not in df.columns:
        msg = "Expected 'season' and 'week' columns for time-series CV."
        raise ValueError(msg)
    season = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    week = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    timepoint = (season * 100 + week).dropna().astype(int)
    return sorted(timepoint.unique().tolist())


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
        msg = "n_splits must be positive."
        raise ValueError(msg)
    if len(points) < (min_train_timepoints + n_splits):
        msg = "Not enough timepoints to create requested CV folds."
        raise ValueError(msg)

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
        msg = "Unable to create valid time-series CV folds with the provided settings."
        raise ValueError(msg)
    return folds


def _prepare_margin_total_targets(
    df: pd.DataFrame, target_columns: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    away_col, home_col = target_columns
    margin = df[home_col].to_numpy() - df[away_col].to_numpy()
    total = df[home_col].to_numpy() + df[away_col].to_numpy()
    return margin, total


def prepare_margin_total_targets_with_anchor(
    df: pd.DataFrame,
    target_columns: tuple[str, str],
    *,
    market_anchor: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return the margin and total targets, as residuals against the market when anchored.

    Also returns the market baselines the residuals subtract (``None`` without anchoring), so
    predictions can add them back.
    """
    margin, total = _prepare_margin_total_targets(df, target_columns)
    if not market_anchor:
        return margin, total, None, None
    baseline_margin, baseline_total = get_market_baseline(df)
    return (
        margin - baseline_margin,
        total - baseline_total,
        baseline_margin,
        baseline_total,
    )


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
    home_win_prob: np.ndarray | None = None,
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


@dataclass(frozen=True)
class FitData:
    """A design matrix with its margin and total targets, and optional per-row weights."""

    x: np.ndarray | spmatrix
    margin: np.ndarray
    total: np.ndarray
    sample_weight: np.ndarray | None = None


def fit_margin_total_models(
    train: FitData,
    params: dict[str, Any],
    *,
    eval_data: FitData | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """Fit the margin and total XGBoost heads.

    Early stopping applies only when ``early_stopping_rounds`` and ``eval_data`` are given. A
    fit that XGBoost rejects for its tree method is retried once with a compatible one.
    """

    def _train_with_params(
        active_params: dict[str, Any],
    ) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
        resolved_early_stopping = None if eval_data is None else early_stopping_rounds
        x_eval = None if eval_data is None else eval_data.x

        active_params = _with_xgb_early_stopping_params(active_params, resolved_early_stopping)
        margin_model = xgb.XGBRegressor(**active_params)
        total_model = xgb.XGBRegressor(**active_params)

        fit_kwargs = _build_xgb_fit_kwargs(
            x_eval,
            None if eval_data is None else eval_data.margin,
            resolved_early_stopping,
        )
        margin_model.fit(train.x, train.margin, sample_weight=train.sample_weight, **fit_kwargs)

        fit_kwargs = _build_xgb_fit_kwargs(
            x_eval,
            None if eval_data is None else eval_data.total,
            resolved_early_stopping,
        )
        total_model.fit(train.x, train.total, sample_weight=train.sample_weight, **fit_kwargs)

        return margin_model, total_model

    try:
        return _train_with_params(params)
    except xgb.core.XGBoostError as exc:
        fallback_params = _coerce_tree_method_on_error(params, exc)
        if fallback_params is None:
            raise
        return _train_with_params(fallback_params)


def validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Return the quantiles as floats, refusing an empty list or any outside (0, 1)."""
    if not quantiles:
        msg = "Quantiles must be non-empty."
        raise ValueError(msg)
    normalized = tuple(float(q) for q in quantiles)
    for q in normalized:
        if not 0.0 < q < 1.0:
            msg = f"Quantile must be in (0, 1): {q}"
            raise ValueError(msg)
    return normalized


def fit_quantile_models(
    x_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    params: dict[str, Any],
    quantiles: Sequence[float],
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[float, xgb.XGBRegressor]:
    """Fit one XGBoost quantile regressor per requested quantile, on its full tree budget.

    Uses `objective='reg:quantileerror'` and passes `quantile_alpha` via model params.
    """
    resolved = validate_quantiles(quantiles)
    models: dict[float, xgb.XGBRegressor] = {}

    def _train_with_params(
        active_params: dict[str, Any],
    ) -> dict[float, xgb.XGBRegressor]:
        fitted: dict[float, xgb.XGBRegressor] = {}
        for quantile in resolved:
            q_params = active_params.copy()
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = quantile
            q_params = _with_xgb_early_stopping_params(q_params, None)
            model = xgb.XGBRegressor(**q_params)
            fit_kwargs = _build_xgb_fit_kwargs(None, None, None)
            model.fit(x_train, y_train, sample_weight=sample_weight, **fit_kwargs)
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


def resolve_calibration(method: str) -> str:
    """Return ``auto`` for either spelling of the deterministic floor.

    Raises:
        ValueError: If ``method`` names a retired calibrator or is unknown.

    """
    normalized = method.lower()
    if normalized in _CALIBRATION_ALIASES:
        return CALIBRATION_FLOOR
    if normalized in RETIRED_CALIBRATIONS:
        msg = (
            f"Calibration {method!r} was retired: every run submits the deterministic floor "
            "(auto, also spelled none)."
        )
        raise ValueError(msg)
    msg = f"Unknown win probability calibration method: {method!r}"
    raise ValueError(msg)


def normalize_no_vig(home_prob: np.ndarray, away_prob: np.ndarray) -> np.ndarray:
    """Normalize raw implied probs so home+away sums to 1 (no-vig), for the market yardstick."""
    total = home_prob + away_prob
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(total > 0, home_prob / total, np.nan)
    return np.clip(normalized, 0.0, 1.0)


def _predict_margin_total_from_model(
    model: MarginTotalModel, games_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    feature_df = apply_feature_spec(games_df, model.feature_spec)
    log.debug("Prediction feature matrix: %d rows x %d columns", *feature_df.shape)
    x_games = transform_matrix(model.preprocessor, feature_df)
    pred_margin = predict_xgb(model.margin_model, x_games)
    pred_total = predict_xgb(model.total_model, x_games)
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

    feature_df = apply_feature_spec(games_df, model.feature_spec)
    x_games = transform_matrix(model.preprocessor, feature_df)

    margin_preds: dict[float, np.ndarray] = {
        q: predict_xgb(q_model, x_games) for q, q_model in margin_models.items()
    }
    total_preds: dict[float, np.ndarray] = {
        q: predict_xgb(q_model, x_games) for q, q_model in total_models.items()
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
    summary_df["predicted_winner"] = np.where(picks_home(home_win_prob), "home", "away")
    summary_df["actual_winner"] = np.where(
        summary_df[home_col] > summary_df[away_col],
        "home",
        np.where(summary_df[home_col] < summary_df[away_col], "away", "tie"),
    )
    summary_df["confidence_rank"] = confidence_ranks(
        home_win_prob,
        tiebreaker=df["game_id"].to_numpy() if "game_id" in df.columns else None,
        groups=(summary_df["season"].to_numpy(), summary_df["week"].to_numpy()),
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


def margin_to_home_win_prob(
    margin: np.ndarray, sigma: float = constants.SCORE_DIFF_STD_DEV
) -> np.ndarray:
    """Return the deterministic floor, ``Phi(margin / sigma)``.

    ``sigma`` defaults to ``SCORE_DIFF_STD_DEV``, the fallback spread; the submitted
    probabilities pass the week's estimated sigma (``floor_sigma``).
    """
    return home_win_prob(margin, sigma)


def model_floor_sigma(model: object) -> tuple[float, bool]:
    """Return the sigma a saved model predicts with, and whether it is the fallback constant.

    A model saved before the sigma was recorded has none and uses the constant.
    """
    record = getattr(model, "floor_sigma", None)
    if record is None:
        return float(constants.SCORE_DIFF_STD_DEV), True
    return float(record.sigma), bool(record.fallback)


# Display-only score snapping: plausible NFL scores built from scoring plays, each play
# carrying a rarity cost. A TD (7) and a FG (3) cost 0, a TD without the extra point (6) and a
# TD with a two-point try (8) cost 1, and a safety (2) costs 2; every play also costs
# SCORING_EVENT_PENALTY, so fewer events are preferred. A raw score snaps to the candidate that
# minimizes |candidate - value| + SNAP_COST_WEIGHT * cost.
MAX_DISPLAY_SCORE = 70
SCORING_EVENT_PENALTY = 0.35
SCORING_PLAY_COSTS = ((3, 0.0), (7, 0.0), (6, 1.0), (8, 1.0), (2, 2.0))
SNAP_COST_WEIGHT = 1.0


@functools.cache
def _nfl_score_costs() -> tuple[np.ndarray, np.ndarray]:
    """Return the reachable scores and each one's minimal rarity cost (Dijkstra)."""
    increments = [(points, cost + SCORING_EVENT_PENALTY) for points, cost in SCORING_PLAY_COSTS]
    best_cost = np.full(MAX_DISPLAY_SCORE + 1, np.inf, dtype=float)
    best_cost[0] = 0.0
    heap: list[tuple[float, int]] = [(0.0, 0)]
    while heap:
        cost, score = heapq.heappop(heap)
        if cost != best_cost[score]:
            continue
        for inc, inc_cost in increments:
            nxt = score + inc
            if nxt > MAX_DISPLAY_SCORE:
                continue
            new_cost = cost + inc_cost
            if new_cost < best_cost[nxt]:
                best_cost[nxt] = new_cost
                heapq.heappush(heap, (new_cost, nxt))
    candidates = np.where(np.isfinite(best_cost))[0]
    return candidates, best_cost[candidates]


def _snap_to_nfl_scores(values: np.ndarray) -> np.ndarray:
    """Snap each raw score to a plausible NFL team score (for display only)."""
    candidates, candidate_costs = _nfl_score_costs()
    snapped: list[float] = []
    for v in np.asarray(values, dtype=float):
        if not np.isfinite(v):
            snapped.append(float(v))
            continue
        v_clip = float(np.clip(v, 0.0, float(MAX_DISPLAY_SCORE)))
        diffs = np.abs(candidates.astype(float) - v_clip)
        scores = diffs + SNAP_COST_WEIGHT * candidate_costs
        snapped.append(float(candidates[int(np.argmin(scores))]))
    return np.asarray(snapped, dtype=float)


def _apply_score_rounding(values: np.ndarray, mode: str) -> np.ndarray:
    """Return scores rounded for display: none, whole points, half points, or NFL scores."""
    mode_norm = (mode or "none").strip().lower()
    if mode_norm == "none":
        return values
    if mode_norm in {"int", "integer"}:
        return np.round(values, 0)
    if mode_norm in {"half", "0.5", "nearest_half"}:
        return np.round(values * 2.0, 0) / 2.0
    if mode_norm in {"nfl", "football"}:
        return _snap_to_nfl_scores(values)
    msg = f"Unknown score rounding mode: {mode}"
    raise ValueError(msg)


def _add_score_columns(
    output_df: pd.DataFrame, pred_away: np.ndarray, pred_home: np.ndarray, score_rounding: str
) -> None:
    """Add the raw scores and the display scores, each with its total and margin."""
    raw_away_scores = np.asarray(pred_away, dtype=float)
    raw_home_scores = np.asarray(pred_home, dtype=float)
    output_df["predicted_away_score_raw"] = np.round(raw_away_scores, 1)
    output_df["predicted_home_score_raw"] = np.round(raw_home_scores, 1)
    output_df["predicted_total_raw"] = np.round(raw_away_scores + raw_home_scores, 1)
    output_df["predicted_margin_raw"] = np.round(raw_home_scores - raw_away_scores, 1)

    display_away_scores = _apply_score_rounding(raw_away_scores, score_rounding)
    display_home_scores = _apply_score_rounding(raw_home_scores, score_rounding)

    output_df["predicted_away_score"] = np.round(display_away_scores, 1)
    output_df["predicted_home_score"] = np.round(display_home_scores, 1)
    output_df["predicted_total"] = np.round(display_away_scores + display_home_scores, 1)
    output_df["predicted_margin"] = np.round(display_home_scores - display_away_scores, 1)


def _add_pick_columns(output_df: pd.DataFrame, home_win_prob: np.ndarray) -> None:
    """Add the published probabilities, the predicted winner, and the confidence ranking."""
    # Round first (for stable output), then clip so values don't collapse to 0.0/1.0
    # at 4-decimal precision (which would make log loss unstable).
    home_win_prob_out = np.clip(np.round(home_win_prob, 4), 0.0001, 0.9999)
    output_df["home_win_prob"] = home_win_prob_out
    output_df["away_win_prob"] = np.round(1.0 - home_win_prob_out, 4)

    # The side and the confidence come from the unrounded probability, as in the walk-forward,
    # so a published 0.5000 or a shared 4-decimal value never decides a pick or a rank.
    unrounded = np.asarray(home_win_prob, dtype=float)
    team_cols = [
        col
        for col in constants.METADATA_COLUMNS
        if col.endswith("_abbr") and col in output_df.columns
    ]
    away_team_col = next((col for col in team_cols if col.startswith("away_")), None)
    home_team_col = next((col for col in team_cols if col.startswith("home_")), None)
    if away_team_col and home_team_col:
        output_df["predicted_winner"] = np.where(
            picks_home(unrounded),
            output_df[home_team_col],
            output_df[away_team_col],
        )

    output_df["confidence_strength"] = confidence_strength(unrounded)
    tiebreaker = output_df["game_id"].to_numpy() if "game_id" in output_df.columns else None
    output_df["confidence_rank"] = confidence_ranks(unrounded, tiebreaker)


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
    output_df = games_df.copy()
    _add_score_columns(output_df, pred_away, pred_home, score_rounding)
    _add_pick_columns(output_df, home_win_prob)
    return output_df


def _boosted_rounds(estimator: object) -> int | None:
    """Return the number of boosting rounds a fitted booster ran, when it can say."""
    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None
    num_boosted_rounds = getattr(get_booster(), "num_boosted_rounds", None)
    rounds = num_boosted_rounds() if callable(num_boosted_rounds) else None
    return int(rounds) if isinstance(rounds, int | np.integer) else None


def _estimator_iteration_info(prefix: str, estimator: object) -> dict[str, Any]:
    """Return a head's early-stopping attributes, its best iteration at the least."""
    info = {
        f"{prefix}.{key}": getattr(estimator, key)
        for key in ("best_iteration", "best_score", "best_ntree_limit")
        if hasattr(estimator, key)
    }
    if f"{prefix}.best_iteration" not in info:
        rounds = _boosted_rounds(estimator)
        if rounds is not None:
            info[f"{prefix}.best_iteration"] = rounds - 1
    return info


def _early_stopping_info(model: object) -> dict[str, Any]:
    """Return every head's early-stopping details, keyed ``<head>.<attribute>``."""
    if not isinstance(model, MarginTotalModel):
        return {}
    heads: dict[str, object] = {
        "margin_model": model.margin_model,
        "total_model": model.total_model,
    }
    heads.update({f"margin_q{q}": est for q, est in (model.margin_quantile_models or {}).items()})
    heads.update({f"total_q{q}": est for q, est in (model.total_quantile_models or {}).items()})
    info: dict[str, Any] = {}
    for prefix, estimator in heads.items():
        info.update(_estimator_iteration_info(prefix, estimator))
    return info


def _load_model_checkpoint(path: Path, model_kind: str) -> MarginTotalModel:
    for cls in (
        FeatureSpec,
        MarginTotalModel,
        BlendedMarginTotalModel,
        MarketProbConfig,
        WinProbCalibrator,
        BlendLayer,
    ):
        setattr(__main__, cls.__name__, cls)

    # XGBoost can emit a noisy warning when unpickling models across versions.
    # We keep our own reproducibility metadata; for most same-env runs this warning
    # is not actionable. If versions differ materially, we log a clearer message below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*If you are loading a serialized model.*",
            category=UserWarning,
        )
        model = joblib.load(path)

    _log_xgboost_version_change(path.with_name("metadata.json"))

    model = _ensure_backward_compatible_model(model)
    if model_kind != "margin_total":
        msg = f"Unknown model kind: {model_kind}"
        raise ValueError(msg)
    if isinstance(model, BlendedMarginTotalModel):
        msg = (
            f"{path} is a blend model: the blend model kind was retired; "
            "train a margin_total model instead."
        )
        raise TypeError(msg)
    if not isinstance(model, MarginTotalModel):
        msg = "Model checkpoint type mismatch; expected MarginTotalModel."
        raise TypeError(msg)
    _drop_retired_calibrator(model, path)
    _drop_retired_market_blend(model, path)
    _drop_retired_uncertainty(model, path)
    log.info("Loaded model checkpoint from %s", path)
    return model


def _drop_retired_calibrator(model: MarginTotalModel, path: Path) -> None:
    """Remove a saved fitted or Elo calibrator from a loaded model, saying what is ignored."""
    calibrator = model.__dict__.pop("calibrator", None)
    if calibrator is None:
        return
    log.warning(
        "%s was saved with the '%s' calibrator; fitted and Elo calibrators were retired, so it "
        "predicts the deterministic floor.",
        path,
        getattr(calibrator, "method", type(calibrator).__name__),
    )


def _drop_retired_market_blend(model: MarginTotalModel, path: Path) -> None:
    """Remove a saved market blend or clamp from a loaded model, saying what is ignored."""
    config = model.__dict__.pop("market_prob_config", None)
    if config is None:
        return
    weight = float(getattr(config, "blend_weight", 0.0))
    clamp = float(getattr(config, "clamp_delta", 0.0))
    if weight or clamp:
        log.warning(
            "%s was saved with a market blend (weight %.2f, clamp %.2f); market probability "
            "blending was retired, so it predicts the deterministic floor.",
            path,
            weight,
            clamp,
        )


def _drop_retired_uncertainty(model: MarginTotalModel, path: Path) -> None:
    """Remove a saved quantile-spread probability flag from a loaded model, saying so."""
    if model.__dict__.pop("win_prob_use_uncertainty", False):
        log.warning(
            "%s was saved with uncertainty-aware win probabilities (a sigma from the margin "
            "quantiles); that path was retired, so it predicts the deterministic floor.",
            path,
        )


def _ensure_backward_compatible_model[T](model: T) -> T:
    """Patch older pickled models missing newer fields."""

    def _safe_set_attr(instance: MarginTotalModel, name: str, value: object) -> None:
        try:
            object.__setattr__(instance, name, value)
        except AttributeError:
            setattr(instance, name, value)

    def _ensure_margin_total(instance: MarginTotalModel) -> None:
        if not hasattr(instance, "margin_quantile_models"):
            _safe_set_attr(instance, "margin_quantile_models", None)
        if not hasattr(instance, "total_quantile_models"):
            _safe_set_attr(instance, "total_quantile_models", None)
        if not hasattr(instance, "quantiles"):
            _safe_set_attr(instance, "quantiles", None)
        if not hasattr(instance, "optuna_summary"):
            _safe_set_attr(instance, "optuna_summary", None)
        if not hasattr(instance, "floor_sigma"):
            _safe_set_attr(instance, "floor_sigma", None)

    if isinstance(model, MarginTotalModel):
        _ensure_margin_total(model)
    return model


def _saved_xgboost_version(meta_path: Path) -> str | None:
    """Return the XGBoost version a sibling ``metadata.json`` records, if it is readable."""
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    versions = meta.get("library_versions") if isinstance(meta, dict) else None
    saved = versions.get("xgboost") if isinstance(versions, dict) else None
    return str(saved) if saved else None


def _log_xgboost_version_change(meta_path: Path) -> None:
    """Say so when a checkpoint was saved under a different XGBoost version."""
    saved_xgb = _saved_xgboost_version(meta_path)
    if saved_xgb and saved_xgb != xgb.__version__:
        log.info(
            "Loaded checkpoint saved with xgboost=%s (current=%s). "
            "For maximum portability, prefer retraining in the current environment.",
            saved_xgb,
            xgb.__version__,
        )


def load_model_checkpoint(path: Path, model_kind: str) -> MarginTotalModel:
    """Load a saved model checkpoint with type validation."""
    return _load_model_checkpoint(path, model_kind)


def get_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return away/home score column names."""
    return _get_target_columns(df)


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
    msg = f"Unknown objective metric: {objective}"
    raise ValueError(msg)


@dataclass(frozen=True)
class FoldSetup:
    """What every tuning fold shares: the targets, the feature selection, the market anchor."""

    target_columns: tuple[str, str]
    selection: FeatureSelection
    market_anchor: bool = False


def _score_margin_total_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    setup: FoldSetup,
    params: dict[str, Any],
    optuna_config: OptunaConfig,
) -> float:
    """Fit on one fold's training rows, early-stopping on its validation rows, and score them."""
    feature_spec = build_feature_spec(train_df, setup.selection)
    preprocessor = build_preprocessor(feature_spec, for_tree=True)

    x_train = fit_transform_matrix(preprocessor, apply_feature_spec(train_df, feature_spec))
    x_val = transform_matrix(preprocessor, apply_feature_spec(val_df, feature_spec))

    y_margin_train, y_total_train, _, _ = prepare_margin_total_targets_with_anchor(
        train_df, setup.target_columns, market_anchor=setup.market_anchor
    )
    (
        y_margin_val,
        y_total_val,
        baseline_margin_val,
        baseline_total_val,
    ) = prepare_margin_total_targets_with_anchor(
        val_df, setup.target_columns, market_anchor=setup.market_anchor
    )

    margin_model, total_model = fit_margin_total_models(
        FitData(x_train, y_margin_train, y_total_train),
        params,
        eval_data=FitData(x_val, y_margin_val, y_total_val),
        early_stopping_rounds=optuna_config.early_stopping_rounds,
    )

    pred_margin = predict_xgb(margin_model, x_val)
    pred_total = predict_xgb(total_model, x_val)
    if setup.market_anchor:
        pred_margin = pred_margin + baseline_margin_val
        pred_total = pred_total + baseline_total_val
    home_win_prob = margin_to_home_win_prob(pred_margin)

    metrics = _evaluate_margin_total_predictions(
        val_df, pred_margin, pred_total, setup.target_columns, home_win_prob
    )
    pool_summary = _summarize_confidence_pool(val_df, home_win_prob, setup.target_columns)
    return _select_objective_score(metrics, pool_summary, optuna_config.objective)


def _cv_fold_scores(
    df: pd.DataFrame, setup: FoldSetup, params: dict[str, Any], optuna_config: OptunaConfig
) -> list[float]:
    """Score ``params`` on each blocked, time-ordered season-week fold."""
    timepoints = _build_season_week_timepoints(df)
    folds = _build_blocked_timepoint_folds(timepoints, n_splits=optuna_config.cv_splits)
    fold_scores: list[float] = []
    for train_points, val_points in folds:
        point_series = df["season"].astype(int) * 100 + df["week"].astype(int)
        train_df = df[point_series.isin(train_points)].copy()
        val_df = df[point_series.isin(val_points)].copy()
        fold_scores.append(
            float(_score_margin_total_fold(train_df, val_df, setup, params, optuna_config))
        )
    return fold_scores


def _evaluate_margin_total_cv(
    df: pd.DataFrame, setup: FoldSetup, params: dict[str, Any], optuna_config: OptunaConfig
) -> float:
    """Return the mean fold score of ``params``."""
    return float(np.mean(_cv_fold_scores(df, setup, params, optuna_config)))


def _evaluate_margin_total_cv_summary(
    df: pd.DataFrame, setup: FoldSetup, params: dict[str, Any], optuna_config: OptunaConfig
) -> dict[str, Any]:
    """Return every fold score of ``params`` with their mean and spread."""
    fold_scores = _cv_fold_scores(df, setup, params, optuna_config)
    return {
        "cv_splits": len(fold_scores),
        "objective": optuna_config.objective,
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)) if fold_scores else None,
        "std": float(np.std(fold_scores)) if fold_scores else None,
    }


def _check_tuning_excludes_holdout(df: pd.DataFrame, holdout_seasons: Sequence[int] | None) -> None:
    """Refuse tuning data that includes a holdout season."""
    if not holdout_seasons:
        return
    if "season" not in df.columns:
        msg = "Optuna tuning requires a season column for holdout checks."
        raise ValueError(msg)
    if df["season"].isin(holdout_seasons).any():
        msg = "Optuna tuning data includes holdout seasons."
        raise ValueError(msg)


def _create_study(optuna_config: OptunaConfig) -> optuna.Study:
    """Create (or, with a storage, resume) the seeded study for the configured objective."""
    study_kwargs: dict[str, Any] = {
        "direction": _optuna_direction(optuna_config.objective),
        "sampler": optuna.samplers.TPESampler(seed=42),
    }
    if optuna_config.storage:
        study_kwargs["storage"] = optuna_config.storage
        study_kwargs["study_name"] = optuna_config.study_name
        study_kwargs["load_if_exists"] = True
    return optuna.create_study(**study_kwargs)


def _suggest_xgb_params(trial: optuna.Trial, optuna_config: OptunaConfig) -> dict[str, Any]:
    """Return one trial's XGBoost parameters: the searched ones over the defaults."""
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
    return resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=trial_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )


def _persist_best_params(
    optuna_config: OptunaConfig, study: optuna.Study, trial: optuna.trial.FrozenTrial
) -> None:
    """Write the best parameters so far to ``best_params_out`` after each completed trial."""
    best_params_out = optuna_config.best_params_out
    if best_params_out is None or trial.state != optuna.trial.TrialState.COMPLETE:
        return
    best_params_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": optuna_config.objective,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    best_params_out.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _trial_logger(
    optuna_config: OptunaConfig,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Return the callback that logs each completed trial and persists the best parameters."""

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        log.info(
            "Optuna trial %d complete: value=%.4f | best=%.4f",
            trial.number,
            trial.value,
            study.best_value,
        )
        _persist_best_params(optuna_config, study, trial)

    return log_trial


def _best_trial(study: optuna.Study) -> tuple[float | None, dict[str, Any], int | None]:
    """Return the best value, raw parameters and trial number, or nothing without a success."""
    try:
        return (
            float(study.best_value),
            dict(study.best_params),
            int(study.best_trial.number),
        )
    except AttributeError, ValueError, TypeError:
        return None, {}, None


def _run_optuna_search(
    df: pd.DataFrame,
    setup: FoldSetup,
    optuna_config: OptunaConfig,
    holdout_seasons: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Tune the XGBoost parameters by time-ordered cross-validation on the training rows.

    Returns the best parameters, their cross-validation summary, and the study summary.
    """
    _check_tuning_excludes_holdout(df, holdout_seasons)
    study = _create_study(optuna_config)

    def objective_fn(trial: optuna.Trial) -> float:
        """Optuna objective function for margin/total model tuning."""
        params = _suggest_xgb_params(trial, optuna_config)
        return _evaluate_margin_total_cv(df, setup, params, optuna_config)

    start_time = time.monotonic()
    study.optimize(
        objective_fn,
        timeout=optuna_config.timeout_seconds,
        n_trials=optuna_config.n_trials,
        callbacks=[_trial_logger(optuna_config)],
    )
    duration_seconds = time.monotonic() - start_time

    best_value, best_params_raw, best_trial_number = _best_trial(study)
    if best_value is None:
        log.warning("Optuna completed without a successful trial.")
    else:
        log.info("Optuna best %s: %.4f", optuna_config.objective, best_value)
        log.info("Optuna best params: %s", best_params_raw)

    best_params = {
        key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key): value
        for key, value in best_params_raw.items()
    }
    resolved_best = resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=best_params,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )
    cv_summary = _evaluate_margin_total_cv_summary(df, setup, resolved_best, optuna_config)
    complete_trials = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )
    optuna_summary = {
        "study_name": optuna_config.study_name,
        "storage": optuna_config.storage,
        "objective": optuna_config.objective,
        "direction": _optuna_direction(optuna_config.objective),
        "best_value": best_value,
        "best_trial": best_trial_number,
        "best_params": best_params_raw,
        "n_trials": len(study.trials),
        "n_complete_trials": int(complete_trials),
        "timeout_seconds": int(optuna_config.timeout_seconds),
        "cv_splits": int(optuna_config.cv_splits),
        "sampler": type(study.sampler).__name__,
        "duration_seconds": float(duration_seconds),
    }
    return best_params, cv_summary, optuna_summary
