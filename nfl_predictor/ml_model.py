"""
Train and evaluate score prediction models for NFL games.

This module uses time-aware splits by season, trains separate models for away/home scores,
reports score-focused metrics, and can generate weekly predictions with confidence ranks.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import spmatrix
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "n_jobs": 4,
}


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
    date_column: Optional[str]


@dataclass(frozen=True)
class ScoreModel:
    """Trained score models and preprocessing state."""

    preprocessor: ColumnTransformer
    feature_spec: FeatureSpec
    away_model: xgb.XGBRegressor
    home_model: xgb.XGBRegressor
    target_columns: tuple[str, str]


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


def _get_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in constants.POLARS_METADATA_COLUMNS:
        if col == "date" and col in df.columns:
            return col
    return None


def _add_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_column], errors="coerce")
    df["game_month"] = dates.dt.month
    df["game_day_of_week"] = dates.dt.dayofweek
    df["game_day_of_year"] = dates.dt.dayofyear
    df = df.drop(columns=[date_column])
    return df


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
) -> FeatureSpec:
    date_column = _get_date_column(df)
    if date_column:
        df = _add_date_features(df, date_column)

    drop_columns = set(_available_columns(df, constants.RESULT_COLUMNS))
    if not include_market:
        drop_columns.update(_available_columns(df, constants.LINES_COLUMNS))
    df = df.drop(columns=sorted(drop_columns))

    df, id_columns = _drop_identifier_columns(df)
    df, constant_columns = _drop_constant_columns(df)
    df, high_cardinality_columns = _drop_high_cardinality_columns(df, max_cardinality_ratio)

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [
        col for col in df.columns if col not in categorical_columns and col not in drop_columns
    ]

    return FeatureSpec(
        feature_columns=df.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        dropped_columns=sorted(drop_columns),
        id_columns=id_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        date_column=date_column,
    )


def _apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    df = df.copy()
    if spec.date_column and spec.date_column in df.columns:
        df = _add_date_features(df, spec.date_column)

    drop_cols = set(spec.dropped_columns)
    drop_cols.update(spec.id_columns)
    drop_cols.update(spec.constant_columns)
    drop_cols.update(spec.high_cardinality_columns)

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric_columns),
            ("cat", categorical_transformer, spec.categorical_columns),
        ],
        remainder="drop",
    )


def _load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def _split_by_season(
    df: pd.DataFrame, holdout_seasons: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    if "season" not in df.columns:
        raise ValueError("Expected a season column for time-aware splits.")
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) <= holdout_seasons:
        raise ValueError("Not enough seasons to create a holdout split.")
    holdout = seasons[-holdout_seasons:]
    train_df = df[~df["season"].isin(holdout)].copy()
    holdout_df = df[df["season"].isin(holdout)].copy()
    return train_df, holdout_df, holdout


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


def _margin_to_home_win_prob(margin: np.ndarray) -> np.ndarray:
    if constants.SCORE_DIFF_STD_DEV <= 0:
        raise ValueError("SCORE_DIFF_STD_DEV must be positive.")
    return norm.cdf(margin / constants.SCORE_DIFF_STD_DEV)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if "squared" in inspect.signature(mean_squared_error).parameters:
        return float(mean_squared_error(y_true, y_pred, squared=False))
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


def train_score_model(
    data_path: Path,
    holdout_seasons: int,
    include_market: bool,
    max_cardinality_ratio: float,
) -> ScoreModel:
    """Train score models using time-aware season splits."""
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    train_df, holdout_df, holdout = _split_by_season(df, holdout_seasons)
    log.info("Training seasons: %s", sorted(train_df["season"].unique()))
    log.info("Holdout seasons: %s", holdout)

    feature_spec = _build_feature_spec(train_df, include_market, max_cardinality_ratio)
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
    x_holdout = preprocessor.transform(x_holdout_df)

    away_model, home_model = _fit_models(x_train, train_df, target_columns)

    pred_away = away_model.predict(x_holdout)
    pred_home = home_model.predict(x_holdout)

    metrics = _evaluate_predictions(holdout_df, pred_away, pred_home, target_columns)
    log.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    return ScoreModel(
        preprocessor=preprocessor,
        feature_spec=feature_spec,
        away_model=away_model,
        home_model=home_model,
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
    x_games = model.preprocessor.transform(feature_df)

    pred_away = model.away_model.predict(x_games)
    pred_home = model.home_model.predict(x_games)

    output_df = games_df.copy()
    output_df["predicted_away_score"] = np.round(pred_away, 1)
    output_df["predicted_home_score"] = np.round(pred_home, 1)
    output_df["predicted_total"] = np.round(pred_away + pred_home, 1)
    output_df["predicted_margin"] = np.round(pred_home - pred_away, 1)

    home_win_prob = _margin_to_home_win_prob(pred_home - pred_away)
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
    tiebreaker = None
    if "game_id" in output_df.columns:
        tiebreaker = output_df["game_id"].to_numpy()
    output_df["confidence_rank"] = _rank_confidence(confidence_strength, tiebreaker)

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
        "--max-cardinality-ratio",
        type=float,
        default=0.5,
        help="Drop categorical columns with unique ratio above this threshold.",
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
    model = train_score_model(
        data_path=args.data_path,
        holdout_seasons=args.holdout_seasons,
        include_market=not args.exclude_market,
        max_cardinality_ratio=args.max_cardinality_ratio,
    )

    if args.predict_path:
        output_path = args.output_path
        if output_path is None:
            output_path = args.predict_path.with_name(f"{args.predict_path.stem}_predictions.csv")
        predict_week(model, args.predict_path, output_path, pretty_output=args.pretty_output)


if __name__ == "__main__":
    main()
