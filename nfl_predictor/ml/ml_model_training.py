"""Training routines for NFL models.

This module hosts the training entrypoints that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import feature_importance
from nfl_predictor.ml.ml_model_core import (
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_QUANTILES,
    DEFAULT_XGB_PARAMS,
    FeatureSpec,
    MarginTotalModel,
    OptunaConfig,
    TrainingResult,
    _apply_feature_spec,
    _build_feature_spec,
    _build_preprocessor,
    _early_stopping_info,
    _evaluate_margin_total_predictions,
    _filter_season_bounds,
    _fit_margin_total_models,
    _fit_quantile_models,
    _fit_transform_matrix,
    _get_target_columns,
    _load_games,
    _margin_to_home_win_prob,
    _predict_xgb,
    _prepare_margin_total_targets_with_anchor,
    _resolve_xgb_params,
    _run_optuna_search,
    _split_by_season,
    _summarize_confidence_pool,
    _summarize_missing_data,
    _transform_matrix,
    _validate_quantiles,
    get_market_baseline,
)
from nfl_predictor.ml.sample_weights import (
    combine_sample_weights,
    compute_postseason_sample_weight,
    compute_recency_sample_weight,
)
from nfl_predictor.utils.logger import log

xgb.set_config(verbosity=0)


def _filter_to_regular_season_for_training(
    df: pd.DataFrame,
    *,
    include_postseason: bool,
) -> pd.DataFrame:
    """Filter a dataset to regular season rows when `game_type` exists.

    This is the default for training and evaluation splits. Prediction can still be run on
    postseason rows (or any rows) as long as features are available.
    """
    if include_postseason:
        return df
    if "game_type" not in df.columns:
        return df

    filtered = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
    dropped = len(df) - len(filtered)
    if dropped:
        log.info("Filtering to regular season for training: dropped %d rows.", dropped)
    return filtered


def _split_train_holdout(
    df: pd.DataFrame, holdout_seasons: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """Split eligible games into the training rows and the evaluation holdout.

    The newest ``holdout_seasons`` whole seasons are held out and scored; every other eligible
    completed game trains the trees, the newest completed week included, as in each
    walk-forward fold. Returns the two frames and the seasons of each, as recorded in
    ``metadata.json`` ``splits``.

    Raises:
        ValueError: If ``holdout_seasons`` is negative or leaves no season to train on.

    """
    if holdout_seasons < 0:
        raise ValueError("Holdout seasons must be non-negative.")
    train_df, holdout_df, holdout = _split_by_season(df, holdout_seasons)
    splits = {
        "train_seasons": sorted(int(season) for season in train_df["season"].dropna().unique()),
        "holdout_seasons": [int(season) for season in holdout],
    }
    return train_df, holdout_df, splits


def train_margin_total_model(
    data_path: Path,
    holdout_seasons: int,
    include_market: bool,
    max_cardinality_ratio: float,
    optuna_config: OptunaConfig,
    market_transform: bool,
    market_anchor: bool,
    include_postseason: bool = False,
    postseason_weight: float = 1.0,
    min_season: int | None = None,
    max_season: int | None = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
    recency_half_life_seasons: float | None = None,
    xgb_params_overrides: dict[str, Any] | None = None,
) -> MarginTotalModel:
    """Train margin/total models; win probabilities are the deterministic floor.

    ``xgb_params_overrides`` replace default XGBoost parameters, as a walk-forward's do; tuned
    parameters, when tuning runs, take precedence over them.
    """
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    df = _filter_season_bounds(df, min_season, max_season)
    df = _filter_to_regular_season_for_training(df, include_postseason=include_postseason)
    train_df, holdout_df, splits = _split_train_holdout(df, holdout_seasons)
    holdout = splits["holdout_seasons"]
    log.info("Training seasons: %s", splits["train_seasons"])
    log.info("Holdout seasons: %s", holdout)
    log.debug("Training rows: %d | Holdout rows: %d", len(train_df), len(holdout_df))
    if market_transform:
        log.info("Market feature transforms enabled.")
    if market_anchor:
        get_market_baseline(train_df)
        log.info("Market anchor enabled: training residuals vs spread/total.")

    tuned_params: dict[str, Any] = {}
    tuned_cv_summary: dict[str, Any] | None = None
    optuna_summary: dict[str, Any] | None = None
    if optuna_config.enabled:
        tuned_params, tuned_cv_summary, optuna_summary = _run_optuna_search(
            train_df,
            target_columns=target_columns,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            optuna_config=optuna_config,
            market_transform=market_transform,
            market_anchor=market_anchor,
            holdout_seasons=holdout,
        )

    params_overrides = {**(xgb_params_overrides or {}), **tuned_params}
    if optuna_config.xgb_n_jobs is not None:
        params_overrides["n_jobs"] = optuna_config.xgb_n_jobs
    params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=params_overrides or None,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )

    def _train_models(
        train_frame: pd.DataFrame,
    ) -> tuple[
        ColumnTransformer,
        FeatureSpec,
        xgb.XGBRegressor,
        xgb.XGBRegressor,
        dict[float, xgb.XGBRegressor],
        dict[float, xgb.XGBRegressor],
        tuple[float, ...],
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

        local_preprocessor = _build_preprocessor(local_spec, for_tree=True)
        x_train = _fit_transform_matrix(
            local_preprocessor, _apply_feature_spec(train_frame, local_spec)
        )
        postseason_weights = compute_postseason_sample_weight(
            train_frame,
            include_postseason=include_postseason,
            postseason_weight=postseason_weight,
        )
        recency_weights = compute_recency_sample_weight(
            train_frame,
            half_life_seasons=recency_half_life_seasons,
        )
        train_weight = combine_sample_weights(postseason_weights, recency_weights)
        (
            y_margin_train,
            y_total_train,
            _,
            _,
        ) = _prepare_margin_total_targets_with_anchor(train_frame, target_columns, market_anchor)

        # Every head runs its full tree budget on every training row, with no eval frame.
        margin_model, total_model = _fit_margin_total_models(
            x_train,
            y_margin_train,
            y_total_train,
            params,
            sample_weight=train_weight,
        )

        quantiles = _validate_quantiles(DEFAULT_QUANTILES)
        margin_quantiles = _fit_quantile_models(
            x_train,
            y_margin_train,
            params,
            quantiles,
            sample_weight=train_weight,
        )
        total_quantiles = _fit_quantile_models(
            x_train,
            y_total_train,
            params,
            quantiles,
            sample_weight=train_weight,
        )

        return (
            local_preprocessor,
            local_spec,
            margin_model,
            total_model,
            margin_quantiles,
            total_quantiles,
            quantiles,
        )

    (
        preprocessor,
        feature_spec,
        margin_model,
        total_model,
        margin_quantile_models,
        total_quantile_models,
        quantiles,
    ) = _train_models(train_df)

    if not holdout_df.empty:
        x_holdout = _transform_matrix(preprocessor, _apply_feature_spec(holdout_df, feature_spec))
        pred_margin = _predict_xgb(margin_model, x_holdout)
        pred_total = _predict_xgb(total_model, x_holdout)
        baseline_margin_holdout: np.ndarray | None = None
        baseline_total_holdout: np.ndarray | None = None
        if market_anchor:
            baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _margin_to_home_win_prob(pred_margin)

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
        margin_quantile_models=margin_quantile_models,
        total_quantile_models=total_quantile_models,
        quantiles=quantiles,
        market_anchor=market_anchor,
        xgb_params=params,
        tuned_params=tuned_params or None,
        tuned_cv_summary=tuned_cv_summary,
        optuna_summary=optuna_summary,
    )


def train_margin_total_model_with_report(
    **kwargs: Any,
) -> TrainingResult:
    """Train a margin/total model and return a structured metrics report payload."""
    data_path: Path = kwargs["data_path"]
    holdout_seasons: int = kwargs["holdout_seasons"]

    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))
    df = _filter_season_bounds(df, kwargs.get("min_season"), kwargs.get("max_season"))
    df = _filter_to_regular_season_for_training(
        df,
        include_postseason=bool(kwargs.get("include_postseason", False)),
    )
    missing_data_summary = _summarize_missing_data(df)
    train_df, holdout_df, splits = _split_train_holdout(df, holdout_seasons)

    # Train the actual model (this will also log holdout metrics).
    model: MarginTotalModel = train_margin_total_model(**kwargs)

    holdout_metrics: dict[str, Any] | None = None
    pool_summary: dict[str, Any] | None = None
    if not holdout_df.empty:
        x_holdout = _transform_matrix(
            model.preprocessor, _apply_feature_spec(holdout_df, model.feature_spec)
        )
        pred_margin = _predict_xgb(model.margin_model, x_holdout)
        pred_total = _predict_xgb(model.total_model, x_holdout)
        if model.market_anchor:
            baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _margin_to_home_win_prob(pred_margin)
        holdout_metrics = _evaluate_margin_total_predictions(
            holdout_df, pred_margin, pred_total, model.target_columns, home_win_prob
        )
        pool_summary = _summarize_confidence_pool(holdout_df, home_win_prob, model.target_columns)

    report = {
        "kind": "train",
        "model_kind": "margin_total",
        "metrics": {"holdout": holdout_metrics},
        "pool": pool_summary,
        "tuning_cv": model.tuned_cv_summary,
        "tuning_optuna": getattr(model, "optuna_summary", None),
        "missing_data": missing_data_summary,
    }

    params = model.xgb_params or DEFAULT_XGB_PARAMS.copy()
    feature_list = list(model.feature_spec.feature_columns)
    feature_importance_report = feature_importance.build_feature_importance_report(
        model, shap_rows=train_df
    )
    return TrainingResult(
        model=model,
        metrics_report=report,
        splits=dict(splits),
        params=params,
        tuned_params=model.tuned_params,
        feature_list=feature_list,
        early_stopping=_early_stopping_info(model),
        feature_importance=feature_importance_report,
    )
