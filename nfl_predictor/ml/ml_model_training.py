"""Training routines for NFL models.

This module hosts the training entrypoints that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import xgboost as xgb

from nfl_predictor import constants
from nfl_predictor.ml import artifacts, feature_importance, floor_sigma
from nfl_predictor.ml.ml_model_core import (
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_QUANTILES,
    DEFAULT_XGB_PARAMS,
    FeatureSelection,
    FeatureSpec,
    FitData,
    FoldSetup,
    MarginTotalModel,
    OptunaConfig,
    TrainingResult,
    _early_stopping_info,
    _evaluate_margin_total_predictions,
    _filter_season_bounds,
    _get_target_columns,
    _load_games,
    _run_optuna_search,
    _split_by_season,
    _summarize_confidence_pool,
    _summarize_missing_data,
    apply_feature_spec,
    build_feature_spec,
    build_preprocessor,
    fit_margin_total_models,
    fit_quantile_models,
    fit_transform_matrix,
    get_market_baseline,
    margin_to_home_win_prob,
    predict_xgb,
    prepare_margin_total_targets_with_anchor,
    resolve_xgb_params,
    transform_matrix,
    validate_quantiles,
)
from nfl_predictor.ml.ml_model_xgb_utils import fitted_xgb_device
from nfl_predictor.ml.sample_weights import (
    combine_sample_weights,
    compute_postseason_sample_weight,
    compute_recency_sample_weight,
)
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer

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
        msg = "Holdout seasons must be non-negative."
        raise ValueError(msg)
    train_df, holdout_df, holdout = _split_by_season(df, holdout_seasons)
    splits = {
        "train_seasons": sorted(int(season) for season in train_df["season"].dropna().unique()),
        "holdout_seasons": [int(season) for season in holdout],
    }
    return train_df, holdout_df, splits


@dataclass(frozen=True, kw_only=True)
class TrainingOptions:
    """Everything a margin/total training run reads besides the games file's contents.

    ``xgb_params_overrides`` replace default XGBoost parameters, as a walk-forward's do; tuned
    parameters, when tuning runs, take precedence over them.

    With ``floor_sigma_pool`` (earlier out-of-fold errors), the model records the floor's
    sigma for ``floor_sigma_week`` (default: the week after the newest game in the data),
    estimated from the pool strictly before that week, and predicts with it. Holdout games are
    scored with their own week's sigma from the same pool. Without a pool the model records
    none and predicts with the constant ``SCORE_DIFF_STD_DEV``.
    """

    data_path: Path
    holdout_seasons: int
    include_market: bool
    max_cardinality_ratio: float
    optuna_config: OptunaConfig
    market_transform: bool
    market_anchor: bool
    include_postseason: bool = False
    postseason_weight: float = 1.0
    min_season: int | None = None
    max_season: int | None = None
    feature_start: str = DEFAULT_FEATURE_START_COLUMN
    feature_end: str = DEFAULT_FEATURE_END_COLUMN
    recency_half_life_seasons: float | None = None
    xgb_params_overrides: dict[str, Any] | None = None
    floor_sigma_pool: floor_sigma.ErrorPool | None = None
    floor_sigma_week: tuple[int, int] | None = None


@dataclass(frozen=True)
class _TrainedHeads:
    """A fitted preprocessor and feature spec with the margin, total and quantile heads."""

    preprocessor: ColumnTransformer
    feature_spec: FeatureSpec
    margin_model: xgb.XGBRegressor
    total_model: xgb.XGBRegressor
    margin_quantile_models: dict[float, xgb.XGBRegressor]
    total_quantile_models: dict[float, xgb.XGBRegressor]
    quantiles: tuple[float, ...]


def _load_training_games(options: TrainingOptions) -> tuple[pd.DataFrame, tuple[str, str]]:
    """Load the completed games inside the season bounds, regular season unless asked."""
    df = _load_games(options.data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))
    df = _filter_season_bounds(df, options.min_season, options.max_season)
    df = _filter_to_regular_season_for_training(df, include_postseason=options.include_postseason)
    return df, target_columns


def _resolve_training_params(
    options: TrainingOptions, tuned_params: dict[str, Any]
) -> dict[str, Any]:
    """Return the XGBoost parameters the final fit uses: defaults, overrides, then tuning."""
    params_overrides = {**(options.xgb_params_overrides or {}), **tuned_params}
    if options.optuna_config.xgb_n_jobs is not None:
        params_overrides["n_jobs"] = options.optuna_config.xgb_n_jobs
    return resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=params_overrides or None,
        tree_method=options.optuna_config.tree_method,
        device=options.optuna_config.device,
    )


def _train_heads(
    train_frame: pd.DataFrame,
    options: TrainingOptions,
    params: dict[str, Any],
    target_columns: tuple[str, str],
) -> _TrainedHeads:
    """Fit the preprocessor and every head on the training rows."""
    feature_spec = build_feature_spec(
        train_frame,
        FeatureSelection(
            include_market=options.include_market,
            max_cardinality_ratio=options.max_cardinality_ratio,
            feature_start=options.feature_start,
            feature_end=options.feature_end,
            market_transform=options.market_transform,
        ),
    )
    log.info(
        "Feature columns: %d (numeric=%d, categorical=%d)",
        len(feature_spec.feature_columns),
        len(feature_spec.numeric_columns),
        len(feature_spec.categorical_columns),
    )

    preprocessor = build_preprocessor(feature_spec, for_tree=True)
    x_train = fit_transform_matrix(preprocessor, apply_feature_spec(train_frame, feature_spec))
    postseason_weights = compute_postseason_sample_weight(
        train_frame,
        include_postseason=options.include_postseason,
        postseason_weight=options.postseason_weight,
    )
    recency_weights = compute_recency_sample_weight(
        train_frame,
        half_life_seasons=options.recency_half_life_seasons,
    )
    train_weight = combine_sample_weights(postseason_weights, recency_weights)
    y_margin_train, y_total_train, _, _ = prepare_margin_total_targets_with_anchor(
        train_frame, target_columns, market_anchor=options.market_anchor
    )

    # Every head runs its full tree budget on every training row, with no eval frame.
    margin_model, total_model = fit_margin_total_models(
        FitData(x_train, y_margin_train, y_total_train, sample_weight=train_weight), params
    )

    quantiles = validate_quantiles(DEFAULT_QUANTILES)
    margin_quantiles = fit_quantile_models(
        x_train,
        y_margin_train,
        params,
        quantiles,
        sample_weight=train_weight,
    )
    total_quantiles = fit_quantile_models(
        x_train,
        y_total_train,
        params,
        quantiles,
        sample_weight=train_weight,
    )
    return _TrainedHeads(
        preprocessor=preprocessor,
        feature_spec=feature_spec,
        margin_model=margin_model,
        total_model=total_model,
        margin_quantile_models=margin_quantiles,
        total_quantile_models=total_quantiles,
        quantiles=quantiles,
    )


def _log_holdout_evaluation(
    holdout_df: pd.DataFrame,
    heads: _TrainedHeads,
    options: TrainingOptions,
    target_columns: tuple[str, str],
) -> None:
    """Log the holdout metrics and confidence-pool summary, or say there is no holdout."""
    if holdout_df.empty:
        log.info("No holdout seasons configured; skipping holdout evaluation.")
        return
    x_holdout = transform_matrix(
        heads.preprocessor, apply_feature_spec(holdout_df, heads.feature_spec)
    )
    pred_margin = predict_xgb(heads.margin_model, x_holdout)
    pred_total = predict_xgb(heads.total_model, x_holdout)
    if options.market_anchor:
        baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
        pred_margin = pred_margin + baseline_margin_holdout
        pred_total = pred_total + baseline_total_holdout
    home_win_prob = _holdout_home_win_prob(pred_margin, holdout_df, options.floor_sigma_pool)

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


def _estimate_floor_sigma(
    options: TrainingOptions, df: pd.DataFrame
) -> floor_sigma.FloorSigma | None:
    """Return the floor sigma the model predicts with, or None without a reference pool."""
    if options.floor_sigma_pool is None:
        return None
    season, week = options.floor_sigma_week or floor_sigma.week_after(df)
    sigma_record = floor_sigma.estimate(
        options.floor_sigma_pool.errors, season, week, sources=options.floor_sigma_pool.sources
    )
    log.info(
        "Floor sigma for season %d week %d: %.4f%s, from %d earlier games in seasons %s",
        season,
        week,
        sigma_record.sigma,
        f" (the constant: fewer than {constants.FLOOR_SIGMA_MIN_POOL_SEASONS} earlier seasons)"
        if sigma_record.fallback
        else "",
        sigma_record.pool_games,
        list(sigma_record.pool_seasons),
    )
    return sigma_record


def train_margin_total_model(options: TrainingOptions) -> MarginTotalModel:
    """Train margin/total models; win probabilities are the deterministic floor."""
    df, target_columns = _load_training_games(options)
    train_df, holdout_df, splits = _split_train_holdout(df, options.holdout_seasons)
    holdout = splits["holdout_seasons"]
    log.info("Training seasons: %s", splits["train_seasons"])
    log.info("Holdout seasons: %s", holdout)
    log.debug("Training rows: %d | Holdout rows: %d", len(train_df), len(holdout_df))
    if options.market_transform:
        log.info("Market feature transforms enabled.")
    if options.market_anchor:
        get_market_baseline(train_df)
        log.info("Market anchor enabled: training residuals vs spread/total.")

    tuned_params: dict[str, Any] = {}
    tuned_cv_summary: dict[str, Any] | None = None
    optuna_summary: dict[str, Any] | None = None
    if options.optuna_config.enabled:
        tuned_params, tuned_cv_summary, optuna_summary = _run_optuna_search(
            train_df,
            FoldSetup(
                target_columns=target_columns,
                selection=FeatureSelection(
                    include_market=options.include_market,
                    max_cardinality_ratio=options.max_cardinality_ratio,
                    feature_start=options.feature_start,
                    feature_end=options.feature_end,
                    market_transform=options.market_transform,
                ),
                market_anchor=options.market_anchor,
            ),
            options.optuna_config,
            holdout_seasons=holdout,
        )

    params = _resolve_training_params(options, tuned_params)
    heads = _train_heads(train_df, options, params, target_columns)
    _log_holdout_evaluation(holdout_df, heads, options, target_columns)
    return MarginTotalModel(
        preprocessor=heads.preprocessor,
        feature_spec=heads.feature_spec,
        margin_model=heads.margin_model,
        total_model=heads.total_model,
        target_columns=target_columns,
        margin_quantile_models=heads.margin_quantile_models,
        total_quantile_models=heads.total_quantile_models,
        quantiles=heads.quantiles,
        market_anchor=options.market_anchor,
        xgb_params=params,
        tuned_params=tuned_params or None,
        tuned_cv_summary=tuned_cv_summary,
        optuna_summary=optuna_summary,
        floor_sigma=_estimate_floor_sigma(options, df),
    )


def _holdout_home_win_prob(
    pred_margin: np.ndarray,
    holdout_df: pd.DataFrame,
    pool: floor_sigma.ErrorPool | None,
) -> np.ndarray:
    """Return holdout probabilities: each week's pool sigma, or the constant without a pool."""
    if pool is None:
        return margin_to_home_win_prob(pred_margin)
    return floor_sigma.weekly_home_win_prob(
        pred_margin,
        holdout_df["season"].to_numpy(dtype=int),
        holdout_df["week"].to_numpy(dtype=int),
        pool.errors,
    )


def train_margin_total_model_with_report(options: TrainingOptions) -> TrainingResult:
    """Train a margin/total model and return a structured metrics report payload."""
    df, _target_columns = _load_training_games(options)
    missing_data_summary = _summarize_missing_data(df)
    train_df, holdout_df, splits = _split_train_holdout(df, options.holdout_seasons)

    # Train the actual model (this will also log holdout metrics).
    model = train_margin_total_model(options)

    holdout_metrics: dict[str, Any] | None = None
    pool_summary: dict[str, Any] | None = None
    if not holdout_df.empty:
        x_holdout = transform_matrix(
            model.preprocessor, apply_feature_spec(holdout_df, model.feature_spec)
        )
        pred_margin = predict_xgb(model.margin_model, x_holdout)
        pred_total = predict_xgb(model.total_model, x_holdout)
        if model.market_anchor:
            baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _holdout_home_win_prob(pred_margin, holdout_df, options.floor_sigma_pool)
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
        "floor_sigma": floor_sigma.model_record(model),
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


@dataclass(frozen=True, kw_only=True)
class TrainingRecord:
    """What every artifact of one training run records about the run."""

    run_id: str
    created_at: str
    dataset_hash: str
    config_payload: dict[str, Any]


def write_training_artifacts(
    result: TrainingResult, record: TrainingRecord, run_dir: Path
) -> artifacts.RunPaths:
    """Save the model with its metrics report, metadata and feature importance.

    The metadata names the device the model was fitted on, read from its fitted heads.
    """
    paths = artifacts.resolve_run_paths(record.run_id, run_dir=run_dir)
    artifacts.save_model(paths.model_path, result.model)

    metrics_report = {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "config": record.config_payload,
        "splits": result.splits,
        "metrics": result.metrics_report,
    }
    artifacts.write_json(paths.metrics_path, metrics_report)

    details = artifacts.TrainedModelDetails(
        feature_list=result.feature_list,
        splits=result.splits,
        params=result.params,
        tuned_params=result.tuned_params,
        early_stopping=result.early_stopping,
        optuna_summary=getattr(result.model, "optuna_summary", None),
        xgb_device=fitted_xgb_device(result.model),
        floor_sigma=floor_sigma.model_record(result.model),
    )
    metadata = artifacts.build_metadata(
        created_at=record.created_at,
        run_id=record.run_id,
        dataset_hash=record.dataset_hash,
        config=record.config_payload,
        details=details,
    )
    artifacts.write_json(paths.metadata_path, metadata)
    if result.feature_importance:
        importance_payload = {
            "run_id": record.run_id,
            "created_at": record.created_at,
            **result.feature_importance,
        }
        artifacts.write_json(paths.feature_importance_path, importance_payload)
    return paths
