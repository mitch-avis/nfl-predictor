"""Training routines for NFL models.

This module hosts the training entrypoints that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml.ml_model_core import (
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_QUANTILES,
    DEFAULT_XGB_PARAMS,
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    MarketProbConfig,
    OptunaConfig,
    ScoreModel,
    TrainingResult,
    _adjust_home_win_prob,
    _apply_feature_spec,
    _build_feature_spec,
    _build_preprocessor,
    _drop_injury_feature_columns,
    _early_stopping_info,
    _evaluate_margin_total_predictions,
    _evaluate_predictions,
    _filter_season_bounds,
    _fit_blend_ridge_constrained,
    _fit_margin_total_models,
    _fit_models,
    _fit_quantile_models,
    _fit_win_prob_calibrator,
    _get_target_columns,
    _load_games,
    _predict_home_win_prob,
    _predict_margin_total_from_model,
    _predict_xgb,
    _prepare_margin_total_targets,
    _prepare_margin_total_targets_with_anchor,
    _resolve_xgb_params,
    _run_optuna_search,
    _split_by_season,
    _split_train_calibration_holdout,
    _summarize_confidence_pool,
    _summarize_missing_data,
    _validate_quantiles,
    get_market_baseline,
)
from nfl_predictor.utils.logger import log


def _filter_to_regular_season_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a dataset to regular season rows when `game_type` exists.

    This is the default for training and evaluation splits. Prediction can still be run on
    postseason rows (or any rows) as long as features are available.
    """

    if "game_type" not in df.columns:
        return df

    filtered = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
    dropped = len(df) - len(filtered)
    if dropped:
        log.info("Filtering to regular season for training: dropped %d rows.", dropped)
    return filtered


def train_score_model(
    data_path: Path,
    holdout_seasons: int,
    include_market: bool,
    include_injuries: bool,
    max_cardinality_ratio: float,
    market_prob_config: Optional[MarketProbConfig],
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
    xgb_tree_method: Optional[str] = None,
    xgb_device: Optional[str] = None,
    xgb_n_jobs: Optional[int] = None,
) -> ScoreModel:
    """Train score models using time-aware season splits."""
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    if not include_injuries:
        df, dropped = _drop_injury_feature_columns(df)
        if dropped:
            log.info("Injury features disabled; dropping %d columns.", len(dropped))

    df = _filter_season_bounds(df, min_season, max_season)
    df = _filter_to_regular_season_for_training(df)
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

    preprocessor = _build_preprocessor(feature_spec, for_tree=True)
    x_train = preprocessor.fit_transform(x_train_df)
    xgb_overrides: dict[str, Any] = {}
    if xgb_n_jobs is not None:
        xgb_overrides["n_jobs"] = xgb_n_jobs
    params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=xgb_overrides or None,
        tree_method=xgb_tree_method,
        device=xgb_device,
    )
    away_model, home_model = _fit_models(x_train, train_df, target_columns, params=params)

    if not holdout_df.empty:
        x_holdout = preprocessor.transform(x_holdout_df)
        pred_away = _predict_xgb(away_model, x_holdout)
        pred_home = _predict_xgb(home_model, x_holdout)

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
        market_prob_config=market_prob_config,
        xgb_params=params,
    )


def train_score_model_with_report(
    **kwargs: Any,
) -> TrainingResult:
    """Train a score model and return a structured metrics report payload."""
    data_path: Path = kwargs["data_path"]
    kwargs.setdefault("include_injuries", True)

    model: ScoreModel = train_score_model(**kwargs)
    df = _load_games(data_path)
    df = df.dropna(subset=list(model.target_columns))
    df = _filter_season_bounds(df, kwargs.get("min_season"), kwargs.get("max_season"))
    df = _filter_to_regular_season_for_training(df)

    missing_data_summary = _summarize_missing_data(df)
    _train_df, holdout_df, holdout = _split_by_season(df, kwargs["holdout_seasons"])
    metrics: dict[str, Any] = {}
    if not holdout_df.empty:
        x_holdout = model.preprocessor.transform(
            _apply_feature_spec(holdout_df, model.feature_spec)
        )
        pred_away = _predict_xgb(model.away_model, x_holdout)
        pred_home = _predict_xgb(model.home_model, x_holdout)
        metrics = _evaluate_predictions(holdout_df, pred_away, pred_home, model.target_columns)

    report = {
        "kind": "train",
        "model_kind": "score",
        "metrics": {"holdout": metrics or None},
        "missing_data": missing_data_summary,
    }
    splits = {
        "train_seasons": sorted(_train_df["season"].dropna().unique().tolist()),
        "holdout_seasons": holdout,
    }
    params = model.xgb_params or DEFAULT_XGB_PARAMS.copy()
    feature_list = list(model.feature_spec.feature_columns)
    return TrainingResult(
        model=model,
        metrics_report=report,
        splits=splits,
        params=params,
        tuned_params=None,
        feature_list=feature_list,
        early_stopping=_early_stopping_info(model),
    )


def train_margin_total_model(
    data_path: Path,
    holdout_seasons: int,
    calibration_seasons: int,
    calibration_weeks: int,
    include_market: bool,
    include_injuries: bool,
    max_cardinality_ratio: float,
    win_prob_calibration: str,
    optuna_config: OptunaConfig,
    market_transform: bool,
    market_anchor: bool,
    market_prob_config: Optional[MarketProbConfig],
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    feature_start: str = DEFAULT_FEATURE_START_COLUMN,
    feature_end: str = DEFAULT_FEATURE_END_COLUMN,
) -> MarginTotalModel:
    """Train margin/total models with optional calibration."""
    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))

    if not include_injuries:
        df, dropped = _drop_injury_feature_columns(df)
        if dropped:
            log.info("Injury features disabled; dropping %d columns.", len(dropped))

    df = _filter_season_bounds(df, min_season, max_season)
    df = _filter_to_regular_season_for_training(df)
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
    if market_prob_config is not None:
        log.info(
            "Market win-prob adjustment: blend=%.2f clamp=%.2f",
            market_prob_config.blend_weight,
            market_prob_config.clamp_delta,
        )
    if market_anchor:
        get_market_baseline(train_df)
        log.info("Market anchor enabled: training residuals vs spread/total.")
    if market_prob_config is not None:
        log.info(
            "Market win-prob adjustment: blend=%.2f clamp=%.2f",
            market_prob_config.blend_weight,
            market_prob_config.clamp_delta,
        )

    tuned_params: dict[str, Any] = {}
    tuned_cv_summary: Optional[dict[str, Any]] = None
    if optuna_config.enabled:
        tuned_params, tuned_cv_summary = _run_optuna_search(
            train_df,
            target_columns=target_columns,
            include_market=include_market,
            max_cardinality_ratio=max_cardinality_ratio,
            feature_start=feature_start,
            feature_end=feature_end,
            optuna_config=optuna_config,
            market_transform=market_transform,
            market_anchor=market_anchor,
            market_prob_config=market_prob_config,
        )

    params_overrides = tuned_params.copy()
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
        calib_frame: pd.DataFrame,
    ) -> tuple[
        ColumnTransformer,
        FeatureSpec,
        xgb.XGBRegressor,
        xgb.XGBRegressor,
        dict[float, xgb.XGBRegressor],
        dict[float, xgb.XGBRegressor],
        tuple[float, ...],
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

        local_preprocessor = _build_preprocessor(local_spec, for_tree=True)
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

        quantiles = _validate_quantiles(DEFAULT_QUANTILES)
        margin_quantiles = _fit_quantile_models(
            x_train,
            y_margin_train,
            params,
            quantiles,
            x_eval=x_calibration,
            y_eval=y_margin_calibration,
            early_stopping_rounds=optuna_config.early_stopping_rounds,
        )
        total_quantiles = _fit_quantile_models(
            x_train,
            y_total_train,
            params,
            quantiles,
            x_eval=x_calibration,
            y_eval=y_total_calibration,
            early_stopping_rounds=optuna_config.early_stopping_rounds,
        )

        return (
            local_preprocessor,
            local_spec,
            margin_model,
            total_model,
            margin_quantiles,
            total_quantiles,
            quantiles,
            x_calibration,
            baseline_margin_calibration,
        )

    (
        preprocessor,
        feature_spec,
        margin_model,
        total_model,
        margin_quantile_models,
        total_quantile_models,
        quantiles,
        x_calibration,
        baseline_margin_calibration,
    ) = _train_models(train_df, calibration_df)

    calibrator = None
    if win_prob_calibration != "none":
        if calibration_df.empty:
            raise ValueError("Calibration requested but no calibration seasons configured.")
        if x_calibration is None:
            raise ValueError("Calibration features are unavailable.")
        pred_margin_calib = _predict_xgb(margin_model, x_calibration)
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
        pred_margin = _predict_xgb(margin_model, x_holdout)
        pred_total = _predict_xgb(total_model, x_holdout)
        if market_anchor:
            baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _predict_home_win_prob(pred_margin, calibrator)
        home_win_prob = _adjust_home_win_prob(holdout_df, home_win_prob, market_prob_config)

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
        margin_quantile_models=margin_quantile_models,
        total_quantile_models=total_quantile_models,
        quantiles=quantiles,
        market_anchor=market_anchor,
        market_prob_config=market_prob_config,
        xgb_params=params,
        tuned_params=tuned_params or None,
        tuned_cv_summary=tuned_cv_summary,
    )


def train_margin_total_model_with_report(
    **kwargs: Any,
) -> TrainingResult:
    """Train a margin/total model and return a structured metrics report payload."""
    data_path: Path = kwargs["data_path"]
    holdout_seasons: int = kwargs["holdout_seasons"]
    calibration_seasons: int = kwargs["calibration_seasons"]
    calibration_weeks: int = kwargs["calibration_weeks"]

    kwargs.setdefault("include_injuries", True)

    df = _load_games(data_path)
    target_columns = _get_target_columns(df)
    df = df.dropna(subset=list(target_columns))
    df = _filter_season_bounds(df, kwargs.get("min_season"), kwargs.get("max_season"))
    missing_data_summary = _summarize_missing_data(df)
    split = _split_train_calibration_holdout(
        df, holdout_seasons, calibration_seasons, calibration_weeks
    )
    holdout_df = split[2]
    train_seasons = split[3]
    calibration = split[4]
    holdout = split[5]
    calibration_season_inseason = split[6]
    calibration_weeks_inseason = split[7]

    # Train the actual model (this will also log holdout metrics).
    model: MarginTotalModel = train_margin_total_model(**kwargs)

    holdout_metrics: Optional[dict[str, Any]] = None
    pool_summary: Optional[dict[str, Any]] = None
    if not holdout_df.empty:
        x_holdout = model.preprocessor.transform(
            _apply_feature_spec(holdout_df, model.feature_spec)
        )
        pred_margin = _predict_xgb(model.margin_model, x_holdout)
        pred_total = _predict_xgb(model.total_model, x_holdout)
        if model.market_anchor:
            baseline_margin_holdout, baseline_total_holdout = get_market_baseline(holdout_df)
            pred_margin = pred_margin + baseline_margin_holdout
            pred_total = pred_total + baseline_total_holdout
        home_win_prob = _predict_home_win_prob(pred_margin, model.calibrator)
        home_win_prob = _adjust_home_win_prob(holdout_df, home_win_prob, model.market_prob_config)
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
        "missing_data": missing_data_summary,
    }

    splits: dict[str, Any] = {
        "train_seasons": train_seasons,
        "calibration_seasons": calibration,
        "holdout_seasons": holdout,
        "calibration_inseason": {
            "season": calibration_season_inseason,
            "weeks": calibration_weeks_inseason,
        },
    }
    params = model.xgb_params or DEFAULT_XGB_PARAMS.copy()
    feature_list = list(model.feature_spec.feature_columns)
    return TrainingResult(
        model=model,
        metrics_report=report,
        splits=splits,
        params=params,
        tuned_params=model.tuned_params,
        feature_list=feature_list,
        early_stopping=_early_stopping_info(model),
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
    market_prob_config: Optional[MarketProbConfig],
    include_injuries: bool = True,
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

    if not include_injuries:
        df, dropped = _drop_injury_feature_columns(df)
        if dropped:
            log.info("Injury features disabled; dropping %d columns.", len(dropped))

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
    tuned_cv_summary: dict[str, Any] = {}
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
                xgb_n_jobs=optuna_config.xgb_n_jobs,
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
                xgb_n_jobs=optuna_config.xgb_n_jobs,
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
                xgb_n_jobs=team_optuna.xgb_n_jobs,
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
                xgb_n_jobs=market_optuna.xgb_n_jobs,
            )

        if tune_scope in {"team", "both"}:
            log.info("Tuning team-feature model hyperparameters...")
            team_params, tuned_cv_summary["team"] = _run_optuna_search(
                train_df,
                target_columns=target_columns,
                include_market=False,
                max_cardinality_ratio=max_cardinality_ratio,
                feature_start=feature_start,
                feature_end=feature_end,
                optuna_config=team_optuna,
                market_only=False,
                market_transform=market_transform,
                market_prob_config=market_prob_config,
            )
        if tune_scope in {"market", "both"}:
            log.info(
                "Skipping market-only model tuning: blended models now use market baseline only."
            )

    team_overrides = team_params.copy()
    if optuna_config.xgb_n_jobs is not None:
        team_overrides["n_jobs"] = optuna_config.xgb_n_jobs
    team_xgb_params = _resolve_xgb_params(
        DEFAULT_XGB_PARAMS,
        overrides=team_overrides or None,
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
    team_preprocessor = _build_preprocessor(team_spec, for_tree=True)

    team_train = team_preprocessor.fit_transform(_apply_feature_spec(train_df, team_spec))
    y_margin_train, y_total_train = _prepare_margin_total_targets(train_df, target_columns)

    team_calib = team_preprocessor.transform(_apply_feature_spec(calibration_df, team_spec))
    y_margin_calib, y_total_calib = _prepare_margin_total_targets(calibration_df, target_columns)
    market_margin_calib, market_total_calib = get_market_baseline(calibration_df)

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
    team_margin_calib = _predict_xgb(team_margin_model, team_calib)
    team_total_calib = _predict_xgb(team_total_model, team_calib)

    margin_blender = _fit_blend_ridge_constrained(
        np.column_stack([team_margin_calib, market_margin_calib]),
        y_margin_calib,
        alpha=1.0,
    )
    total_blender = _fit_blend_ridge_constrained(
        np.column_stack([team_total_calib, market_total_calib]),
        y_total_calib,
        alpha=1.0,
    )

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
        market_margin_holdout, market_total_holdout = get_market_baseline(holdout_df)

        team_margin_holdout = _predict_xgb(team_margin_model, team_holdout)
        team_total_holdout = _predict_xgb(team_total_model, team_holdout)

        blended_margin = margin_blender.predict(
            np.column_stack([team_margin_holdout, market_margin_holdout])
        )
        blended_total = total_blender.predict(
            np.column_stack([team_total_holdout, market_total_holdout])
        )
        home_win_prob = _predict_home_win_prob(blended_margin, calibrator)
        home_win_prob = _adjust_home_win_prob(holdout_df, home_win_prob, market_prob_config)

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
        market_prob_config=market_prob_config,
        tuned_params=team_params or None,
        tuned_cv_summary=tuned_cv_summary.get("team"),
    )
    return BlendedMarginTotalModel(
        team_model=team_model,
        market_model=None,
        blend_layer=BlendLayer(margin_model=margin_blender, total_model=total_blender),
        calibrator=calibrator,
        target_columns=target_columns,
        market_prob_config=market_prob_config,
        xgb_params={"team": team_xgb_params},
        tuned_params={"team": team_params or None},
        tuned_cv_summary=tuned_cv_summary or None,
    )


def train_blended_margin_total_model_with_report(
    **kwargs: Any,
) -> TrainingResult:
    """Train a blended model and return a structured metrics report payload."""
    data_path: Path = kwargs["data_path"]
    kwargs.setdefault("include_injuries", True)

    model: BlendedMarginTotalModel = train_blended_margin_total_model(**kwargs)
    df = _load_games(data_path)
    df = df.dropna(subset=list(model.target_columns))
    df = _filter_season_bounds(df, kwargs.get("min_season"), kwargs.get("max_season"))
    missing_data_summary = _summarize_missing_data(df)
    split = _split_train_calibration_holdout(
        df,
        kwargs["holdout_seasons"],
        kwargs["calibration_seasons"],
        kwargs["calibration_weeks"],
    )
    holdout_df = split[2]
    train_seasons = split[3]
    calibration = split[4]
    holdout = split[5]
    calibration_season_inseason = split[6]
    calibration_weeks_inseason = split[7]

    holdout_metrics: Optional[dict[str, Any]] = None
    if not holdout_df.empty:
        team_margin, team_total = _predict_margin_total_from_model(model.team_model, holdout_df)
        if model.market_model is None:
            market_margin, market_total = get_market_baseline(holdout_df)
        else:
            market_margin, market_total = _predict_margin_total_from_model(
                model.market_model, holdout_df
            )
        blended_margin = model.blend_layer.margin_model.predict(
            np.column_stack([team_margin, market_margin])
        )
        blended_total = model.blend_layer.total_model.predict(
            np.column_stack([team_total, market_total])
        )
        home_win_prob = _predict_home_win_prob(blended_margin, model.calibrator)
        home_win_prob = _adjust_home_win_prob(holdout_df, home_win_prob, model.market_prob_config)
        holdout_metrics = _evaluate_margin_total_predictions(
            holdout_df, blended_margin, blended_total, model.target_columns, home_win_prob
        )

    report = {
        "kind": "train",
        "model_kind": "blend",
        "metrics": {"holdout": holdout_metrics},
        "tuning_cv": model.tuned_cv_summary,
        "missing_data": missing_data_summary,
    }
    splits = {
        "train_seasons": train_seasons,
        "calibration_seasons": calibration,
        "holdout_seasons": holdout,
        "calibration_inseason": {
            "season": calibration_season_inseason,
            "weeks": calibration_weeks_inseason,
        },
    }
    params = model.xgb_params or DEFAULT_XGB_PARAMS.copy()
    feature_list = list(model.team_model.feature_spec.feature_columns)
    return TrainingResult(
        model=model,
        metrics_report=report,
        splits=splits,
        params=params,
        tuned_params=model.tuned_params,
        feature_list=feature_list,
        early_stopping=_early_stopping_info(model),
    )
