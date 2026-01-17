"""CLI entrypoint for training and prediction.

This module hosts argument parsing and `main()` that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nfl_predictor import constants
from nfl_predictor.ml import artifacts
from nfl_predictor.ml.ml_model_core import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_OPTUNA_CV_SPLITS,
    DEFAULT_OPTUNA_TIMEOUT_SECONDS,
    MarketProbConfig,
    OptunaConfig,
    TrainingResult,
    _load_model_checkpoint,
    _with_market_prob_config,
    normalize_win_prob_calibration_method,
)
from nfl_predictor.ml.ml_model_predict import (
    predict_week,
    predict_week_blended,
    predict_week_margin_total,
)
from nfl_predictor.ml.ml_model_training import (
    train_blended_margin_total_model_with_report,
    train_margin_total_model_with_report,
    train_score_model_with_report,
)
from nfl_predictor.utils.logger import log


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
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include postseason games in training when a game_type column exists. "
            "Default: False (train on regular season only)."
        ),
    )
    parser.add_argument(
        "--postseason-weight",
        type=float,
        default=1.0,
        help=(
            "Sample-weight multiplier for postseason rows when --include-postseason is enabled. "
            "Default: 1.0 (no upweight)."
        ),
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
        "--market-prob-weight",
        type=float,
        default=None,
        help=(
            "Market probability weight for post-processing (0=off, 1=market only). "
            "Alias for --market-prob-blend."
        ),
    )
    parser.add_argument(
        "--market-prob-blend",
        type=float,
        default=0.0,
        help="Market probability weight for post-processing (0=off, 1=market only).",
    )
    parser.add_argument(
        "--market-prob-clamp",
        type=float,
        default=0.0,
        help="Clamp model probability within +/- this delta of market (0=off).",
    )
    parser.add_argument(
        "--market-prob-source",
        choices=["raw", "novig"],
        default="raw",
        help="Market probability source for blending/clamping.",
    )
    parser.add_argument(
        "--market-prob-blend-method",
        choices=["prob", "logit"],
        default="prob",
        help="Blend method for market probabilities (prob or logit space).",
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
        "--win-prob-calibration",
        choices=["none", "platt", "isotonic", "elo", "auto", "logistic"],
        default="isotonic",
        help="Calibration method for win probabilities (logistic is an alias for platt).",
    )
    parser.add_argument(
        "--win-prob-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use margin quantiles to derive uncertainty-aware win probabilities.",
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
        help="XGBoost tree_method (e.g., auto, hist, approx, exact).",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default="auto",
        help="XGBoost device (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost parallel threads (default: os.cpu_count()).",
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
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory to write model.joblib, metadata.json, and metrics_report.json. "
            "If set, --model-out must be inside this directory (or omitted)."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id used when writing --run-dir (defaults to directory name).",
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
    parser.add_argument(
        "--score-rounding",
        choices=["none", "int", "half", "nfl"],
        default="none",
        help=(
            "Optional post-processing for predicted scores (does not change training): "
            "none|int|half|nfl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training and prediction."""

    args = _parse_args()
    args.win_prob_calibration = normalize_win_prob_calibration_method(args.win_prob_calibration)
    win_prob_use_uncertainty = bool(args.win_prob_uncertainty)
    if args.model_kind != "margin_total" and win_prob_use_uncertainty:
        log.warning("Uncertainty-aware win prob is only supported for margin_total models.")
        win_prob_use_uncertainty = False

    created_at = artifacts.now_utc_iso()
    dataset_hash = artifacts.sha256_file(args.data_path)

    config_payload: dict[str, Any] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }

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
        xgb_n_jobs=args.xgb_n_jobs,
    )

    market_prob_weight = args.market_prob_weight
    if market_prob_weight is None:
        market_prob_weight = args.market_prob_blend

    market_prob_config = MarketProbConfig(
        blend_weight=float(market_prob_weight),
        clamp_delta=float(args.market_prob_clamp),
        prob_source=args.market_prob_source,
        blend_method=args.market_prob_blend_method,
    )
    if float(market_prob_weight) == 0.0 and float(args.market_prob_clamp) == 0.0:
        market_prob_config = None

    output_path = args.output_path
    if args.predict_path and output_path is None:
        output_path = args.predict_path.with_name(f"{args.predict_path.stem}_predictions.csv")

    if args.model_in is not None:
        if args.tune:
            log.info("Model checkpoint provided; ignoring training and Optuna tuning.")
        model = _load_model_checkpoint(args.model_in, args.model_kind)
        model = _with_market_prob_config(model, market_prob_config)
        if not args.predict_path:
            log.info("No --predict-path provided; exiting after loading model.")
            return
        use_uncertainty = win_prob_use_uncertainty or getattr(
            model, "win_prob_use_uncertainty", False
        )
        if args.model_kind == "score":
            predict_week(
                model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
            )
        elif args.model_kind == "margin_total":
            predict_week_margin_total(
                model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
                win_prob_use_uncertainty=use_uncertainty,
            )
        elif args.model_kind == "blend":
            predict_week_blended(
                model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
            )
        else:
            raise ValueError(f"Unknown model kind: {args.model_kind}")
        return

    def _write_artifacts(result: TrainingResult, model_out: Path) -> None:
        run_dir = args.run_dir or model_out.parent
        if args.run_dir is not None and model_out.parent != args.run_dir:
            raise ValueError("--model-out must be inside --run-dir")

        run_id = args.run_id or run_dir.name
        paths = artifacts.resolve_run_paths(run_id, run_dir=run_dir)

        artifacts.save_model(paths.model_path, result.model)

        metrics_report = {
            "run_id": run_id,
            "created_at": created_at,
            "config": config_payload,
            "splits": result.splits,
            "metrics": result.metrics_report,
        }
        artifacts.write_json(paths.metrics_path, metrics_report)

        metadata = artifacts.build_metadata(
            created_at=created_at,
            run_id=run_id,
            dataset_hash=dataset_hash,
            config=config_payload,
            feature_list=result.feature_list,
            splits=result.splits,
            params=result.params,
            tuned_params=result.tuned_params,
            early_stopping=result.early_stopping,
            optuna_summary=getattr(result.model, "optuna_summary", None),
        )
        artifacts.write_json(paths.metadata_path, metadata)

    if args.model_kind == "score":
        result = train_score_model_with_report(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            include_market=not args.exclude_market,
            max_cardinality_ratio=args.max_cardinality_ratio,
            market_prob_config=market_prob_config,
            include_postseason=args.include_postseason,
            postseason_weight=args.postseason_weight,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
            xgb_tree_method=args.xgb_tree_method,
            xgb_device=args.xgb_device,
            xgb_n_jobs=args.xgb_n_jobs,
        )

        if args.run_dir is not None and args.model_out is None:
            args.model_out = args.run_dir / "model.joblib"
        if args.model_out is not None:
            _write_artifacts(result, args.model_out)
        if args.predict_path:
            predict_week(
                result.model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
            )
        return

    if args.model_kind == "margin_total":
        result = train_margin_total_model_with_report(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            calibration_seasons=args.calibration_seasons,
            calibration_weeks=args.calibration_weeks,
            include_market=not args.exclude_market,
            max_cardinality_ratio=args.max_cardinality_ratio,
            win_prob_calibration=args.win_prob_calibration,
            win_prob_use_uncertainty=win_prob_use_uncertainty,
            optuna_config=optuna_config,
            market_transform=args.market_transform,
            market_anchor=args.market_anchor,
            market_prob_config=market_prob_config,
            include_postseason=args.include_postseason,
            postseason_weight=args.postseason_weight,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
        )

        if args.run_dir is not None and args.model_out is None:
            args.model_out = args.run_dir / "model.joblib"
        if args.model_out is not None:
            _write_artifacts(result, args.model_out)
        if args.predict_path:
            predict_week_margin_total(
                result.model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
                win_prob_use_uncertainty=win_prob_use_uncertainty,
            )
        return

    if args.model_kind == "blend":
        result = train_blended_margin_total_model_with_report(
            data_path=args.data_path,
            holdout_seasons=args.holdout_seasons,
            calibration_seasons=args.calibration_seasons,
            calibration_weeks=args.calibration_weeks,
            max_cardinality_ratio=args.max_cardinality_ratio,
            win_prob_calibration=args.win_prob_calibration,
            optuna_config=optuna_config,
            market_transform=args.market_transform,
            market_anchor=args.market_anchor,
            market_prob_config=market_prob_config,
            include_postseason=args.include_postseason,
            postseason_weight=args.postseason_weight,
            min_season=args.min_season,
            max_season=args.max_season,
            feature_start=args.feature_start,
            feature_end=args.feature_end,
        )

        if args.run_dir is not None and args.model_out is None:
            args.model_out = args.run_dir / "model.joblib"
        if args.model_out is not None:
            _write_artifacts(result, args.model_out)
        if args.predict_path:
            predict_week_blended(
                result.model,
                args.predict_path,
                output_path,
                pretty_output=args.pretty_output,
                score_rounding=args.score_rounding,
            )
        return

    raise ValueError(f"Unknown model kind: {args.model_kind}")


if __name__ == "__main__":
    main()
