"""Train a model, or predict a week with a saved one.

``nfl-predictor train`` and ``nfl-predictor predict`` (which requires ``--model-in``) run this
module's ``main``, as does ``python -m nfl_predictor.ml_model``. It lives outside
``nfl_predictor/ml/`` so that editing the command line never changes a walk-forward
checkpoint fingerprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.cli import options
from nfl_predictor.ml import artifacts, floor_sigma
from nfl_predictor.ml.ml_model_core import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_OPTUNA_CV_SPLITS,
    DEFAULT_OPTUNA_TIMEOUT_SECONDS,
    OptunaConfig,
    load_model_checkpoint,
)
from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
from nfl_predictor.ml.ml_model_training import (
    TrainingOptions,
    TrainingRecord,
    train_margin_total_model_with_report,
    write_training_artifacts,
)
from nfl_predictor.ml.ml_model_xgb_utils import (
    XGB_DEVICE_AUTO,
    XGB_DEVICE_HELP,
    resolve_xgb_device,
    xgb_device_arg,
)
from nfl_predictor.utils.logger import log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFL score prediction models.")
    options.add_model_kind_option(parser, "Model pipeline to use")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=constants.DATA_PATH / "completed_games_ml.csv",
        help="Path to completed games dataset.",
    )
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=2,
        help=(
            "Number of most recent seasons to hold out for evaluation; the model trains on "
            "every other completed game."
        ),
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
        "--recency-half-life-seasons",
        type=float,
        default=None,
        help="Optional exponential half-life in seasons for recency weighting.",
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
        "--win-prob-calibration",
        choices=["auto", "none"],
        default="auto",
        help=(
            "Win-probability calibration: auto, the deterministic floor (the predicted margin "
            "through a normal curve whose spread is the root-mean-square out-of-fold error "
            "before the predicted week); none is the same."
        ),
    )
    parser.add_argument(
        "--floor-sigma-reference-runs",
        type=Path,
        nargs="*",
        default=[Path(path) for path in constants.FLOOR_SIGMA_REFERENCE_RUNS],
        help=(
            "Walk-forward runs (run or fold checkpoint directories, relative to the repository "
            "root) whose out-of-fold margin errors set the floor's sigma, which the saved model "
            "records; their checkpoints are only read. A missing run is an error; give no "
            "paths to train without them (the constant, recorded as the fallback)."
        ),
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
        "--tune-objective",
        "--tune-metric",
        dest="tune_metric",
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
        "--tune-cv-splits",
        "--cv-splits",
        dest="cv_splits",
        type=int,
        default=DEFAULT_OPTUNA_CV_SPLITS,
        help="Number of time-series CV folds for tuning.",
    )
    parser.add_argument(
        "--tune-early-stopping-rounds",
        "--early-stopping-rounds",
        dest="early_stopping_rounds",
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
        type=xgb_device_arg,
        default=XGB_DEVICE_AUTO,
        help=XGB_DEVICE_HELP,
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost parallel threads (default: os.cpu_count()).",
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


def _predicted_week(predict_path: Path | None) -> tuple[int, int] | None:
    """Return the one season and week of a prediction file, else None (the week after the data)."""
    if predict_path is None:
        return None
    weeks = pd.read_csv(predict_path, usecols=["season", "week"]).drop_duplicates()
    if len(weeks) != 1:
        return None
    return int(weeks["season"].iloc[0]), int(weeks["week"].iloc[0])


def _optuna_config(args: argparse.Namespace) -> OptunaConfig:
    """Build the Optuna config, naming a default study when a storage is given without one."""
    study_name = args.tune_study_name
    if args.tune_storage and study_name is None:
        study_name = f"nfl_predictor_{args.model_kind}_{args.tune_metric}"
        log.info("Using default Optuna study name: %s", study_name)

    return OptunaConfig(
        enabled=args.tune,
        timeout_seconds=args.tune_timeout,
        n_trials=args.tune_trials,
        cv_splits=args.cv_splits,
        objective=args.tune_metric,
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method=args.xgb_tree_method,
        device=args.xgb_device,
        storage=args.tune_storage,
        study_name=study_name,
        best_params_out=args.tune_best_params_out,
        xgb_n_jobs=args.xgb_n_jobs,
    )


def _artifact_run_dir(args: argparse.Namespace) -> Path:
    """Return the run directory: ``--run-dir``, which must hold ``--model-out``, or its folder."""
    model_out: Path = args.model_out
    if args.run_dir is not None and model_out.parent != args.run_dir:
        msg = "--model-out must be inside --run-dir"
        raise ValueError(msg)
    return args.run_dir or model_out.parent


def _predict_from_checkpoint(args: argparse.Namespace, output_path: Path | None) -> None:
    """Load ``--model-in`` and predict ``--predict-path`` with it, training nothing."""
    if args.tune:
        log.info("Model checkpoint provided; ignoring training and Optuna tuning.")
    model = load_model_checkpoint(args.model_in, args.model_kind)
    if not args.predict_path:
        log.info("No --predict-path provided; exiting after loading model.")
        return
    predict_week_margin_total(
        model,
        args.predict_path,
        output_path,
        pretty_output=args.pretty_output,
        score_rounding=args.score_rounding,
    )


def _training_options(args: argparse.Namespace, optuna_config: OptunaConfig) -> TrainingOptions:
    """Return the training options the arguments ask for, with the reference sigma pool."""
    return TrainingOptions(
        data_path=args.data_path,
        holdout_seasons=args.holdout_seasons,
        include_market=not args.exclude_market,
        max_cardinality_ratio=args.max_cardinality_ratio,
        optuna_config=optuna_config,
        market_transform=args.market_transform,
        market_anchor=args.market_anchor,
        include_postseason=args.include_postseason,
        postseason_weight=args.postseason_weight,
        recency_half_life_seasons=args.recency_half_life_seasons,
        min_season=args.min_season,
        max_season=args.max_season,
        feature_start=args.feature_start,
        feature_end=args.feature_end,
        floor_sigma_pool=floor_sigma.load_reference_pool(args.floor_sigma_reference_runs),
        floor_sigma_week=_predicted_week(args.predict_path),
    )


def main() -> None:
    """CLI entry point for training and prediction."""
    args = _parse_args()

    created_at = artifacts.now_utc_iso()
    dataset_hash = artifacts.sha256_file(args.data_path)
    # Resolve `auto` once, so the fit and the recorded config name the same device.
    args.xgb_device = resolve_xgb_device(args.xgb_device)

    config_payload: dict[str, Any] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    optuna_config = _optuna_config(args)

    output_path = args.output_path
    if args.predict_path and output_path is None:
        output_path = args.predict_path.with_name(f"{args.predict_path.stem}_predictions.csv")

    if args.model_in is not None:
        _predict_from_checkpoint(args, output_path)
        return

    result = train_margin_total_model_with_report(_training_options(args, optuna_config))

    if args.run_dir is not None and args.model_out is None:
        args.model_out = args.run_dir / "model.joblib"
    if args.model_out is not None:
        run_dir = _artifact_run_dir(args)
        record = TrainingRecord(
            run_id=args.run_id or run_dir.name,
            created_at=created_at,
            dataset_hash=dataset_hash,
            config_payload=config_payload,
        )
        write_training_artifacts(result, record, run_dir)
    if args.predict_path:
        predict_week_margin_total(
            result.model,
            args.predict_path,
            output_path,
            pretty_output=args.pretty_output,
            score_rounding=args.score_rounding,
        )


if __name__ == "__main__":
    main()
