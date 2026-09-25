"""Weekly-run configuration: the JSON/YAML config file, its validation, and the parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core, walk_forward
from nfl_predictor.reporting import power_rankings

_PATH_KEYS = {
    "config",
    "data_path",
    "predict_path",
    "run_dir",
    "output_dir",
    "power_rankings_data_ml",
    "power_rankings_data_schedule",
    "power_rankings_out_dir",
    "power_rankings_strength_snapshots",
}


def _load_config(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs.") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError("Config file must parse to a JSON/YAML object.")
    return payload


def _normalize_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize config defaults for argparse."""
    normalized = dict(config)
    for key in _PATH_KEYS:
        if key in normalized and normalized[key] is not None:
            normalized[key] = Path(normalized[key])
    return normalized


def _allowed_config_keys(parser: argparse.ArgumentParser) -> set[str]:
    """Return allowable config keys based on parser destinations."""
    return {action.dest for action in parser._actions if action.dest != "help"}


def _validate_config_keys(config: dict[str, Any], allowed: set[str]) -> None:
    """Raise if config includes unsupported keys."""
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")


def _build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Weekly orchestration runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults.get("config"),
        help="Optional JSON/YAML config file for arguments.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=defaults.get("data_path", Path(constants.DATA_PATH) / "completed_games_ml.csv"),
        help="Completed games dataset path.",
    )
    parser.add_argument(
        "--predict-path",
        type=Path,
        default=defaults.get("predict_path"),
        help="Optional upcoming games CSV (defaults to newest in data/predict/).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=defaults.get("run_id"),
        help="Optional run id (default: generated).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=defaults.get("run_dir"),
        help="Optional run directory (default: models/<run_id>/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.get("output_dir"),
        help="Optional output directory for weekly artifacts (default: run dir).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("resume", True),
        help="Reuse existing artifacts when inputs match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=defaults.get("dry_run", False),
        help="Print planned outputs without running stages.",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        default=defaults.get("skip_data_refresh", False),
        help="Skip the data collection refresh step.",
    )
    parser.add_argument(
        "--data-collection-args",
        type=str,
        default=defaults.get("data_collection_args"),
        help=(
            "Extra arguments for the data collection refresh, as one shell-quoted string "
            '(for example "--min-season 2010 --stat-prior-blend-games 4").'
        ),
    )
    parser.add_argument(
        "--score-rounding",
        choices=["none", "int", "half", "nfl"],
        default=defaults.get("score_rounding", "none"),
        help="Score rounding for weekly predictions.",
    )
    parser.add_argument(
        "--wf-eval-last-n-seasons",
        type=int,
        default=defaults.get("wf_eval_last_n_seasons", 3),
        help="Walk-forward: evaluate last N seasons.",
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=defaults.get("wf_start_week", 3),
        help="Walk-forward: start week.",
    )
    parser.add_argument(
        "--wf-calibration-weeks",
        type=int,
        default=defaults.get("wf_calibration_weeks", walk_forward.DEFAULT_CALIBRATION_WEEKS),
        help="Walk-forward: calibration weeks.",
    )
    parser.add_argument(
        "--wf-include-postseason",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("wf_include_postseason", False),
        help="Walk-forward: include postseason folds.",
    )
    parser.add_argument(
        "--wf-checkpoint-per-fold",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("wf_checkpoint_per_fold", False),
        help="Walk-forward: emit per-fold progress checkpoints (optional).",
    )
    parser.add_argument(
        "--wf-exclude-incomplete-seasons",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("wf_exclude_incomplete_seasons", False),
        help=("Walk-forward: exclude seasons whose regular season is incomplete in the dataset."),
    )
    parser.add_argument(
        "--wf-recency-half-life-seasons",
        type=float,
        default=defaults.get("wf_recency_half_life_seasons"),
        help="Walk-forward: optional exponential half-life in seasons for recency weighting.",
    )
    parser.add_argument(
        "--wf-market-mode",
        choices=["features", "anchor", "hybrid", "all"],
        default=defaults.get("wf_market_mode", "hybrid"),
        help="Walk-forward: market mode selection.",
    )
    parser.add_argument(
        "--wf-market-prob-source",
        choices=["raw", "novig", "both"],
        default=defaults.get("wf_market_prob_source", "raw"),
        help="Walk-forward: market probability source.",
    )
    parser.add_argument(
        "--wf-market-prob-blend-method",
        choices=["prob", "logit", "both"],
        default=defaults.get("wf_market_prob_blend_method", "prob"),
        help="Walk-forward: market probability blend method.",
    )
    parser.add_argument(
        "--wf-win-prob-uncertainty",
        choices=["off", "on", "both"],
        default=defaults.get("wf_win_prob_uncertainty", "off"),
        help="Walk-forward: use uncertainty-aware win probabilities.",
    )
    parser.add_argument(
        "--wf-include-quantiles",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("wf_include_quantiles", False),
        help="Walk-forward: include quantile models (slow).",
    )
    parser.add_argument(
        "--wf-n-estimators",
        type=int,
        default=defaults.get("wf_n_estimators", ml_model_core.DEFAULT_XGB_PARAMS["n_estimators"]),
        help="Walk-forward: XGBoost n_estimators override.",
    )
    parser.add_argument(
        "--wf-max-depth",
        type=int,
        default=defaults.get("wf_max_depth", ml_model_core.DEFAULT_XGB_PARAMS["max_depth"]),
        help="Walk-forward: XGBoost max_depth override.",
    )
    parser.add_argument(
        "--wf-learning-rate",
        type=float,
        default=defaults.get(
            "wf_learning_rate",
            ml_model_core.DEFAULT_XGB_PARAMS["learning_rate"],
        ),
        help="Walk-forward: XGBoost learning_rate override.",
    )
    parser.add_argument(
        "--wf-n-jobs",
        type=int,
        default=defaults.get("wf_n_jobs", 1),
        help="Walk-forward: XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=defaults.get("holdout_seasons", 0),
        help="Training: holdout seasons.",
    )
    parser.add_argument(
        "--train-calibration-seasons",
        type=int,
        default=defaults.get("train_calibration_seasons", 0),
        help="Training: calibration seasons.",
    )
    parser.add_argument(
        "--train-calibration-weeks",
        type=int,
        default=defaults.get("train_calibration_weeks"),
        help="Training: calibration weeks (default: use wf-calibration-weeks).",
    )
    parser.add_argument(
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("include_postseason", False),
        help="Training: include postseason games.",
    )
    parser.add_argument(
        "--postseason-weight",
        type=float,
        default=defaults.get("postseason_weight", 1.0),
        help="Training: postseason sample weight multiplier.",
    )
    parser.add_argument(
        "--train-recency-half-life-seasons",
        type=float,
        default=defaults.get("train_recency_half_life_seasons"),
        help=(
            "Training: optional exponential half-life in seasons for recency weighting "
            "(default: the walk-forward setting)."
        ),
    )
    parser.add_argument(
        "--market-transform",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("market_transform"),
        help="Training: use transformed market features (auto if omitted).",
    )
    parser.add_argument(
        "--max-cardinality-ratio",
        type=float,
        default=defaults.get("max_cardinality_ratio", 0.5),
        help="Training: max categorical cardinality ratio.",
    )
    parser.add_argument(
        "--feature-start",
        type=str,
        default=defaults.get("feature_start", ml_model_core.DEFAULT_FEATURE_START_COLUMN),
        help="Training: first feature column.",
    )
    parser.add_argument(
        "--feature-end",
        type=str,
        default=defaults.get("feature_end", ml_model_core.DEFAULT_FEATURE_END_COLUMN),
        help="Training: last feature column.",
    )
    parser.add_argument(
        "--train-early-stopping-rounds",
        type=int,
        default=defaults.get(
            "train_early_stopping_rounds",
            ml_model_core.DEFAULT_EARLY_STOPPING_ROUNDS,
        ),
        help=(
            "Training: early stopping rounds for Optuna tuning trials only; the final "
            "in-season fit runs the full n_estimators budget."
        ),
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("tune", False),
        help="Enable Optuna tuning during training.",
    )
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=defaults.get("tune_timeout", ml_model_core.DEFAULT_OPTUNA_TIMEOUT_SECONDS),
        help="Optuna tuning timeout in seconds.",
    )
    parser.add_argument(
        "--tune-n-trials",
        type=int,
        default=defaults.get("tune_n_trials"),
        help="Optuna number of trials (optional).",
    )
    parser.add_argument(
        "--tune-cv-splits",
        type=int,
        default=defaults.get("tune_cv_splits", ml_model_core.DEFAULT_OPTUNA_CV_SPLITS),
        help="Optuna CV splits.",
    )
    parser.add_argument(
        "--tune-objective",
        choices=[
            "margin_mae",
            "total_mae",
            "combined_mae",
            "winner_accuracy",
            "brier",
            "expected_points",
        ],
        default=defaults.get("tune_objective", "brier"),
        help="Optuna objective metric.",
    )
    parser.add_argument(
        "--tune-storage",
        type=str,
        default=defaults.get("tune_storage"),
        help="Optuna storage URL (default: sqlite under run dir).",
    )
    parser.add_argument(
        "--tune-study-name",
        type=str,
        default=defaults.get("tune_study_name"),
        help="Optuna study name.",
    )
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default=defaults.get("xgb_tree_method"),
        help="XGBoost tree_method override.",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default=defaults.get("xgb_device"),
        help="XGBoost device override.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=defaults.get("xgb_n_jobs"),
        help="XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--skip-power-rankings",
        action="store_true",
        default=defaults.get("skip_power_rankings", False),
        help="Skip power rankings generation.",
    )
    parser.add_argument(
        "--power-rankings-season",
        type=int,
        default=defaults.get("power_rankings_season"),
        help="Power rankings season (default: infer from predictions).",
    )
    parser.add_argument(
        "--power-rankings-through-week",
        type=int,
        default=defaults.get("power_rankings_through_week"),
        help="Power rankings through-week (default: infer from predictions).",
    )
    parser.add_argument(
        "--power-rankings-data-ml",
        type=Path,
        default=defaults.get(
            "power_rankings_data_ml",
            Path(constants.DATA_PATH) / "all_data_ml.csv",
        ),
        help="Power rankings ML dataset path.",
    )
    parser.add_argument(
        "--power-rankings-data-schedule",
        type=Path,
        default=defaults.get(
            "power_rankings_data_schedule", Path(constants.DATA_PATH) / "all_data.csv"
        ),
        help="Power rankings schedule dataset path.",
    )
    parser.add_argument(
        "--power-rankings-out-dir",
        type=Path,
        default=defaults.get("power_rankings_out_dir"),
        help="Power rankings output directory (default: output dir).",
    )
    parser.add_argument(
        "--power-rankings-include-postseason",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("power_rankings_include_postseason", False),
        help="Include postseason games in power rankings (default: regular season only).",
    )
    parser.add_argument(
        "--ratings-min-season",
        type=int,
        default=defaults.get("ratings_min_season"),
        help="Optional minimum season for the Bradley-Terry ratings fit.",
    )
    parser.add_argument(
        "--power-rankings-method",
        choices=power_rankings.RANKING_METHODS,
        default=defaults.get("power_rankings_method"),
        help=(
            "Power rankings method: 'composite' (default; the ETL's schedule-adjusted "
            "strength for the week after the through-week) or 'bradley_terry'. "
            "--legacy-franchise-fit implies bradley_terry."
        ),
    )
    parser.add_argument(
        "--power-rankings-strength-snapshots",
        type=Path,
        default=defaults.get(
            "power_rankings_strength_snapshots", power_rankings.DEFAULT_STRENGTH_SNAPSHOTS
        ),
        help="Per-team weekly strength file the composite method reads.",
    )
    parser.add_argument(
        "--ratings-window-seasons",
        type=int,
        default=defaults.get(
            "ratings_window_seasons", power_rankings.DEFAULT_RATINGS_WINDOW_SEASONS
        ),
        help="Bradley-Terry only: seasons the fit sees, counting the current one (0 = all).",
    )
    parser.add_argument(
        "--ratings-prior-season-weight",
        type=float,
        default=defaults.get(
            "ratings_prior_season_weight", power_rankings.DEFAULT_PRIOR_SEASON_WEIGHT
        ),
        help="Bradley-Terry only: weight on games from seasons before the current one.",
    )
    parser.add_argument(
        "--ratings-target",
        choices=("margin", "binary"),
        default=defaults.get("ratings_target", "margin"),
        help="Bradley-Terry only: target for completed games.",
    )
    parser.add_argument(
        "--ratings-include-future",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("ratings_include_future", False),
        help="Bradley-Terry only: feed future games' model win probabilities into the fit.",
    )
    parser.add_argument(
        "--legacy-franchise-fit",
        action="store_true",
        default=defaults.get("legacy_franchise_fit", False),
        help=(
            "Rank with the historical all-seasons, equal-weight Bradley-Terry 'franchise' "
            "fit; overrides the other ranking options."
        ),
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args, loading config defaults when provided."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config:
        config = _load_config(args.config)
        _validate_config_keys(config, _allowed_config_keys(parser))
        parser = _build_parser(_normalize_config_defaults(config))
        args = parser.parse_args(argv)
    try:
        _power_ranking_options(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _power_ranking_options(args: argparse.Namespace) -> power_rankings.RankingOptions:
    """Resolve the ranking options the weekly run's flags describe."""
    return power_rankings.resolve_ranking_options(
        method=args.power_rankings_method,
        legacy_franchise_fit=bool(args.legacy_franchise_fit),
        window_seasons=int(args.ratings_window_seasons),
        prior_season_weight=float(args.ratings_prior_season_weight),
        target=str(args.ratings_target),
        include_future=bool(args.ratings_include_future),
        ratings_min_season=args.ratings_min_season,
        strength_snapshots=Path(args.power_rankings_strength_snapshots),
    )


def _power_rankings_report_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the ranking settings that decide whether a reports stage can be reused."""
    options = _power_ranking_options(args)
    return {
        "power_rankings_method": options.method,
        "power_rankings_strength_snapshots": str(options.strength_snapshots),
        "ratings_window_seasons": options.window_seasons,
        "ratings_prior_season_weight": options.prior_season_weight,
        "ratings_target": options.target,
        "ratings_include_future": options.include_future,
        "ratings_min_season": options.ratings_min_season,
    }
