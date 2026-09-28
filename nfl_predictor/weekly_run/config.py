"""Weekly-run configuration: the JSON/YAML config file, its validation, and the parser."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core, walk_forward
from nfl_predictor.ml.ml_model_core import OptunaConfig
from nfl_predictor.ml.ml_model_xgb_utils import (
    XGB_DEVICE_AUTO,
    XGB_DEVICE_HELP,
    resolve_xgb_device,
    xgb_device_arg,
)
from nfl_predictor.reporting import power_rankings
from nfl_predictor.weekly_run import stage1

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


# The configuration a weekly run reads when no --config is given.
DEFAULT_CONFIG_PATH = Path(constants.ROOT_DIR) / "config" / "weekly_run.yaml"

# Config keys that were renamed with their option; a config file may still use the old name.
_RENAMED_CONFIG_KEYS = {
    "tune_n_trials": "tune_trials",
    "train_early_stopping_rounds": "tune_early_stopping_rounds",
}


# Config keys that were removed, with what replaces them.
_RETIRED_PROBABILITY_OPTION = (
    "retired; the weekly run submits the deterministic floor, with no market blend and no "
    "uncertainty-aware probabilities"
)
_RETIRED_HOLD_OUT_OPTION = (
    "retired; the final fit and every walk-forward fold train on every eligible completed game, "
    "except holdout_seasons (the evaluation holdout), with no held-out calibration weeks and "
    "no XGBoost eval frame"
)
_REMOVED_CONFIG_KEYS = {
    "wf_n_jobs": "use xgb_n_jobs, which sets XGBoost's CPU threads for every stage",
    "wf_market_prob_source": _RETIRED_PROBABILITY_OPTION,
    "wf_market_prob_blend_method": _RETIRED_PROBABILITY_OPTION,
    "wf_win_prob_uncertainty": _RETIRED_PROBABILITY_OPTION,
    "wf_calibration_weeks": _RETIRED_HOLD_OUT_OPTION,
    "train_calibration_weeks": _RETIRED_HOLD_OUT_OPTION,
    "train_calibration_seasons": _RETIRED_HOLD_OUT_OPTION,
}


def _rename_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` with renamed keys under their current names.

    Raises:
        ValueError: If a config sets both the old and the new name of one key, or sets a key
            that was removed.

    """
    removed = sorted(set(config) & set(_REMOVED_CONFIG_KEYS))
    if removed:
        details = "; ".join(f"{key}: {_REMOVED_CONFIG_KEYS[key]}" for key in removed)
        raise ValueError(f"Config sets removed keys ({details}).")
    renamed = dict(config)
    for old, new in _RENAMED_CONFIG_KEYS.items():
        if old in renamed:
            if new in renamed:
                raise ValueError(f"Config sets both {old!r} and its new name {new!r}.")
            renamed[new] = renamed.pop(old)
    return renamed


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


def _validate_config_choices(config: dict[str, Any], parser: argparse.ArgumentParser) -> None:
    """Raise if a config value is outside its option's choices.

    argparse checks ``choices`` only for values given on the command line, not for defaults, so
    a config file's value (for example the retired ``wf_market_mode: all``) is checked here,
    before the run refreshes any data.
    """
    for action in parser._actions:
        if action.choices is None or action.dest not in config:
            continue
        value = config[action.dest]
        if value not in action.choices:
            options = ", ".join(str(choice) for choice in action.choices)
            raise ValueError(f"Config sets {action.dest} to {value!r}; choose one of: {options}.")


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
        help=(
            "JSON/YAML config file for the options (default: config/weekly_run.yaml when it "
            "exists). Command-line options override it."
        ),
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
        help=(
            "Walk-forward: evaluate the last N seasons. The count includes the current season "
            "even before it has a completed week, so the default 3 scores two seasons."
        ),
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=defaults.get("wf_start_week", 3),
        help="Walk-forward: start week.",
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
        choices=["features", "anchor", "hybrid"],
        default=defaults.get("wf_market_mode", "hybrid"),
        help=(
            "Market mode of the walk-forward and the final fit: market lines as features, "
            "the model anchored to them, or both (hybrid)."
        ),
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
        help="Stage 1 and the final fit: XGBoost n_estimators (the tree budget).",
    )
    parser.add_argument(
        "--wf-max-depth",
        type=int,
        default=defaults.get("wf_max_depth", ml_model_core.DEFAULT_XGB_PARAMS["max_depth"]),
        help="Stage 1 and the final fit: XGBoost max_depth.",
    )
    parser.add_argument(
        "--wf-learning-rate",
        type=float,
        default=defaults.get(
            "wf_learning_rate",
            ml_model_core.DEFAULT_XGB_PARAMS["learning_rate"],
        ),
        help="Stage 1 and the final fit: XGBoost learning_rate.",
    )
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=defaults.get("holdout_seasons", 0),
        help=(
            "Training: newest whole seasons held out of the final fit and scored as an "
            "evaluation holdout; the final fit trains on every other completed game."
        ),
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
        "--tune-early-stopping-rounds",
        "--train-early-stopping-rounds",
        dest="tune_early_stopping_rounds",
        type=int,
        default=defaults.get(
            "tune_early_stopping_rounds",
            ml_model_core.DEFAULT_EARLY_STOPPING_ROUNDS,
        ),
        help=(
            "Tuning: early stopping rounds for Optuna tuning trials only; the final "
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
        "--tune-trials",
        "--tune-n-trials",
        dest="tune_trials",
        type=int,
        default=defaults.get("tune_trials"),
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
        type=xgb_device_arg,
        default=defaults.get("xgb_device", XGB_DEVICE_AUTO),
        help=f"{XGB_DEVICE_HELP} Stage 1 and the final fit use the same device.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=defaults.get("xgb_n_jobs"),
        help=(
            "XGBoost CPU threads for stage 1 and the final fit (default: every CPU core). "
            "Results do not depend on it."
        ),
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
    """Parse CLI args over the config file's defaults (the shipped config unless --config)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = args.config
    if config_path is None and DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH
    if config_path is not None:
        config = _rename_config_keys(_load_config(config_path))
        _validate_config_keys(config, _allowed_config_keys(parser))
        _validate_config_choices(config, parser)
        # Recording the path as the default keeps it in args.config when it was not passed.
        parser = _build_parser(_normalize_config_defaults({**config, "config": config_path}))
        args = parser.parse_args(argv)
    try:
        _power_ranking_options(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``argv`` as the weekly run parses its command line, never reading ``sys.argv``.

    Without ``--config`` the shipped config file supplies the defaults, as in a real run.
    """
    return _parse_args(list(argv))


def xgb_thread_count(args: argparse.Namespace) -> int:
    """Return XGBoost's CPU threads for every stage: ``--xgb-n-jobs``, else every core."""
    if args.xgb_n_jobs is not None:
        return int(args.xgb_n_jobs)
    return int(ml_model_core.DEFAULT_XGB_PARAMS["n_jobs"])


def apply_run_defaults(args: argparse.Namespace) -> None:
    """Fill the options that default to another one, and resolve ``auto`` to a device.

    The final fit's recency half-life follows stage 1's when unset. The device is resolved
    once, so stage 1, the final fit and every record name the same device.
    """
    if args.train_recency_half_life_seasons is None:
        args.train_recency_half_life_seasons = args.wf_recency_half_life_seasons
    args.xgb_device = resolve_xgb_device(args.xgb_device)


def xgb_model_params(args: argparse.Namespace) -> dict[str, Any]:
    """Return the XGBoost tree budget, depth and learning rate of stage 1 and the final fit.

    Both stages train with these, so the walk-forward measures the model the weekly run submits.
    """
    return {
        "n_estimators": int(args.wf_n_estimators),
        "max_depth": int(args.wf_max_depth),
        "learning_rate": float(args.wf_learning_rate),
    }


def stage1_options(args: argparse.Namespace) -> dict[str, Any]:
    """Return the configuration options stage 1 walks forward, from resolved run options."""
    xgb_params_overrides: dict[str, Any] = {
        **xgb_model_params(args),
        "n_jobs": xgb_thread_count(args),
        "verbosity": 0,
        "device": args.xgb_device,
    }
    # Stage 1 trains with the final fit's tree method too.
    if args.xgb_tree_method:
        xgb_params_overrides["tree_method"] = args.xgb_tree_method
    return {
        "eval_last_n_seasons": args.wf_eval_last_n_seasons,
        "wf_start_week": args.wf_start_week,
        "include_postseason": bool(args.wf_include_postseason),
        "exclude_incomplete_seasons": bool(args.wf_exclude_incomplete_seasons),
        "recency_half_life_seasons": args.wf_recency_half_life_seasons,
        "market_mode": args.wf_market_mode,
        "xgb_params_overrides": xgb_params_overrides,
        "include_quantiles": bool(args.wf_include_quantiles),
        "market_transform": args.market_transform,
        "max_cardinality_ratio": float(args.max_cardinality_ratio),
    }


def final_fit_options(
    args: argparse.Namespace, frame: pd.DataFrame, *, run_dir: Path | None
) -> tuple[OptunaConfig, dict[str, Any]]:
    """Return the final fit's tuning config and its training settings.

    ``frame`` needs only the dataset's columns: the market settings resolve against the lines
    it has. Without a ``run_dir`` a tuning run gets no default Optuna storage.
    """
    market_mode = str(args.wf_market_mode)
    include_market, market_anchor = stage1.market_mode_flags(market_mode)
    market_transform = args.market_transform
    if market_transform is None and include_market:
        market_transform = True
    include_market, market_transform, market_anchor = walk_forward.resolve_market_settings(
        frame, include_market, market_transform, market_anchor
    )

    optuna_storage = args.tune_storage
    if args.tune and not optuna_storage and run_dir is not None:
        optuna_storage = f"sqlite:///{(run_dir / 'optuna.db').resolve()}"

    optuna_config = OptunaConfig(
        enabled=bool(args.tune),
        timeout_seconds=int(args.tune_timeout),
        n_trials=args.tune_trials,
        cv_splits=int(args.tune_cv_splits),
        objective=str(args.tune_objective),
        early_stopping_rounds=int(args.tune_early_stopping_rounds),
        tree_method=args.xgb_tree_method,
        device=args.xgb_device,
        storage=optuna_storage,
        study_name=args.tune_study_name,
        best_params_out=None,
        xgb_n_jobs=xgb_thread_count(args),
    )

    train_config: dict[str, Any] = {
        "calibration": "auto",
        "market_mode": market_mode,
        "include_market": include_market,
        "market_transform": market_transform,
        "market_anchor": market_anchor,
        "holdout_seasons": int(args.holdout_seasons),
        "include_postseason": bool(args.include_postseason),
        "postseason_weight": float(args.postseason_weight),
        "recency_half_life_seasons": args.train_recency_half_life_seasons,
        "max_cardinality_ratio": float(args.max_cardinality_ratio),
        "feature_start": str(args.feature_start),
        "feature_end": str(args.feature_end),
        "xgb_params_overrides": xgb_model_params(args),
        "optuna": {
            "enabled": optuna_config.enabled,
            "timeout_seconds": optuna_config.timeout_seconds,
            "n_trials": optuna_config.n_trials,
            "cv_splits": optuna_config.cv_splits,
            "objective": optuna_config.objective,
            "early_stopping_rounds": optuna_config.early_stopping_rounds,
            "tree_method": optuna_config.tree_method,
            "device": optuna_config.device,
            "storage": optuna_config.storage,
            "study_name": optuna_config.study_name,
            "xgb_n_jobs": optuna_config.xgb_n_jobs,
        },
    }
    return optuna_config, train_config


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
