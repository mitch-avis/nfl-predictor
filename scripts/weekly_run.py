#!/usr/bin/env python
"""Weekly orchestration for data refresh, model selection, training, and reporting.

This script runs the weekly workflow in one command:
1) refresh data (ETL)
2) walk-forward compare to select calibration + market settings
3) train the selected model configuration
4) generate weekly predictions and reporting outputs

Outputs land under the run directory (default: models/<run_id>/) and include:
- wf_compare.csv / wf_best.json
- model.joblib / metrics_report.json / metadata.json
- *_predictions.csv / *_confidence_picks.csv
- *_betting_report.csv (when market columns exist)
- optional betting template (.xlsx)
- power rankings + projected standings (when data is available)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

try:
    from nfl_predictor import constants, data_collection
    from nfl_predictor.ml import artifacts, ml_model_core, walk_forward
    from nfl_predictor.ml import metrics as metrics_utils
    from nfl_predictor.ml.ml_model_core import MarketProbConfig, OptunaConfig
    from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
    from nfl_predictor.ml.ml_model_training import train_margin_total_model_with_report
    from nfl_predictor.reporting.betting_excel import write_betting_template_xlsx
    from nfl_predictor.utils.logger import log
    from scripts import betting_pipeline, power_rankings
except ModuleNotFoundError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants, data_collection
    from nfl_predictor.ml import artifacts, ml_model_core, walk_forward
    from nfl_predictor.ml import metrics as metrics_utils
    from nfl_predictor.ml.ml_model_core import MarketProbConfig, OptunaConfig
    from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
    from nfl_predictor.ml.ml_model_training import train_margin_total_model_with_report
    from nfl_predictor.reporting.betting_excel import write_betting_template_xlsx
    from nfl_predictor.utils.logger import log
    from scripts import betting_pipeline, power_rankings


_WEEK_FILE_RE = re.compile(r"week_(\d+)_games_to_predict", re.IGNORECASE)

_WF_MATRIX: list[tuple[str, str, float, float]] = [
    ("none_base", "none", 0.0, 0.0),
    ("platt_base", "platt", 0.0, 0.0),
    ("auto_base", "auto", 0.0, 0.0),
    ("isotonic_base", "isotonic", 0.0, 0.0),
    ("elo_base", "elo", 0.0, 0.0),
    ("isotonic_clamp0.10", "isotonic", 0.0, 0.10),
    ("isotonic_blend0.20_clamp0.10", "isotonic", 0.20, 0.10),
    ("elo_clamp0.10", "elo", 0.0, 0.10),
    ("elo_blend0.20_clamp0.10", "elo", 0.20, 0.10),
]

_PATH_KEYS = {
    "config",
    "data_path",
    "predict_path",
    "run_dir",
    "output_dir",
    "betting_template_path",
    "power_rankings_data_ml",
    "power_rankings_data_schedule",
    "power_rankings_out_dir",
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
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
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


def _build_parser(defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
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
        "--wf-recency-half-life-weeks",
        type=float,
        default=defaults.get("wf_recency_half_life_weeks"),
        help=(
            "Walk-forward: optional exponential half-life in weeks for recency weighting. "
            "Use only one of --wf-recency-half-life-weeks or --wf-recency-half-life-seasons."
        ),
    )
    parser.add_argument(
        "--wf-recency-half-life-seasons",
        type=float,
        default=defaults.get("wf_recency_half_life_seasons"),
        help=(
            "Walk-forward: optional exponential half-life in seasons for recency weighting. "
            "Use only one of --wf-recency-half-life-weeks or --wf-recency-half-life-seasons."
        ),
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
        default=defaults.get("wf_n_estimators", 120),
        help="Walk-forward: XGBoost n_estimators override.",
    )
    parser.add_argument(
        "--wf-max-depth",
        type=int,
        default=defaults.get("wf_max_depth", 4),
        help="Walk-forward: XGBoost max_depth override.",
    )
    parser.add_argument(
        "--wf-learning-rate",
        type=float,
        default=defaults.get("wf_learning_rate", 0.07),
        help="Walk-forward: XGBoost learning_rate override.",
    )
    parser.add_argument(
        "--wf-n-jobs",
        type=int,
        default=defaults.get("wf_n_jobs", 1),
        help="Walk-forward: XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--wf-early-stopping-rounds",
        type=int,
        default=defaults.get("wf_early_stopping_rounds", 15),
        help="Walk-forward: early stopping rounds.",
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
        "--train-recency-half-life-weeks",
        type=float,
        default=defaults.get("train_recency_half_life_weeks"),
        help=(
            "Training: optional exponential half-life in weeks for recency weighting. "
            "Use only one of --train-recency-half-life-weeks or "
            "--train-recency-half-life-seasons."
        ),
    )
    parser.add_argument(
        "--train-recency-half-life-seasons",
        type=float,
        default=defaults.get("train_recency_half_life_seasons"),
        help=(
            "Training: optional exponential half-life in seasons for recency weighting. "
            "Use only one of --train-recency-half-life-weeks or "
            "--train-recency-half-life-seasons."
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
        help="Training: early stopping rounds.",
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
        "--betting-template-path",
        type=Path,
        default=defaults.get("betting_template_path"),
        help="Optional output path for betting template (.xlsx).",
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
        "--ratings-min-season",
        type=int,
        default=defaults.get("ratings_min_season"),
        help="Optional minimum season for ratings fit.",
    )
    return parser


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI args, loading config defaults when provided."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config:
        config = _load_config(args.config)
        _validate_config_keys(config, _allowed_config_keys(parser))
        parser = _build_parser(_normalize_config_defaults(config))
        args = parser.parse_args(argv)
    return args


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse namespace to a JSON-friendly dict."""

    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _extract_week(path: Path) -> Optional[int]:
    match = _WEEK_FILE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_predict_path(predict_path: Optional[Path], data_dir: Path) -> Path:
    """Resolve the default prediction file path when not provided."""

    if predict_path is not None:
        if not predict_path.exists():
            raise FileNotFoundError(f"Missing predict dataset: {predict_path}")
        return predict_path

    predict_dir = data_dir / "predict"
    if not predict_dir.exists():
        raise FileNotFoundError(f"Missing predict directory: {predict_dir}")

    candidates = []
    for candidate in predict_dir.glob("week_*_games_to_predict.csv"):
        week = _extract_week(candidate)
        if week is not None:
            candidates.append((week, candidate))
    if not candidates:
        raise FileNotFoundError(f"No week_XX_games_to_predict.csv files found in {predict_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _infer_season_week(df: pd.DataFrame) -> tuple[Optional[int], Optional[int]]:
    """Infer a single season/week from a prediction frame."""

    season = None
    week = None
    if "season" in df.columns:
        seasons = pd.Series(df["season"]).dropna().unique()
        if len(seasons) == 1:
            season = int(seasons[0])
    if "week" in df.columns:
        weeks = pd.Series(df["week"]).dropna().unique()
        if len(weeks) == 1:
            week = int(weeks[0])
    return season, week


def _resolve_output_paths(
    output_dir: Path,
    season: Optional[int],
    week: Optional[int],
) -> dict[str, Path]:
    """Resolve output paths for predictions and reports."""

    if season is not None and week is not None:
        suffix = f"season_{season}_week_{week:02d}"
    else:
        suffix = "weekly"

    return {
        "predictions": output_dir / f"{suffix}_predictions.csv",
        "confidence_picks": output_dir / f"{suffix}_confidence_picks.csv",
        "betting_report": output_dir / f"{suffix}_betting_report.csv",
        "betting_template": output_dir / f"{suffix}_betting_template.xlsx",
    }


def _build_confidence_picks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Extract confidence pool picks from a predictions DataFrame."""

    if "home_win_prob" not in predictions.columns:
        raise ValueError("Predictions missing home_win_prob for confidence picks.")

    picks = predictions.copy()
    if "predicted_winner" not in picks.columns:
        home_col = "home_abbr" if "home_abbr" in picks.columns else None
        away_col = "away_abbr" if "away_abbr" in picks.columns else None
        if home_col and away_col:
            picks["predicted_winner"] = picks[home_col].where(
                picks["home_win_prob"] >= 0.5, picks[away_col]
            )
        else:
            picks["predicted_winner"] = picks["home_win_prob"].map(
                lambda prob: "home" if prob >= 0.5 else "away"
            )

    if "confidence_rank" not in picks.columns:
        strength = (picks["home_win_prob"] - 0.5).abs()
        picks["confidence_rank"] = strength.rank(method="first", ascending=True).astype(int)

    columns = [
        "season",
        "week",
        "date",
        "game_id",
        "away_abbr",
        "home_abbr",
        "predicted_winner",
        "home_win_prob",
        "away_win_prob",
        "confidence_strength",
        "confidence_rank",
    ]
    trimmed = picks[[col for col in columns if col in picks.columns]]
    if "confidence_rank" in trimmed.columns:
        trimmed = trimmed.sort_values("confidence_rank", ascending=False)
    return trimmed.reset_index(drop=True)


def _market_modes(mode: str) -> list[tuple[str, bool, bool]]:
    """Resolve which market modes to evaluate."""

    if mode == "features":
        return [("features", True, False)]
    if mode == "anchor":
        return [("anchor", False, True)]
    if mode == "hybrid":
        return [("hybrid", True, True)]
    return [("features", True, False), ("anchor", False, True), ("hybrid", True, True)]


def _pick_best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick best row by (brier, log_loss) ascending."""

    if not rows:
        raise ValueError("No walk-forward rows produced.")

    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (
            float(row.get("brier", float("inf"))),
            float(row.get("log_loss", float("inf"))),
        )

    return dict(sorted(rows, key=key)[0])


def _run_wf_compare(
    df: pd.DataFrame,
    *,
    eval_last_n_seasons: int,
    wf_start_week: int,
    calibration_weeks: int,
    include_postseason: bool,
    recency_half_life_weeks: Optional[float],
    recency_half_life_seasons: Optional[float],
    market_mode: str,
    market_prob_source: str,
    market_prob_blend_method: str,
    win_prob_uncertainty: str,
    xgb_params_overrides: dict[str, Any],
    early_stopping_rounds: int,
    include_quantiles: bool,
) -> pd.DataFrame:
    """Run a walk-forward comparison matrix and return the result DataFrame."""

    rows: list[dict[str, Any]] = []
    market_sources = ["raw", "novig"] if market_prob_source == "both" else [market_prob_source]
    blend_methods = (
        ["prob", "logit"] if market_prob_blend_method == "both" else [market_prob_blend_method]
    )
    uncertainty_modes = (
        [False, True] if win_prob_uncertainty == "both" else [win_prob_uncertainty == "on"]
    )

    for mode_label, include_market, market_anchor in _market_modes(market_mode):
        for source in market_sources:
            for method in blend_methods:
                for use_uncertainty in uncertainty_modes:
                    uncertainty_label = "uncert" if use_uncertainty else "base"
                    for label, calib, weight, clamp in _WF_MATRIX:
                        run_label = f"{mode_label}_{source}_{method}_{uncertainty_label}_{label}"
                        log.info(
                            "WF %s (calib=%s, market_weight=%.2f, market_clamp=%.2f)",
                            run_label,
                            calib,
                            weight,
                            clamp,
                        )
                        cfg = walk_forward.WalkForwardConfig(
                            eval_seasons=None,
                            eval_last_n_seasons=eval_last_n_seasons,
                            wf_start_week=wf_start_week,
                            calibration=calib,
                            calibration_weeks=calibration_weeks,
                            random_seed=42,
                            include_postseason=include_postseason,
                            recency_half_life_weeks=recency_half_life_weeks,
                            recency_half_life_seasons=recency_half_life_seasons,
                            include_market=include_market,
                            market_transform=None,
                            market_anchor=market_anchor,
                            market_prob_weight=weight,
                            market_prob_clamp=clamp,
                            market_prob_source=source,
                            market_prob_blend_method=method,
                            win_prob_use_uncertainty=use_uncertainty,
                            include_quantiles=include_quantiles,
                            max_cardinality_ratio=0.5,
                            feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
                            feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
                            early_stopping_rounds=early_stopping_rounds,
                            xgb_params_overrides=xgb_params_overrides,
                        )
                        out = walk_forward.run_walk_forward_backtest(df, cfg)
                        overall = out["overall"]
                        reliability = out.get("reliability", [])
                        rows.append(
                            {
                                "label": run_label,
                                "calibration": calib,
                                "market_prob_weight": float(weight),
                                "market_prob_clamp": float(clamp),
                                "market_prob_source": source,
                                "market_prob_blend_method": method,
                                "win_prob_use_uncertainty": bool(use_uncertainty),
                                "market_mode": mode_label,
                                "brier": float(overall.get("brier", float("nan"))),
                                "log_loss": float(overall.get("log_loss", float("nan"))),
                                "reliability_ece": metrics_utils.reliability_ece(reliability),
                                "pick_accuracy": float(overall.get("pick_accuracy", float("nan"))),
                                "margin_mae": float(overall.get("margin_mae", float("nan"))),
                                "total_mae": float(overall.get("total_mae", float("nan"))),
                                "expected_points_avg": float(
                                    overall.get("expected_points_avg", float("nan"))
                                ),
                                "actual_points_avg": float(
                                    overall.get("actual_points_avg", float("nan"))
                                ),
                                "games": int(overall.get("games", 0) or 0),
                                "weeks": int(overall.get("weeks", 0) or 0),
                            }
                        )

    result_df = pd.DataFrame(rows).sort_values(["brier", "log_loss"], ascending=[True, True])
    return result_df


def _stage_marker_path(run_dir: Path, stage: str) -> Path:
    return run_dir / f"{stage}_state.json"


def _stage_can_reuse(
    marker_path: Path,
    dataset_hash: str,
    config_hash: str,
    outputs: list[Path],
) -> bool:
    """Check whether a stage marker matches and outputs exist."""

    if not marker_path.exists():
        return False
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("dataset_hash") != dataset_hash:
        return False
    if payload.get("config_hash") != config_hash:
        return False
    outputs_list = outputs
    if not outputs_list:
        stored = payload.get("outputs", [])
        outputs_list = [Path(item) for item in stored]
    for path in outputs_list:
        if not path.exists():
            return False
    return True


def _write_stage_marker(
    marker_path: Path,
    *,
    dataset_hash: str,
    config_hash: str,
    stage: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "stage": stage,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    artifacts.write_json(marker_path, payload)


def _write_training_artifacts(
    result: ml_model_core.TrainingResult,
    *,
    run_id: str,
    run_dir: Path,
    created_at: str,
    dataset_hash: str,
    config_payload: dict[str, Any],
) -> artifacts.RunPaths:
    """Write model, metrics, and metadata artifacts."""

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
    if result.feature_importance:
        importance_payload = {
            "run_id": run_id,
            "created_at": created_at,
            **result.feature_importance,
        }
        artifacts.write_json(paths.feature_importance_path, importance_payload)
    return paths


def _market_mode_flags(mode: str) -> tuple[bool, bool]:
    """Return include_market and market_anchor flags for a mode label."""

    if mode == "features":
        return True, False
    if mode == "anchor":
        return False, True
    if mode == "hybrid":
        return True, True
    raise ValueError(f"Unknown market mode: {mode}")


def _power_rankings_outputs(out_dir: Path, season: int, through_week: int) -> list[Path]:
    suffix = f"season_{season}_week_{through_week:02d}"
    return [
        out_dir / f"power_rankings_{suffix}.csv",
        out_dir / f"projected_standings_{suffix}.csv",
        out_dir / f"projected_division_standings_{suffix}.csv",
    ]


def main() -> int:
    """CLI entrypoint."""

    args = _parse_args()
    if (
        args.wf_recency_half_life_weeks is not None
        and args.wf_recency_half_life_seasons is not None
    ):
        raise ValueError(
            "Specify only one of --wf-recency-half-life-weeks or --wf-recency-half-life-seasons."
        )
    if (
        args.train_recency_half_life_weeks is not None
        and args.train_recency_half_life_seasons is not None
    ):
        raise ValueError(
            "Specify only one of --train-recency-half-life-weeks or "
            "--train-recency-half-life-seasons."
        )
    if args.train_recency_half_life_weeks is None and args.train_recency_half_life_seasons is None:
        args.train_recency_half_life_weeks = args.wf_recency_half_life_weeks
        args.train_recency_half_life_seasons = args.wf_recency_half_life_seasons

    if not args.skip_data_refresh:
        log.info("Refreshing data...")
        data_collection.main()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

    dataset_hash = artifacts.sha256_file(args.data_path)
    created_at = datetime.now(timezone.utc).isoformat()

    config_payload = _config_payload(args)
    run_id = args.run_id or artifacts.generate_run_id("weekly", dataset_hash, config_payload)
    run_dir = args.run_dir or (Path(constants.ROOT_DIR) / "models" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Run id: %s", run_id)
    log.info("Run dir: %s", run_dir)

    wf_compare_csv = run_dir / "wf_compare.csv"
    wf_best_json = run_dir / "wf_best.json"

    if args.dry_run:
        log.info("Dry-run: would write %s", wf_compare_csv)
        log.info("Dry-run: would write %s", wf_best_json)
        log.info("Dry-run: would write model + metrics under %s", run_dir)
        log.info("Dry-run: outputs would land under %s", output_dir)
        return 0

    # ----------------------
    # Stage 1: WF comparison
    # ----------------------
    wf_config = {
        "eval_last_n_seasons": args.wf_eval_last_n_seasons,
        "wf_start_week": args.wf_start_week,
        "calibration_weeks": args.wf_calibration_weeks,
        "include_postseason": args.wf_include_postseason,
        "recency_half_life_weeks": args.wf_recency_half_life_weeks,
        "recency_half_life_seasons": args.wf_recency_half_life_seasons,
        "market_mode": args.wf_market_mode,
        "market_prob_source": args.wf_market_prob_source,
        "market_prob_blend_method": args.wf_market_prob_blend_method,
        "win_prob_uncertainty": args.wf_win_prob_uncertainty,
        "wf_matrix": _WF_MATRIX,
        "xgb_params_overrides": {
            "n_estimators": int(args.wf_n_estimators),
            "max_depth": int(args.wf_max_depth),
            "learning_rate": float(args.wf_learning_rate),
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": int(args.wf_n_jobs),
            "verbosity": 0,
        },
        "early_stopping_rounds": int(args.wf_early_stopping_rounds),
        "include_quantiles": bool(args.wf_include_quantiles),
    }

    # Propagate GPU/CPU runtime settings to walk-forward folds too.
    if args.xgb_tree_method:
        wf_config["xgb_params_overrides"]["tree_method"] = args.xgb_tree_method
    if args.xgb_device:
        wf_config["xgb_params_overrides"]["device"] = args.xgb_device
    wf_config["xgb_params_overrides"]["n_jobs"] = (
        int(args.xgb_n_jobs) if args.xgb_n_jobs is not None else int(args.wf_n_jobs)
    )

    wf_config_hash = artifacts.stable_short_hash(wf_config)
    wf_marker = _stage_marker_path(run_dir, "wf_compare")

    if args.resume and _stage_can_reuse(
        wf_marker,
        dataset_hash,
        wf_config_hash,
        [wf_compare_csv, wf_best_json],
    ):
        log.info("Stage 1: reuse %s", wf_compare_csv)
        wf_result_df = pd.read_csv(wf_compare_csv)
        best_row = json.loads(wf_best_json.read_text(encoding="utf-8"))
    else:
        log.info("Stage 1: running walk-forward comparison")
        df = walk_forward.load_games(args.data_path)
        wf_result_df = _run_wf_compare(
            df,
            eval_last_n_seasons=args.wf_eval_last_n_seasons,
            wf_start_week=args.wf_start_week,
            calibration_weeks=args.wf_calibration_weeks,
            include_postseason=bool(args.wf_include_postseason),
            recency_half_life_weeks=args.wf_recency_half_life_weeks,
            recency_half_life_seasons=args.wf_recency_half_life_seasons,
            market_mode=args.wf_market_mode,
            market_prob_source=args.wf_market_prob_source,
            market_prob_blend_method=args.wf_market_prob_blend_method,
            win_prob_uncertainty=args.wf_win_prob_uncertainty,
            xgb_params_overrides=wf_config["xgb_params_overrides"],
            early_stopping_rounds=args.wf_early_stopping_rounds,
            include_quantiles=bool(args.wf_include_quantiles),
        )
        wf_result_df.to_csv(wf_compare_csv, index=False)
        best_rows = [
            {str(key): value for key, value in row.items()}
            for row in wf_result_df.to_dict(orient="records")
        ]
        best_row = _pick_best_row(best_rows)
        wf_best_json.write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")
        _write_stage_marker(
            wf_marker,
            dataset_hash=dataset_hash,
            config_hash=wf_config_hash,
            stage="wf_compare",
        )

    log.info("WF best row: %s", best_row)

    # -----------------
    # Stage 2: training
    # -----------------
    market_mode = str(best_row["market_mode"])
    include_market, market_anchor = _market_mode_flags(market_mode)
    market_transform = args.market_transform
    if market_transform is None and include_market:
        market_transform = True

    columns_only = pd.read_csv(args.data_path, nrows=1)
    include_market, market_transform, market_anchor = walk_forward.resolve_market_settings(
        columns_only, include_market, market_transform, market_anchor
    )

    resolved_calibration = ml_model_core.normalize_win_prob_calibration_method(
        str(best_row["calibration"])
    )
    win_prob_use_uncertainty = bool(best_row.get("win_prob_use_uncertainty", False))
    train_calibration_weeks = (
        args.train_calibration_weeks
        if args.train_calibration_weeks is not None
        else args.wf_calibration_weeks
    )
    train_calibration_seasons = int(args.train_calibration_seasons)
    if (
        resolved_calibration in {"platt", "isotonic", "auto"}
        and train_calibration_weeks <= 0
        and train_calibration_seasons <= 0
    ):
        train_calibration_weeks = max(1, int(args.wf_calibration_weeks))
        log.warning(
            "Calibration '%s' requires data; using %s in-season weeks.",
            resolved_calibration,
            train_calibration_weeks,
        )

    market_prob_config = MarketProbConfig(
        blend_weight=float(best_row["market_prob_weight"]),
        clamp_delta=float(best_row["market_prob_clamp"]),
        prob_source=str(best_row["market_prob_source"]),
        blend_method=str(best_row["market_prob_blend_method"]),
    )

    optuna_storage = args.tune_storage
    if args.tune and not optuna_storage:
        optuna_storage = f"sqlite:///{(run_dir / 'optuna.db').resolve()}"

    optuna_config = OptunaConfig(
        enabled=bool(args.tune),
        timeout_seconds=int(args.tune_timeout),
        n_trials=args.tune_n_trials,
        cv_splits=int(args.tune_cv_splits),
        objective=str(args.tune_objective),
        early_stopping_rounds=int(args.train_early_stopping_rounds),
        tree_method=args.xgb_tree_method,
        device=args.xgb_device,
        tune_scope="both",
        storage=optuna_storage,
        study_name=args.tune_study_name,
        best_params_out=None,
        xgb_n_jobs=args.xgb_n_jobs,
    )

    train_config = {
        "calibration": resolved_calibration,
        "win_prob_use_uncertainty": bool(win_prob_use_uncertainty),
        "market_mode": market_mode,
        "include_market": include_market,
        "market_transform": market_transform,
        "market_anchor": market_anchor,
        "market_prob_config": {
            "blend_weight": market_prob_config.blend_weight,
            "clamp_delta": market_prob_config.clamp_delta,
            "prob_source": market_prob_config.prob_source,
            "blend_method": market_prob_config.blend_method,
        },
        "holdout_seasons": int(args.holdout_seasons),
        "calibration_seasons": train_calibration_seasons,
        "calibration_weeks": int(train_calibration_weeks),
        "include_postseason": bool(args.include_postseason),
        "postseason_weight": float(args.postseason_weight),
        "recency_half_life_weeks": args.train_recency_half_life_weeks,
        "recency_half_life_seasons": args.train_recency_half_life_seasons,
        "max_cardinality_ratio": float(args.max_cardinality_ratio),
        "feature_start": str(args.feature_start),
        "feature_end": str(args.feature_end),
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
    train_config_hash = artifacts.stable_short_hash(train_config)
    train_marker = _stage_marker_path(run_dir, "train")
    paths = artifacts.resolve_run_paths(run_id, run_dir=run_dir)

    model = None
    if args.resume and _stage_can_reuse(
        train_marker,
        dataset_hash,
        train_config_hash,
        [paths.model_path, paths.metrics_path, paths.metadata_path],
    ):
        log.info("Stage 2: reuse %s", paths.model_path)
    else:
        log.info("Stage 2: training margin/total model")
        result = train_margin_total_model_with_report(
            data_path=args.data_path,
            holdout_seasons=int(args.holdout_seasons),
            calibration_seasons=train_calibration_seasons,
            calibration_weeks=int(train_calibration_weeks),
            include_market=include_market,
            max_cardinality_ratio=float(args.max_cardinality_ratio),
            win_prob_calibration=resolved_calibration,
            optuna_config=optuna_config,
            market_transform=bool(market_transform),
            market_anchor=bool(market_anchor),
            market_prob_config=market_prob_config,
            win_prob_use_uncertainty=win_prob_use_uncertainty,
            include_postseason=bool(args.include_postseason),
            postseason_weight=float(args.postseason_weight),
            recency_half_life_weeks=args.train_recency_half_life_weeks,
            recency_half_life_seasons=args.train_recency_half_life_seasons,
            min_season=None,
            max_season=None,
            feature_start=str(args.feature_start),
            feature_end=str(args.feature_end),
        )
        model = result.model
        _write_training_artifacts(
            result,
            run_id=run_id,
            run_dir=run_dir,
            created_at=created_at,
            dataset_hash=dataset_hash,
            config_payload={**config_payload, "wf_best": best_row, "train_config": train_config},
        )
        _write_stage_marker(
            train_marker,
            dataset_hash=dataset_hash,
            config_hash=train_config_hash,
            stage="train",
        )

    # ---------------------
    # Stage 3: predictions
    # ---------------------
    predict_path = _resolve_predict_path(args.predict_path, Path(constants.DATA_PATH))
    predict_preview = pd.read_csv(predict_path, nrows=5)
    preview_season, preview_week = _infer_season_week(predict_preview)
    preview_outputs = _resolve_output_paths(output_dir, preview_season, preview_week)
    predictions_path = preview_outputs["predictions"]
    confidence_path = preview_outputs["confidence_picks"]
    predict_hash = artifacts.sha256_file(predict_path)
    model_path = paths.model_path
    if model is None:
        model = ml_model_core.load_model_checkpoint(model_path, "margin_total")
    model_hash = artifacts.sha256_file(model_path)

    predictions_config = {
        "predict_path": str(predict_path),
        "predict_hash": predict_hash,
        "model_hash": model_hash,
        "score_rounding": args.score_rounding,
        "output_dir": str(output_dir),
        "win_prob_use_uncertainty": bool(win_prob_use_uncertainty),
    }
    predictions_hash = artifacts.stable_short_hash(predictions_config)
    predictions_marker = _stage_marker_path(run_dir, "predictions")

    outputs = [predictions_path, confidence_path]
    if args.resume and _stage_can_reuse(
        predictions_marker, dataset_hash, predictions_hash, outputs
    ):
        log.info("Stage 3: reuse predictions output")
    else:
        log.info("Stage 3: predicting %s", predict_path)
        predictions = predict_week_margin_total(
            model,
            games_path=predict_path,
            output_path=None,
            pretty_output=False,
            score_rounding=str(args.score_rounding),
            win_prob_use_uncertainty=win_prob_use_uncertainty,
        )
        season, week = _infer_season_week(predictions)
        output_paths = _resolve_output_paths(output_dir, season, week)
        predictions_path = output_paths["predictions"]
        confidence_path = output_paths["confidence_picks"]
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        log.info("Wrote predictions to %s", predictions_path)

        picks = _build_confidence_picks(predictions)
        picks.to_csv(confidence_path, index=False)
        log.info("Wrote confidence picks to %s", confidence_path)

        _write_stage_marker(
            predictions_marker,
            dataset_hash=dataset_hash,
            config_hash=predictions_hash,
            stage="predictions",
            extra={"outputs": [str(predictions_path), str(confidence_path)]},
        )
    if not predictions_path.exists():
        log.info("Predictions file not found; skipping reports.")
        return 0

    # ------------------
    # Stage 4: reports
    # ------------------
    predictions_hash = artifacts.sha256_file(predictions_path)
    pr_out_dir_default = args.power_rankings_out_dir or output_dir
    report_config = {
        "predictions_hash": predictions_hash,
        "model_hash": model_hash,
        "betting_template_path": (
            str(args.betting_template_path) if args.betting_template_path else None
        ),
        "skip_power_rankings": bool(args.skip_power_rankings),
        "power_rankings_season": args.power_rankings_season,
        "power_rankings_through_week": args.power_rankings_through_week,
        "power_rankings_data_ml": str(args.power_rankings_data_ml),
        "power_rankings_data_schedule": str(args.power_rankings_data_schedule),
        "power_rankings_out_dir": str(pr_out_dir_default),
    }
    report_hash = artifacts.stable_short_hash(report_config)
    report_marker = _stage_marker_path(run_dir, "reports")

    outputs = []
    if args.resume and _stage_can_reuse(report_marker, dataset_hash, report_hash, outputs):
        log.info("Stage 4: reuse reports output")
        return 0

    predictions = pd.read_csv(predictions_path)
    season, week = _infer_season_week(predictions)
    output_paths = _resolve_output_paths(output_dir, season, week)
    betting_report_path = output_paths["betting_report"]

    try:
        report_df = betting_pipeline.build_betting_report(predictions)
    except ValueError as exc:
        log.info("Betting report skipped: %s", exc)
    else:
        report_df.to_csv(betting_report_path, index=False)
        outputs.append(betting_report_path)
        log.info("Wrote betting report to %s", betting_report_path)

    if args.betting_template_path:
        write_betting_template_xlsx(
            predictions=predictions,
            out_path=args.betting_template_path,
        )
        outputs.append(args.betting_template_path)

    if not args.skip_power_rankings:
        pr_season = args.power_rankings_season or season
        pr_week = args.power_rankings_through_week
        if pr_week is None and week is not None:
            pr_week = max(week - 1, 0)

        if pr_season is None or pr_week is None:
            log.info("Power rankings skipped: unable to infer season/week.")
        elif not args.power_rankings_data_ml.exists():
            log.info("Power rankings skipped: missing %s", args.power_rankings_data_ml)
        elif not args.power_rankings_data_schedule.exists():
            log.info("Power rankings skipped: missing %s", args.power_rankings_data_schedule)
        else:
            pr_out_dir = args.power_rankings_out_dir or output_dir
            pr_out_dir.mkdir(parents=True, exist_ok=True)
            model = ml_model_core.load_model_checkpoint(model_path, "margin_total")
            current_records = power_rankings._load_current_records(
                args.power_rankings_data_schedule, season=pr_season, through_week=pr_week
            )
            future_games = power_rankings._predict_future_games(
                model,
                model_kind="margin_total",
                data_ml=args.power_rankings_data_ml,
                season=pr_season,
                through_week=pr_week,
            )
            games_for_ratings = power_rankings._build_games_for_ratings(
                schedule_path=args.power_rankings_data_schedule,
                season=pr_season,
                through_week=pr_week,
                ratings_min_season=args.ratings_min_season,
                future_games_with_probs=future_games,
            )
            result = power_rankings.build_power_rankings_and_standings(
                season=pr_season,
                through_week=pr_week,
                current_records=current_records,
                games_for_ratings=games_for_ratings,
                future_games_with_probs=future_games,
            )
            power_rankings._write_outputs(
                result,
                out_dir=pr_out_dir,
                season=pr_season,
                through_week=pr_week,
            )
            outputs.extend(_power_rankings_outputs(pr_out_dir, pr_season, pr_week))

    _write_stage_marker(
        report_marker,
        dataset_hash=dataset_hash,
        config_hash=report_hash,
        stage="reports",
        extra={"outputs": [str(path) for path in outputs]},
    )

    log.info("Weekly run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
