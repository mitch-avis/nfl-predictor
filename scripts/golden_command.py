#!/usr/bin/env python
"""Golden command: train + walk-forward + predict with one command.

This script is a convenience entrypoint that produces a single run directory under
models/<run_id>/ containing:
- model.joblib
- metrics_report.json (walk-forward)
- metadata.json (walk-forward + config)
- predictions.csv (optional, if --predict-path provided)

It does not fetch data; it operates on existing CSV inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import ParserError

try:
    from nfl_predictor import constants, ml_model
    from nfl_predictor.ml import artifacts, walk_forward
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script: `python scripts/golden_command.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants, ml_model
    from nfl_predictor.ml import artifacts, walk_forward
    from nfl_predictor.utils.logger import log


def _resolve_team_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Resolve the (away, home) team columns used for rankings."""

    for away_col, home_col in (("away_abbr", "home_abbr"), ("away_name", "home_name")):
        if away_col in df.columns and home_col in df.columns:
            return away_col, home_col
    return None, None


def _add_ratings(df: pd.DataFrame, target_columns: tuple[str, str]) -> pd.DataFrame:
    """Add lightweight pre/post-game rating columns for power rankings."""

    rated = df.copy()
    rated["pregame_home_rating"] = (rated["home_win_prob"] * 10).round(2)
    rated["pregame_away_rating"] = ((1 - rated["home_win_prob"]) * 10).round(2)

    rated["postgame_home_rating"] = np.nan
    rated["postgame_away_rating"] = np.nan

    away_col, home_col = target_columns
    if away_col in rated.columns and home_col in rated.columns:
        actual_margin = rated[home_col] - rated[away_col]
        if actual_margin.notna().any():
            actual_home_prob = ml_model.margin_to_home_win_prob(actual_margin.to_numpy())
            rated["postgame_home_rating"] = (actual_home_prob * 10).round(2)
            rated["postgame_away_rating"] = ((1 - actual_home_prob) * 10).round(2)
    return rated


def _build_pregame_power_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-week pregame power rankings from rating columns.

    The produced rating for a given (season, week) is based on:
    - pregame ratings for games at/after that week
    - postgame ratings for games strictly before that week

    This is intentionally simple and deterministic; it is a display artifact and does not
    affect model training or win probabilities.
    """

    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()

    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.upper() == "REG"]

    away_team_col, home_team_col = _resolve_team_columns(df)
    if away_team_col is None or home_team_col is None:
        return pd.DataFrame()

    required_cols = {
        "pregame_away_rating",
        "pregame_home_rating",
        "postgame_away_rating",
        "postgame_home_rating",
    }
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    home_rows = df[
        ["season", "week", home_team_col, "pregame_home_rating", "postgame_home_rating"]
    ].rename(
        columns={
            home_team_col: "team",
            "pregame_home_rating": "pregame_rating",
            "postgame_home_rating": "postgame_rating",
        }
    )
    away_rows = df[
        ["season", "week", away_team_col, "pregame_away_rating", "postgame_away_rating"]
    ].rename(
        columns={
            away_team_col: "team",
            "pregame_away_rating": "pregame_rating",
            "postgame_away_rating": "postgame_rating",
        }
    )
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.dropna(subset=["pregame_rating", "week"])
    long_df["week"] = long_df["week"].astype(int)
    if long_df.empty:
        return pd.DataFrame()

    ranking_rows: list[pd.DataFrame] = []
    for season, season_df in long_df.groupby("season"):
        season_weeks = sorted(season_df["week"].dropna().unique())
        for week in season_weeks:
            ratings = season_df.copy()
            played_mask = (ratings["week"] < week) & ratings["postgame_rating"].notna()
            ratings["rating"] = ratings["pregame_rating"]
            ratings.loc[played_mask, "rating"] = ratings.loc[played_mask, "postgame_rating"]

            team_ratings = (
                ratings.groupby("team", as_index=False)["rating"].mean().assign(season=season)
            )
            team_ratings["week"] = week
            team_ratings = team_ratings.sort_values(["rating", "team"], ascending=[False, True])
            team_ratings["rank"] = np.arange(1, len(team_ratings) + 1)
            ranking_rows.append(team_ratings)

    if not ranking_rows:
        return pd.DataFrame()
    return pd.concat(ranking_rows, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Golden command: train + backtest + predict")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "completed_games_ml.csv",
        help="Training/backtest dataset path.",
    )
    parser.add_argument(
        "--predict-path",
        type=Path,
        default=None,
        help="Optional upcoming games CSV to generate predictions.",
    )
    parser.add_argument(
        "--score-rounding",
        choices=["none", "int", "half", "nfl"],
        default="none",
        help="Optional post-processing for predicted scores: none|int|half|nfl.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: generated).",
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--eval-seasons",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit seasons to evaluate (overrides --eval-last-n-seasons). "
            "Useful when the latest season is incomplete."
        ),
    )
    parser.add_argument(
        "--eval-last-n-seasons",
        type=int,
        default=3,
        help="Evaluate the last N seasons.",
    )
    parser.add_argument(
        "--calibration",
        choices=["platt", "isotonic", "none", "elo"],
        default="platt",
        help="Walk-forward calibration method.",
    )
    parser.add_argument(
        "--wf-calibration-weeks",
        type=int,
        default=walk_forward.DEFAULT_CALIBRATION_WEEKS,
        help="Walk-forward calibration weeks.",
    )
    parser.add_argument(
        "--reuse-wf-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing walk-forward run directory containing "
            "metrics_report.json and metadata.json. If provided, golden_command will copy "
            "and re-stamp those artifacts instead of re-running walk-forward."
        ),
    )
    parser.add_argument(
        "--reuse-wf-allow-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow reusing a walk-forward run whose metadata dataset_hash does not match the "
            "current --data-path. Not recommended."
        ),
    )
    parser.add_argument(
        "--wf-xgb-tree-method",
        type=str,
        default=None,
        help=(
            "Optional XGBoost tree_method override for walk-forward (e.g., auto, hist, approx). "
            "If set, is applied via walk-forward xgb_params_overrides."
        ),
    )
    parser.add_argument(
        "--wf-xgb-device",
        type=str,
        default=None,
        help=(
            "Optional XGBoost device override for walk-forward (e.g., cuda, cpu). "
            "If set, is applied via walk-forward xgb_params_overrides."
        ),
    )
    parser.add_argument(
        "--wf-xgb-n-jobs",
        type=int,
        default=None,
        help=(
            "Optional XGBoost n_jobs override for walk-forward. If set, is applied via "
            "walk-forward xgb_params_overrides."
        ),
    )
    parser.add_argument(
        "--market-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable market anchoring when possible.",
    )
    parser.add_argument(
        "--market-transform",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable market transforms when odds columns exist.",
    )
    parser.add_argument(
        "--market-prob-weight",
        type=float,
        default=0.0,
        help="Market prob blend weight (0=off, 1=market only).",
    )
    parser.add_argument(
        "--market-prob-clamp",
        type=float,
        default=0.0,
        help="Clamp model probability within +/- this delta of market.",
    )

    parser.add_argument(
        "--train-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Optuna tuning for the production model (time-aware CV). "
            "Use --train-tune-timeout-seconds to cap tuning time."
        ),
    )
    parser.add_argument(
        "--train-tune-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Optuna tuning timeout (seconds) for the production model. "
            "Example for 8 hours: 28800. Ignored unless --train-tune is enabled."
        ),
    )
    parser.add_argument(
        "--train-tune-trials",
        type=int,
        default=None,
        help="Optional maximum number of Optuna trials (default: unlimited until timeout).",
    )
    parser.add_argument(
        "--train-tune-metric",
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
        "--train-cv-splits",
        type=int,
        default=ml_model.DEFAULT_OPTUNA_CV_SPLITS,
        help="Number of time-series CV folds for Optuna tuning.",
    )
    parser.add_argument(
        "--train-early-stopping-rounds",
        type=int,
        default=ml_model.DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds for XGBoost during production training.",
    )
    parser.add_argument(
        "--train-xgb-tree-method",
        type=str,
        default="auto",
        help="XGBoost tree_method for production training (e.g., hist, auto).",
    )
    parser.add_argument(
        "--train-xgb-device",
        type=str,
        default="auto",
        help="XGBoost device for production training (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--train-xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost parallel threads for production training (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--write-power-rankings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write power_rankings.csv for the predicted week into the run directory.",
    )
    parser.add_argument(
        "--train-holdout-seasons",
        type=int,
        default=0,
        help="Holdout seasons for the trained model (0 = train on all data).",
    )
    parser.add_argument(
        "--train-calibration-seasons",
        type=int,
        default=1,
        help="Calibration seasons for the trained model.",
    )
    parser.add_argument(
        "--train-calibration-weeks",
        type=int,
        default=0,
        help="Calibration weeks for the trained model.",
    )
    return parser.parse_args()


def main() -> int:
    """Run walk-forward, train a model, and optionally generate predictions."""

    args = _parse_args()
    if not args.data_path.exists():
        log.error("Missing dataset: %s", args.data_path)
        return 2

    dataset_hash = artifacts.sha256_file(args.data_path)
    created_at = datetime.now(timezone.utc).isoformat()

    config_payload = {
        "data_path": str(args.data_path),
        "predict_path": str(args.predict_path) if args.predict_path else None,
        "score_rounding": args.score_rounding,
        "walk_forward": {
            "wf_start_week": args.wf_start_week,
            "eval_seasons": args.eval_seasons,
            "eval_last_n_seasons": args.eval_last_n_seasons,
            "calibration": args.calibration,
            "calibration_weeks": args.wf_calibration_weeks,
            "market_anchor": args.market_anchor,
            "market_transform": args.market_transform,
            "market_prob_weight": args.market_prob_weight,
            "market_prob_clamp": args.market_prob_clamp,
        },
        "train": {
            "holdout_seasons": args.train_holdout_seasons,
            "calibration_seasons": args.train_calibration_seasons,
            "calibration_weeks": args.train_calibration_weeks,
            "tune": bool(args.train_tune),
            "tune_timeout_seconds": int(args.train_tune_timeout_seconds),
            "tune_trials": args.train_tune_trials,
            "tune_metric": args.train_tune_metric,
            "cv_splits": int(args.train_cv_splits),
            "early_stopping_rounds": int(args.train_early_stopping_rounds),
            "xgb_tree_method": args.train_xgb_tree_method,
            "xgb_device": args.train_xgb_device,
            "xgb_n_jobs": args.train_xgb_n_jobs,
        },
    }

    run_id = args.run_id or artifacts.generate_run_id("golden", dataset_hash, config_payload)
    run_dir = Path(constants.ROOT_DIR) / "models" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Walk-forward backtest
    if args.reuse_wf_run_dir is not None:
        src_dir = Path(args.reuse_wf_run_dir)
        src_report_path = src_dir / "metrics_report.json"
        src_metadata_path = src_dir / "metadata.json"
        if not src_report_path.exists() or not src_metadata_path.exists():
            log.error(
                (
                    "Reuse requested but missing artifacts in %s "
                    "(expected metrics_report.json and metadata.json)"
                ),
                src_dir,
            )
            return 2

        src_report = json.loads(src_report_path.read_text())
        src_metadata = json.loads(src_metadata_path.read_text())
        src_dataset_hash = src_metadata.get("dataset_hash")
        if src_dataset_hash != dataset_hash and not bool(args.reuse_wf_allow_mismatch):
            log.error(
                "Refusing to reuse walk-forward run: dataset_hash mismatch (src=%s current=%s).",
                src_dataset_hash,
                dataset_hash,
            )
            log.error("Re-run walk-forward or pass --reuse-wf-allow-mismatch (not recommended).")
            return 2

        reused_from = src_metadata.get("run_id")
        src_report["run_id"] = run_id
        src_report["created_at"] = created_at
        if isinstance(src_report.get("config"), dict):
            src_report["config"]["run_id"] = run_id
            src_report["config"]["reused_from"] = reused_from

        src_metadata["run_id"] = run_id
        src_metadata["created_at"] = created_at
        if isinstance(src_metadata.get("config"), dict):
            src_metadata["config"]["run_id"] = run_id
            src_metadata["config"]["reused_from"] = reused_from
            src_metadata["config"]["data_path"] = str(args.data_path)
        src_metadata["dataset_hash"] = dataset_hash

        (run_dir / "metrics_report.json").write_text(
            json.dumps(src_report, indent=2, sort_keys=True)
        )
        (run_dir / "metadata.json").write_text(json.dumps(src_metadata, indent=2, sort_keys=True))
        log.info("Reused walk-forward artifacts from %s", src_dir)
    else:
        wf_overrides: dict[str, object] = {}
        if args.wf_xgb_tree_method is not None:
            wf_overrides["tree_method"] = str(args.wf_xgb_tree_method)
        if args.wf_xgb_device is not None:
            wf_overrides["device"] = str(args.wf_xgb_device)
        if args.wf_xgb_n_jobs is not None:
            wf_overrides["n_jobs"] = int(args.wf_xgb_n_jobs)

        wf_df = walk_forward.load_games(args.data_path)
        wf_config = walk_forward.WalkForwardConfig(
            eval_seasons=args.eval_seasons,
            eval_last_n_seasons=args.eval_last_n_seasons,
            wf_start_week=args.wf_start_week,
            calibration=args.calibration,
            calibration_weeks=args.wf_calibration_weeks,
            market_anchor=args.market_anchor,
            market_transform=args.market_transform,
            market_prob_weight=float(args.market_prob_weight),
            market_prob_clamp=float(args.market_prob_clamp),
            xgb_params_overrides=(wf_overrides or None),
        )
        wf_results = walk_forward.run_walk_forward_backtest(wf_df, wf_config)

        wf_config_payload = wf_config.to_dict()
        wf_config_payload.update({"run_id": run_id, "data_path": str(args.data_path)})
        wf_config_payload.update(wf_results.get("resolved_settings", {}))
        wf_config_payload["resolved_eval_seasons"] = wf_results.get("resolved_eval_seasons")
        wf_config_payload["feature_list"] = wf_results.get("feature_list")

        wf_report = walk_forward.build_metrics_report(
            run_id, created_at, wf_config_payload, wf_results
        )
        (run_dir / "metrics_report.json").write_text(
            json.dumps(wf_report, indent=2, sort_keys=True)
        )

        wf_config_payload["splits"] = {
            "resolved_eval_seasons": wf_results.get("resolved_eval_seasons"),
            "wf_start_week": args.wf_start_week,
        }
        wf_metadata = walk_forward.build_metadata(created_at, dataset_hash, wf_config_payload)
        (run_dir / "metadata.json").write_text(json.dumps(wf_metadata, indent=2, sort_keys=True))

    # 2) Train model and save checkpoint into same run dir
    train_optuna = ml_model.OptunaConfig(
        enabled=bool(args.train_tune),
        timeout_seconds=int(args.train_tune_timeout_seconds),
        n_trials=args.train_tune_trials,
        cv_splits=int(args.train_cv_splits),
        objective=str(args.train_tune_metric),
        early_stopping_rounds=int(args.train_early_stopping_rounds),
        tree_method=str(args.train_xgb_tree_method),
        device=str(args.train_xgb_device),
        tune_scope="both",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=args.train_xgb_n_jobs,
    )

    train_result = ml_model.train_margin_total_model_with_report(
        data_path=args.data_path,
        holdout_seasons=args.train_holdout_seasons,
        calibration_seasons=args.train_calibration_seasons,
        calibration_weeks=args.train_calibration_weeks,
        include_market=True,
        max_cardinality_ratio=0.5,
        win_prob_calibration="isotonic",
        optuna_config=train_optuna,
        market_transform=(
            bool(args.market_transform) if args.market_transform is not None else False
        ),
        market_anchor=args.market_anchor,
        market_prob_config=(
            ml_model.MarketProbConfig(
                blend_weight=float(args.market_prob_weight),
                clamp_delta=float(args.market_prob_clamp),
            )
            if float(args.market_prob_weight) or float(args.market_prob_clamp)
            else None
        ),
    )
    artifacts.save_model(run_dir / "model.joblib", train_result.model)

    # 3) Predict upcoming games (optional)
    if args.predict_path is not None:
        if not args.predict_path.exists():
            log.error("Missing predict dataset: %s", args.predict_path)
            return 2
        out_path = run_dir / "predictions.csv"
        ml_model.predict_week_margin_total(
            train_result.model,
            games_path=args.predict_path,
            output_path=out_path,
            pretty_output=False,
            score_rounding=args.score_rounding,
        )

    # 4) Power rankings for the predicted week (optional)
    if bool(args.write_power_rankings) and args.predict_path is not None:
        try:
            predict_df = pd.read_csv(args.predict_path)
            if "season" in predict_df.columns and "week" in predict_df.columns:
                season = int(predict_df["season"].iloc[0])
                week = int(predict_df["week"].iloc[0])

                completed_df = pd.read_csv(args.data_path)
                completed_df = completed_df[
                    (completed_df["season"] == season)
                    & (completed_df["game_type"].astype(str).str.upper() == "REG")
                ]
                completed_season_path = run_dir / "_rankings_completed_season.csv"
                completed_df.to_csv(completed_season_path, index=False)

                completed_scored = ml_model.predict_week_margin_total(
                    train_result.model,
                    games_path=completed_season_path,
                    output_path=None,
                    pretty_output=False,
                    score_rounding="none",
                )
                completed_scored = _add_ratings(
                    completed_scored, ml_model.get_target_columns(completed_df)
                )

                future_scored = ml_model.predict_week_margin_total(
                    train_result.model,
                    games_path=args.predict_path,
                    output_path=None,
                    pretty_output=False,
                    score_rounding="none",
                )
                future_scored = _add_ratings(future_scored, ml_model.get_target_columns(predict_df))

                season_scored = pd.concat(
                    [completed_scored, future_scored], ignore_index=True, sort=False
                )
                rankings = _build_pregame_power_rankings(season_scored)
                rankings = rankings[(rankings["season"] == season) & (rankings["week"] == week)]
            else:
                log.warning(
                    "Power rankings require season/week columns in predict CSV: %s",
                    args.predict_path,
                )
                rankings = pd.DataFrame()
            out_rankings = run_dir / "power_rankings.csv"
            rankings.to_csv(out_rankings, index=False)
        except (OSError, ValueError, KeyError, ParserError) as exc:  # pragma: no cover
            log.warning("Power rankings generation failed: %s", exc)

    log.info("Golden run directory: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
