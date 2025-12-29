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

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import artifacts, walk_forward
from nfl_predictor.utils.logger import log


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
        choices=["none", "int", "half"],
        default="none",
        help="Optional post-processing for predicted scores: none|int|half.",
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
        choices=["platt", "isotonic", "none"],
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
        },
    }

    run_id = args.run_id or artifacts.generate_run_id("golden", dataset_hash, config_payload)
    run_dir = Path(constants.ROOT_DIR) / "models" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Walk-forward backtest
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
    )
    wf_results = walk_forward.run_walk_forward_backtest(wf_df, wf_config)

    wf_config_payload = wf_config.to_dict()
    wf_config_payload.update({"run_id": run_id, "data_path": str(args.data_path)})
    wf_config_payload.update(wf_results.get("resolved_settings", {}))
    wf_config_payload["resolved_eval_seasons"] = wf_results.get("resolved_eval_seasons")
    wf_config_payload["feature_list"] = wf_results.get("feature_list")

    wf_report = walk_forward.build_metrics_report(run_id, created_at, wf_config_payload, wf_results)
    (run_dir / "metrics_report.json").write_text(json.dumps(wf_report, indent=2, sort_keys=True))

    wf_config_payload["splits"] = {
        "resolved_eval_seasons": wf_results.get("resolved_eval_seasons"),
        "wf_start_week": args.wf_start_week,
    }
    wf_metadata = walk_forward.build_metadata(created_at, dataset_hash, wf_config_payload)
    (run_dir / "metadata.json").write_text(json.dumps(wf_metadata, indent=2, sort_keys=True))

    # 2) Train model and save checkpoint into same run dir
    train_result = ml_model.train_margin_total_model_with_report(
        data_path=args.data_path,
        holdout_seasons=args.train_holdout_seasons,
        calibration_seasons=args.train_calibration_seasons,
        calibration_weeks=args.train_calibration_weeks,
        include_market=True,
        max_cardinality_ratio=0.5,
        win_prob_calibration="isotonic",
        optuna_config=ml_model.OptunaConfig(
            enabled=False,
            timeout_seconds=0,
            n_trials=None,
            cv_splits=0,
            objective="combined_mae",
            early_stopping_rounds=ml_model.DEFAULT_EARLY_STOPPING_ROUNDS,
            tree_method="auto",
            device="auto",
            tune_scope="both",
            storage=None,
            study_name=None,
            best_params_out=None,
            xgb_n_jobs=None,
        ),
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

    log.info("Golden run directory: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
