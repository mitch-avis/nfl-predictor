#!/usr/bin/env python
"""Walk-forward backtest for margin/total NFL predictions.

This script trains a new model for each week in the evaluation window,
predicts that week, and reports per-week and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from nfl_predictor import constants
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:
    # Allow running as a script: `python scripts/walk_forward_backtest.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log


def _trend_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return trend/season-phase columns to drop for ablation runs."""
    trend_bases = set(constants.TREND_FEATURE_COLUMNS)
    drop_columns = set(constants.SEASON_PHASE_COLUMNS)
    suffix = "_diff"

    for column in df.columns:
        if column.endswith(suffix):
            base = column[: -len(suffix)]
            if base in trend_bases:
                drop_columns.add(column)
            continue

        for prefix in ("away_", "home_"):
            if column.startswith(prefix):
                base = column[len(prefix) :]
                if base in trend_bases:
                    drop_columns.add(column)
                break

    return sorted(col for col in drop_columns if col in df.columns)


def _parse_feature_groups(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature group list into a tuple of stripped, non-empty names."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtests.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "completed_games_ml.csv",
        help="Path to completed games dataset.",
    )
    parser.add_argument(
        "--eval-last-n-seasons",
        type=int,
        default=3,
        help="Evaluate the last N seasons in the dataset.",
    )
    parser.add_argument(
        "--eval-seasons",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seasons to evaluate (overrides --eval-last-n-seasons).",
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--calibration",
        choices=["platt", "isotonic", "none", "elo", "auto", "logistic"],
        default="platt",
        help="Win-prob calibration method (logistic is an alias for platt).",
    )
    parser.add_argument(
        "--wf-calibration-weeks",
        type=int,
        default=walk_forward.DEFAULT_CALIBRATION_WEEKS,
        help="Number of prior weeks (same season) used for time-aware calibration.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=walk_forward.DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--include-postseason",
        action="store_true",
        help="Include postseason games in walk-forward evaluation.",
    )
    parser.add_argument(
        "--exclude-incomplete-seasons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude seasons whose regular season is incomplete in the dataset "
            "(useful when the current season is partial)."
        ),
    )
    parser.add_argument(
        "--recency-half-life-weeks",
        type=float,
        default=None,
        help=(
            "Optional exponential half-life in weeks for recency weighting. "
            "Use only one of --recency-half-life-weeks or --recency-half-life-seasons."
        ),
    )
    parser.add_argument(
        "--recency-half-life-seasons",
        type=float,
        default=None,
        help=(
            "Optional exponential half-life in seasons for recency weighting. "
            "Use only one of --recency-half-life-weeks or --recency-half-life-seasons."
        ),
    )
    parser.add_argument(
        "--market-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable market anchoring when spread/total lines exist.",
    )
    parser.add_argument(
        "--market-transform",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use transformed market features when odds columns exist.",
    )
    parser.add_argument(
        "--market-prob-weight",
        type=float,
        default=None,
        help=(
            "Blend weight for market implied probability (0=off, 1=market only). "
            "Alias for --market-prob-blend."
        ),
    )
    parser.add_argument(
        "--market-prob-blend",
        type=float,
        default=0.0,
        help="Blend weight for market implied probability (0=off, 1=market only).",
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
        "--win-prob-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use margin quantiles to derive uncertainty-aware win probabilities.",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable the feature pruning list.",
    )
    parser.add_argument(
        "--disable-trend-features",
        action="store_true",
        help="Drop trend + season-phase features for ablation comparisons.",
    )
    parser.add_argument(
        "--disable-feature-groups",
        type=str,
        default=None,
        help=(
            "Comma-separated feature group names to drop for ablation comparisons "
            "(e.g. 'pbp' or 'pbp,other'). See constants.FEATURE_GROUP_COLUMN_MARKERS."
        ),
    )
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default=None,
        help="XGBoost tree_method override (e.g., hist).",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default=None,
        help="XGBoost device override (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to metrics_report.json (default: models/<run_id>/metrics_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for walk-forward backtests."""
    args = _parse_args()
    if args.recency_half_life_weeks is not None and args.recency_half_life_seasons is not None:
        raise ValueError(
            "Specify only one of --recency-half-life-weeks or --recency-half-life-seasons."
        )

    df = walk_forward.load_games(args.data_path)
    drop_columns: list[str] = []
    if args.disable_trend_features:
        drop_columns = _trend_feature_columns(df)
        if drop_columns:
            df = df.drop(columns=drop_columns)
        log.info(
            "Trend feature ablation enabled; dropped %d columns.",
            len(drop_columns),
        )

    disabled_feature_groups = _parse_feature_groups(args.disable_feature_groups)
    dropped_feature_group_columns: list[str] = []
    if disabled_feature_groups:
        dropped_feature_group_columns = walk_forward.resolve_feature_group_columns(
            list(df.columns), disabled_feature_groups
        )
        if dropped_feature_group_columns:
            df = df.drop(columns=dropped_feature_group_columns)
        log.info(
            "Feature group ablation enabled for %s; dropped %d columns.",
            list(disabled_feature_groups),
            len(dropped_feature_group_columns),
        )

    market_prob_weight = args.market_prob_weight
    if market_prob_weight is None:
        market_prob_weight = args.market_prob_blend

    xgb_overrides: dict[str, Any] = {}
    if args.xgb_tree_method is not None:
        xgb_overrides["tree_method"] = str(args.xgb_tree_method)
    if args.xgb_device is not None:
        xgb_overrides["device"] = str(args.xgb_device)
    if args.xgb_n_jobs is not None:
        xgb_overrides["n_jobs"] = int(args.xgb_n_jobs)

    config = walk_forward.WalkForwardConfig(
        eval_seasons=args.eval_seasons,
        eval_last_n_seasons=args.eval_last_n_seasons,
        wf_start_week=args.wf_start_week,
        calibration=args.calibration,
        calibration_weeks=args.wf_calibration_weeks,
        random_seed=args.random_seed,
        include_postseason=args.include_postseason,
        exclude_incomplete_seasons=args.exclude_incomplete_seasons,
        recency_half_life_weeks=args.recency_half_life_weeks,
        recency_half_life_seasons=args.recency_half_life_seasons,
        market_anchor=args.market_anchor,
        market_transform=args.market_transform,
        market_prob_weight=float(market_prob_weight),
        market_prob_clamp=float(args.market_prob_clamp),
        market_prob_source=args.market_prob_source,
        market_prob_blend_method=args.market_prob_blend_method,
        win_prob_use_uncertainty=bool(args.win_prob_uncertainty),
        disable_pruning=bool(args.disable_pruning),
        disabled_feature_groups=disabled_feature_groups,
        xgb_params_overrides=xgb_overrides or None,
    )

    dataset_hash = walk_forward.dataset_fingerprint(args.data_path)
    run_id = walk_forward.generate_run_id(dataset_hash, config)
    created_at = datetime.now(UTC).isoformat()

    results = walk_forward.run_walk_forward_backtest(df, config)

    run_dir = Path(constants.ROOT_DIR) / "models" / run_id
    out_json = args.out_json or (run_dir / "metrics_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    config_payload = config.to_dict()
    config_payload["data_path"] = str(args.data_path)
    config_payload["out_json"] = str(out_json)
    config_payload["disable_trend_features"] = bool(args.disable_trend_features)
    if args.disable_trend_features:
        config_payload["dropped_trend_columns"] = drop_columns
    config_payload["disabled_feature_groups"] = list(disabled_feature_groups)
    if disabled_feature_groups:
        config_payload["dropped_feature_group_columns"] = dropped_feature_group_columns
    config_payload.update(results.get("resolved_settings", {}))
    if "resolved_eval_seasons" in results:
        config_payload["resolved_eval_seasons"] = results["resolved_eval_seasons"]
    if "eval_window" in results:
        config_payload["eval_window"] = results["eval_window"]
    if "excluded_incomplete_seasons" in results:
        config_payload["excluded_incomplete_seasons"] = results["excluded_incomplete_seasons"]

    report = walk_forward.build_metrics_report(run_id, created_at, config_payload, results)
    config_payload["run_id"] = run_id
    config_payload["feature_list"] = results.get("feature_list")
    config_payload["splits"] = {
        "resolved_eval_seasons": results.get("resolved_eval_seasons"),
        "wf_start_week": config_payload.get("wf_start_week"),
        "eval_window": results.get("eval_window"),
    }
    metadata = walk_forward.build_metadata(created_at, dataset_hash, config_payload)

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    metadata_path = out_json.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    log.info("Saved metrics report to %s", out_json)
    log.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
