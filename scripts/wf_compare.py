#!/usr/bin/env python
"""Compare walk-forward settings across calibration and market-prob adjustments.

This script runs a small matrix of walk-forward backtests and prints a summary table.
It is intended for local experimentation and does not require any external services.

Outputs:
- prints a summary sorted by Brier score
- writes a CSV summary under models/ (or --out)

Notes:
- Uses regular-season folds only (per walk_forward.filter_regular_season)
- Disables quantile models by default for speed
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from nfl_predictor.ml import metrics as metrics_utils
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script: `python scripts/wf_compare.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor.ml import metrics as metrics_utils
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("wfcmp_%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare walk-forward calibration/prob settings")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/completed_games_ml.csv"),
        help="Path to completed games ML CSV.",
    )
    parser.add_argument(
        "--eval-last-n-seasons",
        type=int,
        default=3,
        help="Evaluate the last N seasons (regular season only).",
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--calibration-weeks",
        type=int,
        default=4,
        help="Number of prior weeks used for time-aware calibration.",
    )
    parser.add_argument(
        "--market-mode",
        choices=["features", "anchor", "hybrid", "all"],
        default="hybrid",
        help="Market mode to evaluate (features, anchor, hybrid, or all).",
    )
    parser.add_argument(
        "--market-prob-source",
        choices=["raw", "novig", "both"],
        default="raw",
        help="Market probability source for blending/clamping.",
    )
    parser.add_argument(
        "--market-prob-blend-method",
        choices=["prob", "logit", "both"],
        default="prob",
        help="Blend method for market probabilities (prob or logit space).",
    )
    parser.add_argument(
        "--win-prob-uncertainty",
        choices=["off", "on", "both"],
        default="off",
        help="Use uncertainty-aware win probabilities (off, on, both).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path (default: models/<run_id>/wf_compare.csv).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=120,
        help="XGBoost n_estimators override (smaller = faster).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="XGBoost max_depth override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.07,
        help="XGBoost learning_rate override.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="XGBoost n_jobs override (use 1 for deterministic local comparisons).",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=15,
        help="Early stopping rounds (smaller = faster).",
    )
    parser.add_argument(
        "--include-quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train quantile models (slow).",
    )
    return parser.parse_args()


def _run_one(
    df: pd.DataFrame,
    *,
    label: str,
    eval_last_n_seasons: int,
    wf_start_week: int,
    calibration: str,
    calibration_weeks: int,
    include_market: bool,
    market_anchor: bool,
    market_mode: str,
    market_prob_source: str,
    market_prob_blend_method: str,
    win_prob_use_uncertainty: bool,
    market_prob_weight: float,
    market_prob_clamp: float,
    xgb_params_overrides: dict[str, Any],
    early_stopping_rounds: int,
    include_quantiles: bool,
) -> dict[str, Any]:
    cfg = walk_forward.WalkForwardConfig(
        eval_seasons=None,
        eval_last_n_seasons=eval_last_n_seasons,
        wf_start_week=wf_start_week,
        calibration=calibration,
        calibration_weeks=calibration_weeks,
        random_seed=42,
        include_market=include_market,
        market_transform=None,
        market_anchor=market_anchor,
        market_prob_weight=market_prob_weight,
        market_prob_clamp=market_prob_clamp,
        market_prob_source=market_prob_source,
        market_prob_blend_method=market_prob_blend_method,
        win_prob_use_uncertainty=win_prob_use_uncertainty,
        include_quantiles=include_quantiles,
        max_cardinality_ratio=0.5,
        feature_start="away_rest",
        feature_end="home_moneyline",
        early_stopping_rounds=early_stopping_rounds,
        xgb_params_overrides=xgb_params_overrides,
    )

    out = walk_forward.run_walk_forward_backtest(df, cfg)
    overall = out["overall"]
    reliability = out.get("reliability", [])
    return {
        "label": label,
        "calibration": calibration,
        "market_prob_weight": market_prob_weight,
        "market_prob_clamp": market_prob_clamp,
        "market_prob_source": market_prob_source,
        "market_prob_blend_method": market_prob_blend_method,
        "win_prob_use_uncertainty": bool(win_prob_use_uncertainty),
        "market_mode": market_mode,
        "brier": float(overall.get("brier", float("nan"))),
        "log_loss": float(overall.get("log_loss", float("nan"))),
        "reliability_ece": metrics_utils.reliability_ece(reliability),
        "pick_accuracy": float(overall.get("pick_accuracy", float("nan"))),
        "margin_mae": float(overall.get("margin_mae", float("nan"))),
        "total_mae": float(overall.get("total_mae", float("nan"))),
        "expected_points_avg": float(overall.get("expected_points_avg", float("nan"))),
        "actual_points_avg": float(overall.get("actual_points_avg", float("nan"))),
        "games": int(overall.get("games", 0) or 0),
        "weeks": int(overall.get("weeks", 0) or 0),
    }


def _market_modes(mode: str) -> list[tuple[str, bool, bool]]:
    """Resolve which market modes to evaluate."""

    if mode == "features":
        return [("features", True, False)]
    if mode == "anchor":
        return [("anchor", False, True)]
    if mode == "hybrid":
        return [("hybrid", True, True)]
    return [("features", True, False), ("anchor", False, True), ("hybrid", True, True)]


def main() -> int:
    """Main CLI entrypoint."""

    args = _parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

    log.info("Loading %s", args.data_path)
    df = pd.read_csv(args.data_path)

    xgb_params_overrides = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }

    matrix = [
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

    rows: list[dict[str, Any]] = []
    market_sources = (
        ["raw", "novig"] if args.market_prob_source == "both" else [args.market_prob_source]
    )
    blend_methods = (
        ["prob", "logit"]
        if args.market_prob_blend_method == "both"
        else [args.market_prob_blend_method]
    )
    uncertainty_modes = (
        [False, True]
        if args.win_prob_uncertainty == "both"
        else [args.win_prob_uncertainty == "on"]
    )
    for mode_label, include_market, market_anchor in _market_modes(args.market_mode):
        for source in market_sources:
            for method in blend_methods:
                for use_uncertainty in uncertainty_modes:
                    mode_label_suffix = "uncert" if use_uncertainty else "base"
                    for label, calib, weight, clamp in matrix:
                        run_label = f"{mode_label}_{source}_{method}_{mode_label_suffix}_{label}"
                        log.info(
                            "Running %s (calib=%s, market_weight=%.2f, market_clamp=%.2f)",
                            run_label,
                            calib,
                            weight,
                            clamp,
                        )
                        row = _run_one(
                            df,
                            label=run_label,
                            eval_last_n_seasons=args.eval_last_n_seasons,
                            wf_start_week=args.wf_start_week,
                            calibration=calib,
                            calibration_weeks=args.calibration_weeks,
                            include_market=include_market,
                            market_anchor=market_anchor,
                            market_mode=mode_label,
                            market_prob_source=source,
                            market_prob_blend_method=method,
                            win_prob_use_uncertainty=use_uncertainty,
                            market_prob_weight=weight,
                            market_prob_clamp=clamp,
                            xgb_params_overrides=xgb_params_overrides,
                            early_stopping_rounds=args.early_stopping_rounds,
                            include_quantiles=bool(args.include_quantiles),
                        )
                        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["brier", "log_loss"], ascending=[True, True])

    run_id = _now_run_id()
    out_path = args.out
    if out_path is None:
        out_dir = Path("models") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wf_compare.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_path, index=False)

    log.info("Wrote %s", out_path)
    log.info("Top results (lower brier/log_loss is better):")
    # Keep console output short and scannable.
    display_cols = [
        "label",
        "market_mode",
        "market_prob_source",
        "market_prob_blend_method",
        "win_prob_use_uncertainty",
        "brier",
        "log_loss",
        "reliability_ece",
        "pick_accuracy",
        "actual_points_avg",
        "expected_points_avg",
        "margin_mae",
        "total_mae",
    ]
    log.info("\n%s", result_df[display_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
