#!/usr/bin/env python
"""End-to-end betting-oriented run orchestration.

This is a single entrypoint that:

1) Walk-forward compares win-prob calibration + market probability post-processing
     (regular season folds only), and selects the best combination by (Brier, log loss).

2) Runs GPU Optuna tuning for the blended margin/total model (resumable).

3) Trains a final full-history blended model (optionally upweighting postseason) and
     produces:
     - model predictions for a provided week file (e.g., Wildcard)
     - a betting-oriented report that highlights model-vs-market disagreements

Important limitations (read this):
- This script is a decision-support tool. It does not guarantee profit.
- Stage 1 is *regular season only* by design (it is meant to pick stable calibration/
    probability post-processing settings on larger sample sizes).
- Stage 2 Optuna tuning optimizes a fold metric based on model predictions and market
    features. It does not directly “tune” fitted calibrators (Platt/Isotonic), which are
    trained after the margin model.
- Stage 3 defaults to win-prob calibration='elo' so you can train on *all* rows without
    holding out calibration seasons. Note: blended models still require a small
    time-aware calibration window to fit the blend layer.

Outputs (written under the run directory):
- wf_compare.csv: stage-1 comparison table
- wf_best.json: selected stage-1 row (best by Brier then log loss)
- optuna.db: resumable Optuna study database (unless overridden)
- tuned_model.joblib / tuned_metrics_report.json / tuned_metadata.json
- final_model.joblib + model.joblib (same final model, two filenames)
- predictions.csv: week predictions
- betting_report.csv: betting-oriented summary derived from predictions.csv

Resuming / stopping / restarting:
- Stopping: you can safely Ctrl+C at any time.
- Resuming: re-run the same command. With --resume (default):
    - stage 1 reuses wf_compare.csv + wf_best.json if present
    - stage 2 resumes Optuna if you keep the same --tune-storage + --tune-study-name
        (default uses sqlite:///<run_dir>/optuna.db)
    - stage 3 is skipped if final_model.joblib + predictions.csv exist

GPU notes:
- For XGBoost >= 2.0, GPU is selected via `device=cuda` with `tree_method=hist`.
    Recommended:
        --xgb-tree-method hist --xgb-device cuda
- Even with GPU, CPU work still happens (feature preprocessing, data movement).
    --xgb-n-jobs controls XGBoost CPU thread usage; it may still matter a bit for throughput.

Using the betting report with FanDuel (or any book):
- The report includes model-derived “fair” moneylines for each side.
    Compare the sportsbook line to model fair line:
    - For favorites (negative ML): -120 is better than -140.
    - For underdogs (positive ML): +170 is better than +150.
- Because live odds can differ from the CSV, treat the report as a template:
    replace the market line with the live line in the same comparison.

Examples

One-shot end-to-end run (2h tuning, GPU, Week 19 predictions):

    python scripts/betting_pipeline.py \
        --tune-timeout 7200 \
        --xgb-tree-method hist --xgb-device cuda \
        --include-postseason --postseason-weight 1.15 \
        --predict-path data/predict/week_19_games_to_predict.csv \
        --output-predictions data/predict/week_19_wildcard_predictions.csv

Dry-run (prints planned paths; does not train):

    python scripts/betting_pipeline.py --dry-run

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from nfl_predictor import constants
    from nfl_predictor.ml import artifacts, walk_forward
    from nfl_predictor.ml.ml_model_core import (
        MarketProbConfig,
        OptunaConfig,
        TrainingResult,
        normalize_win_prob_calibration_method,
    )
    from nfl_predictor.ml.ml_model_predict import predict_week_blended
    from nfl_predictor.ml.ml_model_training import (
        train_blended_margin_total_model_with_report,
    )
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script: `python scripts/betting_pipeline.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.ml import artifacts, walk_forward
    from nfl_predictor.ml.ml_model_core import (
        MarketProbConfig,
        OptunaConfig,
        TrainingResult,
        normalize_win_prob_calibration_method,
    )
    from nfl_predictor.ml.ml_model_predict import predict_week_blended
    from nfl_predictor.ml.ml_model_training import (
        train_blended_margin_total_model_with_report,
    )
    from nfl_predictor.utils.logger import log


def _moneyline_to_implied_prob(moneyline: float) -> float:
    """Convert American moneyline to implied probability.

    Returns NaN for non-finite inputs.
    """

    try:
        ml = float(moneyline)
    except (TypeError, ValueError):
        return float("nan")
    if not pd.notna(ml):
        return float("nan")
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def _implied_prob_to_moneyline(prob: float) -> float:
    """Convert implied probability to an American moneyline.

    Returns NaN for invalid probabilities.
    """

    try:
        p = float(prob)
    except (TypeError, ValueError):
        return float("nan")
    if not 0.0 < p < 1.0:
        return float("nan")
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def _novig_pair(p_home_raw: float, p_away_raw: float) -> tuple[float, float]:
    """Normalize two implied probabilities to remove vig (sum to 1)."""

    if not pd.notna(p_home_raw) or not pd.notna(p_away_raw):
        return float("nan"), float("nan")
    denom = float(p_home_raw) + float(p_away_raw)
    if denom <= 0:
        return float("nan"), float("nan")
    return float(p_home_raw) / denom, float(p_away_raw) / denom


def _edge_to_confidence_1_to_10(edge: float) -> int:
    """Map absolute probability edge to a 1..10 confidence score.

    This is a heuristic scale for readability, not bankroll management.
    """

    e = float(abs(edge))
    if e < 0.01:
        return 1
    if e < 0.02:
        return 2
    if e < 0.03:
        return 3
    if e < 0.04:
        return 4
    if e < 0.05:
        return 5
    if e < 0.06:
        return 6
    if e < 0.07:
        return 7
    if e < 0.08:
        return 8
    if e < 0.10:
        return 9
    return 10


def _edge_to_action(edge: float) -> str:
    """Map absolute probability edge to a simple action label."""

    e = float(abs(edge))
    if e < 0.02:
        return "PASS"
    if e < 0.04:
        return "LEAN"
    if e < 0.07:
        return "SMALL"
    if e < 0.10:
        return "MEDIUM"
    return "STRONG"


def build_betting_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a betting-oriented report from a predictions table.

    Expects the output schema from ml_model_predict.predict_week_blended (or margin_total).

    The report focuses on moneyline value signals (probability calibration) and also
    includes simple spread/total deltas (without claiming cover probabilities).
    """

    required = {
        "away_abbr",
        "home_abbr",
        "home_win_prob",
        "predicted_margin",
        "predicted_total",
        "home_moneyline",
        "away_moneyline",
    }
    missing = sorted([c for c in required if c not in predictions.columns])
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    df = predictions.copy()

    df["market_home_prob_raw"] = df["home_moneyline"].map(_moneyline_to_implied_prob)
    df["market_away_prob_raw"] = df["away_moneyline"].map(_moneyline_to_implied_prob)

    novig = df.apply(
        lambda row: _novig_pair(row["market_home_prob_raw"], row["market_away_prob_raw"]),
        axis=1,
        result_type="expand",
    )
    df["market_home_prob_novig"] = novig[0]
    df["market_away_prob_novig"] = novig[1]

    df["model_home_prob"] = df["home_win_prob"].astype(float)
    df["model_away_prob"] = 1.0 - df["model_home_prob"]

    df["edge_home_prob"] = df["model_home_prob"] - df["market_home_prob_novig"]
    df["edge_away_prob"] = df["model_away_prob"] - df["market_away_prob_novig"]

    df["model_fair_home_moneyline"] = df["model_home_prob"].map(_implied_prob_to_moneyline)
    df["model_fair_away_moneyline"] = df["model_away_prob"].map(_implied_prob_to_moneyline)

    df["moneyline_value_side"] = df.apply(
        lambda row: row["home_abbr"] if row["edge_home_prob"] >= 0 else row["away_abbr"],
        axis=1,
    )
    df["moneyline_edge_prob"] = df[["edge_home_prob", "edge_away_prob"]].abs().max(axis=1)
    df["moneyline_confidence_1_10"] = df["moneyline_edge_prob"].map(_edge_to_confidence_1_to_10)
    df["moneyline_action"] = df["moneyline_edge_prob"].map(_edge_to_action)

    # Spread / total deltas (no probability claims)
    if "home_spread" in df.columns:
        df["spread_edge_points_home"] = df["predicted_margin"].astype(float) + df[
            "home_spread"
        ].astype(float)
        df["spread_value_side"] = df.apply(
            lambda row: (
                row["home_abbr"] if row["spread_edge_points_home"] >= 0 else row["away_abbr"]
            ),
            axis=1,
        )
        df["spread_edge_points"] = df["spread_edge_points_home"].abs()
    else:
        df["spread_value_side"] = None
        df["spread_edge_points"] = float("nan")

    if "total_line" in df.columns:
        df["total_edge_points"] = (
            df["predicted_total"].astype(float) - df["total_line"].astype(float)
        ).abs()
        df["total_value_side"] = df.apply(
            lambda row: "OVER" if row["predicted_total"] >= row["total_line"] else "UNDER",
            axis=1,
        )
    else:
        df["total_value_side"] = None
        df["total_edge_points"] = float("nan")

    # Friendly display columns
    df["matchup"] = df["away_abbr"].astype(str) + " @ " + df["home_abbr"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    cols: list[str] = [
        "game_id",
        "date",
        "matchup",
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "predicted_margin",
        "model_home_prob",
        "market_home_prob_novig",
        "edge_home_prob",
        "away_moneyline",
        "home_moneyline",
        "model_fair_away_moneyline",
        "model_fair_home_moneyline",
        "moneyline_value_side",
        "moneyline_edge_prob",
        "moneyline_confidence_1_10",
        "moneyline_action",
        "away_spread",
        "home_spread",
        "spread_value_side",
        "spread_edge_points",
        "total_line",
        "total_value_side",
        "total_edge_points",
    ]
    cols_present = [c for c in cols if c in df.columns]
    report = df[cols_present].copy()
    report = report.sort_values(
        ["moneyline_edge_prob", "total_edge_points"],
        ascending=[False, False],
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WF comparison -> Optuna tuning -> full-history training -> predictions in one run."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "completed_games_ml.csv",
        help="Path to completed games ML CSV.",
    )
    parser.add_argument(
        "--predict-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "predict" / "week_19_games_to_predict.csv",
        help="Path to games-to-predict CSV.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: generated).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory (default: models/<run_id>).",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help=(
            "Optional predictions CSV output path. Default: <run_dir>/predictions.csv. "
            "If provided, the script also writes a copy to this path."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages whose output artifacts already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log planned actions and exit without training.",
    )

    # Stage 1: walk-forward comparison (regular season only)
    parser.add_argument(
        "--wf-eval-last-n-seasons",
        type=int,
        default=3,
        help="Evaluate last N seasons (regular season only).",
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--wf-calibration-weeks",
        type=int,
        default=4,
        help="Time-aware calibration weeks for walk-forward.",
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
        "--wf-n-estimators",
        type=int,
        default=200,
        help="XGBoost n_estimators override for walk-forward compare.",
    )
    parser.add_argument(
        "--wf-max-depth",
        type=int,
        default=4,
        help="XGBoost max_depth override for walk-forward compare.",
    )
    parser.add_argument(
        "--wf-learning-rate",
        type=float,
        default=0.07,
        help="XGBoost learning_rate override for walk-forward compare.",
    )
    parser.add_argument(
        "--wf-xgb-n-jobs",
        type=int,
        default=1,
        help=("XGBoost n_jobs override for walk-forward compare (1 is most deterministic)."),
    )

    # Shared modeling knobs
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default="hist",
        help=(
            "XGBoost tree_method (valid: auto, hist, approx, exact; recommended: hist; "
            "GPU selected via --xgb-device cuda)."
        ),
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default="cuda",
        help="XGBoost device (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help=(
            "XGBoost n_jobs for training/tuning (defaults to os.cpu_count()). "
            "Even with GPU, some CPU parallelism can still be used for preprocessing."
        ),
    )

    # Stage 2: tuning
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=7200,
        help="Optuna timeout in seconds for the tuning stage.",
    )
    parser.add_argument(
        "--tune-metric",
        choices=["brier", "expected_points", "combined_mae", "margin_mae", "total_mae"],
        default="brier",
        help="Objective metric for Optuna tuning.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=8,
        help="Time-series CV folds for Optuna tuning.",
    )
    parser.add_argument(
        "--tune-storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL for resumable tuning (e.g., sqlite:////abs/path/optuna.db). "
            "If omitted, defaults to sqlite:///<run_dir>/optuna.db."
        ),
    )
    parser.add_argument(
        "--tune-study-name",
        type=str,
        default="week19_blend_brier",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--tune-best-params-out",
        type=Path,
        default=None,
        help="Optional path to write best Optuna params JSON.",
    )

    # Data inclusion
    parser.add_argument(
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include postseason games in training (when game_type exists).",
    )
    parser.add_argument(
        "--postseason-weight",
        type=float,
        default=1.15,
        help="Sample-weight multiplier for postseason rows.",
    )

    # Final step win-prob calibration: default to elo to allow calibration_seasons=0
    parser.add_argument(
        "--final-win-prob-calibration",
        choices=["elo", "isotonic", "platt", "none", "auto", "logistic"],
        default="elo",
        help=(
            "Win-prob calibration for final training (logistic is an alias for platt). "
            "Default elo enables calibration_seasons=0."
        ),
    )
    parser.add_argument(
        "--score-rounding",
        choices=["none", "int", "half", "nfl"],
        default="nfl",
        help="Optional display-only score rounding used for predictions.",
    )

    # Market integration (kept aligned with repo philosophy)
    parser.add_argument(
        "--market-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable market transforms when supported by the dataset.",
    )
    parser.add_argument(
        "--market-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable market anchoring when supported by the dataset.",
    )

    return parser.parse_args()


def _write_training_artifacts(
    *,
    result: TrainingResult,
    run_id: str,
    run_dir: Path,
    created_at: str,
    dataset_hash: str,
    config_payload: dict[str, Any],
    prefix: str,
) -> Path:
    """Write model + metadata + metrics artifacts and return model path."""

    paths = artifacts.resolve_run_paths(run_id, run_dir=run_dir)

    model_path = run_dir / f"{prefix}_model.joblib"
    metrics_path = run_dir / f"{prefix}_metrics_report.json"
    metadata_path = run_dir / f"{prefix}_metadata.json"

    artifacts.save_model(model_path, result.model)

    metrics_report = {
        "run_id": run_id,
        "created_at": created_at,
        "config": config_payload,
        "splits": result.splits,
        "metrics": result.metrics_report,
    }
    artifacts.write_json(metrics_path, metrics_report)

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
    )
    artifacts.write_json(metadata_path, metadata)

    # Keep the standard filenames pointing at the *final* run.
    # (We still write prefixed files so earlier stages don't clobber the final ones.)
    if prefix == "final":
        artifacts.save_model(paths.model_path, result.model)
        artifacts.write_json(paths.metrics_path, metrics_report)
        artifacts.write_json(paths.metadata_path, metadata)

    return model_path


def _wf_compare_matrix() -> list[tuple[str, str, float, float]]:
    """Default matrix: (label, calibration, market_prob_weight, market_prob_clamp)."""

    return [
        ("platt_base", "platt", 0.0, 0.0),
        ("isotonic_base", "isotonic", 0.0, 0.0),
        ("elo_base", "elo", 0.0, 0.0),
        ("isotonic_clamp0.10", "isotonic", 0.0, 0.10),
        ("isotonic_blend0.20_clamp0.10", "isotonic", 0.20, 0.10),
        ("elo_clamp0.10", "elo", 0.0, 0.10),
        ("elo_blend0.20_clamp0.10", "elo", 0.20, 0.10),
    ]


def _pick_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick best by (brier, log_loss) ascending."""

    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (
            float(row.get("brier", float("inf"))),
            float(row.get("log_loss", float("inf"))),
        )

    if not rows:
        raise ValueError("No walk-forward rows produced.")
    return sorted(rows, key=key)[0]


def main() -> int:
    """CLI entrypoint."""

    args = _parse_args()

    if not args.data_path.exists():
        log.error("Missing dataset: %s", args.data_path)
        return 2

    dataset_hash = artifacts.sha256_file(args.data_path)
    created_at = datetime.now(timezone.utc).isoformat()

    # Create run directory
    config_payload: dict[str, Any] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    run_id = args.run_id or artifacts.generate_run_id("betting", dataset_hash, config_payload)
    run_dir = args.run_dir or (Path(constants.ROOT_DIR) / "models" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Run id: %s", run_id)
    log.info("Run dir: %s", run_dir)

    wf_compare_csv = run_dir / "wf_compare.csv"
    wf_best_json = run_dir / "wf_best.json"

    if args.dry_run:
        log.info("Dry-run enabled; exiting after planning.")
        log.info("Stage 1 would write: %s", wf_compare_csv)
        log.info("Stage 2 would use Optuna storage under run dir unless overridden.")
        log.info("Stage 3 would write final model + predictions under: %s", run_dir)
        return 0

    # Load data once (used by stage 1)
    log.info("Loading %s", args.data_path)
    df = pd.read_csv(args.data_path)

    # ----------------------
    # Stage 1: WF comparison
    # ----------------------
    if args.resume and wf_compare_csv.exists() and wf_best_json.exists():
        log.info("Stage 1: reuse %s", wf_compare_csv)
        wf_result_df = pd.read_csv(wf_compare_csv)
        best_row = json.loads(wf_best_json.read_text(encoding="utf-8"))
    else:
        log.info(
            "Stage 1: running walk-forward comparison (tree_method=%s, device=%s)",
            args.xgb_tree_method,
            args.xgb_device,
        )

        xgb_params_overrides: dict[str, Any] = {
            "n_estimators": int(args.wf_n_estimators),
            "max_depth": int(args.wf_max_depth),
            "learning_rate": float(args.wf_learning_rate),
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": int(args.wf_xgb_n_jobs),
            "verbosity": 0,
            "tree_method": str(args.xgb_tree_method),
            "device": str(args.xgb_device),
        }

        rows: list[dict[str, Any]] = []
        for label, calibration, weight, clamp in _wf_compare_matrix():
            log.info(
                "WF %s (calib=%s, market_blend=%.2f, clamp=%.2f)",
                label,
                calibration,
                weight,
                clamp,
            )
            cfg = walk_forward.WalkForwardConfig(
                eval_seasons=None,
                eval_last_n_seasons=int(args.wf_eval_last_n_seasons),
                wf_start_week=int(args.wf_start_week),
                calibration=str(calibration),
                calibration_weeks=int(args.wf_calibration_weeks),
                random_seed=42,
                include_market=True,
                market_transform=None,
                market_anchor=True,
                market_prob_weight=float(weight),
                market_prob_clamp=float(clamp),
                market_prob_source=args.market_prob_source,
                market_prob_blend_method=args.market_prob_blend_method,
                include_quantiles=False,
                max_cardinality_ratio=0.5,
                feature_start="away_rest",
                feature_end="home_moneyline",
                early_stopping_rounds=15,
                xgb_params_overrides=xgb_params_overrides,
            )
            out = walk_forward.run_walk_forward_backtest(df, cfg)
            overall = out["overall"]
            rows.append(
                {
                    "label": label,
                    "calibration": calibration,
                    "market_prob_weight": float(weight),
                    "market_prob_clamp": float(clamp),
                    "market_prob_source": args.market_prob_source,
                    "market_prob_blend_method": args.market_prob_blend_method,
                    "brier": float(overall.get("brier", float("nan"))),
                    "log_loss": float(overall.get("log_loss", float("nan"))),
                    "pick_accuracy": float(overall.get("pick_accuracy", float("nan"))),
                    "margin_mae": float(overall.get("margin_mae", float("nan"))),
                    "total_mae": float(overall.get("total_mae", float("nan"))),
                    "expected_points_avg": float(overall.get("expected_points_avg", float("nan"))),
                    "actual_points_avg": float(overall.get("actual_points_avg", float("nan"))),
                    "games": int(overall.get("games", 0) or 0),
                    "weeks": int(overall.get("weeks", 0) or 0),
                }
            )

        wf_result_df = pd.DataFrame(rows).sort_values(["brier", "log_loss"], ascending=[True, True])
        wf_result_df.to_csv(wf_compare_csv, index=False)
        best_row = _pick_best_row(rows)
        wf_best_json.write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")

        log.info("Stage 1 best: %s", best_row)

    best_market_prob_config = MarketProbConfig(
        blend_weight=float(best_row["market_prob_weight"]),
        clamp_delta=float(best_row["market_prob_clamp"]),
        prob_source=str(best_row.get("market_prob_source", args.market_prob_source)),
        blend_method=str(best_row.get("market_prob_blend_method", args.market_prob_blend_method)),
    )

    best_calibration = str(best_row["calibration"]).lower()

    # ----------------------
    # Stage 2: Optuna tuning
    # ----------------------
    tuned_marker = run_dir / "tuning_done.json"
    if args.resume and tuned_marker.exists():
        log.info("Stage 2: reuse tuning artifacts (%s exists)", tuned_marker)
    else:
        log.info(
            "Stage 2: Optuna tuning (timeout=%ss, metric=%s)",
            args.tune_timeout,
            args.tune_metric,
        )

        if args.market_anchor:
            log.warning(
                "Blended model training does not support market anchoring; continuing with "
                "market_anchor=False (blend layer uses market baselines)."
            )

        storage = args.tune_storage
        if storage is None:
            # Optuna expects an absolute path for reliability.
            optuna_db = (run_dir / "optuna.db").resolve()
            storage = f"sqlite:///{optuna_db}"

        best_params_out = args.tune_best_params_out
        if best_params_out is None:
            best_params_out = run_dir / "optuna_best_params.json"

        optuna_config = OptunaConfig(
            enabled=True,
            timeout_seconds=int(args.tune_timeout),
            n_trials=None,
            cv_splits=int(args.cv_splits),
            objective=str(args.tune_metric),
            early_stopping_rounds=50,
            tree_method=str(args.xgb_tree_method),
            device=str(args.xgb_device),
            tune_scope="both",
            storage=str(storage),
            study_name=str(args.tune_study_name),
            best_params_out=Path(best_params_out),
            xgb_n_jobs=args.xgb_n_jobs,
        )

        # For tuning, we keep a small holdout for sanity-check metrics.
        tune_result = train_blended_margin_total_model_with_report(
            data_path=args.data_path,
            holdout_seasons=2,
            calibration_seasons=1,
            calibration_weeks=0,
            max_cardinality_ratio=0.5,
            win_prob_calibration=best_calibration,
            optuna_config=optuna_config,
            market_transform=bool(args.market_transform),
            market_anchor=False,
            market_prob_config=best_market_prob_config,
            include_postseason=bool(args.include_postseason),
            postseason_weight=float(args.postseason_weight),
            min_season=None,
            max_season=None,
            feature_start="away_rest",
            feature_end="home_moneyline",
        )

        _write_training_artifacts(
            result=tune_result,
            run_id=run_id,
            run_dir=run_dir,
            created_at=created_at,
            dataset_hash=dataset_hash,
            config_payload={
                "stage": "tune",
                **config_payload,
                "wf_best": best_row,
                "optuna": asdict(optuna_config),
            },
            prefix="tuned",
        )

        artifacts.write_json(
            tuned_marker,
            {
                "created_at": created_at,
                "optuna_storage": storage,
                "study_name": args.tune_study_name,
                "wf_best": best_row,
            },
        )

    # ----------------------------------------------
    # Stage 3: full-history training + week prediction
    # ----------------------------------------------
    final_model_path = run_dir / "final_model.joblib"
    final_predictions_path = run_dir / "predictions.csv"

    if args.resume and final_model_path.exists() and final_predictions_path.exists():
        log.info("Stage 3: reuse %s and %s", final_model_path, final_predictions_path)
    else:
        final_calibration = normalize_win_prob_calibration_method(
            str(args.final_win_prob_calibration)
        )

        # If the user asked for a fitted calibrator but also wants full-history/no calibration,
        # we force elo (deterministic) to avoid needing to hold out rows.
        if final_calibration in {"platt", "isotonic"}:
            log.warning(
                "Final win-prob calibration '%s' requires calibration data; forcing 'elo' for "
                "full-history training with calibration_seasons=0.",
                final_calibration,
            )
            final_calibration = "elo"

        log.info("Stage 3: training final model (calibration=%s)", final_calibration)

        final_calibration_weeks = max(1, int(args.wf_calibration_weeks))
        if args.wf_calibration_weeks <= 0:
            log.warning(
                "Blended models require calibration; using %s in-season calibration weeks.",
                final_calibration_weeks,
            )
        final_result = train_blended_margin_total_model_with_report(
            data_path=args.data_path,
            holdout_seasons=0,
            calibration_seasons=0,
            calibration_weeks=final_calibration_weeks,
            max_cardinality_ratio=0.5,
            win_prob_calibration=final_calibration,
            optuna_config=OptunaConfig(
                enabled=False,
                timeout_seconds=0,
                n_trials=None,
                cv_splits=0,
                objective="brier",
                early_stopping_rounds=50,
                tree_method=str(args.xgb_tree_method),
                device=str(args.xgb_device),
                tune_scope="both",
                storage=None,
                study_name=None,
                best_params_out=None,
                xgb_n_jobs=args.xgb_n_jobs,
            ),
            market_transform=bool(args.market_transform),
            market_anchor=False,
            market_prob_config=best_market_prob_config,
            include_postseason=bool(args.include_postseason),
            postseason_weight=float(args.postseason_weight),
            min_season=None,
            max_season=None,
            feature_start="away_rest",
            feature_end="home_moneyline",
        )

        model_path = _write_training_artifacts(
            result=final_result,
            run_id=run_id,
            run_dir=run_dir,
            created_at=created_at,
            dataset_hash=dataset_hash,
            config_payload={"stage": "final", **config_payload, "wf_best": best_row},
            prefix="final",
        )

        if not args.predict_path.exists():
            log.error("Missing predict dataset: %s", args.predict_path)
            return 2
        log.info("Predicting %s -> %s", args.predict_path, final_predictions_path)
        output_df = predict_week_blended(
            final_result.model,
            games_path=args.predict_path,
            output_path=final_predictions_path,
            pretty_output=False,
            score_rounding=str(args.score_rounding),
        )

        report_path = run_dir / "betting_report.csv"
        report_df = build_betting_report(output_df)
        report_df.to_csv(report_path, index=False)
        log.info("Saved betting report to %s", report_path)

        if args.output_predictions is not None:
            out_copy = Path(args.output_predictions)
            out_copy.parent.mkdir(parents=True, exist_ok=True)
            out_copy.write_text(
                final_predictions_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            log.info("Wrote predictions copy: %s", out_copy)

        # Ensure the model path is consistent.
        if model_path != final_model_path:
            # _write_training_artifacts already wrote final_model.joblib.
            pass

    log.info("Done. Run directory: %s", run_dir)
    log.info("WF compare: %s", wf_compare_csv)
    log.info("WF best: %s", wf_best_json)
    log.info("Predictions: %s", final_predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
