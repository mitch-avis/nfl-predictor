#!/usr/bin/env python
"""Objective, time-aware comparison of two model artifacts.

This script compares two saved model artifacts under identical walk-forward
splits by retraining per fold and scoring out-of-sample.

It exists specifically to avoid the common pitfall:
- scoring a model on games it was trained on (optimistic / biased)

Example:

  python scripts/objective_compare_models.py \
    --model-a models/week19_fullhistory_postseason_elo_gpu_24h/model.joblib \
    --model-b models/betting_20260110_065502_ede59f27/model.joblib \
    --data-path data/completed_games_ml.csv \
    --eval-last-n-seasons 3

Outputs are written under --out-dir (default: models/compare_<timestamp>/).

Note on .xlsb:
This script does not produce spreadsheets; see scripts/betting_report_excel.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from nfl_predictor.ml.model_compare import (
        CompareConfig,
        load_model,
        recipe_from_model,
        run_objective_compare,
        write_compare_outputs,
    )
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor.ml.model_compare import (
        CompareConfig,
        load_model,
        recipe_from_model,
        run_objective_compare,
        write_compare_outputs,
    )
    from nfl_predictor.utils.logger import log


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("compare_%Y%m%d_%H%M%S")
    return Path("models") / stamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Objective walk-forward comparison of two models")
    parser.add_argument("--model-a", type=Path, required=True, help="Model A path or run dir")
    parser.add_argument("--model-b", type=Path, required=True, help="Model B path or run dir")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/completed_games_ml.csv"),
        help="Completed games ML CSV (targets required).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--eval-last-n-seasons", type=int, default=3)
    parser.add_argument(
        "--eval-seasons",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit seasons to evaluate (overrides --eval-last-n-seasons).",
    )
    parser.add_argument("--wf-start-week", type=int, default=3)
    parser.add_argument("--calibration-weeks", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument("--feature-start", type=str, default="away_rest")
    parser.add_argument("--feature-end", type=str, default="home_moneyline")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Optional number of bootstrap resamples for rough metric CIs (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for objective model comparison script."""

    args = _parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

    out_dir = args.out_dir or _default_out_dir()

    log.info("Loading dataset %s", args.data_path)
    df = pd.read_csv(args.data_path)

    log.info("Loading models")
    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)

    recipe_a = recipe_from_model(model_a, label="A")
    recipe_b = recipe_from_model(model_b, label="B")

    cfg = CompareConfig(
        eval_seasons=list(args.eval_seasons) if args.eval_seasons else None,
        eval_last_n_seasons=int(args.eval_last_n_seasons),
        wf_start_week=int(args.wf_start_week),
        calibration_weeks=int(args.calibration_weeks),
        random_seed=int(args.random_seed),
        feature_start=str(args.feature_start),
        feature_end=str(args.feature_end),
        early_stopping_rounds=int(args.early_stopping_rounds),
        include_quantiles=False,
    )

    log.info("Running objective compare: %s vs %s", recipe_a.kind, recipe_b.kind)
    results = run_objective_compare(
        df,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        cfg=cfg,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    write_compare_outputs(out_dir, results)

    overall = results["overall"]
    log.info("Objective compare overall:")
    for label, metrics in overall.items():
        log.info(
            "%s: brier=%.4f log_loss=%.4f margin_mae=%.3f total_mae=%.3f games=%d",
            label,
            float(metrics.get("brier", float("nan"))),
            float(metrics.get("log_loss", float("nan"))),
            float(metrics.get("margin_mae", float("nan"))),
            float(metrics.get("total_mae", float("nan"))),
            int(metrics.get("games", 0)),
        )

    if results.get("skipped_blend_folds"):
        log.info(
            "Skipped %d blended folds due to insufficient calibration.",
            results["skipped_blend_folds"],
        )

    log.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
