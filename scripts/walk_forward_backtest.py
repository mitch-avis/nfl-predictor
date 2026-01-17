#!/usr/bin/env python
"""
Walk-forward backtest for margin/total NFL predictions.

This script trains a new model for each week in the evaluation window,
predicts that week, and reports per-week and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from nfl_predictor import constants
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script: `python scripts/walk_forward_backtest.py`.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.ml import walk_forward
    from nfl_predictor.utils.logger import log


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
        choices=["platt", "isotonic", "none"],
        default="platt",
        help="Win-prob calibration method.",
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
        "--out-json",
        type=Path,
        default=None,
        help="Path to metrics_report.json (default: models/<run_id>/metrics_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for walk-forward backtests."""

    args = _parse_args()

    df = walk_forward.load_games(args.data_path)

    market_prob_weight = args.market_prob_weight
    if market_prob_weight is None:
        market_prob_weight = args.market_prob_blend

    config = walk_forward.WalkForwardConfig(
        eval_seasons=args.eval_seasons,
        eval_last_n_seasons=args.eval_last_n_seasons,
        wf_start_week=args.wf_start_week,
        calibration=args.calibration,
        calibration_weeks=args.wf_calibration_weeks,
        random_seed=args.random_seed,
        include_postseason=args.include_postseason,
        market_anchor=args.market_anchor,
        market_transform=args.market_transform,
        market_prob_weight=float(market_prob_weight),
        market_prob_clamp=float(args.market_prob_clamp),
    )

    dataset_hash = walk_forward.dataset_fingerprint(args.data_path)
    run_id = walk_forward.generate_run_id(dataset_hash, config)
    created_at = datetime.now(timezone.utc).isoformat()

    results = walk_forward.run_walk_forward_backtest(df, config)

    run_dir = Path(constants.ROOT_DIR) / "models" / run_id
    out_json = args.out_json or (run_dir / "metrics_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    config_payload = config.to_dict()
    config_payload["data_path"] = str(args.data_path)
    config_payload["out_json"] = str(out_json)
    config_payload.update(results.get("resolved_settings", {}))
    if "resolved_eval_seasons" in results:
        config_payload["resolved_eval_seasons"] = results["resolved_eval_seasons"]
    if "eval_window" in results:
        config_payload["eval_window"] = results["eval_window"]

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
