"""Walk-forward backtest for margin/total NFL predictions.

This script trains a new model for each week in the evaluation window,
predicts that week, and reports per-week and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nfl_predictor import constants
from nfl_predictor.cli import options
from nfl_predictor.ml import floor_sigma, walk_forward
from nfl_predictor.ml.ml_model_xgb_utils import XGB_DEVICE_AUTO, XGB_DEVICE_HELP, xgb_device_arg
from nfl_predictor.reporting import production_settings, run_comparison
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    import pandas as pd


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtests.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=constants.DATA_PATH / "completed_games_ml.csv",
        help="Path to completed games dataset.",
    )
    options.add_wf_window_options(parser)
    parser.add_argument(
        "--eval-seasons",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seasons to evaluate (overrides --wf-eval-last-n-seasons).",
    )
    parser.add_argument(
        "--win-prob-calibration",
        "--calibration",
        dest="calibration",
        choices=["auto", "none"],
        default="auto",
        help=(
            "Win-probability calibration: auto, the deterministic floor (the predicted margin "
            "through a normal curve whose spread is the root-mean-square out-of-fold error "
            "before the predicted week); none is the same."
        ),
    )
    parser.add_argument(
        "--floor-sigma-reference-runs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Walk-forward runs (run or fold checkpoint directories, relative to the repository "
            "root) whose out-of-fold margin errors join this run's own earlier weeks in each "
            "week's floor sigma; their checkpoints are only read, and a week this run predicts "
            "replaces theirs. Default: none, so each week's sigma comes from this run's earlier "
            "weeks (the constant until they span three earlier seasons, any weeks)."
        ),
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
        "--recency-half-life-seasons",
        type=float,
        default=None,
        help="Optional exponential half-life in seasons for recency weighting.",
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
        "--disable-pruning",
        action="store_true",
        help="Disable the feature pruning list.",
    )
    parser.add_argument(
        "--disable-trend-features",
        action="store_true",
        help="Drop trend + season-phase features for ablation comparisons.",
    )
    options.add_feature_group_option(parser)
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default=None,
        help="XGBoost tree_method override (e.g., hist).",
    )
    parser.add_argument(
        "--xgb-device",
        type=xgb_device_arg,
        default=XGB_DEVICE_AUTO,
        help=XGB_DEVICE_HELP,
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=None,
        help="Optional XGBoost min_child_weight override.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional XGBoost gamma override.",
    )
    parser.add_argument(
        "--wf-n-estimators",
        "--n-estimators",
        dest="n_estimators",
        type=int,
        default=None,
        help=(
            "Optional XGBoost n_estimators override (the tree budget; every in-season fit "
            "runs the full budget)."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restore weeks already finished by an identical earlier run (same data, config, "
            "code and library versions) instead of training them again. Use --no-resume to "
            "retrain every week and overwrite its checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=walk_forward.DEFAULT_CHECKPOINT_DIR,
        help=(
            "Root for per-week checkpoints; each run uses a subdirectory named for its "
            "fingerprint (default: models/wf_checkpoints)."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to metrics_report.json (default: models/<run_id>/metrics_report.json).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class _Ablation:
    """The columns a run dropped, and why."""

    trend_columns: list[str]
    disabled_groups: tuple[str, ...]
    group_columns: list[str]


def _apply_ablations(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, _Ablation]:
    """Drop the trend features and disabled feature groups the arguments ask to leave out."""
    trend_columns: list[str] = []
    if args.disable_trend_features:
        trend_columns = _trend_feature_columns(df)
        if trend_columns:
            df = df.drop(columns=trend_columns)
        log.info(
            "Trend feature ablation enabled; dropped %d columns.",
            len(trend_columns),
        )

    disabled_groups = options.parse_feature_groups(args.disable_feature_groups)
    group_columns: list[str] = []
    if disabled_groups:
        group_columns = walk_forward.resolve_feature_group_columns(
            list(df.columns), disabled_groups
        )
        if group_columns:
            df = df.drop(columns=group_columns)
        log.info(
            "Feature group ablation enabled for %s; dropped %d columns.",
            list(disabled_groups),
            len(group_columns),
        )
    return df, _Ablation(trend_columns, disabled_groups, group_columns)


def _xgb_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return the XGBoost parameters the arguments override."""
    overrides: dict[str, Any] = {}
    if args.xgb_tree_method is not None:
        overrides["tree_method"] = str(args.xgb_tree_method)
    overrides["device"] = str(args.xgb_device)
    if args.xgb_n_jobs is not None:
        overrides["n_jobs"] = int(args.xgb_n_jobs)
    if args.min_child_weight is not None:
        overrides["min_child_weight"] = float(args.min_child_weight)
    if args.gamma is not None:
        overrides["gamma"] = float(args.gamma)
    if args.n_estimators is not None:
        overrides["n_estimators"] = int(args.n_estimators)
    return overrides


def _walk_forward_config(
    args: argparse.Namespace, disabled_groups: tuple[str, ...]
) -> walk_forward.WalkForwardConfig:
    """Build the walk-forward config, with the XGBoost device resolved to the one it uses."""
    config = walk_forward.WalkForwardConfig(
        eval_seasons=args.eval_seasons,
        eval_last_n_seasons=args.eval_last_n_seasons,
        wf_start_week=args.wf_start_week,
        calibration=args.calibration,
        random_seed=args.random_seed,
        include_postseason=args.include_postseason,
        exclude_incomplete_seasons=args.exclude_incomplete_seasons,
        recency_half_life_seasons=args.recency_half_life_seasons,
        market_anchor=args.market_anchor,
        market_transform=args.market_transform,
        disable_pruning=bool(args.disable_pruning),
        disabled_feature_groups=disabled_groups,
        xgb_params_overrides=_xgb_overrides(args),
    )
    # Resolve `auto` now, so the run id and the recorded config name the device used.
    return walk_forward.with_resolved_xgb_device(config)


def _config_payload(
    args: argparse.Namespace,
    config: walk_forward.WalkForwardConfig,
    results: dict[str, Any],
    ablation: _Ablation,
    xgb_params: dict[str, Any],
) -> dict[str, Any]:
    """Return the run's recorded config: its settings, inputs, ablations and resolved values."""
    payload = config.to_dict()
    payload["data_path"] = str(args.data_path)
    payload["checkpoint"] = results.get("checkpoint")
    payload["floor_sigma"] = results.get("floor_sigma")
    payload["xgb_device"] = (config.xgb_params_overrides or {})["device"]
    payload[production_settings.RESOLVED_XGB_PARAMS_KEY] = xgb_params
    payload["disable_trend_features"] = bool(args.disable_trend_features)
    if args.disable_trend_features:
        payload["dropped_trend_columns"] = ablation.trend_columns
    payload["disabled_feature_groups"] = list(ablation.disabled_groups)
    if ablation.disabled_groups:
        payload["dropped_feature_group_columns"] = ablation.group_columns
    payload.update(results.get("resolved_settings", {}))
    for key in ("resolved_eval_seasons", "eval_window", "excluded_incomplete_seasons"):
        if key in results:
            payload[key] = results[key]
    return payload


def _add_stability(report: dict[str, Any], results: dict[str, Any]) -> None:
    """Add the stability block, rescored exactly as ``nfl-predictor compare`` scores the run."""
    predictions = results.get("predictions")
    if predictions is None:
        return
    # Rescored with the comparison's definitions and bootstrap defaults, so this block
    # equals what ``nfl-predictor compare`` reports for the run on the same games.
    stability = run_comparison.stability_report(predictions)
    report["metrics"]["stability"] = stability
    for line in run_comparison.format_stability(stability):
        log.info("%s", line)


def _add_versus_production(
    report: dict[str, Any],
    args: argparse.Namespace,
    config: walk_forward.WalkForwardConfig,
    df: pd.DataFrame,
    xgb_params: dict[str, Any],
) -> None:
    """Add the section listing where the run's settings differ from the production run's."""
    record = production_settings.RunRecord(
        disable_trend_features=bool(args.disable_trend_features),
        xgb_params=xgb_params,
        scope={
            "data_path": str(args.data_path),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
    )
    versus_production = production_settings.settings_versus_production(config, df, record)
    report[production_settings.SECTION_KEY] = versus_production
    for line in production_settings.format_section(versus_production):
        log.info("%s", line)


def main() -> None:
    """CLI entrypoint for walk-forward backtests."""
    args = _parse_args()
    df, ablation = _apply_ablations(walk_forward.load_games(args.data_path), args)
    config = _walk_forward_config(args, ablation.disabled_groups)
    log.info("XGBoost device: %s", (config.xgb_params_overrides or {})["device"])

    dataset_hash = walk_forward.dataset_fingerprint(args.data_path)
    run_id = walk_forward.generate_run_id(dataset_hash, config)
    created_at = datetime.now(UTC).isoformat()

    history = (
        None
        if args.floor_sigma_reference_runs is None
        else floor_sigma.load_reference_pool(args.floor_sigma_reference_runs)
    )
    results = walk_forward.run_walk_forward_backtest(
        df,
        config,
        checkpoints=walk_forward.FoldCheckpoints(args.checkpoint_dir, resume=bool(args.resume)),
        floor_sigma_history=history,
    )

    run_dir = Path(constants.ROOT_DIR) / "models" / run_id
    out_json = args.out_json or (run_dir / "metrics_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Every parameter the folds trained with, so later comparisons need not infer defaults.
    xgb_params = walk_forward.xgb_params_for_config(config)
    config_payload = _config_payload(args, config, results, ablation, xgb_params)
    config_payload["out_json"] = str(out_json)
    report = walk_forward.build_metrics_report(run_id, created_at, config_payload, results)
    _add_stability(report, results)
    _add_versus_production(report, args, config, df, xgb_params)
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
