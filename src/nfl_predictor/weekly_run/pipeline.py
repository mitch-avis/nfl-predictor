"""Weekly orchestration for data refresh, evaluation, training, and reporting.

The weekly run does the whole workflow in one command:
1) refresh data (ETL)
2) walk-forward the production configuration over the recent seasons, to report how it scores
3) train the production configuration on every completed game
4) generate weekly predictions and reporting outputs

Production submits the deterministic floor, Phi(predicted margin / sigma), with no fitted
calibrator and no market blend or clamp. Sigma is the root-mean-square out-of-fold margin error
before the predicted week: stage 1 pools the configured reference runs' fold checkpoints with its
own earlier weeks, and the final fit pools the reference runs with stage 1's weeks before the
predicted week that the reference runs have no rows for, then records the sigma in the saved
model.

Outputs land under the run directory (default: models/<run_id>/) and include:
- wf_compare.csv / wf_best.json (the walk-forward summary row) and wf_compare/
- model.joblib / metrics_report.json / metadata.json
- *_predictions.csv / *_confidence_picks.csv
- *_betting_report.csv (when market columns exist)
- power rankings + projected standings (when data is available)
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nfl_predictor import constants, data_collection
from nfl_predictor.ml import artifacts, floor_sigma, metrics, ml_model_core, walk_forward
from nfl_predictor.ml.metrics import confidence_ranks
from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
from nfl_predictor.ml.ml_model_training import (
    TrainingOptions,
    TrainingRecord,
    train_margin_total_model_with_report,
    write_training_artifacts,
)
from nfl_predictor.reporting import power_rankings
from nfl_predictor.reporting.betting_report import build_betting_report
from nfl_predictor.utils import fingerprints
from nfl_predictor.utils.logger import log
from nfl_predictor.weekly_run import config, inputs, stage1

if TYPE_CHECKING:
    import argparse


def _data_collection_argv(spec: str | None) -> list[str] | None:
    """Split a pass-through argument string for the data collection stage.

    Args:
        spec: Shell-quoted arguments for ``data_collection.main``, or ``None``.

    Returns:
        The parsed argument list, or ``None`` when nothing was requested.

    """
    if spec is None:
        return None
    argv = shlex.split(spec)
    return argv or None


def _refresh_data(spec: str | None) -> None:
    """Run the data collection refresh, forwarding optional pass-through arguments.

    Args:
        spec: Shell-quoted arguments for ``data_collection.main``, or ``None`` to
            run the refresh with its own defaults.

    """
    argv = _data_collection_argv(spec)
    if argv is None:
        log.info("Refreshing data...")
        data_collection.main()
        return
    log.info("Refreshing data with arguments: %s", " ".join(argv))
    data_collection.main(argv)


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse namespace to a JSON-friendly dict."""
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _build_confidence_picks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Extract confidence pool picks from a predictions DataFrame."""
    if "home_win_prob" not in predictions.columns:
        msg = "Predictions missing home_win_prob for confidence picks."
        raise ValueError(msg)

    picks = predictions.copy()
    if "predicted_winner" not in picks.columns:
        home_col = "home_abbr" if "home_abbr" in picks.columns else None
        away_col = "away_abbr" if "away_abbr" in picks.columns else None
        if home_col and away_col:
            picks["predicted_winner"] = picks[home_col].where(
                metrics.picks_home(picks["home_win_prob"].to_numpy()), picks[away_col]
            )
        else:
            picks["predicted_winner"] = np.where(
                metrics.picks_home(picks["home_win_prob"].to_numpy()), "home", "away"
            )

    if "confidence_rank" not in picks.columns:
        tiebreaker = picks["game_id"].to_numpy() if "game_id" in picks.columns else None
        picks["confidence_rank"] = confidence_ranks(
            picks["home_win_prob"].to_numpy(float), tiebreaker
        )

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
    return all(path.exists() for path in outputs_list)


def _write_stage_marker(
    marker_path: Path,
    *,
    dataset_hash: str,
    config_hash: str,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    artifacts.write_json(marker_path, payload)


def _production_floor_sigma(
    args: argparse.Namespace, run_dir: Path, reference: floor_sigma.ErrorPool
) -> tuple[floor_sigma.ErrorPool, tuple[int, int]]:
    """Return the final fit's error pool and the week its sigma is for.

    The week is the one being predicted (the prediction file's single season and week), or,
    for a file spanning several weeks, the week after the newest completed game. The pool is
    the reference runs' errors plus stage 1's errors before that week at every (season, week)
    the reference runs have no rows for: the seasons after the reference, and the predicted
    season's earlier weeks. Where both have a week, the reference's errors are used.
    """
    predict_path = inputs.resolve_predict_path(args.predict_path, constants.DATA_PATH)
    season, week = inputs.infer_season_week(pd.read_csv(predict_path, usecols=["season", "week"]))
    if season is None or week is None:
        completed = pd.read_csv(args.data_path, usecols=["season", "week"])
        season, week = floor_sigma.week_after(completed)
        log.info(
            "%s spans several weeks; the floor sigma is for season %d week %d.",
            predict_path,
            season,
            week,
        )
    own = stage1.read_margin_errors(run_dir)
    before = (own["season"] < season) | ((own["season"] == season) & (own["week"] < week))
    pool = floor_sigma.ErrorPool(
        floor_sigma.fill_gaps(reference.errors, own[before]),
        (*reference.sources, str(stage1.margin_errors_path(run_dir))),
    )
    return pool, (int(season), int(week))


def _power_rankings_outputs(out_dir: Path, season: int, through_week: int) -> list[Path]:
    suffix = f"season_{season}_week_{through_week:02d}"
    return [
        out_dir / f"power_rankings_{suffix}.csv",
        out_dir / f"projected_standings_{suffix}.csv",
        out_dir / f"projected_division_standings_{suffix}.csv",
    ]


@dataclass(frozen=True)
class _RunContext:
    """The weekly run's arguments, identity and dataset, which every stage reads."""

    args: argparse.Namespace
    run_id: str
    run_dir: Path
    output_dir: Path
    created_at: str
    dataset_hash: str
    dataset_fingerprint: dict[str, Any]
    config_payload: dict[str, Any]


def _start_run(args: argparse.Namespace) -> _RunContext:
    """Check the dataset, fingerprint it, and create the run and output directories."""
    if not args.data_path.exists():
        msg = f"Missing dataset: {args.data_path}"
        raise FileNotFoundError(msg)

    dataset_hash = artifacts.sha256_file(args.data_path)
    dataset_fingerprint = fingerprints.dataset_fingerprint(args.data_path)
    created_at = datetime.now(UTC).isoformat()

    config_payload = _config_payload(args)
    run_id = args.run_id or artifacts.generate_run_id("weekly", dataset_hash, config_payload)
    run_dir = args.run_dir or (Path(constants.ROOT_DIR) / "models" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run id: %s", run_id)
    log.info("Run dir: %s", run_dir)
    return _RunContext(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        output_dir=output_dir,
        created_at=created_at,
        dataset_hash=dataset_hash,
        dataset_fingerprint=dataset_fingerprint,
        config_payload=config_payload,
    )


def _wf_run_fingerprint(
    ctx: _RunContext, options: stage1.ProductionOptions, reference_pool: floor_sigma.ErrorPool
) -> str:
    """Return the fingerprint of stage 1's walk-forward: its dataset, settings and code."""
    args = ctx.args
    return fingerprints.wf_run_fingerprint(
        ctx.dataset_fingerprint,
        {
            "eval_last_n_seasons": args.wf_eval_last_n_seasons,
            "wf_start_week": args.wf_start_week,
            "include_postseason": bool(args.wf_include_postseason),
            "exclude_incomplete_seasons": bool(args.wf_exclude_incomplete_seasons),
            "recency_half_life_seasons": args.wf_recency_half_life_seasons,
            "market_mode": args.wf_market_mode,
            "include_quantiles": bool(args.wf_include_quantiles),
            "market_transform": args.market_transform,
            "max_cardinality_ratio": float(args.max_cardinality_ratio),
            "feature_start": ml_model_core.DEFAULT_FEATURE_START_COLUMN,
            "feature_end": ml_model_core.DEFAULT_FEATURE_END_COLUMN,
            "xgb_params_overrides": options.xgb_params_overrides,
            "floor_sigma_history": reference_pool.digest(),
        },
        code_version=artifacts.git_commit_hash(),
    )


def _run_stage1(ctx: _RunContext, reference_pool: floor_sigma.ErrorPool) -> dict[str, Any]:
    """Stage 1: walk forward the production configuration, or reuse a finished walk."""
    args = ctx.args
    wf_compare_csv = ctx.run_dir / "wf_compare.csv"
    wf_best_json = ctx.run_dir / "wf_best.json"
    options = config.stage1_options(args)
    wf_config = {
        **asdict(options),
        "checkpoint_per_fold": bool(args.wf_checkpoint_per_fold),
        "floor_sigma_history": reference_pool.digest(),
    }
    wf_run_fingerprint = _wf_run_fingerprint(ctx, options, reference_pool)

    wf_config_hash = artifacts.stable_short_hash(wf_config)
    wf_marker = _stage_marker_path(ctx.run_dir, "wf_compare")

    if args.resume and _stage_can_reuse(
        wf_marker,
        ctx.dataset_hash,
        wf_config_hash,
        [wf_compare_csv, wf_best_json, stage1.margin_errors_path(ctx.run_dir)],
    ):
        log.info("Stage 1: reuse %s", wf_best_json)
        wf_summary = json.loads(wf_best_json.read_text(encoding="utf-8"))
    else:
        df = walk_forward.load_games(args.data_path)
        run = stage1.Stage1Run(
            run_dir=ctx.run_dir,
            resume=bool(args.resume),
            dataset_fingerprint=ctx.dataset_fingerprint,
            wf_run_fingerprint=wf_run_fingerprint,
            checkpoint_per_fold=bool(args.wf_checkpoint_per_fold),
            floor_sigma_history=reference_pool,
        )
        wf_summary = stage1.evaluate_production(df, options, run)
        stage1.write_summary(ctx.run_dir, wf_summary)
        _write_stage_marker(
            wf_marker,
            dataset_hash=ctx.dataset_hash,
            config_hash=wf_config_hash,
            stage="wf_compare",
        )

    log.info("Walk-forward summary: %s", wf_summary)
    return wf_summary


def _run_stage2(
    ctx: _RunContext, reference_pool: floor_sigma.ErrorPool, wf_summary: dict[str, Any]
) -> tuple[ml_model_core.MarginTotalModel | None, artifacts.RunPaths]:
    """Stage 2: train the final model, or reuse a finished fit (then the model is not loaded)."""
    args = ctx.args
    columns_only = pd.read_csv(args.data_path, nrows=1)
    optuna_config, train_config = config.final_fit_options(args, columns_only, run_dir=ctx.run_dir)
    sigma_pool, sigma_week = _production_floor_sigma(args, ctx.run_dir, reference_pool)
    train_config["floor_sigma"] = {
        "sources": list(sigma_pool.sources),
        "pool_digest": sigma_pool.digest(),
        "week": list(sigma_week),
    }
    train_config_hash = artifacts.stable_short_hash(train_config)
    train_marker = _stage_marker_path(ctx.run_dir, "train")
    paths = artifacts.resolve_run_paths(ctx.run_id, run_dir=ctx.run_dir)

    if args.resume and _stage_can_reuse(
        train_marker,
        ctx.dataset_hash,
        train_config_hash,
        [paths.model_path, paths.metrics_path, paths.metadata_path],
    ):
        log.info("Stage 2: reuse %s", paths.model_path)
        return None, paths

    log.info("Stage 2: training margin/total model")
    result = train_margin_total_model_with_report(
        TrainingOptions(
            data_path=args.data_path,
            holdout_seasons=int(args.holdout_seasons),
            include_market=train_config["include_market"],
            max_cardinality_ratio=float(args.max_cardinality_ratio),
            optuna_config=optuna_config,
            market_transform=bool(train_config["market_transform"]),
            market_anchor=bool(train_config["market_anchor"]),
            include_postseason=bool(args.include_postseason),
            postseason_weight=float(args.postseason_weight),
            recency_half_life_seasons=args.train_recency_half_life_seasons,
            min_season=None,
            max_season=None,
            feature_start=str(args.feature_start),
            feature_end=str(args.feature_end),
            xgb_params_overrides=train_config["xgb_params_overrides"],
            floor_sigma_pool=sigma_pool,
            floor_sigma_week=sigma_week,
        )
    )
    record = TrainingRecord(
        run_id=ctx.run_id,
        created_at=ctx.created_at,
        dataset_hash=ctx.dataset_hash,
        config_payload={
            **ctx.config_payload,
            "wf_best": wf_summary,
            "train_config": train_config,
        },
    )
    write_training_artifacts(result, record, ctx.run_dir)
    _write_stage_marker(
        train_marker,
        dataset_hash=ctx.dataset_hash,
        config_hash=train_config_hash,
        stage="train",
    )
    return result.model, paths


def _run_stage3(
    ctx: _RunContext, model: ml_model_core.MarginTotalModel | None, model_path: Path
) -> tuple[Path, str]:
    """Stage 3: predict the week and write the confidence picks, or reuse them.

    Returns the predictions path and the hash of the model that made them.
    """
    args = ctx.args
    predict_path = inputs.resolve_predict_path(args.predict_path, constants.DATA_PATH)
    predict_preview = pd.read_csv(predict_path, nrows=5)
    preview_season, preview_week = inputs.infer_season_week(predict_preview)
    preview_outputs = inputs.resolve_output_paths(ctx.output_dir, preview_season, preview_week)
    predictions_path = preview_outputs["predictions"]
    confidence_path = preview_outputs["confidence_picks"]
    predict_hash = artifacts.sha256_file(predict_path)
    if model is None:
        model = ml_model_core.load_model_checkpoint(model_path, "margin_total")
    model_hash = artifacts.sha256_file(model_path)

    predictions_config = {
        "predict_path": str(predict_path),
        "predict_hash": predict_hash,
        "model_hash": model_hash,
        "score_rounding": args.score_rounding,
        "output_dir": str(ctx.output_dir),
    }
    predictions_hash = artifacts.stable_short_hash(predictions_config)
    predictions_marker = _stage_marker_path(ctx.run_dir, "predictions")

    outputs = [predictions_path, confidence_path]
    if args.resume and _stage_can_reuse(
        predictions_marker, ctx.dataset_hash, predictions_hash, outputs
    ):
        log.info("Stage 3: reuse predictions output")
        return predictions_path, model_hash

    log.info("Stage 3: predicting %s", predict_path)
    predictions = predict_week_margin_total(
        model,
        games_path=predict_path,
        output_path=None,
        pretty_output=False,
        score_rounding=str(args.score_rounding),
    )
    season, week = inputs.infer_season_week(predictions)
    output_paths = inputs.resolve_output_paths(ctx.output_dir, season, week)
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
        dataset_hash=ctx.dataset_hash,
        config_hash=predictions_hash,
        stage="predictions",
        extra={"outputs": [str(predictions_path), str(confidence_path)]},
    )
    return predictions_path, model_hash


def _write_power_rankings(
    ctx: _RunContext, season: int | None, week: int | None, model_path: Path
) -> list[Path]:
    """Write the power rankings and projected standings; return the files written."""
    args = ctx.args
    pr_season = args.power_rankings_season or season
    pr_week = args.power_rankings_through_week
    if pr_week is None:
        pr_week = inputs.default_power_rankings_through_week(pr_season, week)

    if pr_season is None or pr_week is None:
        log.info("Power rankings skipped: unable to infer season/week.")
        return []
    if not args.power_rankings_data_ml.exists():
        log.info("Power rankings skipped: missing %s", args.power_rankings_data_ml)
        return []
    if not args.power_rankings_data_schedule.exists():
        log.info("Power rankings skipped: missing %s", args.power_rankings_data_schedule)
        return []
    pr_out_dir = args.power_rankings_out_dir or ctx.output_dir
    pr_out_dir.mkdir(parents=True, exist_ok=True)
    model = ml_model_core.load_model_checkpoint(model_path, "margin_total")
    try:
        ranking_inputs = power_rankings.RankingInputs(
            data_ml=args.power_rankings_data_ml,
            data_schedule=args.power_rankings_data_schedule,
            season=pr_season,
            through_week=pr_week,
            include_postseason=bool(args.power_rankings_include_postseason),
        )
        result = power_rankings.compute_power_rankings(
            model, ranking_inputs, config.power_ranking_options(args)
        )
    except power_rankings.StrengthSnapshotUnavailableError as exc:
        log.warning("Power rankings skipped: %s", exc)
        return []
    power_rankings.write_ranking_outputs(
        result,
        out_dir=pr_out_dir,
        season=pr_season,
        through_week=pr_week,
    )
    return _power_rankings_outputs(pr_out_dir, pr_season, pr_week)


def _run_stage4(
    ctx: _RunContext, predictions_path: Path, model_hash: str, model_path: Path
) -> None:
    """Stage 4: write the betting report and the power rankings, or reuse them."""
    args = ctx.args
    predictions_hash = artifacts.sha256_file(predictions_path)
    pr_out_dir_default = args.power_rankings_out_dir or ctx.output_dir
    report_config = {
        "predictions_hash": predictions_hash,
        "model_hash": model_hash,
        "skip_power_rankings": bool(args.skip_power_rankings),
        "power_rankings_season": args.power_rankings_season,
        "power_rankings_through_week": args.power_rankings_through_week,
        "power_rankings_data_ml": str(args.power_rankings_data_ml),
        "power_rankings_data_schedule": str(args.power_rankings_data_schedule),
        "power_rankings_out_dir": str(pr_out_dir_default),
        "power_rankings_include_postseason": bool(args.power_rankings_include_postseason),
        **config.power_rankings_report_config(args),
    }
    report_hash = artifacts.stable_short_hash(report_config)
    report_marker = _stage_marker_path(ctx.run_dir, "reports")

    outputs: list[Path] = []
    if args.resume and _stage_can_reuse(report_marker, ctx.dataset_hash, report_hash, outputs):
        log.info("Stage 4: reuse reports output")
        return

    predictions = pd.read_csv(predictions_path)
    season, week = inputs.infer_season_week(predictions)
    output_paths = inputs.resolve_output_paths(ctx.output_dir, season, week)
    betting_report_path = output_paths["betting_report"]

    try:
        report_df = build_betting_report(predictions)
    except ValueError as exc:
        log.info("Betting report skipped: %s", exc)
    else:
        report_df.to_csv(betting_report_path, index=False)
        outputs.append(betting_report_path)
        log.info("Wrote betting report to %s", betting_report_path)

    if not args.skip_power_rankings:
        outputs.extend(_write_power_rankings(ctx, season, week, model_path))

    _write_stage_marker(
        report_marker,
        dataset_hash=ctx.dataset_hash,
        config_hash=report_hash,
        stage="reports",
        extra={"outputs": [str(path) for path in outputs]},
    )
    log.info("Weekly run complete.")


def main() -> int:
    """CLI entrypoint."""
    args = config.parse_args(sys.argv[1:])
    config.apply_run_defaults(args)

    if not args.skip_data_refresh:
        if args.dry_run:
            log.info("Dry-run: would refresh data before running the stages")
        else:
            _refresh_data(args.data_collection_args)

    ctx = _start_run(args)
    if args.dry_run:
        log.info("Dry-run: would write %s", ctx.run_dir / "wf_compare.csv")
        log.info("Dry-run: would write %s", ctx.run_dir / "wf_best.json")
        log.info("Dry-run: would write model + metrics under %s", ctx.run_dir)
        log.info("Dry-run: outputs would land under %s", ctx.output_dir)
        return 0

    # The reference runs' out-of-fold errors, for the floor sigma of stage 1 and the final fit.
    # A configured run that is missing stops the run here rather than changing the sigma.
    reference_pool = floor_sigma.load_reference_pool(args.floor_sigma_reference_runs)
    log.info(
        "Floor sigma reference: %d games from %s",
        len(reference_pool.errors),
        list(reference_pool.sources) or "no reference runs",
    )

    wf_summary = _run_stage1(ctx, reference_pool)
    model, paths = _run_stage2(ctx, reference_pool, wf_summary)
    predictions_path, model_hash = _run_stage3(ctx, model, paths.model_path)
    if not predictions_path.exists():
        log.info("Predictions file not found; skipping reports.")
        return 0
    _run_stage4(ctx, predictions_path, model_hash, paths.model_path)
    return 0
