"""Weekly orchestration for data refresh, model selection, training, and reporting.

The weekly run does the whole workflow in one command:
1) refresh data (ETL)
2) walk-forward compare to select calibration + market settings
3) train the selected model configuration
4) generate weekly predictions and reporting outputs

Outputs land under the run directory (default: models/<run_id>/) and include:
- wf_compare.csv / wf_best.json
- model.joblib / metrics_report.json / metadata.json
- *_predictions.csv / *_confidence_picks.csv
- *_betting_report.csv (when market columns exist)
- power rankings + projected standings (when data is available)
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor import constants, data_collection
from nfl_predictor.ml import artifacts, ml_model_core, walk_forward
from nfl_predictor.ml.ml_model_core import MarketProbConfig, OptunaConfig
from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
from nfl_predictor.ml.ml_model_training import train_margin_total_model_with_report
from nfl_predictor.reporting import power_rankings
from nfl_predictor.reporting.betting_report import build_betting_report
from nfl_predictor.utils import fingerprints
from nfl_predictor.utils.logger import log
from nfl_predictor.weekly_run import config, inputs, stage1


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
    args = config._parse_args()
    if args.train_recency_half_life_seasons is None:
        args.train_recency_half_life_seasons = args.wf_recency_half_life_seasons

    if not args.skip_data_refresh:
        _refresh_data(args.data_collection_args)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

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
        "recency_half_life_seasons": args.wf_recency_half_life_seasons,
        "market_mode": args.wf_market_mode,
        "market_prob_source": args.wf_market_prob_source,
        "market_prob_blend_method": args.wf_market_prob_blend_method,
        "win_prob_uncertainty": args.wf_win_prob_uncertainty,
        "exclude_incomplete_seasons": bool(args.wf_exclude_incomplete_seasons),
        "checkpoint_per_fold": bool(args.wf_checkpoint_per_fold),
        "wf_matrix": stage1._WF_MATRIX,
        "xgb_params_overrides": {
            "n_estimators": int(args.wf_n_estimators),
            "max_depth": int(args.wf_max_depth),
            "learning_rate": float(args.wf_learning_rate),
            "n_jobs": config.xgb_thread_count(args),
            "verbosity": 0,
        },
        "include_quantiles": bool(args.wf_include_quantiles),
    }

    # Propagate GPU/CPU runtime settings to walk-forward folds too.
    if args.xgb_tree_method:
        wf_config["xgb_params_overrides"]["tree_method"] = args.xgb_tree_method
    if args.xgb_device:
        wf_config["xgb_params_overrides"]["device"] = args.xgb_device

    wf_run_fingerprint = fingerprints.wf_run_fingerprint(
        dataset_fingerprint,
        {
            "eval_last_n_seasons": args.wf_eval_last_n_seasons,
            "wf_start_week": args.wf_start_week,
            "calibration_weeks": args.wf_calibration_weeks,
            "include_postseason": bool(args.wf_include_postseason),
            "exclude_incomplete_seasons": bool(args.wf_exclude_incomplete_seasons),
            "recency_half_life_seasons": args.wf_recency_half_life_seasons,
            "market_mode": args.wf_market_mode,
            "market_prob_source": args.wf_market_prob_source,
            "market_prob_blend_method": args.wf_market_prob_blend_method,
            "win_prob_uncertainty": args.wf_win_prob_uncertainty,
            "include_quantiles": bool(args.wf_include_quantiles),
            "feature_start": ml_model_core.DEFAULT_FEATURE_START_COLUMN,
            "feature_end": ml_model_core.DEFAULT_FEATURE_END_COLUMN,
            "xgb_params_overrides": wf_config["xgb_params_overrides"],
        },
        code_version=artifacts.git_commit_hash(),
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
        wf_result_df = stage1._run_wf_compare(
            df,
            run_dir=run_dir,
            resume=bool(args.resume),
            dataset_fingerprint=dataset_fingerprint,
            wf_run_fingerprint=wf_run_fingerprint,
            checkpoint_per_fold=bool(args.wf_checkpoint_per_fold),
            eval_last_n_seasons=args.wf_eval_last_n_seasons,
            wf_start_week=args.wf_start_week,
            calibration_weeks=args.wf_calibration_weeks,
            include_postseason=bool(args.wf_include_postseason),
            exclude_incomplete_seasons=bool(args.wf_exclude_incomplete_seasons),
            recency_half_life_seasons=args.wf_recency_half_life_seasons,
            market_mode=args.wf_market_mode,
            market_prob_source=args.wf_market_prob_source,
            market_prob_blend_method=args.wf_market_prob_blend_method,
            win_prob_uncertainty=args.wf_win_prob_uncertainty,
            xgb_params_overrides=wf_config["xgb_params_overrides"],
            include_quantiles=bool(args.wf_include_quantiles),
        )
        if not wf_result_df.empty:
            wf_result_df = stage1._rank_summary(wf_result_df)
        stage1._atomic_write_csv(wf_compare_csv, wf_result_df)
        best_rows = [
            {str(key): value for key, value in row.items()}
            for row in wf_result_df.to_dict(orient="records")
        ]
        best_row = stage1._pick_best_row(best_rows)
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
        n_trials=args.tune_trials,
        cv_splits=int(args.tune_cv_splits),
        objective=str(args.tune_objective),
        early_stopping_rounds=int(args.tune_early_stopping_rounds),
        tree_method=args.xgb_tree_method,
        device=args.xgb_device,
        storage=optuna_storage,
        study_name=args.tune_study_name,
        best_params_out=None,
        xgb_n_jobs=config.xgb_thread_count(args),
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
    predict_path = inputs._resolve_predict_path(args.predict_path, Path(constants.DATA_PATH))
    predict_preview = pd.read_csv(predict_path, nrows=5)
    preview_season, preview_week = inputs._infer_season_week(predict_preview)
    preview_outputs = inputs._resolve_output_paths(output_dir, preview_season, preview_week)
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
        season, week = inputs._infer_season_week(predictions)
        output_paths = inputs._resolve_output_paths(output_dir, season, week)
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
        "skip_power_rankings": bool(args.skip_power_rankings),
        "power_rankings_season": args.power_rankings_season,
        "power_rankings_through_week": args.power_rankings_through_week,
        "power_rankings_data_ml": str(args.power_rankings_data_ml),
        "power_rankings_data_schedule": str(args.power_rankings_data_schedule),
        "power_rankings_out_dir": str(pr_out_dir_default),
        "power_rankings_include_postseason": bool(args.power_rankings_include_postseason),
        **config._power_rankings_report_config(args),
    }
    report_hash = artifacts.stable_short_hash(report_config)
    report_marker = _stage_marker_path(run_dir, "reports")

    outputs = []
    if args.resume and _stage_can_reuse(report_marker, dataset_hash, report_hash, outputs):
        log.info("Stage 4: reuse reports output")
        return 0

    predictions = pd.read_csv(predictions_path)
    season, week = inputs._infer_season_week(predictions)
    output_paths = inputs._resolve_output_paths(output_dir, season, week)
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
        pr_season = args.power_rankings_season or season
        pr_week = args.power_rankings_through_week
        if pr_week is None:
            pr_week = inputs._default_power_rankings_through_week(pr_season, week)

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
            try:
                result = power_rankings.compute_power_rankings(
                    model,
                    model_kind="margin_total",
                    data_ml=args.power_rankings_data_ml,
                    data_schedule=args.power_rankings_data_schedule,
                    season=pr_season,
                    through_week=pr_week,
                    include_postseason=bool(args.power_rankings_include_postseason),
                    options=config._power_ranking_options(args),
                )
            except power_rankings.StrengthSnapshotUnavailableError as exc:
                log.warning("Power rankings skipped: %s", exc)
            else:
                power_rankings.write_ranking_outputs(
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
