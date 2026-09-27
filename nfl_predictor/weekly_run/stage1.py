"""Weekly-run stage 1: one walk-forward of the production configuration.

The weekly run submits the deterministic floor (the predicted margin through the fixed normal
curve, with no fitted calibrator and no market blend), trained with the configured market mode.
Stage 1 scores that same configuration by walk-forward over the recent seasons, so every run
reports how production would have done against the market on games it had not seen. The
finished weeks are checkpointed under the run directory, so a stopped run resumes at its next
unfinished week.

Outputs under ``<run_dir>/wf_compare/``: the fold checkpoints (``wf_folds/``), one evaluation
artifact (``wf_candidate_<key>.json``, with the per-week, per-season and overall metrics and the
reliability table) and, with ``checkpoint_per_fold``, a progress line per finished week
(``wf_fold_progress.jsonl``). The pipeline writes the summary row as ``wf_compare.csv`` and
``wf_best.json``, the names older runs used for their candidate table and its winner.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.ml import ml_model_core, walk_forward, wf_compare_utils
from nfl_predictor.utils import fingerprints
from nfl_predictor.utils.logger import log

PRODUCTION_LABEL = "production"
_MARKET_MODE_FLAGS = {
    "features": (True, False),
    "anchor": (False, True),
    "hybrid": (True, True),
}


def market_mode_flags(mode: str) -> tuple[bool, bool]:
    """Return ``(include_market, market_anchor)`` for a market mode."""
    try:
        return _MARKET_MODE_FLAGS[mode]
    except KeyError:
        raise ValueError(f"Unknown market mode: {mode}") from None


def _wf_compare_dir(run_dir: Path) -> Path:
    """Return the walk-forward evaluation artifact directory."""
    return run_dir / "wf_compare"


def _candidate_artifact_path(run_dir: Path, candidate_key: str) -> Path:
    """Return the evaluation artifact path for a configuration key."""
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_key)
    return _wf_compare_dir(run_dir) / f"wf_candidate_{safe_key}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(fingerprints.to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write CSV to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _build_summary_row(
    candidate_key: str,
    market_mode: str,
    results: dict[str, Any],
    *,
    dataset_sha256: str,
    wf_run_fingerprint: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Build the summary row of the production configuration's walk-forward."""
    overall = results.get("overall", {})
    reliability = results.get("reliability", [])

    def metric(key: str) -> float:
        return float(overall.get(key, float("nan")))

    row: dict[str, Any] = {
        "candidate_key": candidate_key,
        "label": PRODUCTION_LABEL,
        "market_mode": market_mode,
        "brier": metric("brier"),
        "log_loss": metric("log_loss"),
        "deterministic_brier": metric("deterministic_brier"),
        "deterministic_log_loss": metric("deterministic_log_loss"),
        "market_brier": metric("market_brier"),
        "market_log_loss": metric("market_log_loss"),
        "deterministic_brier_vs_market": metric("deterministic_brier_vs_market"),
        "deterministic_log_loss_vs_market": metric("deterministic_log_loss_vs_market"),
        "deterministic_brier_vs_market_ci_low": metric("deterministic_brier_vs_market_ci_low"),
        "deterministic_brier_vs_market_ci_high": metric("deterministic_brier_vs_market_ci_high"),
        "deterministic_log_loss_vs_market_ci_low": metric(
            "deterministic_log_loss_vs_market_ci_low"
        ),
        "deterministic_log_loss_vs_market_ci_high": metric(
            "deterministic_log_loss_vs_market_ci_high"
        ),
        "reliability_ece": metrics_utils.reliability_ece(reliability),
        "pick_accuracy": metric("pick_accuracy"),
        "deterministic_pick_accuracy": metric("deterministic_pick_accuracy"),
        "market_pick_accuracy": metric("market_pick_accuracy"),
        "margin_mae": metric("margin_mae"),
        "total_mae": metric("total_mae"),
        "expected_points_avg": metric("expected_points_avg"),
        "actual_points_avg": metric("actual_points_avg"),
        "market_margin_resid_mae": metric("market_margin_resid_mae"),
        "market_total_resid_mae": metric("market_total_resid_mae"),
        "games": int(overall.get("games", 0) or 0),
        "weeks": int(overall.get("weeks", 0) or 0),
        "dataset_fingerprint": dataset_sha256,
        "wf_run_fingerprint": wf_run_fingerprint,
        "duration_seconds": float(duration_seconds),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    return row


def _append_fold_progress(
    path: Path,
    *,
    candidate_key: str,
    metrics: dict[str, Any],
) -> None:
    """Append a JSONL fold progress entry."""
    payload = {
        "candidate_key": candidate_key,
        "season": metrics.get("season"),
        "week": metrics.get("week"),
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fingerprints.to_jsonable(payload)) + "\n")


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    """Write the summary row as ``wf_compare.csv`` (one row) and ``wf_best.json``."""
    _atomic_write_csv(run_dir / "wf_compare.csv", pd.DataFrame([summary]))
    _atomic_write_json(run_dir / "wf_best.json", summary)


def evaluate_production(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    resume: bool,
    dataset_fingerprint: dict[str, Any],
    wf_run_fingerprint: str,
    checkpoint_per_fold: bool,
    eval_last_n_seasons: int,
    wf_start_week: int,
    calibration_weeks: int,
    include_postseason: bool,
    exclude_incomplete_seasons: bool,
    recency_half_life_seasons: float | None,
    market_mode: str,
    xgb_params_overrides: dict[str, Any],
    include_quantiles: bool,
    market_transform: bool | None = None,
    max_cardinality_ratio: float = 0.5,
) -> dict[str, Any]:
    """Walk the production configuration forward and return its summary row.

    ``market_transform`` and ``max_cardinality_ratio`` are the final fit's own options, so
    the walk-forward scores the configuration the week's picks come from.

    The walk-forward saves every finished week under ``wf_compare/wf_folds/``; with ``resume``
    an identical rerun restores those weeks instead of training them again.
    """
    wf_dir = _wf_compare_dir(run_dir)
    wf_dir.mkdir(parents=True, exist_ok=True)
    include_market, market_anchor = market_mode_flags(market_mode)
    candidate_key = wf_compare_utils.build_candidate_key(
        model_kind="margin_total",
        feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
        feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
        market_mode=market_mode,
        include_quantiles=include_quantiles,
        xgb_params_overrides=xgb_params_overrides,
    )
    cfg = walk_forward.WalkForwardConfig(
        eval_seasons=None,
        eval_last_n_seasons=eval_last_n_seasons,
        wf_start_week=wf_start_week,
        calibration="auto",
        calibration_weeks=calibration_weeks,
        random_seed=walk_forward.DEFAULT_RANDOM_SEED,
        include_postseason=include_postseason,
        exclude_incomplete_seasons=exclude_incomplete_seasons,
        recency_half_life_seasons=recency_half_life_seasons,
        include_market=include_market,
        market_transform=market_transform,
        market_anchor=market_anchor,
        include_quantiles=include_quantiles,
        max_cardinality_ratio=max_cardinality_ratio,
        feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
        feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
        xgb_params_overrides=xgb_params_overrides,
    )

    fold_callback: Callable[[dict[str, Any], walk_forward.WalkForwardFold], None] | None = None
    if checkpoint_per_fold:
        progress_path = wf_dir / "wf_fold_progress.jsonl"

        def fold_progress_callback(
            metrics: dict[str, Any], _fold: walk_forward.WalkForwardFold
        ) -> None:
            """Write a per-fold progress record."""
            _append_fold_progress(progress_path, candidate_key=candidate_key, metrics=metrics)

        fold_callback = fold_progress_callback

    log.info("Stage 1: walk-forward of the production configuration (%s)", candidate_key)
    start = time.monotonic()
    out = walk_forward.run_walk_forward_backtest(
        df,
        cfg,
        fold_callback=fold_callback,
        checkpoint_dir=wf_dir / "wf_folds",
        resume=resume,
    )
    duration = time.monotonic() - start
    summary_row = _build_summary_row(
        candidate_key,
        market_mode,
        out,
        dataset_sha256=str(dataset_fingerprint.get("sha256")),
        wf_run_fingerprint=wf_run_fingerprint,
        duration_seconds=duration,
    )
    _atomic_write_json(
        _candidate_artifact_path(run_dir, candidate_key),
        {
            "candidate_key": candidate_key,
            "config": cfg.to_dict(),
            "dataset_fingerprint": dataset_fingerprint,
            "wf_run_fingerprint": wf_run_fingerprint,
            "metrics": {
                "per_week": out.get("per_week"),
                "per_season": out.get("per_season"),
                "overall": out.get("overall"),
                "reliability": out.get("reliability"),
                "resolved_settings": out.get("resolved_settings"),
            },
            "summary": summary_row,
            "created_at": datetime.now(UTC).isoformat(),
            "duration_seconds": float(duration),
        },
    )
    log.info(
        "Stage 1 done in %.0fs: Brier %.4f (market %.4f), pick accuracy %.4f",
        duration,
        summary_row["brier"],
        summary_row["market_brier"],
        summary_row["pick_accuracy"],
    )
    return summary_row
