"""Weekly-run stage 1: one walk-forward of the production configuration.

The weekly run submits the deterministic floor (the predicted margin through the fixed normal
curve, with no fitted calibrator and no market blend), trained with the configured market mode.
Stage 1 scores that same configuration by walk-forward over the recent seasons, so every run
reports how production would have done against the market on games it had not seen. The
finished weeks are checkpointed under the run directory, so a stopped run resumes at its next
unfinished week.

Each week's floor uses the sigma estimated from the reference runs' out-of-fold errors
(``floor_sigma_history``) and stage 1's own earlier weeks.

Outputs under ``<run_dir>/wf_compare/``: the fold checkpoints (``wf_folds/``), one evaluation
artifact (``wf_candidate_<key>.json``, with the per-week, per-season and overall metrics and the
reliability table), the squared margin error of every game stage 1 predicted
(``wf_margin_errors.csv``, which the final fit's sigma reads for the weeks the reference runs
lack) and, with
``checkpoint_per_fold``, a progress line per finished week (``wf_fold_progress.jsonl``). The
pipeline writes the summary row as ``wf_compare.csv`` and ``wf_best.json``, the names older runs
used for their candidate table and its winner.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from nfl_predictor.ml import artifacts, floor_sigma, ml_model_core, walk_forward, wf_compare_utils
from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
        msg = f"Unknown market mode: {mode}"
        raise ValueError(msg) from None


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
        json.dumps(artifacts.to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write CSV to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def margin_errors_path(run_dir: Path) -> Path:
    """Return the file holding stage 1's squared margin error per game."""
    return _wf_compare_dir(run_dir) / "wf_margin_errors.csv"


def write_margin_errors(run_dir: Path, errors: pd.DataFrame) -> None:
    """Write stage 1's squared margin errors (``floor_sigma.ERROR_COLUMNS``)."""
    _atomic_write_csv(margin_errors_path(run_dir), errors.loc[:, list(floor_sigma.ERROR_COLUMNS)])


def read_margin_errors(run_dir: Path) -> pd.DataFrame:
    """Read stage 1's squared margin errors.

    Raises:
        FileNotFoundError: If stage 1 has not written them.

    """
    path = margin_errors_path(run_dir)
    if not path.exists():
        msg = f"Stage 1 margin errors are missing: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"game_id": str})


def _build_summary_row(
    candidate_key: str,
    market_mode: str,
    results: dict[str, Any],
    run: Stage1Run,
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
        "dataset_fingerprint": str(run.dataset_fingerprint.get("sha256")),
        "wf_run_fingerprint": run.wf_run_fingerprint,
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
        handle.write(json.dumps(artifacts.to_jsonable(payload)) + "\n")


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    """Write the summary row as ``wf_compare.csv`` (one row) and ``wf_best.json``."""
    _atomic_write_csv(run_dir / "wf_compare.csv", pd.DataFrame([summary]))
    _atomic_write_json(run_dir / "wf_best.json", summary)


@dataclass(frozen=True, kw_only=True)
class ProductionOptions:
    """The production configuration's walk-forward options, from the weekly run's settings.

    ``market_transform`` and ``max_cardinality_ratio`` are the final fit's own options, so
    the walk-forward scores the configuration the week's picks come from.
    """

    eval_last_n_seasons: int
    wf_start_week: int
    include_postseason: bool
    exclude_incomplete_seasons: bool
    recency_half_life_seasons: float | None
    market_mode: str
    xgb_params_overrides: dict[str, Any]
    include_quantiles: bool
    market_transform: bool | None = None
    max_cardinality_ratio: float = 0.5


@dataclass(frozen=True, kw_only=True)
class Stage1Run:
    """Where stage 1 writes, whether it may resume, and the provenance it records.

    With ``resume`` an identical rerun restores the finished weeks saved under
    ``wf_compare/wf_folds/`` instead of training them again. ``floor_sigma_history`` (the
    reference runs' errors) joins the walk-forward's own earlier weeks in each week's sigma.
    """

    run_dir: Path
    resume: bool
    dataset_fingerprint: dict[str, Any]
    wf_run_fingerprint: str
    checkpoint_per_fold: bool
    floor_sigma_history: floor_sigma.ErrorPool | None = None


def production_walk_forward_config(options: ProductionOptions) -> walk_forward.WalkForwardConfig:
    """Return the walk-forward configuration stage 1 runs for these options."""
    include_market, market_anchor = market_mode_flags(options.market_mode)
    return walk_forward.WalkForwardConfig(
        eval_seasons=None,
        eval_last_n_seasons=options.eval_last_n_seasons,
        wf_start_week=options.wf_start_week,
        calibration="auto",
        random_seed=walk_forward.DEFAULT_RANDOM_SEED,
        include_postseason=options.include_postseason,
        exclude_incomplete_seasons=options.exclude_incomplete_seasons,
        recency_half_life_seasons=options.recency_half_life_seasons,
        include_market=include_market,
        market_transform=options.market_transform,
        market_anchor=market_anchor,
        include_quantiles=options.include_quantiles,
        max_cardinality_ratio=options.max_cardinality_ratio,
        feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
        feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
        xgb_params_overrides=options.xgb_params_overrides,
    )


def evaluate_production(
    df: pd.DataFrame, options: ProductionOptions, run: Stage1Run
) -> dict[str, Any]:
    """Walk the production configuration forward and return its summary row.

    The squared margin errors of every predicted game are written to ``wf_margin_errors.csv``.
    """
    run_dir = run.run_dir
    wf_dir = _wf_compare_dir(run_dir)
    wf_dir.mkdir(parents=True, exist_ok=True)
    candidate_key = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(
            model_kind="margin_total",
            feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
            feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
            market_mode=options.market_mode,
            include_quantiles=options.include_quantiles,
            xgb_params_overrides=options.xgb_params_overrides,
        )
    )
    cfg = production_walk_forward_config(options)

    fold_callback: Callable[[dict[str, Any], walk_forward.WalkForwardFold], None] | None = None
    if run.checkpoint_per_fold:
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
        checkpoints=walk_forward.FoldCheckpoints(wf_dir / "wf_folds", resume=run.resume),
        floor_sigma_history=run.floor_sigma_history,
    )
    write_margin_errors(run_dir, floor_sigma.margin_errors(out["predictions"]))
    duration = time.monotonic() - start
    summary_row = _build_summary_row(candidate_key, options.market_mode, out, run, duration)
    _atomic_write_json(
        _candidate_artifact_path(run_dir, candidate_key),
        {
            "candidate_key": candidate_key,
            "config": cfg.to_dict(),
            "dataset_fingerprint": run.dataset_fingerprint,
            "wf_run_fingerprint": run.wf_run_fingerprint,
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
