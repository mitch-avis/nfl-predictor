"""Weekly-run stage 1: the resumable walk-forward comparison of probability candidates.

Every candidate in ``_WF_MATRIX`` is scored by walk-forward with checkpoints under the
run directory, and the candidates are ranked by Brier score, then log loss.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor.cli import options
from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.ml import ml_model_core, walk_forward, wf_compare_utils
from nfl_predictor.utils import fingerprints
from nfl_predictor.utils.logger import log

_WF_MATRIX: list[tuple[str, str, float, float]] = [
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


def _pick_best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick best row by deterministic Brier then deterministic log loss."""
    if not rows:
        raise ValueError("No walk-forward rows produced.")

    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (
            float(row.get("deterministic_brier", row.get("brier", float("inf")))),
            float(row.get("deterministic_log_loss", row.get("log_loss", float("inf")))),
        )

    return dict(sorted(rows, key=key)[0])


def _wf_compare_dir(run_dir: Path) -> Path:
    """Return the walk-forward comparison artifact directory."""
    return run_dir / "wf_compare"


def _candidate_artifact_path(run_dir: Path, candidate_key: str) -> Path:
    """Return the per-candidate artifact path for a candidate key."""
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


def _load_candidate_artifact(path: Path) -> dict[str, Any]:
    """Load a candidate artifact JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


def _mark_corrupt_artifact(path: Path) -> None:
    """Move a corrupt artifact aside with a timestamp suffix."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    corrupt_path = path.with_suffix(f"{path.suffix}.corrupt.{timestamp}")
    os.replace(path, corrupt_path)


def _candidate_artifact_valid(
    payload: dict[str, Any],
    *,
    candidate_key: str,
    dataset_sha256: str,
    wf_run_fingerprint: str,
) -> bool:
    """Return True when a candidate artifact matches the expected fingerprints."""
    if not payload:
        return False
    if payload.get("candidate_key") != candidate_key:
        return False
    if payload.get("wf_run_fingerprint") != wf_run_fingerprint:
        return False
    dataset_fp = payload.get("dataset_fingerprint", {})
    if dataset_fp.get("sha256") != dataset_sha256:
        return False
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    for key in ("per_week", "per_season", "overall"):
        if key not in metrics:
            return False
    summary = payload.get("summary")
    return isinstance(summary, dict) and bool(summary)


def _build_wf_candidates(
    *,
    eval_last_n_seasons: int,
    wf_start_week: int,
    calibration_weeks: int,
    include_postseason: bool,
    exclude_incomplete_seasons: bool,
    recency_half_life_seasons: float | None,
    market_mode: str,
    market_prob_source: str,
    market_prob_blend_method: str,
    win_prob_uncertainty: str,
    xgb_params_overrides: dict[str, Any],
    include_quantiles: bool,
) -> list[dict[str, Any]]:
    """Enumerate walk-forward candidates for comparison."""
    rows: list[dict[str, Any]] = []
    market_sources = ["raw", "novig"] if market_prob_source == "both" else [market_prob_source]
    blend_methods = (
        ["prob", "logit"] if market_prob_blend_method == "both" else [market_prob_blend_method]
    )
    uncertainty_modes = (
        [False, True] if win_prob_uncertainty == "both" else [win_prob_uncertainty == "on"]
    )

    for mode_label, include_market, market_anchor in options.market_modes(market_mode):
        for source in market_sources:
            for method in blend_methods:
                for use_uncertainty in uncertainty_modes:
                    uncertainty_label = "uncert" if use_uncertainty else "base"
                    for label, calib, weight, clamp in _WF_MATRIX:
                        run_label = f"{mode_label}_{source}_{method}_{uncertainty_label}_{label}"
                        candidate_key = wf_compare_utils.build_candidate_key(
                            model_kind="margin_total",
                            feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
                            feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
                            calibration=calib,
                            market_mode=mode_label,
                            market_prob_source=source,
                            market_prob_blend_method=method,
                            win_prob_use_uncertainty=use_uncertainty,
                            market_prob_weight=weight,
                            market_prob_clamp=clamp,
                            include_quantiles=include_quantiles,
                            xgb_params_overrides=xgb_params_overrides,
                        )
                        rows.append(
                            {
                                "candidate_key": candidate_key,
                                "label": run_label,
                                "calibration": calib,
                                "market_prob_weight": float(weight),
                                "market_prob_clamp": float(clamp),
                                "market_prob_source": source,
                                "market_prob_blend_method": method,
                                "win_prob_use_uncertainty": bool(use_uncertainty),
                                "market_mode": mode_label,
                                "include_market": include_market,
                                "market_anchor": market_anchor,
                                "eval_last_n_seasons": eval_last_n_seasons,
                                "wf_start_week": wf_start_week,
                                "calibration_weeks": calibration_weeks,
                                "include_postseason": include_postseason,
                                "exclude_incomplete_seasons": exclude_incomplete_seasons,
                                "recency_half_life_seasons": recency_half_life_seasons,
                                "xgb_params_overrides": xgb_params_overrides,
                                "include_quantiles": include_quantiles,
                                "feature_start": ml_model_core.DEFAULT_FEATURE_START_COLUMN,
                                "feature_end": ml_model_core.DEFAULT_FEATURE_END_COLUMN,
                            }
                        )

    return rows


def _build_summary_row(
    candidate: dict[str, Any],
    results: dict[str, Any],
    *,
    dataset_sha256: str,
    wf_run_fingerprint: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Build a summary row from walk-forward results."""
    overall = results.get("overall", {})
    reliability = results.get("reliability", [])
    row: dict[str, Any] = {
        "candidate_key": candidate["candidate_key"],
        "label": candidate.get("label"),
        "calibration": candidate["calibration"],
        "market_prob_weight": float(candidate["market_prob_weight"]),
        "market_prob_clamp": float(candidate["market_prob_clamp"]),
        "market_prob_source": candidate["market_prob_source"],
        "market_prob_blend_method": candidate["market_prob_blend_method"],
        "win_prob_use_uncertainty": bool(candidate["win_prob_use_uncertainty"]),
        "market_mode": candidate["market_mode"],
        "brier": float(overall.get("brier", float("nan"))),
        "log_loss": float(overall.get("log_loss", float("nan"))),
        "deterministic_brier": float(overall.get("deterministic_brier", float("nan"))),
        "deterministic_log_loss": float(overall.get("deterministic_log_loss", float("nan"))),
        "market_brier": float(overall.get("market_brier", float("nan"))),
        "market_log_loss": float(overall.get("market_log_loss", float("nan"))),
        "deterministic_brier_vs_market": float(
            overall.get("deterministic_brier_vs_market", float("nan"))
        ),
        "deterministic_log_loss_vs_market": float(
            overall.get("deterministic_log_loss_vs_market", float("nan"))
        ),
        "deterministic_brier_vs_market_ci_low": float(
            overall.get("deterministic_brier_vs_market_ci_low", float("nan"))
        ),
        "deterministic_brier_vs_market_ci_high": float(
            overall.get("deterministic_brier_vs_market_ci_high", float("nan"))
        ),
        "deterministic_log_loss_vs_market_ci_low": float(
            overall.get("deterministic_log_loss_vs_market_ci_low", float("nan"))
        ),
        "deterministic_log_loss_vs_market_ci_high": float(
            overall.get("deterministic_log_loss_vs_market_ci_high", float("nan"))
        ),
        "reliability_ece": metrics_utils.reliability_ece(reliability),
        "pick_accuracy": float(overall.get("pick_accuracy", float("nan"))),
        "deterministic_pick_accuracy": float(
            overall.get("deterministic_pick_accuracy", float("nan"))
        ),
        "market_pick_accuracy": float(overall.get("market_pick_accuracy", float("nan"))),
        "margin_mae": float(overall.get("margin_mae", float("nan"))),
        "total_mae": float(overall.get("total_mae", float("nan"))),
        "expected_points_avg": float(overall.get("expected_points_avg", float("nan"))),
        "actual_points_avg": float(overall.get("actual_points_avg", float("nan"))),
        "market_margin_resid_mae": float(overall.get("market_margin_resid_mae", float("nan"))),
        "market_total_resid_mae": float(overall.get("market_total_resid_mae", float("nan"))),
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


def _upsert_summary(
    summary_df: pd.DataFrame,
    summary_row: dict[str, Any] | None,
    *,
    full_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Insert or replace a summary row by candidate key."""
    if full_frame is not None:
        return full_frame.reset_index(drop=True)
    if summary_row is None:
        return summary_df
    if summary_df.empty:
        return pd.DataFrame([summary_row])
    filtered = summary_df[summary_df["candidate_key"] != summary_row["candidate_key"]]
    return pd.concat([filtered, pd.DataFrame([summary_row])], ignore_index=True)


def _rank_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a rank column using deterministic Brier then deterministic log loss."""
    if frame.empty:
        return frame
    primary_brier = "deterministic_brier" if "deterministic_brier" in frame.columns else "brier"
    primary_log_loss = (
        "deterministic_log_loss" if "deterministic_log_loss" in frame.columns else "log_loss"
    )
    ranked = frame.sort_values(
        [primary_brier, primary_log_loss],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _run_wf_compare(
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
    market_prob_source: str,
    market_prob_blend_method: str,
    win_prob_uncertainty: str,
    xgb_params_overrides: dict[str, Any],
    include_quantiles: bool,
) -> pd.DataFrame:
    """Run a walk-forward comparison matrix with resumable checkpoints."""
    wf_dir = _wf_compare_dir(run_dir)
    wf_dir.mkdir(parents=True, exist_ok=True)
    summary_path = wf_dir / "wf_summary.csv"
    fold_progress_path = wf_dir / "wf_fold_progress.jsonl"

    candidates = _build_wf_candidates(
        eval_last_n_seasons=eval_last_n_seasons,
        wf_start_week=wf_start_week,
        calibration_weeks=calibration_weeks,
        include_postseason=include_postseason,
        exclude_incomplete_seasons=exclude_incomplete_seasons,
        recency_half_life_seasons=recency_half_life_seasons,
        market_mode=market_mode,
        market_prob_source=market_prob_source,
        market_prob_blend_method=market_prob_blend_method,
        win_prob_uncertainty=win_prob_uncertainty,
        xgb_params_overrides=xgb_params_overrides,
        include_quantiles=include_quantiles,
    )

    summary_df = pd.DataFrame()
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)

    results_rows: list[dict[str, Any]] = []
    total = len(candidates)
    dataset_sha256 = str(dataset_fingerprint.get("sha256"))

    for idx, candidate in enumerate(candidates, start=1):
        candidate_key = candidate["candidate_key"]
        artifact_path = _candidate_artifact_path(run_dir, candidate_key)
        if resume and artifact_path.exists():
            try:
                payload = _load_candidate_artifact(artifact_path)
            except json.JSONDecodeError:
                payload = {}
            if _candidate_artifact_valid(
                payload,
                candidate_key=candidate_key,
                dataset_sha256=dataset_sha256,
                wf_run_fingerprint=wf_run_fingerprint,
            ):
                log.info("WF candidate %d/%d skipped (resume): %s", idx, total, candidate_key)
                summary_row = payload.get("summary")
                if isinstance(summary_row, dict):
                    results_rows.append(summary_row)
                    summary_df = _upsert_summary(summary_df, summary_row)
                    _atomic_write_csv(summary_path, summary_df)
                continue
            if artifact_path.exists():
                _mark_corrupt_artifact(artifact_path)

        log.info("WF candidate %d/%d starting: %s", idx, total, candidate_key)
        start = time.monotonic()
        cfg = walk_forward.WalkForwardConfig(
            eval_seasons=None,
            eval_last_n_seasons=eval_last_n_seasons,
            wf_start_week=wf_start_week,
            calibration=candidate["calibration"],
            calibration_weeks=calibration_weeks,
            random_seed=42,
            include_postseason=include_postseason,
            exclude_incomplete_seasons=exclude_incomplete_seasons,
            recency_half_life_seasons=recency_half_life_seasons,
            include_market=bool(candidate["include_market"]),
            market_transform=None,
            market_anchor=bool(candidate["market_anchor"]),
            market_prob_weight=float(candidate["market_prob_weight"]),
            market_prob_clamp=float(candidate["market_prob_clamp"]),
            market_prob_source=str(candidate["market_prob_source"]),
            market_prob_blend_method=str(candidate["market_prob_blend_method"]),
            win_prob_use_uncertainty=bool(candidate["win_prob_use_uncertainty"]),
            include_quantiles=include_quantiles,
            max_cardinality_ratio=0.5,
            feature_start=ml_model_core.DEFAULT_FEATURE_START_COLUMN,
            feature_end=ml_model_core.DEFAULT_FEATURE_END_COLUMN,
            xgb_params_overrides=xgb_params_overrides,
        )

        fold_callback: Callable[[dict[str, Any], walk_forward.WalkForwardFold], None] | None = None
        if checkpoint_per_fold:

            def fold_progress_callback(
                metrics: dict[str, Any],
                _fold: walk_forward.WalkForwardFold,
                *,
                candidate_key: str = candidate_key,
                path: Path = fold_progress_path,
            ) -> None:
                """Write a per-fold progress record for the current candidate."""
                _append_fold_progress(path, candidate_key=candidate_key, metrics=metrics)

            fold_callback = fold_progress_callback

        # Finished weeks are saved inside the run, so a candidate stopped partway resumes
        # at its next unfinished week rather than from the start.
        out = walk_forward.run_walk_forward_backtest(
            df,
            cfg,
            fold_callback=fold_callback,
            checkpoint_dir=wf_dir / "wf_folds",
            resume=resume,
        )
        duration = time.monotonic() - start
        summary_row = _build_summary_row(
            candidate,
            out,
            dataset_sha256=dataset_sha256,
            wf_run_fingerprint=wf_run_fingerprint,
            duration_seconds=duration,
        )
        candidate_payload = {
            "candidate_key": candidate_key,
            "candidate": candidate,
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
        }
        _atomic_write_json(artifact_path, candidate_payload)
        summary_df = _upsert_summary(summary_df, summary_row)
        _atomic_write_csv(summary_path, summary_df)

        minutes = int(duration // 60)
        seconds = duration - (minutes * 60)
        log.info(
            "WF candidate %d/%d done in %dm%.1fs: brier=%.4f logloss=%.4f",
            idx,
            total,
            minutes,
            seconds,
            summary_row.get("brier", float("nan")),
            summary_row.get("log_loss", float("nan")),
        )
        results_rows.append(summary_row)

    result_df = pd.DataFrame(results_rows)
    if not result_df.empty:
        result_df = _rank_summary(result_df)
        result_df = result_df.sort_values(["brier", "log_loss"], ascending=[True, True])
        summary_df = _upsert_summary(summary_df, None, full_frame=result_df)
        _atomic_write_csv(summary_path, summary_df)
    return result_df
