"""Leakage audit helpers.

This module provides a lightweight, time-aware-ish leakage audit for the ML dataset.
It is intentionally heuristic: it can *prove* certain forms of leakage (e.g. targets
included as features) and can *flag* suspicious columns (e.g. near-perfect correlation
with the label).

The audit is designed to be run on the *completed games* dataset used for training/
walk-forward evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_predictor import ml_model
from nfl_predictor.utils.logger import log

# Fewer finite pairs than this give no correlation.
MIN_PAIRS_FOR_CORRELATION = 3
# A feature equal to a score column in at least this share of rows is the score itself.
NEAR_IDENTICAL_SHARE = 0.999999


@dataclass(frozen=True)
class LeakageAuditConfig:
    """Configuration for leakage audit."""

    feature_start: str = ml_model.DEFAULT_FEATURE_START_COLUMN
    feature_end: str = ml_model.DEFAULT_FEATURE_END_COLUMN
    include_market: bool = True
    market_transform: bool = False
    max_cardinality_ratio: float = 0.5

    suspicious_corr_threshold: float = 0.995
    fail_corr_threshold: float = 0.999999
    equality_rate_threshold: float = 0.98


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < MIN_PAIRS_FOR_CORRELATION:
        return None
    x_m = x[mask]
    y_m = y[mask]
    if np.nanstd(x_m) == 0.0 or np.nanstd(y_m) == 0.0:
        return None
    corr = float(np.corrcoef(x_m, y_m)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _equality_rate(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return None
    return float(np.mean(np.isclose(x[mask], y[mask], atol=1e-9, rtol=0.0)))


@dataclass(frozen=True)
class _ScoreTargets:
    """The label-space targets a feature must not reproduce."""

    away: np.ndarray
    home: np.ndarray
    margin: np.ndarray
    total: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame, target_away: str, target_home: str) -> _ScoreTargets:
        """Read the away and home scores and derive the margin and total."""
        away = pd.to_numeric(df[target_away], errors="coerce").to_numpy(dtype=float)
        home = pd.to_numeric(df[target_home], errors="coerce").to_numpy(dtype=float)
        return cls(away=away, home=home, margin=home - away, total=home + away)


def _at_least(value: float | None, threshold: float) -> bool:
    """Return whether a measured value exists and reaches the threshold."""
    return value is not None and value >= threshold


def _audit_column(
    column: pd.Series, name: str, targets: _ScoreTargets, config: LeakageAuditConfig
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check one feature against the targets: a warning flag, if any, and its hard failures."""
    x = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    corr_margin = _safe_corr(x, targets.margin)
    corr_total = _safe_corr(x, targets.total)
    eq_home = _equality_rate(x, targets.home)
    eq_away = _equality_rate(x, targets.away)
    abs_margin = None if corr_margin is None else abs(corr_margin)
    abs_total = None if corr_total is None else abs(corr_total)

    reasons = [
        reason
        for reason, flagged in (
            ("high_corr_margin", _at_least(abs_margin, config.suspicious_corr_threshold)),
            ("high_corr_total", _at_least(abs_total, config.suspicious_corr_threshold)),
            ("matches_home_score", _at_least(eq_home, config.equality_rate_threshold)),
            ("matches_away_score", _at_least(eq_away, config.equality_rate_threshold)),
        )
        if flagged
    ]
    flag = (
        {
            "column": name,
            "reason": ",".join(reasons),
            "corr_margin": corr_margin,
            "corr_total": corr_total,
            "matches_home_score_rate": eq_home,
            "matches_away_score_rate": eq_away,
        }
        if reasons
        else None
    )

    # Hard fail only for near-perfect correlation or near-perfect equality.
    failures: list[str] = []
    if _at_least(abs_margin, config.fail_corr_threshold) or _at_least(
        abs_total, config.fail_corr_threshold
    ):
        failures.append(f"Near-perfect correlation detected for feature '{name}'.")
    if _at_least(eq_home, NEAR_IDENTICAL_SHARE) or _at_least(eq_away, NEAR_IDENTICAL_SHARE):
        failures.append(f"Feature '{name}' nearly equals a target score column.")
    return flag, failures


def run_leakage_audit(df: pd.DataFrame, config: LeakageAuditConfig) -> dict[str, Any]:
    """Run a leakage audit on a dataset and return a JSON-serializable report."""
    target_away, target_home = ml_model.get_target_columns(df)
    target_cols = (target_away, target_home)

    spec = ml_model.build_feature_spec(
        df,
        ml_model.FeatureSelection(
            include_market=config.include_market,
            max_cardinality_ratio=config.max_cardinality_ratio,
            feature_start=config.feature_start,
            feature_end=config.feature_end,
            market_transform=config.market_transform,
        ),
    )

    feature_cols = list(spec.feature_columns)
    failures: list[str] = []
    warnings: list[str] = []
    flagged_columns: list[dict[str, Any]] = []

    leaked_targets = [col for col in target_cols if col in feature_cols]
    if leaked_targets:
        failures.append(f"Target columns present in features: {leaked_targets}")

    # Heuristic name-based flags (warn only).
    suspicious_name_tokens = ("winner", "result", "final", "post", "boxscore")
    for col in feature_cols:
        lower = col.lower()
        if any(tok in lower for tok in suspicious_name_tokens):
            flagged_columns.append({"column": col, "reason": "suspicious_name"})

    # Correlation / equality checks vs label-space targets.
    targets = _ScoreTargets.from_frame(df, target_away, target_home)
    for col in feature_cols:
        if col in target_cols:
            continue
        flag, column_failures = _audit_column(df[col], col, targets, config)
        if flag is not None:
            flagged_columns.append(flag)
        failures.extend(column_failures)

    ok = not failures
    report: dict[str, Any] = {
        "ok": ok,
        "failures": failures,
        "warnings": warnings,
        "row_count": len(df),
        "feature_count": len(feature_cols),
        "target_columns": list(target_cols),
        "config": {
            "feature_start": config.feature_start,
            "feature_end": config.feature_end,
            "include_market": config.include_market,
            "market_transform": config.market_transform,
            "max_cardinality_ratio": config.max_cardinality_ratio,
            "suspicious_corr_threshold": config.suspicious_corr_threshold,
            "fail_corr_threshold": config.fail_corr_threshold,
            "equality_rate_threshold": config.equality_rate_threshold,
        },
        "flagged_columns": flagged_columns,
    }

    if ok:
        log.info("Leakage audit OK (%d features, %d rows).", len(feature_cols), len(df))
    else:
        log.info(
            "Leakage audit FAILED: %d failures, %d flagged columns.",
            len(failures),
            len(flagged_columns),
        )

    return report


def write_report(report: dict[str, Any], out_path: Path | str) -> None:
    """Write a leakage audit report to disk."""
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    log.info("Wrote leakage audit report to %s", out_path)
