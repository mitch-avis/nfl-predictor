"""Metrics helpers for walk-forward evaluation.

This module centralizes evaluation metrics used by the walk-forward backtest:
- margin/total regression metrics
- win probability metrics (Brier + log loss)
- confidence pool summaries
- calibration reliability table
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

PROB_EPSILON = 1e-15
METRIC_STRATEGY: dict[str, list[dict[str, str]]] = {
    "primary": [
        {"metric": "brier", "direction": "lower"},
        {"metric": "log_loss", "direction": "lower"},
        {"metric": "reliability_ece", "direction": "lower"},
    ],
    "secondary": [
        {"metric": "expected_points_avg", "direction": "higher"},
        {"metric": "actual_points_avg", "direction": "higher"},
        {"metric": "pick_accuracy", "direction": "higher"},
    ],
    "tertiary": [
        {"metric": "margin_mae", "direction": "lower"},
        {"metric": "total_mae", "direction": "lower"},
        {"metric": "market_margin_resid_mae", "direction": "lower"},
        {"metric": "market_total_resid_mae", "direction": "lower"},
    ],
}


def clip_probabilities(probs: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Clip probabilities to [0, 1] or [eps, 1-eps] for stability."""
    if eps is None:
        return np.clip(probs, 0.0, 1.0)
    return np.clip(probs, eps, 1.0 - eps)


def margin_total_metrics(
    actual_margin: np.ndarray,
    actual_total: np.ndarray,
    pred_margin: np.ndarray,
    pred_total: np.ndarray,
) -> dict[str, float]:
    """Compute MAE for margin and total."""
    return {
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
    }


def probability_metrics(actual_home_win: np.ndarray, home_win_prob: np.ndarray) -> dict[str, float]:
    """Compute Brier score and log loss for home win probabilities."""
    probs = clip_probabilities(home_win_prob)
    probs_eps = clip_probabilities(probs, eps=PROB_EPSILON)
    return {
        "brier": float(brier_score_loss(actual_home_win, probs)),
        "log_loss": float(log_loss(actual_home_win, probs_eps, labels=[0, 1])),
    }


def confidence_pool_columns(
    home_win_prob: np.ndarray,
    home_score: np.ndarray,
    away_score: np.ndarray,
    tiebreaker: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-game confidence pool columns.

    Uses confidence strength = abs(p - 0.5) to assign unique ranks 1..N.

    When `tiebreaker` is provided, it is used to deterministically break ties in
    confidence strength (important because some workflows round probabilities for output).
    """
    strength = np.abs(home_win_prob - 0.5)
    if tiebreaker is None:
        order = np.argsort(strength, kind="mergesort")
    else:
        order = np.lexsort((tiebreaker, strength))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(strength) + 1)

    predicted_home = home_win_prob >= 0.5
    actual_outcome = np.where(home_score > away_score, 1, np.where(home_score < away_score, -1, 0))
    predicted_outcome = np.where(predicted_home, 1, -1)
    pick_correct = (predicted_outcome == actual_outcome) & (actual_outcome != 0)
    pick_win_prob = np.where(predicted_home, home_win_prob, 1 - home_win_prob)

    expected_points = ranks * pick_win_prob
    actual_points = ranks * pick_correct.astype(int)

    return {
        "confidence_rank": ranks,
        "pick_correct": pick_correct,
        "expected_points": expected_points,
        "actual_points": actual_points,
    }


def confidence_pool_summary(confidence_cols: dict[str, np.ndarray]) -> dict[str, Any]:
    """Aggregate confidence-pool points across a week (or any set of games)."""
    return {
        "expected_points": float(confidence_cols["expected_points"].sum()),
        "actual_points": float(confidence_cols["actual_points"].sum()),
        "picks_correct": int(confidence_cols["pick_correct"].sum()),
        "games": int(len(confidence_cols["confidence_rank"])),
    }


def reliability_table(
    home_win_prob: np.ndarray, actual_home_win: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    """Return a binned calibration reliability table."""
    probs = clip_probabilities(home_win_prob)
    actual = actual_home_win.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=True)

    rows: list[dict[str, Any]] = []
    for idx in range(bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        avg_pred = float(probs[mask].mean()) if count else None
        avg_actual = float(actual[mask].mean()) if count else None
        rows.append(
            {
                "bin_lower": float(edges[idx]),
                "bin_upper": float(edges[idx + 1]),
                "count": count,
                "avg_pred": avg_pred,
                "avg_actual": avg_actual,
            }
        )
    return rows


def reliability_ece(bins: list[dict[str, Any]]) -> float:
    """Compute expected calibration error from a reliability table."""
    total = sum(int(row.get("count", 0) or 0) for row in bins)
    if total <= 0:
        return float("nan")
    ece = 0.0
    for row in bins:
        count = int(row.get("count", 0) or 0)
        if count <= 0:
            continue
        avg_pred = row.get("avg_pred")
        avg_actual = row.get("avg_actual")
        if avg_pred is None or avg_actual is None:
            continue
        ece += (count / total) * abs(float(avg_pred) - float(avg_actual))
    return float(ece)
