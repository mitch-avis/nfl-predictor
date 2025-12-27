from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

PROB_EPSILON = 1e-15


def clip_probabilities(probs: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
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
    return {
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
    }


def probability_metrics(actual_home_win: np.ndarray, home_win_prob: np.ndarray) -> dict[str, float]:
    probs = clip_probabilities(home_win_prob)
    probs_eps = clip_probabilities(probs, eps=PROB_EPSILON)
    return {
        "brier": float(brier_score_loss(actual_home_win, probs)),
        "log_loss": float(log_loss(actual_home_win, probs_eps, labels=[0, 1])),
    }


def confidence_pool_columns(
    home_win_prob: np.ndarray, home_score: np.ndarray, away_score: np.ndarray
) -> dict[str, np.ndarray]:
    strength = np.abs(home_win_prob - 0.5)
    order = np.argsort(strength, kind="mergesort")
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
    return {
        "expected_points": float(confidence_cols["expected_points"].sum()),
        "actual_points": float(confidence_cols["actual_points"].sum()),
        "picks_correct": int(confidence_cols["pick_correct"].sum()),
        "games": int(len(confidence_cols["confidence_rank"])),
    }


def reliability_table(
    home_win_prob: np.ndarray, actual_home_win: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
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
