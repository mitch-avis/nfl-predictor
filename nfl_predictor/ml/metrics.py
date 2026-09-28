"""Metrics helpers for walk-forward evaluation.

This module centralizes evaluation metrics used by the walk-forward backtest:
- margin/total regression metrics
- win probability metrics (Brier + log loss)
- confidence pool summaries
- calibration reliability table
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

PROB_EPSILON = 1e-15
# Confidence |p - 0.5| is rounded to this many decimals before games are ranked. Probabilities
# that are mathematically equally confident (a home favorite and a home underdog by the same
# spread) differ by about 1e-16 in floating point, which would otherwise decide their order;
# 1e-12 is far above that noise and far below any meaningful difference in probability.
CONFIDENCE_DECIMALS = 12
METRIC_STRATEGY: dict[str, list[dict[str, str]]] = {
    "primary": [
        {"metric": "brier", "direction": "lower"},
        {"metric": "log_loss", "direction": "lower"},
        {"metric": "deterministic_brier", "direction": "lower"},
        {"metric": "deterministic_log_loss", "direction": "lower"},
        {"metric": "market_brier", "direction": "lower"},
        {"metric": "market_log_loss", "direction": "lower"},
        {"metric": "deterministic_brier_vs_market", "direction": "lower"},
        {"metric": "deterministic_log_loss_vs_market", "direction": "lower"},
        {"metric": "reliability_ece", "direction": "lower"},
    ],
    "secondary": [
        {"metric": "expected_points_avg", "direction": "higher"},
        {"metric": "actual_points_avg", "direction": "higher"},
        {"metric": "pick_accuracy", "direction": "higher"},
        {"metric": "deterministic_pick_accuracy", "direction": "higher"},
        {"metric": "market_pick_accuracy", "direction": "higher"},
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


def probability_losses_per_row(
    actual_home_win: np.ndarray, home_win_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's squared error and log loss, the terms `probability_metrics` averages.

    Squared error is ``(y - p)**2`` with ``p`` clipped to ``[0, 1]``; log loss is
    ``-(y * log(p) + (1 - y) * log(1 - p))`` with ``p`` also clipped to
    ``[PROB_EPSILON, 1 - PROB_EPSILON]``, which is what scikit-learn computes for these inputs.
    """
    outcome = np.asarray(actual_home_win, dtype=float)
    probs = clip_probabilities(np.asarray(home_win_prob, dtype=float))
    probs_eps = clip_probabilities(probs, eps=PROB_EPSILON)
    squared_error = (outcome - probs) ** 2
    log_loss_terms = -(outcome * np.log(probs_eps) + (1.0 - outcome) * np.log(1.0 - probs_eps))
    return squared_error, log_loss_terms


def probability_pick_accuracy(actual_margin: np.ndarray, home_win_prob: np.ndarray) -> float:
    """Compute pick accuracy, treating tied games as incorrect for either side."""
    margins = np.asarray(actual_margin, dtype=float)
    probs = clip_probabilities(home_win_prob)
    if len(margins) == 0:
        return 0.0
    predicted_home = picks_home(probs)
    actual_home = margins > 0
    correct = (predicted_home == actual_home) & (margins != 0)
    return float(np.mean(correct))


def probability_summary(
    actual_home_win: np.ndarray,
    actual_margin: np.ndarray,
    home_win_prob: np.ndarray,
    *,
    prefix: str | None = None,
) -> dict[str, float]:
    """Return Brier, log loss, and pick accuracy for one probability path."""
    metrics = probability_metrics(actual_home_win, home_win_prob)
    metrics["pick_accuracy"] = probability_pick_accuracy(actual_margin, home_win_prob)
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def picks_home(home_win_prob: np.ndarray) -> np.ndarray:
    """Return whether each game picks the home side: ``p >= 0.5``, so an exact 0.5 picks home.

    Every pick uses the unrounded probability; published 4-decimal values never decide a side.
    """
    return np.asarray(home_win_prob, dtype=float) >= 0.5


def confidence_strength(home_win_prob: np.ndarray) -> np.ndarray:
    """Return each game's confidence ``|p - 0.5|`` rounded to ``CONFIDENCE_DECIMALS``."""
    return np.round(np.abs(np.asarray(home_win_prob, dtype=float) - 0.5), CONFIDENCE_DECIMALS)


def confidence_ranks(
    home_win_prob: np.ndarray,
    tiebreaker: np.ndarray | None = None,
    groups: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Return confidence-pool ranks ``1..N``, least confident first.

    Games are ordered by ``confidence_strength`` ascending; equal confidences are ordered by
    ``tiebreaker`` (the ``game_id``) and then by input order. With ``groups`` (for example the
    season and week columns), ranks restart at 1 within each distinct combination of them.
    """
    strength = confidence_strength(home_win_prob)
    n = len(strength)
    position = np.arange(n)
    ties = (position,) if tiebreaker is None else (position, np.asarray(tiebreaker))
    group_keys = tuple(np.asarray(group) for group in groups)
    order = np.lexsort((*ties, strength, *reversed(group_keys)))
    starts = np.zeros(n, dtype=int)
    if group_keys and n:
        new_group = np.zeros(n, dtype=bool)
        new_group[0] = True
        for key in group_keys:
            ordered = key[order]
            new_group[1:] |= ordered[1:] != ordered[:-1]
        starts = np.maximum.accumulate(np.where(new_group, position, 0))
    ranks = np.empty(n, dtype=int)
    ranks[order] = position - starts + 1
    return ranks


def confidence_pool_columns(
    home_win_prob: np.ndarray,
    home_score: np.ndarray,
    away_score: np.ndarray,
    tiebreaker: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-game confidence pool columns.

    Ranks come from ``confidence_ranks``: unique ranks 1..N by rounded confidence
    ``|p - 0.5|``, with equal confidences ordered by ``tiebreaker`` (the ``game_id``).
    """
    ranks = confidence_ranks(home_win_prob, tiebreaker)

    predicted_home = picks_home(home_win_prob)
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
