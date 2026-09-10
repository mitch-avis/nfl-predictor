"""Pure market-math helpers shared by the predictions and betting readers.

These mirror the formulas in ``scripts/betting_pipeline.py`` and the betting workbook:
American moneyline to implied probability, vig removal, and the action ladder.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TypeIs

NORMAL = NormalDist()
NORMAL_Z_P90 = 1.281551565545
P10_P90_TO_SIGMA_DENOM = 2.0 * NORMAL_Z_P90
DEFAULT_SIGMA_MARGIN = 13.0
DEFAULT_SIGMA_TOTAL = 16.0
DEFAULT_SPREAD_TOTAL_ODDS = -110.0
ACTION_LADDER: tuple[tuple[float, str], ...] = (
    (0.10, "STRONG"),
    (0.07, "MEDIUM"),
    (0.04, "SMALL"),
    (0.02, "LEAN"),
)


def is_number(value: object) -> TypeIs[int | float]:
    """Return whether ``value`` is a finite number (``bool`` excluded)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def moneyline_to_prob(moneyline: object) -> float | None:
    """Convert American odds to the implied probability (vig included)."""
    if not is_number(moneyline):
        return None
    ml = float(moneyline)
    if ml == 0:
        return None
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


def prob_to_moneyline(prob: object) -> float | None:
    """Convert a probability to the fair American odds."""
    if not is_number(prob):
        return None
    p = float(prob)
    if not 0.0 < p < 1.0:
        return None
    if p >= 0.5:
        return round(-100.0 * p / (1.0 - p))
    return round(100.0 * (1.0 - p) / p)


def novig_pair(
    p_home_raw: float | None, p_away_raw: float | None
) -> tuple[float | None, float | None]:
    """Normalize two implied probabilities so they sum to one."""
    if p_home_raw is None or p_away_raw is None:
        return None, None
    denom = p_home_raw + p_away_raw
    if denom <= 0:
        return None, None
    return p_home_raw / denom, p_away_raw / denom


def profit_per_unit(moneyline: float) -> float:
    """Return the profit on a one-unit stake at ``moneyline``."""
    return moneyline / 100.0 if moneyline > 0 else 100.0 / abs(moneyline)


def expected_value(prob: float, moneyline: float) -> float:
    """Return the expected profit per unit staked: ``p * profit - (1 - p)``."""
    return prob * profit_per_unit(moneyline) - (1.0 - prob)


def action_label(edge: float | None) -> str:
    """Map a probability edge to PASS / LEAN / SMALL / MEDIUM / STRONG."""
    if edge is None or edge < 0.02:
        return "PASS"
    for threshold, label in ACTION_LADDER:
        if edge >= threshold:
            return label
    return "PASS"


def confidence_1_to_10(edge: float | None) -> int:
    """Map a probability edge to a 1..10 ladder (one step per point of edge, capped)."""
    if edge is None or edge <= 0:
        return 1
    return max(1, min(10, int(math.floor(edge * 100.0)) + 1))


def sigma_from_quantiles(p10: object, p90: object, default: float) -> float:
    """Derive a normal standard deviation from the 10th and 90th percentiles."""
    if is_number(p10) and is_number(p90) and p90 > p10:
        return (float(p90) - float(p10)) / P10_P90_TO_SIGMA_DENOM
    return default


def prob_exceeds(threshold: float, mu: float, sigma: float) -> float:
    """Return ``P(X > threshold)`` for ``X ~ Normal(mu, sigma)``."""
    if sigma <= 0:
        return 1.0 if mu > threshold else 0.0
    return 1.0 - NORMAL.cdf((threshold - mu) / sigma)
