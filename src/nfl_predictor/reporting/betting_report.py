"""Betting-oriented report built from a week's predictions.

The report compares the model's win probability with the market's no-vig moneyline
probability for each game, turns the difference into a fair moneyline, a 1-10 confidence
score and an action label, and adds the spread and total differences in points without any
cover probability. Totals carry ``TOTAL_SIGNAL_STATUS`` because the total head trails the
market's line in walk-forward. The report is decision support only and makes no profitability
claim; the weekly run writes it next to its predictions.

Given the floor's sigma, the home edge is split in two: ``market_home_prob_spread`` is the
floor's curve at the market's margin, ``Phi(-home_spread / sigma)``;
``edge_beyond_spread_home_prob`` (the model's probability minus that) is what the model adds
beyond the spread; and ``spread_moneyline_gap_home_prob`` (that minus the no-vig moneyline
probability) is the gap between the spread and the moneyline, with no information from the
model's margin. The two parts sum to ``edge_home_prob``. The confidence score and the action label
still read the whole edge.
"""

from __future__ import annotations

import math

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.ml.floor_sigma import home_win_prob as floor_home_win_prob

# Upper edge bounds (exclusive) of the 1..9 confidence scores; a larger edge scores 10.
CONFIDENCE_EDGE_BOUNDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)
# Upper edge bounds (exclusive) of each action label; a larger edge is STRONG.
ACTION_EDGE_BOUNDS = ((0.02, "PASS"), (0.04, "LEAN"), (0.07, "SMALL"), (0.10, "MEDIUM"))


def _moneyline_to_implied_prob(moneyline: float) -> float:
    """Convert American moneyline to implied probability.

    Returns NaN for non-finite inputs.
    """
    try:
        ml = float(moneyline)
    except TypeError, ValueError:
        return float("nan")
    if not pd.notna(ml):
        return float("nan")
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def _implied_prob_to_moneyline(prob: float) -> float:
    """Convert implied probability to an American moneyline.

    Returns NaN for invalid probabilities.
    """
    try:
        p = float(prob)
    except TypeError, ValueError:
        return float("nan")
    if not 0.0 < p < 1.0:
        return float("nan")
    if p >= constants.EVEN_ODDS_PROBABILITY:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def _novig_pair(p_home_raw: float, p_away_raw: float) -> tuple[float, float]:
    """Normalize two implied probabilities to remove vig (sum to 1)."""
    if not pd.notna(p_home_raw) or not pd.notna(p_away_raw):
        return float("nan"), float("nan")
    denom = float(p_home_raw) + float(p_away_raw)
    if denom <= 0:
        return float("nan"), float("nan")
    return float(p_home_raw) / denom, float(p_away_raw) / denom


def _novig_row(row: pd.Series) -> tuple[float, float]:
    """Return the no-vig home and away probabilities of one prediction row."""
    return _novig_pair(row["market_home_prob_raw"], row["market_away_prob_raw"])


def _moneyline_value_side(row: pd.Series) -> str:
    """Return the team whose moneyline the model favors over the market."""
    return row["home_abbr"] if row["edge_home_prob"] >= 0 else row["away_abbr"]


def _spread_value_side(row: pd.Series) -> str:
    """Return the team the model's margin favors against the spread."""
    return row["home_abbr"] if row["spread_edge_points_home"] >= 0 else row["away_abbr"]


def _total_value_side(row: pd.Series) -> str:
    """Return ``OVER`` when the predicted total reaches the line, else ``UNDER``."""
    return "OVER" if row["predicted_total"] >= row["total_line"] else "UNDER"


def _edge_to_confidence_1_to_10(edge: float) -> int:
    """Map absolute probability edge to a 1..10 confidence score.

    This is a heuristic scale for readability, not bankroll management.
    """
    e = float(abs(edge))
    for score, upper_bound in enumerate(CONFIDENCE_EDGE_BOUNDS, start=1):
        if e < upper_bound:
            return score
    return len(CONFIDENCE_EDGE_BOUNDS) + 1


def _edge_to_action(edge: float) -> str:
    """Map absolute probability edge to a simple action label."""
    e = float(abs(edge))
    for upper_bound, label in ACTION_EDGE_BOUNDS:
        if e < upper_bound:
            return label
    return "STRONG"


TOTAL_SIGNAL_STATUS = "diagnostic_only"
"""Label written next to the total (over/under) columns of every betting report row.

In the 2023-2025 walk-forward the model's total trails the market's own total line (weeks 3-18
total MAE ``10.3152`` unanchored and ``10.2295`` anchored against ``10.0847`` for the line), and
its deviation from the line has no correlation with the actual deviation. Over/under leans are
therefore diagnostics, not betting signals, until a total model beats the line in walk-forward.
"""


def _spread_home_prob(df: pd.DataFrame, floor_sigma: float | None) -> pd.Series:
    """Return the floor's home win probability at the market's margin.

    That is ``Phi(-home_spread / floor_sigma)``. Every game is NaN when ``floor_sigma`` is None
    or the predictions carry no ``home_spread``, and a game without a spread is NaN.
    """
    if floor_sigma is None or "home_spread" not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    spread = pd.to_numeric(df["home_spread"], errors="coerce").to_numpy(dtype=float)
    return pd.Series(floor_home_win_prob(-spread, floor_sigma), index=df.index, dtype=float)


def build_betting_report(
    predictions: pd.DataFrame, *, floor_sigma: float | None = None
) -> pd.DataFrame:
    """Build a betting-oriented report from a predictions table.

    Expects the output schema of ``ml_model_predict.predict_week_margin_total``.

    The report focuses on moneyline value signals (probability calibration) and also
    includes simple spread/total deltas (without claiming cover probabilities).

    Args:
        predictions: The week's predictions with the market's moneylines (and spreads).
        floor_sigma: The sigma of the floor that gave ``home_win_prob``. With it, the moneyline
            edge is split into the part beyond the spread and the spread-moneyline gap; without
            it those columns are NaN.

    Returns:
        The report, sorted by the moneyline edge.

    Raises:
        ValueError: If ``predictions`` lacks a required column, or ``floor_sigma`` is given and
            is not finite and positive.

    """
    if floor_sigma is not None and not (math.isfinite(floor_sigma) and floor_sigma > 0):
        msg = f"The floor's sigma must be positive and finite, got {floor_sigma!r}."
        raise ValueError(msg)
    required = {
        "away_abbr",
        "home_abbr",
        "home_win_prob",
        "predicted_margin",
        "predicted_total",
        "home_moneyline",
        "away_moneyline",
    }
    missing = sorted([c for c in required if c not in predictions.columns])
    if missing:
        msg = f"Predictions missing required columns: {missing}"
        raise ValueError(msg)

    df = predictions.copy()

    df["market_home_prob_raw"] = df["home_moneyline"].map(_moneyline_to_implied_prob)
    df["market_away_prob_raw"] = df["away_moneyline"].map(_moneyline_to_implied_prob)

    novig = df.apply(_novig_row, axis=1, result_type="expand")
    df["market_home_prob_novig"] = novig[0]
    df["market_away_prob_novig"] = novig[1]

    df["model_home_prob"] = df["home_win_prob"].astype(float)
    df["model_away_prob"] = 1.0 - df["model_home_prob"]

    df["edge_home_prob"] = df["model_home_prob"] - df["market_home_prob_novig"]
    df["edge_away_prob"] = df["model_away_prob"] - df["market_away_prob_novig"]

    # The home edge is the part beyond the spread plus the spread-moneyline gap.
    df["market_home_prob_spread"] = _spread_home_prob(df, floor_sigma)
    df["edge_beyond_spread_home_prob"] = df["model_home_prob"] - df["market_home_prob_spread"]
    df["spread_moneyline_gap_home_prob"] = (
        df["market_home_prob_spread"] - df["market_home_prob_novig"]
    )

    df["model_fair_home_moneyline"] = df["model_home_prob"].map(_implied_prob_to_moneyline)
    df["model_fair_away_moneyline"] = df["model_away_prob"].map(_implied_prob_to_moneyline)

    df["moneyline_value_side"] = df.apply(_moneyline_value_side, axis=1)
    df["moneyline_edge_prob"] = df[["edge_home_prob", "edge_away_prob"]].abs().max(axis=1)
    df["moneyline_confidence_1_10"] = df["moneyline_edge_prob"].map(_edge_to_confidence_1_to_10)
    df["moneyline_action"] = df["moneyline_edge_prob"].map(_edge_to_action)

    # Spread / total deltas (no probability claims)
    if "home_spread" in df.columns:
        df["spread_edge_points_home"] = df["predicted_margin"].astype(float) + df[
            "home_spread"
        ].astype(float)
        df["spread_value_side"] = df.apply(_spread_value_side, axis=1)
        df["spread_edge_points"] = df["spread_edge_points_home"].abs()
    else:
        df["spread_value_side"] = None
        df["spread_edge_points"] = float("nan")

    if "total_line" in df.columns:
        df["total_edge_points"] = (
            df["predicted_total"].astype(float) - df["total_line"].astype(float)
        ).abs()
        df["total_value_side"] = df.apply(_total_value_side, axis=1)
    else:
        df["total_value_side"] = None
        df["total_edge_points"] = float("nan")
    df["total_signal"] = TOTAL_SIGNAL_STATUS

    # Friendly display columns
    df["matchup"] = df["away_abbr"].astype(str) + " @ " + df["home_abbr"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    cols: list[str] = [
        "game_id",
        "date",
        "matchup",
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "predicted_margin",
        "model_home_prob",
        "market_home_prob_novig",
        "edge_home_prob",
        "market_home_prob_spread",
        "edge_beyond_spread_home_prob",
        "spread_moneyline_gap_home_prob",
        "away_moneyline",
        "home_moneyline",
        "model_fair_away_moneyline",
        "model_fair_home_moneyline",
        "moneyline_value_side",
        "moneyline_edge_prob",
        "moneyline_confidence_1_10",
        "moneyline_action",
        "away_spread",
        "home_spread",
        "spread_value_side",
        "spread_edge_points",
        "total_line",
        "total_value_side",
        "total_edge_points",
        "total_signal",
    ]
    cols_present = [c for c in cols if c in df.columns]
    report = df[cols_present].copy()
    return report.sort_values(
        ["moneyline_edge_prob", "total_edge_points"],
        ascending=[False, False],
    )
