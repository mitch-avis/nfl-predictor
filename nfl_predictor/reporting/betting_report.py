"""Betting-oriented report built from a week's predictions.

The report compares the model's win probability with the market's no-vig moneyline
probability for each game, turns the difference into a fair moneyline, a 1-10 confidence
score and an action label, and adds the spread and total differences in points without any
cover probability. Totals carry ``TOTAL_SIGNAL_STATUS`` because the total head trails the
market's line in walk-forward. The report is decision support only and makes no profitability
claim; the weekly run writes it next to its predictions.
"""

from __future__ import annotations

import pandas as pd


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
    if p >= 0.5:
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


def _edge_to_confidence_1_to_10(edge: float) -> int:
    """Map absolute probability edge to a 1..10 confidence score.

    This is a heuristic scale for readability, not bankroll management.
    """
    e = float(abs(edge))
    if e < 0.01:
        return 1
    if e < 0.02:
        return 2
    if e < 0.03:
        return 3
    if e < 0.04:
        return 4
    if e < 0.05:
        return 5
    if e < 0.06:
        return 6
    if e < 0.07:
        return 7
    if e < 0.08:
        return 8
    if e < 0.10:
        return 9
    return 10


def _edge_to_action(edge: float) -> str:
    """Map absolute probability edge to a simple action label."""
    e = float(abs(edge))
    if e < 0.02:
        return "PASS"
    if e < 0.04:
        return "LEAN"
    if e < 0.07:
        return "SMALL"
    if e < 0.10:
        return "MEDIUM"
    return "STRONG"


TOTAL_SIGNAL_STATUS = "diagnostic_only"
"""Label written next to the total (over/under) columns of every betting report row.

In the 2023-2025 walk-forward the model's total trails the market's own total line (weeks 3-18
total MAE ``10.3152`` unanchored and ``10.2295`` anchored against ``10.0847`` for the line), and
its deviation from the line has no correlation with the actual deviation. Over/under leans are
therefore diagnostics, not betting signals, until a total model beats the line in walk-forward.
"""


def build_betting_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a betting-oriented report from a predictions table.

    Expects the output schema from ml_model_predict.predict_week_blended (or margin_total).

    The report focuses on moneyline value signals (probability calibration) and also
    includes simple spread/total deltas (without claiming cover probabilities).
    """
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
        raise ValueError(f"Predictions missing required columns: {missing}")

    df = predictions.copy()

    df["market_home_prob_raw"] = df["home_moneyline"].map(_moneyline_to_implied_prob)
    df["market_away_prob_raw"] = df["away_moneyline"].map(_moneyline_to_implied_prob)

    novig = df.apply(
        lambda row: _novig_pair(row["market_home_prob_raw"], row["market_away_prob_raw"]),
        axis=1,
        result_type="expand",
    )
    df["market_home_prob_novig"] = novig[0]
    df["market_away_prob_novig"] = novig[1]

    df["model_home_prob"] = df["home_win_prob"].astype(float)
    df["model_away_prob"] = 1.0 - df["model_home_prob"]

    df["edge_home_prob"] = df["model_home_prob"] - df["market_home_prob_novig"]
    df["edge_away_prob"] = df["model_away_prob"] - df["market_away_prob_novig"]

    df["model_fair_home_moneyline"] = df["model_home_prob"].map(_implied_prob_to_moneyline)
    df["model_fair_away_moneyline"] = df["model_away_prob"].map(_implied_prob_to_moneyline)

    df["moneyline_value_side"] = df.apply(
        lambda row: row["home_abbr"] if row["edge_home_prob"] >= 0 else row["away_abbr"],
        axis=1,
    )
    df["moneyline_edge_prob"] = df[["edge_home_prob", "edge_away_prob"]].abs().max(axis=1)
    df["moneyline_confidence_1_10"] = df["moneyline_edge_prob"].map(_edge_to_confidence_1_to_10)
    df["moneyline_action"] = df["moneyline_edge_prob"].map(_edge_to_action)

    # Spread / total deltas (no probability claims)
    if "home_spread" in df.columns:
        df["spread_edge_points_home"] = df["predicted_margin"].astype(float) + df[
            "home_spread"
        ].astype(float)
        df["spread_value_side"] = df.apply(
            lambda row: (
                row["home_abbr"] if row["spread_edge_points_home"] >= 0 else row["away_abbr"]
            ),
            axis=1,
        )
        df["spread_edge_points"] = df["spread_edge_points_home"].abs()
    else:
        df["spread_value_side"] = None
        df["spread_edge_points"] = float("nan")

    if "total_line" in df.columns:
        df["total_edge_points"] = (
            df["predicted_total"].astype(float) - df["total_line"].astype(float)
        ).abs()
        df["total_value_side"] = df.apply(
            lambda row: "OVER" if row["predicted_total"] >= row["total_line"] else "UNDER",
            axis=1,
        )
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
    report = report.sort_values(
        ["moneyline_edge_prob", "total_edge_points"],
        ascending=[False, False],
    )
    return report
