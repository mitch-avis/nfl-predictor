"""Unit tests for the betting report and its moneyline helpers.

These tests validate deterministic conversions and report construction.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nfl_predictor.reporting import betting_report


def test_moneyline_to_implied_prob_negative() -> None:
    """-110 should be about 0.5238."""
    p = betting_report._moneyline_to_implied_prob(-110)
    assert abs(p - (110 / 210)) < 1e-6


def test_moneyline_to_implied_prob_positive() -> None:
    """+150 should be 0.4."""
    p = betting_report._moneyline_to_implied_prob(150)
    assert abs(p - 0.4) < 1e-6


def test_implied_prob_to_moneyline_round_trip_close() -> None:
    """Converting p->ml->p should be approximately consistent."""
    p = 0.62
    ml = betting_report._implied_prob_to_moneyline(p)
    p2 = betting_report._moneyline_to_implied_prob(ml)
    assert abs(p2 - p) < 1e-6


def test_novig_pair_sums_to_one() -> None:
    """No-vig normalization should sum to 1."""
    ph, pa = betting_report._novig_pair(0.55, 0.52)
    assert abs((ph + pa) - 1.0) < 1e-12


def test_build_betting_report_basic_columns() -> None:
    """The report should include key moneyline fields and sort by edge."""
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": ["2026-01-12", "2026-01-13"],
            "away_abbr": ["A", "C"],
            "home_abbr": ["B", "D"],
            "home_win_prob": [0.60, 0.52],
            "predicted_margin": [3.0, 0.5],
            "predicted_total": [44.0, 41.0],
            "predicted_away_score": [20.5, 20.0],
            "predicted_home_score": [23.5, 21.0],
            "home_moneyline": [-120, -110],
            "away_moneyline": [110, 100],
            "home_spread": [-2.5, -1.0],
            "away_spread": [2.5, 1.0],
            "total_line": [43.5, 40.5],
        }
    )

    report = betting_report.build_betting_report(df)

    assert "moneyline_edge_prob" in report.columns
    assert "moneyline_confidence_1_10" in report.columns
    assert "model_fair_home_moneyline" in report.columns

    # Should sort descending by moneyline_edge_prob
    assert report.iloc[0]["moneyline_edge_prob"] >= report.iloc[1]["moneyline_edge_prob"]

    # Fair moneylines should be finite
    assert math.isfinite(float(report.iloc[0]["model_fair_home_moneyline"]))


def test_build_betting_report_labels_totals_diagnostic_only() -> None:
    """Every row says the total columns are diagnostic, next to the total columns themselves."""
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "away_abbr": ["A", "C"],
            "home_abbr": ["B", "D"],
            "home_win_prob": [0.60, 0.52],
            "predicted_margin": [3.0, 0.5],
            "predicted_total": [44.0, 41.0],
            "home_moneyline": [-120, -110],
            "away_moneyline": [110, 100],
            "total_line": [43.5, 40.5],
        }
    )

    report = betting_report.build_betting_report(df)

    assert report["total_signal"].tolist() == ["diagnostic_only", "diagnostic_only"]
    columns = report.columns.tolist()
    assert columns.index("total_signal") == columns.index("total_edge_points") + 1


@pytest.mark.parametrize(
    ("edge", "score"),
    [
        (0.0, 1),
        (0.0099, 1),
        (0.01, 2),
        (-0.035, 4),
        (0.0799, 8),
        (0.08, 9),
        (0.0999, 9),
        (0.10, 10),
    ],
)
def test_edge_confidence_scores_step_at_each_bound(edge: float, score: int) -> None:
    """Each bound starts the next score; the edge's sign does not matter."""
    assert betting_report._edge_to_confidence_1_to_10(edge) == score


@pytest.mark.parametrize(
    ("edge", "action"),
    [(0.0199, "PASS"), (0.02, "LEAN"), (-0.05, "SMALL"), (0.07, "MEDIUM"), (0.10, "STRONG")],
)
def test_edge_actions_step_at_each_bound(edge: float, action: str) -> None:
    """Each bound starts the next action label; the edge's sign does not matter."""
    assert betting_report._edge_to_action(edge) == action


def test_a_missing_edge_lands_on_the_top_rung() -> None:
    """A NaN edge fails every bound, so it scores 10 and reads STRONG (unchanged behavior)."""
    assert betting_report._edge_to_confidence_1_to_10(float("nan")) == 10
    assert betting_report._edge_to_action(float("nan")) == "STRONG"
