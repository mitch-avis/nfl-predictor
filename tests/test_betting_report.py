"""Unit tests for the betting report and its moneyline helpers.

These tests validate deterministic conversions and report construction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_predictor.reporting import betting_report

_SIGMA = 13.5
_SPLIT_COLUMNS = [
    "market_home_prob_spread",
    "edge_beyond_spread_home_prob",
    "spread_moneyline_gap_home_prob",
]


def _spread_predictions(home_spread: list[float]) -> pd.DataFrame:
    """Return two games whose home win probabilities are the floor of their margins."""
    margins = np.array([3.0, -2.0])
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "away_abbr": ["A", "C"],
            "home_abbr": ["B", "D"],
            "home_win_prob": norm.cdf(margins / _SIGMA),
            "predicted_margin": margins,
            "predicted_total": [44.0, 41.0],
            "home_moneyline": [-120.0, 135.0],
            "away_moneyline": [110.0, -150.0],
            "home_spread": home_spread,
            "away_spread": [-value for value in home_spread],
            "total_line": [43.5, 40.5],
        }
    )


def test_moneyline_to_implied_prob_negative() -> None:
    """-110 should be about 0.5238."""
    # Act
    p = betting_report._moneyline_to_implied_prob(-110)

    # Assert
    assert abs(p - (110 / 210)) < 1e-6


def test_moneyline_to_implied_prob_positive() -> None:
    """+150 should be 0.4."""
    # Act
    p = betting_report._moneyline_to_implied_prob(150)

    # Assert
    assert abs(p - 0.4) < 1e-6


def test_implied_prob_to_moneyline_round_trip_close() -> None:
    """Converting p->ml->p should be approximately consistent."""
    # Arrange
    p = 0.62

    # Act
    p2 = betting_report._moneyline_to_implied_prob(betting_report._implied_prob_to_moneyline(p))

    # Assert
    assert abs(p2 - p) < 1e-6


def test_novig_pair_sums_to_one() -> None:
    """No-vig normalization should sum to 1."""
    # Act
    ph, pa = betting_report._novig_pair(0.55, 0.52)

    # Assert
    assert abs((ph + pa) - 1.0) < 1e-12


def test_build_betting_report_basic_columns() -> None:
    """The report should include key moneyline fields and sort by edge."""
    # Arrange
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

    # Act
    report = betting_report.build_betting_report(df)

    # Assert
    assert "moneyline_edge_prob" in report.columns
    assert "moneyline_confidence_1_10" in report.columns
    assert "model_fair_home_moneyline" in report.columns
    # Should sort descending by moneyline_edge_prob
    assert report.iloc[0]["moneyline_edge_prob"] >= report.iloc[1]["moneyline_edge_prob"]
    # Fair moneylines should be finite
    assert math.isfinite(float(report.iloc[0]["model_fair_home_moneyline"]))


def test_build_betting_report_labels_totals_diagnostic_only() -> None:
    """Every row says the total columns are diagnostic, next to the total columns themselves."""
    # Arrange
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

    # Act
    report = betting_report.build_betting_report(df)

    # Assert
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
    # Act
    result = betting_report._edge_to_confidence_1_to_10(edge)

    # Assert
    assert result == score


@pytest.mark.parametrize(
    ("edge", "action"),
    [(0.0199, "PASS"), (0.02, "LEAN"), (-0.05, "SMALL"), (0.07, "MEDIUM"), (0.10, "STRONG")],
)
def test_edge_actions_step_at_each_bound(edge: float, action: str) -> None:
    """Each bound starts the next action label; the edge's sign does not matter."""
    # Act
    result = betting_report._edge_to_action(edge)

    # Assert
    assert result == action


def test_a_missing_edge_scores_the_top_confidence() -> None:
    """A NaN edge fails every bound, so it scores 10 (unchanged behavior)."""
    # Act
    score = betting_report._edge_to_confidence_1_to_10(float("nan"))

    # Assert
    assert score == 10


def test_a_missing_edge_reads_strong() -> None:
    """A NaN edge fails every bound, so it reads STRONG (unchanged behavior)."""
    # Act
    action = betting_report._edge_to_action(float("nan"))

    # Assert
    assert action == "STRONG"


def test_build_betting_report_edge_is_the_sum_of_beyond_and_gap() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0])

    # Act
    report = betting_report.build_betting_report(predictions, floor_sigma=_SIGMA)

    # Assert
    np.testing.assert_allclose(
        report["edge_home_prob"],
        report["edge_beyond_spread_home_prob"] + report["spread_moneyline_gap_home_prob"],
        rtol=0,
        atol=1e-15,
    )


def test_build_betting_report_spread_probability_is_the_floor_at_the_market_margin() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0])

    # Act
    report = betting_report.build_betting_report(predictions, floor_sigma=_SIGMA)

    # Assert
    by_game = report.set_index("game_id")["market_home_prob_spread"]
    expected = norm.cdf(np.array([2.5, -3.0]) / _SIGMA)
    np.testing.assert_allclose(by_game.loc[["g1", "g2"]], expected, rtol=0, atol=1e-15)


def test_build_betting_report_without_floor_sigma_leaves_the_split_missing() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0])

    # Act
    report = betting_report.build_betting_report(predictions)

    # Assert
    assert report[_SPLIT_COLUMNS].isna().all().all()


def test_build_betting_report_game_without_a_spread_leaves_its_split_missing() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, float("nan")])

    # Act
    report = betting_report.build_betting_report(predictions, floor_sigma=_SIGMA)

    # Assert
    missing = report.set_index("game_id")[_SPLIT_COLUMNS].isna()
    assert missing.loc[["g1", "g2"]].to_numpy().tolist() == [[False] * 3, [True] * 3]


def test_build_betting_report_without_a_spread_column_leaves_the_split_missing() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0]).drop(columns=["home_spread", "away_spread"])

    # Act
    report = betting_report.build_betting_report(predictions, floor_sigma=_SIGMA)

    # Assert
    assert report[_SPLIT_COLUMNS].isna().all().all()


def test_build_betting_report_floor_sigma_leaves_the_existing_columns_unchanged() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0])
    without_sigma = betting_report.build_betting_report(predictions)

    # Act
    with_sigma = betting_report.build_betting_report(predictions, floor_sigma=_SIGMA)

    # Assert
    existing = [column for column in without_sigma.columns if column not in _SPLIT_COLUMNS]
    pd.testing.assert_frame_equal(with_sigma[existing], without_sigma[existing])


def test_build_betting_report_non_positive_floor_sigma_raises() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0])

    # Act & Assert
    with pytest.raises(ValueError, match="sigma must be positive"):
        betting_report.build_betting_report(predictions, floor_sigma=0.0)


def test_build_betting_report_non_positive_floor_sigma_without_spreads_raises() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0]).drop(columns=["home_spread", "away_spread"])

    # Act & Assert
    with pytest.raises(ValueError, match="sigma must be positive"):
        betting_report.build_betting_report(predictions, floor_sigma=-1.0)


def test_build_betting_report_nan_floor_sigma_raises() -> None:
    # Arrange
    predictions = _spread_predictions([-2.5, 3.0]).drop(columns=["home_spread", "away_spread"])

    # Act & Assert
    with pytest.raises(ValueError, match="sigma must be positive"):
        betting_report.build_betting_report(predictions, floor_sigma=float("nan"))
