"""Tests for the market math helpers."""

from __future__ import annotations

import pytest

from nfl_predictor.api.readers import market


def test_moneyline_conversions() -> None:
    """American odds round-trip through implied probabilities."""
    assert market.moneyline_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert market.moneyline_to_prob(150) == pytest.approx(0.4)
    assert market.moneyline_to_prob(0) is None
    assert market.moneyline_to_prob(None) is None
    assert market.moneyline_to_prob(float("nan")) is None
    assert market.moneyline_to_prob(True) is None
    assert market.prob_to_moneyline(0.6) == -150
    assert market.prob_to_moneyline(0.4) == 150
    assert market.prob_to_moneyline(1.5) is None
    assert market.prob_to_moneyline("x") is None


def test_novig_and_ev() -> None:
    """Vig removal normalizes to one and EV follows p * profit - (1 - p)."""
    home, away = market.novig_pair(0.55, 0.50)
    assert home == pytest.approx(0.55 / 1.05)
    assert away is not None and home is not None and home + away == pytest.approx(1.0)
    assert market.novig_pair(None, 0.5) == (None, None)
    assert market.novig_pair(0.0, 0.0) == (None, None)
    assert market.expected_value(0.5, 100) == pytest.approx(0.0)
    assert market.expected_value(0.6, -150) == pytest.approx(0.6 * (100 / 150) - 0.4)


def test_action_ladder_and_confidence() -> None:
    """Edges map to the workbook's PASS/LEAN/SMALL/MEDIUM/STRONG rungs and 1..10 ladder."""
    assert market.action_label(None) == "PASS"
    assert market.action_label(0.019) == "PASS"
    assert market.action_label(0.02) == "LEAN"
    assert market.action_label(0.05) == "SMALL"
    assert market.action_label(0.08) == "MEDIUM"
    assert market.action_label(0.12) == "STRONG"
    assert market.confidence_1_to_10(None) == 1
    assert market.confidence_1_to_10(0.0) == 1
    assert market.confidence_1_to_10(0.015) == 2
    assert market.confidence_1_to_10(0.095) == 10
    assert market.confidence_1_to_10(0.5) == 10


def test_sigma_and_normal_tail() -> None:
    """Sigma comes from the p10/p90 spread with a fallback; tail probabilities behave."""
    assert market.sigma_from_quantiles(-17.0, 17.0, 13.0) == pytest.approx(34 / 2.563103, abs=1e-4)
    assert market.sigma_from_quantiles(None, 17.0, 13.0) == 13.0
    assert market.sigma_from_quantiles(5.0, 1.0, 13.0) == 13.0
    assert market.prob_exceeds(0.0, 0.0, 10.0) == pytest.approx(0.5)
    assert market.prob_exceeds(0.0, 3.0, 0.0) == 1.0
    assert market.prob_exceeds(5.0, 3.0, 0.0) == 0.0
    assert market.prob_exceeds(-3.0, 3.0, 13.0) > 0.5
