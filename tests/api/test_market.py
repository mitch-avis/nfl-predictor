"""Tests for the market math helpers."""

from __future__ import annotations

import pytest

from nfl_predictor.api.readers import market


@pytest.mark.parametrize(
    ("moneyline", "expected"),
    [
        (-110, pytest.approx(0.5238, abs=1e-4)),
        (150, pytest.approx(0.4)),
    ],
)
def test_moneyline_conversions(moneyline: float, expected: object) -> None:
    """American odds convert to implied probabilities."""
    # Act
    prob = market.moneyline_to_prob(moneyline)

    # Assert
    assert prob == expected


@pytest.mark.parametrize("moneyline", [0, None, float("nan")])
def test_moneyline_to_prob_invalid_returns_none(moneyline: float | None) -> None:
    """Zero, missing and NaN moneylines have no implied probability."""
    # Act
    prob = market.moneyline_to_prob(moneyline)

    # Assert
    assert prob is None


def test_moneyline_to_prob_bool_keyword_returns_none() -> None:
    """A boolean moneyline has no implied probability."""
    # Act
    prob = market.moneyline_to_prob(moneyline=True)

    # Assert
    assert prob is None


@pytest.mark.parametrize(("prob", "expected"), [(0.6, -150), (0.4, 150), (1.5, None), ("x", None)])
def test_prob_to_moneyline(prob: object, expected: int | None) -> None:
    """Probabilities convert back to American odds; invalid inputs give ``None``."""
    # Act
    moneyline = market.prob_to_moneyline(prob)

    # Assert
    if expected is None:
        assert moneyline is None
    else:
        assert moneyline == expected


def test_novig_and_ev() -> None:
    """Vig removal normalizes to one."""
    # Act
    home, away = market.novig_pair(0.55, 0.50)

    # Assert
    assert home == pytest.approx(0.55 / 1.05)
    assert away is not None
    assert home is not None
    assert home + away == pytest.approx(1.0)


@pytest.mark.parametrize(("home", "away"), [(None, 0.5), (0.0, 0.0)])
def test_novig_pair_invalid_returns_none_pair(home: float | None, away: float) -> None:
    """A missing or all-zero pair has no vig-free probabilities."""
    # Act
    pair = market.novig_pair(home, away)

    # Assert
    assert pair == (None, None)


@pytest.mark.parametrize(
    ("prob", "moneyline", "expected"),
    [
        (0.5, 100, pytest.approx(0.0)),
        (0.6, -150, pytest.approx(0.6 * (100 / 150) - 0.4)),
    ],
)
def test_expected_value(prob: float, moneyline: int, expected: object) -> None:
    """EV follows p * profit - (1 - p)."""
    # Act
    value = market.expected_value(prob, moneyline)

    # Assert
    assert value == expected


@pytest.mark.parametrize(
    ("edge", "expected"),
    [
        (None, "PASS"),
        (0.019, "PASS"),
        (0.02, "LEAN"),
        (0.05, "SMALL"),
        (0.08, "MEDIUM"),
        (0.12, "STRONG"),
    ],
)
def test_action_ladder_and_confidence(edge: float | None, expected: str) -> None:
    """Edges map to the workbook's PASS/LEAN/SMALL/MEDIUM/STRONG rungs."""
    # Act
    label = market.action_label(edge)

    # Assert
    assert label == expected


@pytest.mark.parametrize(
    ("edge", "expected"),
    [(None, 1), (0.0, 1), (0.015, 2), (0.095, 10), (0.5, 10)],
)
def test_confidence_1_to_10_ladder(edge: float | None, expected: int) -> None:
    """Edges map to the workbook's 1..10 confidence ladder."""
    # Act
    confidence = market.confidence_1_to_10(edge)

    # Assert
    assert confidence == expected


@pytest.mark.parametrize(
    ("p10", "p90", "expected"),
    [
        (-17.0, 17.0, pytest.approx(34 / 2.563103, abs=1e-4)),
        (None, 17.0, 13.0),
        (5.0, 1.0, 13.0),
    ],
)
def test_sigma_and_normal_tail(p10: float | None, p90: float, expected: object) -> None:
    """Sigma comes from the p10/p90 spread with a fallback."""
    # Act
    sigma = market.sigma_from_quantiles(p10, p90, 13.0)

    # Assert
    assert sigma == expected


@pytest.mark.parametrize(
    ("threshold", "mu", "sigma", "expected"),
    [
        (0.0, 0.0, 10.0, pytest.approx(0.5)),
        (0.0, 3.0, 0.0, 1.0),
        (5.0, 3.0, 0.0, 0.0),
    ],
)
def test_prob_exceeds(threshold: float, mu: float, sigma: float, expected: object) -> None:
    """Normal tail probabilities behave at the center and with zero sigma."""
    # Act
    prob = market.prob_exceeds(threshold, mu, sigma)

    # Assert
    assert prob == expected


def test_prob_exceeds_threshold_below_mean_above_half() -> None:
    """A threshold below the mean gives a tail probability above one half."""
    # Act
    prob = market.prob_exceeds(-3.0, 3.0, 13.0)

    # Assert
    assert prob > 0.5
