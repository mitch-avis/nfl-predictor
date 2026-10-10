"""Tests for ranking on a rating source in points: the composite, ported, model and blend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.reporting import power_rankings


def _snapshot(composites: dict[str, float], *, points_per_unit: float) -> pd.DataFrame:
    """Return one week's strength snapshot whose SRS is an exact multiple of the composite."""
    teams = sorted(composites)
    return pd.DataFrame(
        {
            "team_abbr": teams,
            "adj_strength_composite": [composites[team] for team in teams],
            "adj_srs": [composites[team] * points_per_unit for team in teams],
        }
    )


def test_composite_points_vs_average_keeps_every_digit() -> None:
    # Arrange
    snapshot = _snapshot({"AAA": -1.0, "BBB": 0.5, "CCC": 2.0}, points_per_unit=6.123457)

    # Act
    points = power_rankings.composite_points_vs_average(snapshot)

    # Assert
    assert points == pytest.approx([-9.1851855, 0.0, 9.1851855], rel=1e-9, abs=1e-12)


def test_composite_points_vs_average_without_a_scale_is_all_nan() -> None:
    # Arrange
    snapshot = _snapshot({"AAA": -1.0, "BBB": 0.5, "CCC": 2.0}, points_per_unit=-2.0)

    # Act
    points = power_rankings.composite_points_vs_average(snapshot)

    # Assert
    assert np.isnan(points).all()
