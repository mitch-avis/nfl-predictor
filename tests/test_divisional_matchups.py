"""Unit tests for divisional matchup feature engineering."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor.utils import polars_utils


def test_add_divisional_matchup_feature_flags_true_for_division_game() -> None:
    """Divisional opponents should yield `is_divisional_matchup == 1`."""
    # Arrange
    df = pl.DataFrame(
        {
            "away_abbr": ["PHI"],
            "home_abbr": ["DAL"],
        }
    )

    # Act
    out = polars_utils.add_divisional_matchup_feature(df)

    # Assert
    assert out.select("is_divisional_matchup").to_series().to_list() == [1]


def test_add_divisional_matchup_feature_flags_false_for_non_division_game() -> None:
    """Non-divisional opponents should yield `is_divisional_matchup == 0`."""
    # Arrange
    df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    # Act
    out = polars_utils.add_divisional_matchup_feature(df)

    # Assert
    assert out.select("is_divisional_matchup").to_series().to_list() == [0]


def test_add_divisional_matchup_feature_unknown_team_defaults_false() -> None:
    """Unknown team abbreviations should not crash and default to non-divisional."""
    # Arrange
    df = pl.DataFrame(
        {
            "away_abbr": ["XXX"],
            "home_abbr": ["DAL"],
        }
    )

    # Act
    out = polars_utils.add_divisional_matchup_feature(df)

    # Assert
    assert out.select("is_divisional_matchup").to_series().to_list() == [0]


def test_add_divisional_matchup_feature_validates_schema() -> None:
    """Helper should raise a clear error if required columns are missing."""
    # Arrange
    df = pl.DataFrame({"away_abbr": ["PHI"]})

    # Act & Assert
    with pytest.raises(ValueError, match="missing required columns"):
        polars_utils.add_divisional_matchup_feature(df)


def test_add_divisional_matchup_feature_uses_historical_alignment_by_season() -> None:
    """Season-aware divisional flags should match the schedule flag in every season."""
    # Arrange
    rows: list[dict[str, int | str]] = []
    for season in range(1999, 2027):
        if season < 2002:
            rows.extend(
                [
                    {"season": season, "away_abbr": "ARI", "home_abbr": "DAL", "division": 1},
                    {"season": season, "away_abbr": "ARI", "home_abbr": "SF", "division": 0},
                ]
            )
        else:
            rows.extend(
                [
                    {"season": season, "away_abbr": "ARI", "home_abbr": "DAL", "division": 0},
                    {"season": season, "away_abbr": "ARI", "home_abbr": "SF", "division": 1},
                ]
            )
    df = pl.DataFrame(rows).with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("division").cast(pl.Int32),
        ]
    )

    # Act
    out = polars_utils.add_divisional_matchup_feature(df)

    # Assert
    assert out["is_divisional_matchup"].to_list() == out["division"].to_list()
