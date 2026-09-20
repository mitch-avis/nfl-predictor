"""Unit tests for divisional matchup feature engineering."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils import polars_utils


def test_add_divisional_matchup_feature_flags_true_for_division_game() -> None:
    """Divisional opponents should yield `is_divisional_matchup == 1`."""
    df = pl.DataFrame(
        {
            "away_abbr": ["PHI"],
            "home_abbr": ["DAL"],
        }
    )

    out = polars_utils.add_divisional_matchup_feature(df)
    assert out.select("is_divisional_matchup").to_series().to_list() == [1]


def test_add_divisional_matchup_feature_flags_false_for_non_division_game() -> None:
    """Non-divisional opponents should yield `is_divisional_matchup == 0`."""
    df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    out = polars_utils.add_divisional_matchup_feature(df)
    assert out.select("is_divisional_matchup").to_series().to_list() == [0]


def test_add_divisional_matchup_feature_unknown_team_defaults_false() -> None:
    """Unknown team abbreviations should not crash and default to non-divisional."""
    df = pl.DataFrame(
        {
            "away_abbr": ["XXX"],
            "home_abbr": ["DAL"],
        }
    )

    out = polars_utils.add_divisional_matchup_feature(df)
    assert out.select("is_divisional_matchup").to_series().to_list() == [0]


def test_add_divisional_matchup_feature_validates_schema() -> None:
    """Helper should raise a clear error if required columns are missing."""
    df = pl.DataFrame({"away_abbr": ["PHI"]})

    try:
        polars_utils.add_divisional_matchup_feature(df)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")


def test_add_divisional_matchup_feature_uses_historical_alignment_by_season() -> None:
    """Season-aware divisional flags should match the schedule flag in every season."""
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

    out = polars_utils.add_divisional_matchup_feature(df)

    assert out["is_divisional_matchup"].to_list() == out["division"].to_list()
