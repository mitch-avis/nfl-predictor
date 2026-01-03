"""Unit tests for lookahead (next-week context) feature engineering."""

from __future__ import annotations

import datetime as dt

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils


def test_add_lookahead_features_next_week_opponent_and_location_change() -> None:
    """Next-week opponent lookup and location-change flags should match schedule."""

    schedule_df = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2],
            "game_type": ["REG", "REG", "REG", "REG"],
            "date": [
                dt.date(2024, 9, 8),
                dt.date(2024, 9, 8),
                dt.date(2024, 9, 15),
                dt.date(2024, 9, 15),
            ],
            "away_abbr": ["BUF", "PHI", "BUF", "PHI"],
            "home_abbr": ["KC", "DAL", "DAL", "BUF"],
        }
    )

    games_df = pl.DataFrame(
        {
            "away_abbr": ["BUF", "PHI"],
            "home_abbr": ["KC", "DAL"],
            "away_win_pct": [0.0, 0.0],
            "home_win_pct": [0.0, 0.0],
        }
    )

    out = polars_utils.add_lookahead_features(games_df, schedule_df, season=2024, week=1)

    for col in constants.LOOKAHEAD_FEATURE_COLUMNS:
        assert col in out.columns

    # BUF plays @KC in week 1, then @DAL in week 2 => next_is_home=0, no location change
    buf_row = out.filter(pl.col("away_abbr") == "BUF").row(0, named=True)
    assert buf_row["away_next_opponent_abbr"] == "DAL"
    assert buf_row["away_next_is_home"] == 0
    assert buf_row["away_next_location_change"] == 0
    assert buf_row["away_days_to_next_game"] == 7

    # PHI plays @DAL in week 1, then @BUF in week 2 => next_is_home=0, no location change
    phi_row = out.filter(pl.col("away_abbr") == "PHI").row(0, named=True)
    assert phi_row["away_next_opponent_abbr"] == "BUF"
    assert phi_row["away_next_is_home"] == 0
    assert phi_row["away_next_location_change"] == 0
    assert phi_row["away_days_to_next_game"] == 7

    # Next opponent win pct is derived from current week record win_pct.
    # Here BUF has away_win_pct=0.0 in the input, so PHI's next opponent win pct is 0.0.
    assert phi_row["away_next_opponent_win_pct"] == 0.0


def test_add_lookahead_features_missing_next_week_yields_nulls() -> None:
    """When there is no week+1 game, next-week fields should be null."""

    schedule_df = pl.DataFrame(
        {
            "season": [2024],
            "week": [18],
            "game_type": ["REG"],
            "date": [dt.date(2025, 1, 5)],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    games_df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_win_pct": [0.5],
            "home_win_pct": [0.5],
        }
    )

    out = polars_utils.add_lookahead_features(games_df, schedule_df, season=2024, week=18)

    assert out.schema["away_next_opponent_abbr"] == pl.Utf8
    assert out.schema["away_next_is_home"] == pl.Int32
    assert out.schema["away_days_to_next_game"] == pl.Int32
    assert out.schema["away_next_location_change"] == pl.Int32
    assert out.schema["away_next_is_divisional_matchup"] == pl.Int32
    assert out.schema["away_next_opponent_win_pct"] == pl.Float32

    row = out.row(0, named=True)
    assert row["away_next_opponent_abbr"] is None
    assert row["away_next_is_home"] is None
    assert row["away_days_to_next_game"] is None
    assert row["away_next_location_change"] is None
    assert row["away_next_is_divisional_matchup"] is None
    assert row["away_next_opponent_win_pct"] is None


def test_compute_team_next_week_context_validates_schema() -> None:
    """Helper should raise a clear error if required schedule columns are missing."""

    bad_schedule_df = pl.DataFrame({"season": [2024], "week": [1]})

    try:
        polars_utils.compute_team_next_week_context(bad_schedule_df, season=2024, week=1)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")
