"""Unit tests for team-week injury burden feature engineering."""

from __future__ import annotations

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils


def test_compute_team_week_injury_burdens_aggregates_total_and_positions() -> None:
    """Aggregates should sum weights and split burdens by position group."""

    injuries_df = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "team": ["BUF", "BUF", "BUF", "KC"],
            "position": ["QB", "WR", "S", "RB"],
            "report_status": ["Out", "Questionable", "Probable", "Doubtful"],
        }
    )

    out = polars_utils.compute_team_week_injury_burdens(injuries_df, season=2024, week=1)
    by_team = {row["team_abbr"]: row for row in out.iter_rows(named=True)}

    buf = by_team["BUF"]
    kc = by_team["KC"]

    # Total burden weights: BUF = 1.0 + 0.5 + 0.25 = 1.75, KC = 0.75
    assert abs(float(buf["injury_burden_total"]) - 1.75) < 1e-6
    assert abs(float(kc["injury_burden_total"]) - 0.75) < 1e-6

    # Positional splits (per constants.INJURY_POSITION_GROUPS)
    assert abs(float(buf["injury_burden_qb"]) - 1.0) < 1e-6
    assert abs(float(buf["injury_burden_wr"]) - 0.5) < 1e-6
    assert abs(float(buf["injury_burden_db"]) - 0.25) < 1e-6

    assert abs(float(kc["injury_burden_rb"]) - 0.75) < 1e-6


def test_add_injury_burden_features_adds_nulls_when_unavailable() -> None:
    """When injuries are unavailable, columns are present and null (explicit missing-data)."""

    games_df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    out = polars_utils.add_injury_burden_features(games_df, None, season=2007, week=1)
    for col in constants.INJURY_FEATURE_COLUMNS:
        assert col in out.columns
        assert out.select(col).to_series().to_list() == [None]


def test_add_injury_burden_features_joins_and_fills_zero_for_no_injuries() -> None:
    """When injury data is available for the week, missing teams should get 0.0 burden."""

    games_df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    injuries_df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "team": ["BUF"],
            "position": ["QB"],
            "report_status": ["Out"],
        }
    )

    out = polars_utils.add_injury_burden_features(games_df, injuries_df, season=2024, week=1)

    assert out.select("away_injury_burden_total").to_series().to_list() == [1.0]
    assert out.select("home_injury_burden_total").to_series().to_list() == [0.0]


def test_compute_team_week_injury_burdens_validates_schema() -> None:
    """Helper should raise a clear error if required columns are missing."""

    bad_df = pl.DataFrame({"season": [2024], "week": [1]})

    try:
        polars_utils.compute_team_week_injury_burdens(bad_df, season=2024, week=1)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")
