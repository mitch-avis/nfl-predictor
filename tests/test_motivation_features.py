"""Tests for standings-based motivational asymmetry features.

These features are computed strictly from games prior to the prediction week.
"""

from __future__ import annotations

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils


def _schedule_rows(*rows: dict) -> pl.DataFrame:
    """Create a minimal schedule DataFrame for standings computation."""
    return pl.DataFrame(list(rows)).with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("away_score").cast(pl.Int32),
            pl.col("home_score").cast(pl.Int32),
        ]
    )


def _games_to_predict(away: str, home: str) -> pl.DataFrame:
    """Create a minimal games_df with required team columns."""
    return pl.DataFrame({"away_abbr": [away], "home_abbr": [home]})


def test_motivation_features_use_only_prior_weeks() -> None:
    """Week N motivation must ignore results from week N and later."""
    season = 2024

    # Pick two teams in the same division to keep comparisons meaningful.
    away = "BUF"  # AFC East
    home = "MIA"  # AFC East

    schedule = _schedule_rows(
        {
            "season": season,
            "week": 1,
            "game_type": "REG",
            "away_abbr": away,
            "home_abbr": home,
            "away_score": 21,
            "home_score": 14,
        },
        # This game should NOT affect week 2 standings.
        {
            "season": season,
            "week": 2,
            "game_type": "REG",
            "away_abbr": away,
            "home_abbr": home,
            "away_score": 0,
            "home_score": 70,
        },
    )

    games_df = _games_to_predict(away, home)

    out = polars_utils.add_motivation_features(
        games_df,
        schedule,
        season=season,
        week=2,
        include_postseason=False,
    )

    # Based on week 1 only, BUF should lead the division and have 0 games behind.
    assert out["away_division_rank"].to_list() == [1]
    assert out["away_division_games_behind"].to_list() == [0.0]


def test_motivation_features_invariant_schema_on_missing_scores() -> None:
    """When schedule scores are missing, motivation columns should exist as nulls."""
    season = 2024

    schedule_missing_scores = pl.DataFrame(
        {
            "season": [season],
            "week": [1],
            "game_type": ["REG"],
            "away_abbr": ["BUF"],
            "home_abbr": ["MIA"],
        }
    )

    games_df = _games_to_predict("BUF", "MIA")

    out = polars_utils.add_motivation_features(
        games_df,
        schedule_missing_scores,
        season=season,
        week=2,
        include_postseason=False,
    )

    for col in constants.MOTIVATION_FEATURE_COLUMNS:
        assert col in out.columns
        assert out[col].to_list() == [None]


def test_motivation_division_clinch_proxy_uses_max_wins_other() -> None:
    """Division clinch proxy should consider other teams' max possible wins.

    This test uses a minimal scheduled season for the AFC East teams so the leader can clinch
    early in a toy setup.
    """
    season = 2024

    schedule = _schedule_rows(
        {
            "season": season,
            "week": 1,
            "game_type": "REG",
            "away_abbr": "BUF",
            "home_abbr": "MIA",
            "away_score": 21,
            "home_score": 14,
        },
        {
            "season": season,
            "week": 1,
            "game_type": "REG",
            "away_abbr": "NE",
            "home_abbr": "NYJ",
            "away_score": 0,
            "home_score": 0,
        },
    )

    games_df = _games_to_predict("BUF", "MIA")
    out = polars_utils.add_motivation_features(
        games_df,
        schedule,
        season=season,
        week=2,
        include_postseason=False,
    )

    assert out["away_division_rank"].to_list() == [1]
    assert out["away_division_clinched_proxy"].to_list() == [1]
