"""Tests for trend/recency feature helpers."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils


def test_build_qb_trends_groups_by_qb_name() -> None:
    """QB trends are computed per QB name, not team."""

    elo_df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["CCC", "DDD"],
            "away_elo_pre": [1500.0, 1510.0],
            "home_elo_pre": [1490.0, 1485.0],
            "away_qb": ["QB1", "QB1"],
            "home_qb": ["QB2", "QB3"],
            "away_qb_elo_pre": [1000.0, 1100.0],
            "home_qb_elo_pre": [900.0, 950.0],
            "away_qb_value_pre": [1.0, 2.0],
            "home_qb_value_pre": [0.5, 0.6],
        }
    )

    qb_trends = polars_utils.build_qb_trends(elo_df, season=2023)
    qb1_week2 = qb_trends.filter((pl.col("qb_name") == "QB1") & (pl.col("week") == 2))

    assert qb1_week2.height == 1
    qb1_trend = cast(float, qb1_week2["qb_elo_4wk_trend"][0])
    assert qb1_trend == pytest.approx(100.0)


def test_build_team_elo_trends_use_prior_games_only() -> None:
    """Team ELO trend uses prior games only and is zero for the first game."""

    elo_df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "away_abbr": ["AAA", "EEE"],
            "home_abbr": ["FFF", "AAA"],
            "away_elo_pre": [1500.0, 1400.0],
            "home_elo_pre": [1490.0, 1510.0],
        }
    )

    trends = polars_utils.build_team_elo_trends(elo_df, season=2023)
    aaa_week1 = trends.filter((pl.col("team_abbr") == "AAA") & (pl.col("week") == 1))
    aaa_week2 = trends.filter((pl.col("team_abbr") == "AAA") & (pl.col("week") == 2))

    assert aaa_week1.height == 1
    assert aaa_week2.height == 1
    assert aaa_week1["elo_4wk_trend"][0] == pytest.approx(0.0)
    assert aaa_week2["elo_4wk_trend"][0] == pytest.approx(10.0)


def test_build_team_stat_trends_compare_recent_to_season_mean() -> None:
    """Performance trends compare recent mean to season-to-date mean."""

    stats = pl.DataFrame(
        {
            "season": [2023] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "team_abbr": ["AAA"] * 6,
            "scoring_margin": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0],
            "turnover_margin": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        }
    )

    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin", "turnover_margin"],
        window=4,
    )

    week6 = trends.filter((pl.col("team_abbr") == "AAA") & (pl.col("week") == 6))
    assert week6.height == 1

    scoring_trend = cast(float, week6["scoring_margin_4wk_trend"][0])
    turnover_trend = cast(float, week6["turnover_margin_4wk_trend"][0])

    assert scoring_trend == pytest.approx(0.45, abs=1e-2)
    assert turnover_trend == pytest.approx(0.1, abs=1e-2)


def test_add_season_phase_features_buckets() -> None:
    """Season-phase buckets align with early/mid/late week ranges."""

    base = pl.DataFrame({"game_id": [1]})
    season = 2023
    regular_weeks = constants.get_regular_season_weeks(season)

    early = polars_utils.add_season_phase_features(base, season=season, week=1)
    assert early["season_phase_early"][0] == 1
    assert early["season_phase_mid"][0] == 0
    assert early["season_phase_late"][0] == 0
    assert early["week_in_season_norm"][0] == pytest.approx(1.0 / regular_weeks)

    mid_week = int((regular_weeks / 3.0) + 1)
    mid = polars_utils.add_season_phase_features(base, season=season, week=mid_week)
    assert mid["season_phase_early"][0] == 0
    assert mid["season_phase_mid"][0] == 1
    assert mid["season_phase_late"][0] == 0

    late_week = int((2.0 * regular_weeks / 3.0) + 1)
    late = polars_utils.add_season_phase_features(base, season=season, week=late_week)
    assert late["season_phase_early"][0] == 0
    assert late["season_phase_mid"][0] == 0
    assert late["season_phase_late"][0] == 1


def test_add_season_phase_features_caps_postseason() -> None:
    """Season phase caps at 1.0 for postseason weeks."""

    base = pl.DataFrame({"game_id": [1]})
    season = 2023
    regular_weeks = constants.get_regular_season_weeks(season)
    postseason_week = regular_weeks + 2

    result = polars_utils.add_season_phase_features(base, season=season, week=postseason_week)

    assert result["week_in_season_norm"][0] == pytest.approx(1.0)
    assert result["season_phase_early"][0] == 0
    assert result["season_phase_mid"][0] == 0
    assert result["season_phase_late"][0] == 1
