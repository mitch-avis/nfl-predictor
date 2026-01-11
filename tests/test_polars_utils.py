"""Tests for Polars-based utility transforms."""

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils


def test_combine_stats_creates_combined_and_drops() -> None:
    """Derived columns are computed and source columns are dropped."""

    df = pl.DataFrame(
        {
            "sack_fumbles": [1],
            "rushing_fumbles": [2],
            "receiving_fumbles": [3],
            "sack_fumbles_lost": [1],
            "rushing_fumbles_lost": [1],
            "receiving_fumbles_lost": [0],
            "passing_first_downs": [5],
            "rushing_first_downs": [4],
            "passing_2pt_conversions": [1],
            "rushing_2pt_conversions": [0],
            "receiving_2pt_conversions": [1],
            "fumble_recovery_own": [1],
            "fumble_recovery_opp": [2],
            "def_interceptions": [1],
            "interceptions_thrown": [2],
            "pass_yards": [250],
            "rush_yards": [100],
            "yards_lost_from_sacks": [10],
        }
    )

    combined = polars_utils.combine_stats(df)

    assert combined.select("fumbles").item() == 6
    assert combined.select("fumbles_lost").item() == 2
    assert combined.select("first_downs").item() == 9
    assert combined.select("2pt_conversions").item() == 2
    assert combined.select("fumble_recoveries").item() == 3
    assert combined.select("turnover_margin").item() == -1
    assert combined.select("total_yards").item() == 340

    for col in (
        "sack_fumbles",
        "rushing_fumbles",
        "receiving_fumbles",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "passing_first_downs",
        "rushing_first_downs",
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fumble_recovery_own",
        "fumble_recovery_opp",
    ):
        assert col not in combined.columns


def test_add_scoring_data_to_team_stats() -> None:
    """Schedule scores are joined into per-team stats."""

    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [24],
            "home_score": [17],
        }
    )

    scored = polars_utils.add_scoring_data_to_team_stats(team_stats, schedule)

    away_row = scored.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    home_row = scored.filter(pl.col("team_abbr") == "KC").row(0, named=True)

    assert away_row["points_scored"] == 24
    assert away_row["points_allowed"] == 17
    assert away_row["scoring_margin"] == 7
    assert home_row["points_scored"] == 17
    assert home_row["points_allowed"] == 24
    assert home_row["scoring_margin"] == -7


def test_add_per_game_opponent_stats() -> None:
    """Opponent stats are attached per game based on opponent_abbr."""

    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [275, 310],
        }
    )

    result = polars_utils.add_per_game_opponent_stats(team_stats)

    away_row = result.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    home_row = result.filter(pl.col("team_abbr") == "KC").row(0, named=True)

    assert away_row["opponent_pass_yards"] == 310
    assert home_row["opponent_pass_yards"] == 275


def test_aggregate_team_stats_to_week_regular() -> None:
    """Rolling averages aggregate correctly up to a target week."""

    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "team_abbr": ["BUF", "BUF", "KC", "KC"],
            "opponent_abbr": ["KC", "KC", "BUF", "BUF"],
            "pass_yards": [200, 300, 250, 350],
            "points_scored": [20, 30, 24, 28],
            "points_allowed": [17, 21, 20, 24],
        }
    )

    agg = polars_utils.aggregate_team_stats_to_week(team_stats, target_week=3, season=2024)

    buf_row = agg.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    assert buf_row["games_played"] == 2
    assert buf_row["pass_yards"] == pytest.approx(250.0)
    assert buf_row["points_scored"] == pytest.approx(25.0)


def test_calculate_stat_differentials_skips_non_numeric() -> None:
    """Stat differentials are only computed for numeric columns."""

    df = pl.DataFrame(
        {
            "away_elo_pre": [1500.0],
            "home_elo_pre": [1400.0],
            "away_qb": ["QB1"],
            "home_qb": ["QB2"],
        }
    )

    result = polars_utils.calculate_stat_differentials(df, ["elo_pre", "qb"])

    assert "elo_pre_diff" in result.columns
    assert result.select("elo_pre_diff").item() == pytest.approx(100.0)
    assert "qb_diff" not in result.columns


def test_build_final_column_order_metadata_first() -> None:
    """Final column ordering starts with metadata columns."""

    final_order = polars_utils.build_final_column_order()
    assert final_order[: len(constants.METADATA_COLUMNS)] == constants.METADATA_COLUMNS


def test_build_final_column_order_has_no_duplicates_and_includes_lines_results() -> None:
    """Final column ordering has no duplicates and includes lines/result columns."""

    final_order = polars_utils.build_final_column_order()
    assert len(final_order) == len(set(final_order))

    for col in (*constants.LINES_COLUMNS, *constants.RESULT_COLUMNS):
        assert col in final_order
