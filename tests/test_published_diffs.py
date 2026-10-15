"""Tests for which away-minus-home companions the published game schema carries."""

from __future__ import annotations

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars.finalize import build_final_column_order


def test_the_strength_game_count_is_published_per_side_without_a_diff() -> None:
    # Act
    order = build_final_column_order()

    # Assert
    assert {"away_strength_games_played", "home_strength_games_played"} <= set(order)
    assert "strength_games_played_diff" not in order


def test_the_week_rows_diff_every_strength_stat_but_the_game_count() -> None:
    # Act
    stats = polars_utils.get_stats_for_diff()

    # Assert
    assert set(constants.ADJUSTED_STRENGTH_STATS) - set(stats) == {"strength_games_played"}
