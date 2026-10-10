"""Tests for the fixed row order of the season tables."""

import pytest

from nfl_predictor.season_stats.row_order import data_file_row_order, row_identity_keys


def test_row_identity_keys_order_team_bins_by_unit_then_bin() -> None:
    # Arrange
    columns = ["game_id", "week", "team", "opponent_team", "wp_unit", "wp_bin", "wp_bin_plays"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("team", "week", "game_id", "wp_unit", "wp_bin")


def test_row_identity_keys_put_the_passer_before_week_and_bin() -> None:
    # Arrange
    columns = ["game_id", "week", "qb_id", "wp_bin", "qb_wp_bin_dropbacks"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("qb_id", "week", "game_id", "wp_bin")


def test_row_identity_keys_are_empty_without_an_identity_column() -> None:
    # Act
    keys = row_identity_keys(["week", "wp_bin"])

    # Assert
    assert keys == ()


def test_row_identity_keys_order_team_pairs_by_team_then_compared_team() -> None:
    # Arrange
    columns = ["team", "other_team", "team_rated_above_probability", "team_pair_share"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("team", "other_team")


def test_row_identity_keys_order_passer_pairs_by_passer_then_compared_passer() -> None:
    # Arrange
    columns = ["qb_id", "other_qb_id", "qb_rated_above_probability", "qb_pair_share"]

    # Act
    keys = row_identity_keys(columns)

    # Assert
    assert keys == ("qb_id", "other_qb_id")


def test_data_file_row_order_keeps_a_published_order() -> None:
    # Act
    order = data_file_row_order("ratings", ["team", "team_rating"])

    # Assert
    assert order == (("team_rating", True), ("team", False))


def test_data_file_row_order_sorts_other_files_by_their_identity_keys() -> None:
    # Arrange
    columns = ["game_id", "week", "team", "opponent_team", "points_for"]

    # Act
    order = data_file_row_order("team_game_logs", columns)

    # Assert
    assert order == (("team", False), ("week", False), ("game_id", False))


def test_data_file_row_order_rejects_a_file_without_a_row_key() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="has no row order"):
        data_file_row_order("league_totals", ["week", "plays"])
