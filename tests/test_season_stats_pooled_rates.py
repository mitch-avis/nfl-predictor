"""Tests for rates pooled over games (numerator and denominator summed, then divided)."""

import polars as pl

from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    drop_rate_parts,
    mean_with_parts,
    numerator_column,
    numerator_for_value,
    pooled_rate,
    rate_parts,
    ratio_with_parts,
    summed_ratio_with_parts,
)


def _plays() -> pl.DataFrame:
    """Return two games' plays: game 1 has three passes (one complete), game 2 has none."""
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2"],
            "attempt": [1, 1, 1, 0],
            "complete": [1, 0, 0, 0],
            "cpoe": [12.0, None, -6.0, None],
        }
    )


def test_summed_ratio_with_parts_divides_the_group_sums_and_keeps_both() -> None:
    # Arrange
    plays = _plays()

    # Act
    result = plays.group_by("game_id").agg(
        summed_ratio_with_parts("completion_pct", pl.col("complete"), pl.col("attempt"))
    )

    # Assert
    assert result.sort("game_id").to_dicts() == [
        {
            "game_id": "g1",
            "completion_pct": 1 / 3,
            numerator_column("completion_pct"): 1,
            denominator_column("completion_pct"): 3,
        },
        {
            "game_id": "g2",
            "completion_pct": None,
            numerator_column("completion_pct"): 0,
            denominator_column("completion_pct"): 0,
        },
    ]


def test_mean_with_parts_averages_the_known_values_and_counts_them() -> None:
    # Arrange
    plays = _plays()

    # Act
    result = plays.group_by("game_id").agg(mean_with_parts("cpoe_mean", pl.col("cpoe")))

    # Assert
    assert result.sort("game_id").rows() == [("g1", 3.0, 6.0, 2), ("g2", None, 0.0, 0)]


def test_ratio_with_parts_divides_columns_of_a_game_row() -> None:
    # Arrange
    games = pl.DataFrame({"points": [21, 0], "snaps": [60, 0]})

    # Act
    result = games.with_columns(ratio_with_parts("points_per_snap", "points", "snaps"))

    # Assert
    assert result.get_column("points_per_snap").to_list() == [21 / 60, None]
    assert result.get_column(numerator_column("points_per_snap")).to_list() == [21, 0]
    assert result.get_column(denominator_column("points_per_snap")).to_list() == [60, 0]


def test_pooled_rate_divides_season_sums_over_games_with_both_parts() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "DEN", "KC"],
            numerator_column("completion_pct"): [1, 1, None, None],
            denominator_column("completion_pct"): [1, 9, 4, None],
        }
    )

    # Act
    result = games.group_by("team").agg(pooled_rate("completion_pct"))

    # Assert
    assert dict(result.sort("team").rows()) == {"DEN": 0.2, "KC": None}


def test_pooled_rate_is_empty_when_no_game_has_a_denominator() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "team": ["DEN", "DEN"],
            numerator_column("fourth_down_pct"): [0, 0],
            denominator_column("fourth_down_pct"): [0, 0],
        }
    )

    # Act
    result = games.group_by("team").agg(pooled_rate("fourth_down_pct"))

    # Assert
    assert result.get_column("fourth_down_pct").to_list() == [None]


def test_rate_parts_lists_each_rate_with_both_parts() -> None:
    # Arrange
    columns = [
        "completions",
        numerator_column("completion_pct"),
        denominator_column("completion_pct"),
        numerator_column("orphan"),
    ]

    # Act
    rates = rate_parts(columns)

    # Assert
    assert rates == ["completion_pct"]


def test_drop_rate_parts_removes_every_numerator_and_denominator() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "completion_pct": [0.5],
            numerator_column("completion_pct"): [1],
            denominator_column("completion_pct"): [2],
            numerator_column("orphan"): [3],
        }
    )

    # Act
    result = drop_rate_parts(games)

    # Assert
    assert result.columns == ["completion_pct"]


def test_numerator_for_value_weighs_a_replaced_rate_by_its_denominator() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "cpoe": [2.5],
            numerator_column("cpoe"): [4.0],
            denominator_column("cpoe"): [2],
        }
    )

    # Act
    result = numerator_for_value(games, "cpoe")

    # Assert
    assert result.get_column(numerator_column("cpoe")).to_list() == [5.0]


def test_numerator_for_value_leaves_a_rate_without_parts_alone() -> None:
    # Arrange
    games = pl.DataFrame({"cpoe": [2.5]})

    # Act
    result = numerator_for_value(games, "cpoe")

    # Assert
    assert result.equals(games)
