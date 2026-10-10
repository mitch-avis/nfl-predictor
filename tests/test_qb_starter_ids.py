"""Tests for the expected starters' ids that every quarterback feature reads."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils.polars import qb_stats

_IDENTITY = pl.DataFrame(
    {"qb_name": ["Name A", "Name B"], "qb_id": ["A", "B"]},
    schema={"qb_name": pl.String, "qb_id": pl.String},
)


def _qb_games(passers: list[tuple[str, str]]) -> pl.DataFrame:
    """Return one quarterback-game row per ``(qb_id, passer_name)`` with zero sums."""
    rows = [
        {
            "season": 2023,
            "week": 1,
            "team_abbr": "NE",
            "opponent_abbr": "NYJ",
            "qb_id": qb_id,
            "passer_name": name,
            **dict.fromkeys(qb_stats.QB_GAME_SUM_COLUMNS, 0.0),
        }
        for qb_id, name in passers
    ]
    return pl.DataFrame(rows, schema=qb_stats.QB_GAME_SCHEMA)


def _games(pairs: list[tuple[str | None, str | None]]) -> pl.DataFrame:
    """Return game rows with the given ``(away_qb, home_qb)`` names."""
    return pl.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(len(pairs))],
            "season": [2023] * len(pairs),
            "week": [2] * len(pairs),
            "away_qb": [pair[0] for pair in pairs],
            "home_qb": [pair[1] for pair in pairs],
        },
        schema={
            "game_id": pl.String,
            "season": pl.Int64,
            "week": pl.Int64,
            "away_qb": pl.String,
            "home_qb": pl.String,
        },
    )


def test_expected_starter_ids_map_names_through_the_identity() -> None:
    # Arrange
    games = _games([("Name A", "Name B")])

    # Act
    starters = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY)

    # Assert
    assert starters.rows() == [("A", "B")]


def test_expected_starter_ids_fall_back_to_a_unique_abbreviated_passer_name() -> None:
    # Arrange
    games = _games([("Joe Burrow", "Name A")])
    passers = _qb_games([("JB", "J.Burrow")])

    # Act
    starters = qb_stats.expected_starter_ids(games, passers, _IDENTITY)

    # Assert
    assert starters.rows() == [("JB", "A")]


def test_expected_starter_ids_leave_an_ambiguous_abbreviation_unknown() -> None:
    # Arrange
    games = _games([("Josh Allen", "Name A")])
    passers = _qb_games([("JA1", "J.Allen"), ("JA2", "J.Allen")])

    # Act
    starters = qb_stats.expected_starter_ids(games, passers, _IDENTITY)

    # Assert
    assert starters.rows() == [(None, "A")]


def test_expected_starter_ids_leave_unknown_and_missing_names_unknown() -> None:
    # Arrange
    games = _games([("Nobody Known", None)])

    # Act
    starters = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY)

    # Assert
    assert starters.rows() == [(None, None)]


def test_expected_starter_ids_keep_one_row_per_game_in_the_games_order() -> None:
    # Arrange
    games = _games([("Name B", "Name A"), ("Nobody Known", "Name B"), ("Name A", "Name A")])

    # Act
    starters = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY)

    # Assert
    assert starters.columns == ["away_qb_id", "home_qb_id"]
    assert starters.rows() == [("B", "A"), (None, "B"), ("A", "A")]


def test_expected_starter_ids_without_starter_columns_are_unknown() -> None:
    # Arrange
    games = _games([("Name A", "Name B")]).drop("away_qb", "home_qb")

    # Act
    starters = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY)

    # Assert
    assert starters.rows() == [(None, None)]
