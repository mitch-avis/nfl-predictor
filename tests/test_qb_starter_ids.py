"""Tests for the expected starters' ids that every quarterback feature reads."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils.polars import qb_starters, qb_stats

_IDENTITY = pl.DataFrame(
    {"qb_name": ["Name A", "Name B"], "qb_id": ["A", "B"]},
    schema={"qb_name": pl.String, "qb_id": pl.String},
)


def _qb_games(passers: list[tuple[str, str]]) -> pl.DataFrame:
    """Return one 2023 week-1 quarterback-game row per ``(qb_id, passer_name)``."""
    return _qb_games_at([(qb_id, name, 2023, 1) for qb_id, name in passers])


def _qb_games_at(passers: list[tuple[str, str, int, int]]) -> pl.DataFrame:
    """Return one quarterback-game row per ``(qb_id, passer_name, season, week)``, zero sums."""
    rows = [
        {
            "season": season,
            "week": week,
            "team_abbr": "NE",
            "opponent_abbr": "NYJ",
            "qb_id": qb_id,
            "passer_name": name,
            **dict.fromkeys(qb_stats.QB_GAME_SUM_COLUMNS, 0.0),
        }
        for qb_id, name, season, week in passers
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


def test_a_later_season_passer_with_the_same_abbreviation_leaves_the_name_resolved() -> None:
    """A rookie three seasons later must not make an earlier row's name ambiguous."""
    # Arrange
    games = _games([("Jayden Daniels", "Name A")])
    passers = _qb_games_at([("JD24", "J.Daniels", 2023, 1), ("JD26", "J.Daniels", 2026, 1)])

    # Act
    starters = qb_stats.expected_starter_ids(games, passers, _IDENTITY)

    # Assert
    assert starters.rows() == [("JD24", "A")]


def test_a_passer_first_seen_in_the_rows_week_or_later_does_not_name_it() -> None:
    """The fallback reads only games before the row's week, as the row's own features do."""
    # Arrange
    games = _games([("Joe Burrow", "Name A"), ("Joe Burrow", "Name B")])
    games = games.with_columns(pl.Series("week", [2, 3]))
    passers = _qb_games_at([("JB", "J.Burrow", 2023, 2)])

    # Act
    starters = qb_stats.expected_starter_ids(games, passers, _IDENTITY)

    # Assert
    assert starters.rows() == [(None, "A"), ("JB", "B")]


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


def _with_teams(games: pl.DataFrame) -> pl.DataFrame:
    """Return ``games`` with NYJ visiting NE in every row."""
    return games.with_columns(pl.lit("NYJ").alias("away_abbr"), pl.lit("NE").alias("home_abbr"))


def _starters(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """Return play-by-play starters from ``(game_id, team_abbr, qb_id)`` rows."""
    return pl.DataFrame(
        [
            {"game_id": game, "season": 2023, "week": 2, "team_abbr": team, "qb_id": qb_id}
            for game, team, qb_id in rows
        ],
        schema=qb_starters.STARTERS_SCHEMA,
    )


def test_expected_starter_ids_take_the_play_by_play_starter_over_the_named_one() -> None:
    # Arrange
    games = _with_teams(_games([("Name A", "Name B"), ("Name A", "Name B")]))
    starters = _starters([("g0", "NE", "PBP"), ("g1", "NYJ", "PBP")])

    # Act
    ids = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY, starters=starters)

    # Assert
    assert ids.rows() == [("A", "PBP"), ("PBP", "B")]


def test_expected_starter_ids_name_every_side_without_a_play_by_play_starter() -> None:
    # Arrange
    games = _with_teams(_games([("Name A", "Name B")]))
    starters = _starters([("an upcoming game", "NE", "PBP")])

    # Act
    ids = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY, starters=starters)

    # Assert
    assert ids.rows() == [("A", "B")]


def test_expected_starter_ids_without_game_ids_or_teams_use_the_named_starters() -> None:
    # Arrange
    games = _games([("Name A", "Name B")]).drop("game_id")
    starters = _starters([("g0", "NE", "PBP")])

    # Act
    ids = qb_stats.expected_starter_ids(games, _qb_games([]), _IDENTITY, starters=starters)

    # Assert
    assert ids.rows() == [("A", "B")]
