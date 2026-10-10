"""Tests for the team-game and quarterback-game input rows of the ratings package."""

import polars as pl
import pytest

from nfl_predictor.ratings.inputs import (
    QB_GAME_SCHEMA,
    TEAM_GAME_SCHEMA,
    qb_game_rows,
    team_game_rows,
)


def _team_games() -> pl.DataFrame:
    """Return two team-game rows of one game, with an extra column and narrow integer types."""
    return pl.DataFrame(
        {
            "game_id": ["2024_01_AAA_BBB", "2024_01_AAA_BBB"],
            "week": pl.Series([1, 1], dtype=pl.Int32),
            "team": ["BBB", "AAA"],
            "opponent_team": ["AAA", "BBB"],
            "is_home": [True, False],
            "offensive_snaps": [62, 58],
            "offensive_epa": [3.5, -1.25],
            "st_plays": [11, 13],
            "st_epa": [0.5, -0.75],
            "point_margin": pl.Series([7, -7], dtype=pl.Int32),
            "passing_yards": [250.0, 180.0],
        }
    )


def _qb_games() -> pl.DataFrame:
    """Return two quarterback-game rows of one game, with an extra column."""
    return pl.DataFrame(
        {
            "game_id": ["2024_01_AAA_BBB", "2024_01_AAA_BBB"],
            "week": pl.Series([1, 1], dtype=pl.Int32),
            "team": ["BBB", "AAA"],
            "opponent_team": ["AAA", "BBB"],
            "is_home": [True, False],
            "qb_id": ["00-0000001", "00-0000002"],
            "qb_name": ["B.Passer", "A.Passer"],
            "qb_dropbacks": [38, 41],
            "qb_dropback_epa": [6.25, -2.5],
            "qb_attempts": [35, 37],
            "qb_passer_rating": [101.2, 77.4],
        }
    )


def test_team_game_rows_keep_the_schema_columns_in_order() -> None:
    # Act
    rows = team_game_rows(_team_games())

    # Assert
    assert rows.schema == pl.Schema(TEAM_GAME_SCHEMA)


def test_team_game_rows_keep_the_row_order() -> None:
    # Act
    rows = team_game_rows(_team_games())

    # Assert
    assert rows.get_column("team").to_list() == ["BBB", "AAA"]


def test_team_game_rows_missing_columns_raise_naming_them() -> None:
    # Arrange
    games = _team_games().drop("st_epa", "point_margin")

    # Act & Assert
    with pytest.raises(ValueError, match="point_margin, st_epa"):
        team_game_rows(games)


def test_team_game_rows_null_keys_raise() -> None:
    # Arrange
    games = _team_games().with_columns(
        pl.when(pl.col("team") == "AAA").then(None).otherwise(pl.col("team")).alias("team")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="null team"):
        team_game_rows(games)


def test_team_game_rows_duplicate_team_games_raise() -> None:
    # Arrange
    games = pl.concat([_team_games(), _team_games().head(1)])

    # Act & Assert
    with pytest.raises(ValueError, match="more than one row"):
        team_game_rows(games)


def test_team_game_rows_values_that_do_not_fit_the_schema_raise() -> None:
    # Arrange
    games = _team_games().with_columns(pl.lit("many").alias("offensive_snaps"))

    # Act & Assert
    with pytest.raises(pl.exceptions.InvalidOperationError):
        team_game_rows(games)


def test_qb_game_rows_keep_the_schema_columns_in_order() -> None:
    # Act
    rows = qb_game_rows(_qb_games())

    # Assert
    assert rows.schema == pl.Schema(QB_GAME_SCHEMA)


def test_qb_game_rows_missing_columns_raise_naming_them() -> None:
    # Arrange
    games = _qb_games().drop("qb_dropback_epa")

    # Act & Assert
    with pytest.raises(ValueError, match="qb_dropback_epa"):
        qb_game_rows(games)


def test_qb_game_rows_duplicate_passer_games_raise() -> None:
    # Arrange
    games = pl.concat([_qb_games(), _qb_games().tail(1)])

    # Act & Assert
    with pytest.raises(ValueError, match="more than one row"):
        qb_game_rows(games)


def test_qb_game_rows_allow_a_missing_home_flag_for_a_neutral_site() -> None:
    # Arrange
    games = _qb_games().with_columns(pl.lit(None, dtype=pl.Boolean).alias("is_home"))

    # Act
    rows = qb_game_rows(games)

    # Assert
    assert rows.get_column("is_home").null_count() == 2


@pytest.mark.parametrize("column", ["game_id", "team", "opponent_team"])
def test_team_game_rows_empty_identity_values_raise(column: str) -> None:
    # Arrange
    games = _team_games().with_columns(
        pl.when(pl.col("team") == "AAA").then(pl.lit("")).otherwise(pl.col(column)).alias(column)
    )

    # Act & Assert
    with pytest.raises(ValueError, match=f"1 empty {column} value"):
        team_game_rows(games)


def test_team_game_rows_blank_team_raises() -> None:
    # Arrange
    games = _team_games().with_columns(
        pl.when(pl.col("team") == "AAA").then(pl.lit("  ")).otherwise(pl.col("team")).alias("team")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="empty team"):
        team_game_rows(games)


def test_qb_game_rows_empty_passer_id_raises() -> None:
    # Arrange
    games = _qb_games().with_columns(
        pl.when(pl.col("team") == "AAA").then(pl.lit("")).otherwise(pl.col("qb_id")).alias("qb_id")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="1 empty qb_id value"):
        qb_game_rows(games)
