"""Tests for the adapter from the game logs ``nfl-sos-ratings`` published."""

import polars as pl

from nfl_predictor.ratings.inputs import QB_GAME_SCHEMA, TEAM_GAME_SCHEMA
from nfl_predictor.ratings.published_logs import (
    qb_games_from_published,
    team_games_from_published,
)
from nfl_predictor.ratings.qb_rating import QB_EPA_PER_DROPBACK_COLUMN, qb_rating_rows


def _published_team_logs() -> pl.DataFrame:
    """Return published-style team game logs: sorted by team, week, and game, with extras."""
    return pl.DataFrame(
        {
            "game_id": ["2024_01_BBB_AAA", "2024_02_AAA_CCC", "2024_01_BBB_AAA"],
            "week": pl.Series([1, 2, 1], dtype=pl.Int32),
            "team": ["AAA", "AAA", "BBB"],
            "opponent_team": ["BBB", "CCC", "AAA"],
            "passing_yards": [210.0, 305.0, 188.0],
            "offensive_snaps": [61, 66, 59],
            "is_home": [True, False, False],
            "point_margin": pl.Series([3, -10, -3], dtype=pl.Int32),
            "offensive_epa": [2.75, -4.5, 1.0],
            "st_plays": [12, 10, 13],
            "st_epa": [-0.5, 1.25, 0.25],
        }
    )


def _published_qb_logs() -> pl.DataFrame:
    """Return published-style QB game logs: sorted by passer, week, and game, with extras."""
    return pl.DataFrame(
        {
            "game_id": ["2024_01_BBB_AAA", "2024_02_AAA_CCC", "2024_01_BBB_AAA", "2024_01_BBB_AAA"],
            "week": pl.Series([1, 2, 1, 1], dtype=pl.Int32),
            "team": ["AAA", "AAA", "BBB", "AAA"],
            "opponent_team": ["BBB", "CCC", "AAA", "BBB"],
            "qb_id": ["00-01", "00-01", "00-02", "00-03"],
            "qb_name": ["Z.Starter", "Z.Starter", "B.Passer", "A.Backup"],
            "qb_dropbacks": [37, 44, 35, 3],
            "qb_attempts": [33, 40, 31, 3],
            "qb_passing_epa": [7.0, -3.3, 2.9, -0.1],
            "qb_spike_epa": [-0.2, 0.0, -0.3, 0.0],
            "qb_epa_per_dropback": [7.2 / 37, -3.3 / 44, 3.2 / 35, -0.1 / 3],
            "is_home": [True, False, False, True],
        }
    )


def test_team_games_from_published_keep_the_published_row_order() -> None:
    # Act
    rows = team_games_from_published(_published_team_logs())

    # Assert
    assert rows.select("team", "week").rows() == [("AAA", 1), ("AAA", 2), ("BBB", 1)]


def test_team_games_from_published_have_the_input_schema() -> None:
    # Act
    rows = team_games_from_published(_published_team_logs())

    # Assert
    assert rows.schema == pl.Schema(TEAM_GAME_SCHEMA)


def test_qb_games_from_published_have_the_input_schema() -> None:
    # Act
    rows = qb_games_from_published(_published_qb_logs())

    # Assert
    assert rows.schema == pl.Schema(QB_GAME_SCHEMA)


def test_qb_games_from_published_take_the_dropback_epa_without_spikes() -> None:
    # Act
    rows = qb_games_from_published(_published_qb_logs())

    # Assert
    by_passer = dict(rows.filter(pl.col("week") == 1).select("qb_id", "qb_dropback_epa").rows())
    assert by_passer == {"00-01": 7.0 - -0.2, "00-02": 2.9 - -0.3, "00-03": -0.1 - 0.0}


def test_qb_games_from_published_sort_like_the_rows_the_fit_used() -> None:
    """The source fit its QB rows sorted by team, week, game, and passer name."""
    # Act
    rows = qb_games_from_published(_published_qb_logs())

    # Assert
    assert rows.get_column("qb_name").to_list() == [
        "A.Backup",
        "Z.Starter",
        "Z.Starter",
        "B.Passer",
    ]


def test_qb_games_from_published_rebuild_the_published_rate_bit_for_bit() -> None:
    # Arrange
    published = _published_qb_logs().with_columns(
        ((pl.col("qb_passing_epa") - pl.col("qb_spike_epa")) / pl.col("qb_dropbacks")).alias(
            "qb_epa_per_dropback"
        )
    )
    expected = published.sort(["team", "week", "game_id", "qb_name"]).get_column(
        "qb_epa_per_dropback"
    )

    # Act
    rows = qb_rating_rows(qb_games_from_published(published))

    # Assert
    assert rows.get_column(QB_EPA_PER_DROPBACK_COLUMN).to_list() == expected.to_list()
