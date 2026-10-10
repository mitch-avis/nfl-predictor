"""Tests for the season-table inputs rebuilt from the game logs nfl-sos-ratings published."""

import polars as pl
import pytest

from nfl_predictor.season_stats import game_logs, published_logs
from nfl_predictor.season_stats.pooled_rates import denominator_column, numerator_column
from nfl_predictor.season_stats.team_stats import compute_all_teams_per_game


def _team_logs() -> pl.DataFrame:
    """Return two DEN games' published team logs, both sides of each game, in team order."""
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "week": [1, 1, 2, 2],
            "team": ["DEN", "KC", "DEN", "LV"],
            "opponent_team": ["KC", "DEN", "LV", "DEN"],
            "passing_yards": [250.0, 180.0, 200.0, 150.0],
            "offensive_snaps": [60, 55, 50, 65],
            "passing_yards_per_offensive_snap": [250 / 60, 180 / 55, 200 / 50, 150 / 65],
            "completions": [20, 15, 10, 30],
            "attempts": [30, 25, 20, 40],
            "completion_pct": [20 / 30, 15 / 25, 10 / 20, 30 / 40],
            "dropbacks": [35, 28, 22, 44],
            "epa_per_dropback": [0.1, -0.05, 0.2, 0.0],
            "completion_pct_allowed": [15 / 25, 20 / 30, 30 / 40, 10 / 20],
        }
    ).sort("team", "week", "game_id")


def test_team_games_from_published_rebuilds_per_snap_rate_parts() -> None:
    # Arrange
    rate = "passing_yards_per_offensive_snap"

    # Act
    games = published_logs.team_games_from_published(_team_logs())

    # Assert
    assert games.get_column(numerator_column(rate)).to_list() == [250.0, 200.0, 180.0, 150.0]
    assert games.get_column(denominator_column(rate)).to_list() == [60, 50, 55, 65]


def test_team_games_from_published_rebuilds_a_ratio_of_published_columns() -> None:
    # Act
    games = published_logs.team_games_from_published(_team_logs())

    # Assert
    parts = games.select(
        numerator_column("completion_pct"), denominator_column("completion_pct")
    ).rows()
    assert parts == [(20, 30), (10, 20), (15, 25), (30, 40)]


def test_team_games_from_published_leaves_a_ratio_of_a_dropped_column_without_parts() -> None:
    """EPA per dropback divides the dropbacks' EPA, a working column the logs leave out."""
    # Act
    games = published_logs.team_games_from_published(_team_logs())

    # Assert
    assert numerator_column("epa_per_dropback") not in games.columns
    assert "epa_per_dropback" not in published_logs.recovered_team_rates(_team_logs().columns)


def test_team_games_from_published_takes_a_mirrored_rates_parts_from_the_opponent() -> None:
    # Act
    games = published_logs.team_games_from_published(_team_logs())

    # Assert
    parts = games.select(
        numerator_column("completion_pct_allowed"), denominator_column("completion_pct_allowed")
    ).rows()
    assert parts == [(15, 25), (30, 40), (20, 30), (10, 20)]


def test_team_games_from_published_keeps_the_published_rows_and_columns() -> None:
    # Act
    games = published_logs.team_games_from_published(_team_logs())

    # Assert
    assert games.select(_team_logs().columns).equals(_team_logs())


def test_a_season_row_from_published_logs_pools_a_recovered_rate() -> None:
    # Arrange
    games = published_logs.team_games_from_published(_team_logs())

    # Act
    season = compute_all_teams_per_game(games).filter(pl.col("team") == "DEN")

    # Assert
    assert season.get_column("completion_pct").item() == pytest.approx(30 / 50)
    assert season.get_column("passing_yards_per_offensive_snap").item() == pytest.approx(450 / 110)


def test_recovered_team_rates_lists_each_rate_it_rebuilds() -> None:
    # Act
    rates = published_logs.recovered_team_rates(_team_logs().columns)

    # Assert
    assert rates == ("passing_yards_per_offensive_snap", "completion_pct", "completion_pct_allowed")


def test_qb_games_from_published_reads_the_passers_rows_without_the_game_context() -> None:
    # Arrange
    logs = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "qb_id": ["00-0031234"],
            "qb_name": ["John Doe"],
            "qb_dropbacks": [35],
            "is_home": [True],
            "points_for": [24],
            "points_allowed": [17],
            "point_margin": [7],
            "win_value": [1.0],
            "turnover_margin": [1],
        }
    )

    # Act
    games = published_logs.qb_games_from_published(logs)

    # Assert
    assert games.columns == ["game_id", "week", "team_abbr", "qb_id", "qb_name", "qb_dropbacks"]
    assert set(game_logs.QB_LOG_CONTEXT_COLUMNS).isdisjoint(games.columns)


def test_team_games_from_published_returns_rows_in_the_builders_order() -> None:
    """The season rows sum each team's games in this order, as from the play-by-play build."""
    # Arrange
    logs = _team_logs().reverse()

    # Act
    games = published_logs.team_games_from_published(logs)

    # Assert
    assert games.select(game_logs.TEAM_GAME_ORDER).equals(
        _team_logs().select(game_logs.TEAM_GAME_ORDER).sort(game_logs.TEAM_GAME_ORDER)
    )


def test_qb_games_from_published_returns_rows_in_the_builders_order() -> None:
    # Arrange
    logs = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "week": [1, 1, 2],
            "team": ["KC", "DEN", "DEN"],
            "qb_id": ["00-2", "00-1", "00-1"],
            "qb_name": ["B", "A", "A"],
        }
    )

    # Act
    games = published_logs.qb_games_from_published(logs)

    # Assert
    assert games.select("team_abbr", "week").rows() == [("DEN", 1), ("DEN", 2), ("KC", 1)]
