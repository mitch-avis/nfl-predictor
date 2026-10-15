"""The season statistics and the ratings package summarize a passer's season alike.

Both packages carry a port of ``nfl-sos-ratings``' passer season summary: the season row of
``season_stats.qb_stats`` and ``ratings.qb_rating.qb_season_summary``, which ranks passers. Their
games, totals, primary team, and ranking qualifier must agree, here on the published game logs of
two seasons (the parity fixtures).
"""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.ratings import published_logs as ratings_published_logs
from nfl_predictor.ratings.qb_rating import QB_SUMMARY_COLUMNS, qb_season_summary
from nfl_predictor.season_stats import published_logs
from nfl_predictor.season_stats.qb_stats import compute_qb_season_stats

FIXTURES = Path(__file__).parent / "fixtures" / "sos_season_stats_parity"
# The parity audit's float tolerance: the two sum each passer's games in different orders.
FLOAT_TOLERANCE = 1e-9


@pytest.mark.parametrize("season", [1999, 2024])
def test_season_stats_and_ratings_summarize_each_passers_season_alike(season: int) -> None:
    # Arrange
    qb_logs = pl.read_parquet(FIXTURES / f"{season}_season_qb_game_logs.parquet")
    team_games = published_logs.team_stat_rows_from_published(
        pl.read_parquet(FIXTURES / f"{season}_season_team_game_logs.parquet")
    )
    ratings_summary = qb_season_summary(
        ratings_published_logs.qb_games_from_published(qb_logs), team_games
    )

    # Act
    season_rows = compute_qb_season_stats(
        published_logs.qb_stat_rows_from_published(qb_logs), weekly_df=team_games
    )

    # Assert
    assert_frame_equal(
        season_rows.select(QB_SUMMARY_COLUMNS).sort("team", "qb_id"),
        ratings_summary.sort("team", "qb_id"),
        check_exact=False,
        rel_tol=0,
        abs_tol=FLOAT_TOLERANCE,
    )
