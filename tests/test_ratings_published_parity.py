"""Pin the ratings port to files ``nfl-sos-ratings`` published at its frozen commit.

The fixtures in ``tests/fixtures/sos_ratings_parity/`` are cut from that project's published
Parquet files (``PROVENANCE.md`` there lists the source, the cut, and the hashes). Each test fits
the cut game logs with the port and expects the published table: the same rows in the same order,
the same columns and types, and identical values. The one exception is a passer's raw season EPA
per dropback, whose sum runs in another order there and so may differ in the last bit.
"""

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.ratings.pre_week import RatingsHistory
from nfl_predictor.ratings.published_logs import (
    qb_games_from_published,
    team_games_from_published,
)
from nfl_predictor.ratings.season import (
    BootstrapSettings,
    QbSeasonRatings,
    TeamSeasonRatings,
    build_qb_season,
    build_team_season,
)
from nfl_predictor.ratings.team_prior import PriorHistory

FIXTURES = Path(__file__).parent / "fixtures" / "sos_ratings_parity"
# A passer's raw season rate sums his games in another order there.
RAW_RATE_TOLERANCE = 1e-15


def _fixture(season: int, name: str) -> pl.DataFrame:
    """Return one fixture file."""
    return pl.read_parquet(FIXTURES / f"{season}_{name}.parquet")


def _team_games(season: int) -> pl.DataFrame:
    """Return one fixture season's team-game rows."""
    return team_games_from_published(_fixture(season, "team_game_logs"))


def _qb_games(season: int) -> pl.DataFrame:
    """Return one fixture season's quarterback-game rows."""
    return qb_games_from_published(_fixture(season, "qb_game_logs"))


def _rows_like(ours: pl.DataFrame, published: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Return the rows of ``ours`` whose keys the (possibly cut) published table holds."""
    return ours.join(published.select(keys), on=keys, how="semi", maintain_order="left")


@pytest.fixture(scope="module")
def team_1999() -> TeamSeasonRatings:
    """Return the 1999 team tables, with the published 1000-resample bootstrap."""
    return build_team_season(1999, _team_games(1999), PriorHistory(_team_games))


@pytest.fixture(scope="module")
def team_2003() -> TeamSeasonRatings:
    """Return the 2003 team tables (the bootstrap is not compared for 2003)."""
    return build_team_season(
        2003, _team_games(2003), PriorHistory(_team_games), bootstrap=BootstrapSettings(1)
    )


@pytest.fixture(scope="module")
def qb_1999() -> QbSeasonRatings:
    """Return the 1999 quarterback tables, with the published bootstrap."""
    return build_qb_season(_qb_games(1999), _team_games(1999))


@pytest.fixture(scope="module")
def qb_2024() -> QbSeasonRatings:
    """Return the 2024 quarterback tables, with the published bootstrap."""
    return build_qb_season(_qb_games(2024), _team_games(2024))


def test_fixtures_match_the_hashes_in_their_manifest() -> None:
    # Arrange
    manifest = json.loads((FIXTURES / "manifest.json").read_text())

    # Act
    mismatched = [
        record["fixture"]
        for record in manifest["files"]
        if hashlib.sha256((FIXTURES / record["fixture"]).read_bytes()).hexdigest()
        != record["sha256"]
    ]

    # Assert
    assert manifest["source_commit"] == "9081bced89b0d51c99e28f80f8f62b9fcd0118e0"
    assert mismatched == []


def test_first_season_team_ratings_match_the_published_file(team_1999: TeamSeasonRatings) -> None:
    # Arrange
    published = _fixture(1999, "ratings")

    # Act
    ratings = team_1999.ratings

    # Assert
    assert_frame_equal(ratings, published, check_exact=True)


def test_first_season_team_rank_ranges_match_the_published_file(
    team_1999: TeamSeasonRatings,
) -> None:
    # Arrange
    published = _fixture(1999, "rating_ranges")

    # Act
    ranges = _rows_like(team_1999.rating_ranges, published, ["team"])

    # Assert
    assert_frame_equal(ranges, published, check_exact=True)


def test_first_season_team_pairs_match_the_published_file(team_1999: TeamSeasonRatings) -> None:
    # Arrange
    published = _fixture(1999, "rating_pairs")

    # Act
    pairs = _rows_like(team_1999.rating_pairs, published, ["team", "other_team"])

    # Assert
    assert_frame_equal(pairs, published, check_exact=True)


def test_first_prior_season_team_ratings_match_the_published_file(
    team_2003: TeamSeasonRatings,
) -> None:
    """2003 rates with 2002's penalties, which rated with 2001's, and so on back to 1999."""
    # Arrange
    published = _fixture(2003, "ratings")

    # Act
    ratings = team_2003.ratings

    # Assert
    assert_frame_equal(ratings, published, check_exact=True)


def test_first_prior_season_rating_history_matches_the_published_file(
    team_2003: TeamSeasonRatings,
) -> None:
    """The first preseason prior shapes 2003's weeks 1-8; its slopes pool 2000-2002."""
    # Arrange
    published = _fixture(2003, "ratings_by_week")

    # Act
    history = team_2003.ratings_by_week

    # Assert
    assert_frame_equal(history, published, check_exact=True)


def test_first_season_qb_ratings_match_the_published_file(qb_1999: QbSeasonRatings) -> None:
    # Arrange
    published = _fixture(1999, "qb_ratings")

    # Act
    ratings = qb_1999.qb_ratings

    # Assert
    assert_frame_equal(ratings, published, check_exact=True)


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("qb_ratings_by_week", ["qb_id", "week"]),
        ("qb_rating_ranges", ["qb_id"]),
        ("qb_rating_pairs", ["qb_id", "other_qb_id"]),
    ],
)
def test_first_season_qb_tables_match_the_published_files(
    qb_1999: QbSeasonRatings, name: str, keys: list[str]
) -> None:
    # Arrange
    published = _fixture(1999, name)

    # Act
    table = _rows_like(getattr(qb_1999, name), published, keys)

    # Assert
    assert_frame_equal(table, published, check_exact=True)


def test_recent_season_qb_ratings_match_the_published_file(qb_2024: QbSeasonRatings) -> None:
    # Arrange
    published = _fixture(2024, "qb_ratings")

    # Act
    ratings = qb_2024.qb_ratings

    # Assert
    assert_frame_equal(
        ratings.drop("qb_epa_per_dropback"),
        published.drop("qb_epa_per_dropback"),
        check_exact=True,
    )
    assert_frame_equal(
        ratings.select("qb_epa_per_dropback"),
        published.select("qb_epa_per_dropback"),
        check_exact=False,
        rel_tol=0.0,
        abs_tol=RAW_RATE_TOLERANCE,
    )


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("qb_ratings_by_week", ["qb_id", "week"]),
        ("qb_rating_ranges", ["qb_id"]),
        ("qb_rating_pairs", ["qb_id", "other_qb_id"]),
    ],
)
def test_recent_season_qb_tables_match_the_published_files(
    qb_2024: QbSeasonRatings, name: str, keys: list[str]
) -> None:
    # Arrange
    published = _fixture(2024, name)

    # Act
    table = _rows_like(getattr(qb_2024, name), published, keys)

    # Assert
    assert_frame_equal(table, published, check_exact=True)


def test_pre_week_team_ratings_are_the_published_history_of_the_week_before() -> None:
    # Arrange
    published = _fixture(2003, "ratings_by_week").filter(pl.col("week") == 4).drop("week")
    history = RatingsHistory(_team_games, _qb_games)

    # Act
    teams = history.pre_week_teams(2003, 5, published.get_column("team").to_list())

    # Assert
    assert_frame_equal(teams.select(published.columns), published, check_exact=True)
