"""The page payloads over cuts of real published tables from the ``nfl-sos-ratings`` files.

The parity fixtures under ``tests/fixtures/`` are cut from the published 2024 tables (provenance
and hashes beside them). They keep every row, but only some of the columns of the season-statistics
tables (the team and quarterback tables and their game logs); the rating tables are whole. Written
into a ratings folder under the file names the pages read, with Washington recoded to this
repository's ``WSH``, every column a fixture holds must reach the pages described. The columns the
fixtures leave out are checked where the season tables are built
(``nfl_predictor.season_stats.season.finalize_table``), which rejects any column the metric
registry cannot describe.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.api.ratings import files, sections, tables
from nfl_predictor.api.readers import cache

if TYPE_CHECKING:
    from nfl_predictor.api.schemas.ratings import EntityKind

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SEASON_STATS = FIXTURES / "sos_season_stats_parity"
RATINGS = FIXTURES / "sos_ratings_parity"
PUBLISHED = {
    "combined": SEASON_STATS / "2024_season_combined.parquet",
    "qb_combined": SEASON_STATS / "2024_season_qb_combined.parquet",
    "team_game_logs": SEASON_STATS / "2024_season_team_game_logs.parquet",
    "qb_game_logs": SEASON_STATS / "2024_season_qb_game_logs.parquet",
    "ratings": SEASON_STATS / "2024_season_ratings.parquet",
    "qb_rating_ranges": RATINGS / "2024_qb_rating_ranges.parquet",
    "qb_rating_pairs": RATINGS / "2024_qb_rating_pairs.parquet",
    "qb_ratings_by_week": RATINGS / "2024_qb_ratings_by_week.parquet",
}
TEAM_CODE_COLUMNS = ("team", "opponent_team")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty table cache."""
    cache.clear()


@pytest.fixture
def ratings_dir(tmp_path: Path) -> Path:
    """Return a ratings folder holding the published 2024 tables, Washington as ``WSH``."""
    for kind, source in PUBLISHED.items():
        frame = pl.read_parquet(source)
        frame = frame.with_columns(
            pl.col(column).replace("WAS", "WSH")
            for column in TEAM_CODE_COLUMNS
            if column in frame.columns
        )
        frame.write_parquet(files.table_path(tmp_path, 2024, kind))
    return tmp_path


def test_published_seasons_are_listed(ratings_dir: Path) -> None:
    """The published season has both index tables."""
    # Act
    seasons = files.available_seasons(ratings_dir)

    # Assert
    assert seasons == [2024]


@pytest.mark.parametrize(("kind", "table_kind"), [("teams", "combined"), ("qbs", "qb_combined")])
def test_published_index_tables_keep_every_fixture_column(
    ratings_dir: Path, kind: EntityKind, table_kind: str
) -> None:
    """Every column the team and quarterback table fixtures hold reaches the page."""
    # Arrange
    frame = files.read_table(ratings_dir, 2024, table_kind)

    # Act
    table = tables.index_table(kind, frame)

    # Assert
    assert sorted(table.visible_columns) == sorted(frame.columns)
    assert len(table.rows) == frame.height


def test_published_team_game_logs_keep_every_fixture_column(ratings_dir: Path) -> None:
    """Every column of a team's game log fixture reaches the page, with its opponents' ratings."""
    # Arrange
    logs = files.entity_rows(files.read_table(ratings_dir, 2024, "team_game_logs"), "team", "WSH")
    ratings = files.read_table(ratings_dir, 2024, "ratings")

    # Act
    table = sections.game_log_table("teams", logs, ratings)

    # Assert
    assert set(logs.columns) <= set(table.visible_columns)
    assert len(table.rows) == 17
    assert all(row["opp_team_rating"] is not None for row in table.rows)


def test_published_qb_sections_keep_every_column(ratings_dir: Path) -> None:
    """The quarterback rank ranges, pairs and history, whole published tables, reach the page."""
    # Arrange
    ranges = files.read_table(ratings_dir, 2024, "qb_rating_ranges")
    qb_id = str(ranges.get_column("qb_id")[0])
    pairs = files.entity_rows(
        files.read_table(ratings_dir, 2024, "qb_rating_pairs"), "qb_id", qb_id
    )
    history = files.entity_rows(
        files.read_table(ratings_dir, 2024, "qb_ratings_by_week"), "qb_id", qb_id
    )

    # Act
    payloads = (
        sections.rank_range_table("qbs", ranges),
        sections.pair_table("qbs", pairs),
        sections.rating_history_table("qbs", history),
    )

    # Assert
    for payload, frame in zip(payloads, (ranges, pairs, history), strict=True):
        assert sorted(payload.visible_columns) == sorted(frame.columns)
