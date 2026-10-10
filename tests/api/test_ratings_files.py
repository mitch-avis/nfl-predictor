"""Tests for the ratings data contract: season discovery, table reads and entity rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from tests.api import ratings_factories as factories

from nfl_predictor.api.ratings import files
from nfl_predictor.api.readers import cache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty table cache."""
    cache.clear()


def test_available_seasons_lists_seasons_with_both_index_tables_newest_first(
    tmp_path: Path,
) -> None:
    """A season shows once its team and quarterback tables both exist."""
    # Arrange
    factories.write_season(tmp_path, 2024)
    factories.write_season(tmp_path, 2025)
    factories.write_season(tmp_path, 2023, kinds=("combined", "ratings"))
    (tmp_path / "notes.parquet").write_bytes(b"")
    (tmp_path / "2022_combined.csv").write_text("team\n", encoding="utf-8")

    # Act
    seasons = files.available_seasons(tmp_path)

    # Assert
    assert seasons == [2025, 2024]


def test_available_seasons_is_empty_without_a_directory(tmp_path: Path) -> None:
    """A data directory the ETL has not written ratings to yet has no seasons."""
    # Act
    seasons = files.available_seasons(tmp_path / "ratings")

    # Assert
    assert seasons == []


def test_read_table_returns_the_kind_for_the_season(tmp_path: Path) -> None:
    """The table comes back as written."""
    # Arrange
    factories.write_season(tmp_path, 2025)

    # Act
    frame = files.read_table(tmp_path, 2025, "ratings")

    # Assert
    assert frame.get_column("team").to_list() == ["BUF", "KC", "NE", "WSH"]


def test_read_table_names_the_missing_file(tmp_path: Path) -> None:
    """A missing table raises with the season and kind, for a clear 404."""
    # Arrange
    factories.write_season(tmp_path, 2025, kinds=("combined",))

    # Act & Assert
    with pytest.raises(
        files.RatingsTableMissingError, match=r"2025_rating_pairs\.parquet"
    ) as raised:
        files.read_table(tmp_path, 2025, "rating_pairs")
    assert (raised.value.season, raised.value.kind) == (2025, "rating_pairs")


@pytest.mark.parametrize(("kind", "expected"), [("combined", True), ("ratings", False)])
def test_has_table_reports_whether_the_file_exists(
    tmp_path: Path, kind: str, *, expected: bool
) -> None:
    """Optional tables are checked before a page asks for them."""
    # Arrange
    factories.write_season(tmp_path, 2025, kinds=("combined",))

    # Act
    present = files.has_table(tmp_path, 2025, kind)

    # Assert
    assert present is expected


def test_entity_rows_keeps_one_team_by_week(tmp_path: Path) -> None:
    """Game logs narrow to one team, in week order."""
    # Arrange
    frame = factories.team_game_logs(2025).sample(fraction=1.0, shuffle=True, seed=3)

    # Act
    rows = files.entity_rows(frame, "team", "WSH")

    # Assert
    assert rows.get_column("team").unique().to_list() == ["WSH"]
    assert rows.get_column("week").to_list() == [1, 2, 3, 4, 5, 6]


def test_entity_rows_raises_for_an_unknown_entity() -> None:
    """An id with no rows is a lookup error the route turns into a 404."""
    # Arrange
    frame = factories.team_game_logs(2025)

    # Act & Assert
    with pytest.raises(files.EntityNotFoundError, match="LV"):
        files.entity_rows(frame, "team", "LV")


def test_entity_rows_raises_when_the_id_column_is_missing() -> None:
    """A table without the id column cannot be narrowed to one entity."""
    # Arrange
    frame = pl.DataFrame({"week": [1]})

    # Act & Assert
    with pytest.raises(files.EntityNotFoundError, match="team"):
        files.entity_rows(frame, "team", "BUF")
