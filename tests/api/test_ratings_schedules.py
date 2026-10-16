"""Tests for the all-time schedule-strength leaderboard and the glossary's registry payload."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.api.ratings import files, glossary, schedules
from nfl_predictor.api.readers import cache
from tests.api import ratings_factories as factories

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty table cache."""
    cache.clear()


def _write_ratings(ratings_dir: Path, season: int, sos: dict[str, float], games: int) -> None:
    """Write one season's ratings table with the given schedule strengths."""
    ratings_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "team": list(sos),
            "games_played": [games] * len(sos),
            "team_rating": [0.0] * len(sos),
            "sos": list(sos.values()),
        },
        schema_overrides={"games_played": pl.UInt32},
    ).write_parquet(ratings_dir / f"{season}_ratings.parquet")


def test_schedule_ranks_rank_every_completed_team_season_softest_first(tmp_path: Path) -> None:
    """Team-seasons from every completed season share one ranking."""
    # Arrange
    _write_ratings(tmp_path, 2023, {"BUF": 1.5, "KC": -0.5}, games=17)
    _write_ratings(tmp_path, 2024, {"BUF": -2.0, "WSH": 0.25}, games=17)

    # Act
    ranks = schedules.schedule_ranks(tmp_path)

    # Assert
    assert ranks.count == 4
    assert ranks.seasons == [2023, 2024]
    assert ranks.skipped_seasons == []
    assert [(row["season"], row["team"]) for row in ranks.table.rows] == [
        (2024, "BUF"),
        (2023, "KC"),
        (2024, "WSH"),
        (2023, "BUF"),
    ]
    assert [row["softest_rank"] for row in ranks.table.rows] == [1, 2, 3, 4]
    assert [row["hardest_rank"] for row in ranks.table.rows] == [4, 3, 2, 1]


def test_schedule_ranks_skip_a_season_in_progress(tmp_path: Path) -> None:
    """A season whose teams have not played a full schedule is listed as skipped."""
    # Arrange
    _write_ratings(tmp_path, 2025, {"BUF": 1.0, "KC": 0.0}, games=17)
    _write_ratings(tmp_path, 2026, {"BUF": 3.0, "KC": -3.0}, games=5)

    # Act
    ranks = schedules.schedule_ranks(tmp_path)

    # Assert
    assert (ranks.seasons, ranks.skipped_seasons, ranks.count) == ([2025], [2026], 2)


def test_schedule_ranks_skip_a_season_whose_table_vanished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table removed between the folder scan and its read drops out instead of failing."""
    # Arrange
    _write_ratings(tmp_path, 2024, {"BUF": -2.0, "WSH": 0.25}, games=17)
    _write_ratings(tmp_path, 2025, {"BUF": 1.0, "KC": 0.0}, games=17)
    read = files.read_table

    def vanish_2024(ratings_dir: Path, season: int, kind: str) -> pl.DataFrame:
        if season == 2024:
            raise files.RatingsTableMissingError(season, kind)
        return read(ratings_dir, season, kind)

    monkeypatch.setattr(schedules, "read_table", vanish_2024)

    # Act
    ranks = schedules.schedule_ranks(tmp_path)

    # Assert
    assert (ranks.seasons, ranks.skipped_seasons, ranks.count) == ([2025], [], 2)


def test_schedule_ranks_skip_a_season_whose_table_is_unreadable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cut-short season drops out of the ranking, with a warning that names its file."""
    # Arrange
    caplog.set_level(logging.WARNING)
    _write_ratings(tmp_path, 2024, {"BUF": -2.0, "WSH": 0.25}, games=17)
    _write_ratings(tmp_path, 2025, {"BUF": 1.0, "KC": 0.0}, games=17)
    broken = tmp_path / "2024_ratings.parquet"
    broken.write_bytes(broken.read_bytes()[:100])

    # Act
    ranks = schedules.schedule_ranks(tmp_path)

    # Assert
    assert (ranks.seasons, ranks.skipped_seasons, ranks.count) == ([2025], [], 2)
    assert [message for message in caplog.messages if broken.name in message]


def test_schedule_ranks_ignore_other_kinds_and_names(tmp_path: Path) -> None:
    """Only ``{season}_ratings.parquet`` files feed the leaderboard."""
    # Arrange
    factories.write_season(tmp_path, 2024, kinds=("combined", "qb_ratings", "ratings_by_week"))
    _write_ratings(tmp_path, 2025, {"BUF": 1.0}, games=17)

    # Act
    ranks = schedules.schedule_ranks(tmp_path)

    # Assert
    assert ranks.seasons == [2025]


def test_schedule_ranks_describe_every_column(tmp_path: Path) -> None:
    """The leaderboard's columns carry labels and descriptions, the ranks included."""
    # Arrange
    _write_ratings(tmp_path, 2024, {"BUF": 1.0}, games=17)

    # Act
    table = schedules.schedule_ranks(tmp_path).table

    # Assert
    assert table.visible_columns == ["season", "team", "sos", "softest_rank", "hardest_rank"]
    assert all(table.column_metadata[column].description for column in table.visible_columns)
    assert table.column_metadata["sos"].contextual is True
    assert table.column_metadata["season"].kind == "text"


def test_schedule_ranks_without_ratings_are_empty(tmp_path: Path) -> None:
    """No ratings tables means an empty leaderboard, not an error."""
    # Act
    ranks = schedules.schedule_ranks(tmp_path / "ratings")

    # Assert
    assert (ranks.count, ranks.seasons, ranks.table.rows) == (0, [], [])


def test_glossary_registry_lists_both_entities_and_their_metrics() -> None:
    """The glossary payload carries every category and metric the pages show."""
    # Act
    registry = glossary.metric_registry()

    # Assert
    assert set(registry.entities) == {"team", "qb"}
    assert registry.entities["team"].categories[0].name == "Schedule-Adjusted Ratings"
    assert registry.metrics["team_rating"].label == "Team Rating"
    assert registry.metrics["adj_qb_epa_per_dropback"].entity == "qb"


def test_glossary_categories_do_not_describe_the_garbage_time_filter() -> None:
    """No category description points readers to a filter view the app does not have."""
    # Act
    registry = glossary.metric_registry()

    # Assert
    descriptions = [
        category.description
        for entity in registry.entities.values()
        for category in entity.categories
    ]
    assert not [text for text in descriptions if "garbage" in text.lower()]


def test_glossary_registry_leaves_out_the_garbage_time_filter() -> None:
    """The filter's own metrics and prefix belong to a view this app does not have."""
    # Act
    registry = glossary.metric_registry()

    # Assert
    assert not [name for name in registry.metrics if "wp_" in name]
    assert "filtered_" not in [rule.prefix for rule in registry.prefix_rules]
    assert "opp_" in [rule.prefix for rule in registry.prefix_rules]
