"""Tests for the ratings frames a cached season build carries, and the code they are keyed on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.testing import assert_frame_equal

from nfl_predictor.utils import season_cache

if TYPE_CHECKING:
    from pathlib import Path


def _ratings() -> dict[str, pl.DataFrame]:
    """Return two named ratings frames, in a deliberate order."""
    return {
        "pre_week_teams": pl.DataFrame({"season": [2021], "team": ["AAA"], "team_rating": [1.25]}),
        "ratings": pl.DataFrame({"team": ["BBB", "AAA"], "team_rating": [0.5, -0.5]}),
    }


def _build(ratings: dict[str, pl.DataFrame] | None = None) -> season_cache.SeasonBuild:
    """Return a season build with one game row, one snapshot row and ``ratings``."""
    return season_cache.SeasonBuild(
        games=pl.DataFrame({"game_id": ["g1"], "season": [2021]}),
        snapshots=pl.DataFrame({"season": [2021], "team_abbr": ["AAA"], "adj_srs": [0.25]}),
        ratings=_ratings() if ratings is None else ratings,
    )


def test_a_stored_builds_ratings_frames_read_back_exactly_and_in_order(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is not None
    assert list(loaded.ratings) == list(_ratings())
    for name, frame in _ratings().items():
        assert_frame_equal(loaded.ratings[name], frame, check_exact=True)


def test_a_build_without_ratings_reads_back_without_ratings(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build({}))

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is not None
    assert dict(loaded.ratings) == {}


def test_a_damaged_ratings_frame_makes_the_entry_a_miss(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    damaged = next(path for path in tmp_path.rglob("*.arrow") if "pre_week_teams" in path.name)
    damaged.write_bytes(b"not an arrow file")

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def test_a_missing_ratings_frame_makes_the_entry_a_miss(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    next(path for path in tmp_path.rglob("*.arrow") if "ratings_ratings" in path.name).unlink()

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def _package(root: Path) -> Path:
    """Write a minimal package tree: ETL, utility, ratings and season statistics modules."""
    for relative in (
        "data_collection.py",
        "constants.py",
        "utils/helpers.py",
        "ratings/fits.py",
        "season_stats/tables.py",
        "ml/model.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    return root


def test_the_code_fingerprint_covers_the_ratings_package(tmp_path: Path) -> None:
    # Arrange
    package = _package(tmp_path)
    before = season_cache.etl_code_fingerprint(package)
    (package / "ratings" / "fits.py").write_text("# changed\n", encoding="utf-8")

    # Act
    after = season_cache.etl_code_fingerprint(package)

    # Assert
    assert after != before


def test_the_code_fingerprint_covers_the_season_statistics_package(tmp_path: Path) -> None:
    # Arrange
    package = _package(tmp_path)
    before = season_cache.etl_code_fingerprint(package)
    (package / "season_stats" / "tables.py").write_text("# changed\n", encoding="utf-8")

    # Act
    after = season_cache.etl_code_fingerprint(package)

    # Assert
    assert after != before


def test_the_code_fingerprint_still_ignores_the_model_code(tmp_path: Path) -> None:
    # Arrange
    package = _package(tmp_path)
    before = season_cache.etl_code_fingerprint(package)
    (package / "ml" / "model.py").write_text("# changed\n", encoding="utf-8")

    # Act
    after = season_cache.etl_code_fingerprint(package)

    # Assert
    assert after == before
