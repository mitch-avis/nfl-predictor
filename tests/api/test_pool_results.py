"""Tests for reading final scores from the dataset for pool scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.pool.results import final_scores
from nfl_predictor.api.readers import cache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty artifact cache."""
    cache.clear()


def test_final_scores_read_played_games_from_the_dataset(tmp_path: Path) -> None:
    """Played games map to ``(away, home)``; unplayed games are left out."""
    # Arrange
    (tmp_path / "all_data.csv").write_text(
        "game_id,season,week,away_score,home_score,other\n"
        "2026_04_ATL_NO,2026,4,45,24,x\n"
        "2026_05_TB_DAL,2026,5,,,y\n",
        encoding="utf-8",
    )

    # Act
    scores = final_scores(tmp_path)

    # Assert
    assert scores == {"2026_04_ATL_NO": (45, 24)}


def test_final_scores_without_the_dataset_are_empty(tmp_path: Path) -> None:
    """Before the first ETL there is no dataset and so no final score."""
    # Act
    scores = final_scores(tmp_path)

    # Assert
    assert scores == {}


def test_final_scores_without_score_columns_are_empty(tmp_path: Path) -> None:
    """A dataset without score columns yields no scores rather than an error."""
    # Arrange
    (tmp_path / "all_data.csv").write_text("game_id,season\n2026_04_ATL_NO,2026\n", "utf-8")

    # Act
    scores = final_scores(tmp_path)

    # Assert
    assert scores == {}
