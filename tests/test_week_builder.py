"""Tests for building a games-to-predict file for a future week."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nfl_predictor import week_builder

ROWS: list[dict[str, Any]] = [
    # season, week, teams, scores: a played game, two upcoming ones, and another week.
    {"game_id": "2026_01_DEN_KC", "season": 2026, "week": 1, "away_abbr": "DEN", "home_abbr": "KC",
     "away_score": 17, "home_score": 24, "total_line": 44.0, "home_spread": -3.0,
     "away_spread": 3.0, "home_moneyline": -150, "away_moneyline": 130, "home_rest": 7},
    {"game_id": "2026_03_BUF_NYJ", "season": 2026, "week": 3, "away_abbr": "BUF",
     "home_abbr": "NYJ", "away_score": None, "home_score": None, "total_line": None,
     "home_spread": None, "away_spread": None, "home_moneyline": None, "away_moneyline": None,
     "home_rest": 7},
    {"game_id": "2026_03_GB_MIN", "season": 2026, "week": 3, "away_abbr": "GB", "home_abbr": "MIN",
     "away_score": None, "home_score": None, "total_line": None, "home_spread": None,
     "away_spread": None, "home_moneyline": None, "away_moneyline": None, "home_rest": 6},
    {"game_id": "2025_03_GB_MIN", "season": 2025, "week": 3, "away_abbr": "GB", "home_abbr": "MIN",
     "away_score": None, "home_score": None, "total_line": 40.0, "home_spread": -1.0,
     "away_spread": 1.0, "home_moneyline": -120, "away_moneyline": 100, "home_rest": 7},
]  # fmt: skip


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write an ``all_data_ml.csv`` covering a played week and a future one."""
    (tmp_path / "predict").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(ROWS).write_csv(tmp_path / "all_data_ml.csv")
    return tmp_path


def test_build_writes_only_the_upcoming_games_of_that_week(data_dir: Path) -> None:
    """The file holds the season's unplayed games for the requested week, with every column."""
    result = week_builder.build_week_file(2026, 3, data_dir=data_dir)

    assert result.created is True
    assert result.games == 2
    frame = pl.read_csv(result.path)
    assert result.path.name == "week_03_games_to_predict.csv"
    assert frame["game_id"].to_list() == ["2026_03_BUF_NYJ", "2026_03_GB_MIN"]
    assert frame.columns == pl.read_csv(data_dir / "all_data_ml.csv").columns


def test_build_reports_the_market_columns_that_are_still_empty(data_dir: Path) -> None:
    """A future week has no lines yet; the caller is told which columns are blank."""
    result = week_builder.build_week_file(2026, 3, data_dir=data_dir)

    assert set(result.missing_market_columns) == {
        "total_line",
        "away_spread",
        "home_spread",
        "away_moneyline",
        "home_moneyline",
    }


def test_build_keeps_an_existing_file_unless_asked_to_overwrite(data_dir: Path) -> None:
    """An existing week file is left alone by default and replaced with ``overwrite``."""
    path = data_dir / "predict" / "week_03_games_to_predict.csv"
    path.write_text("hand-made\n", encoding="utf-8")

    kept = week_builder.build_week_file(2026, 3, data_dir=data_dir)
    assert kept.created is False
    assert path.read_text(encoding="utf-8") == "hand-made\n"

    replaced = week_builder.build_week_file(2026, 3, data_dir=data_dir, overwrite=True)
    assert replaced.created is True
    assert pl.read_csv(path).height == 2


def test_build_refuses_a_week_with_no_upcoming_games(data_dir: Path) -> None:
    """Week 1 is already played, so there is nothing to predict."""
    with pytest.raises(ValueError, match="no upcoming games"):
        week_builder.build_week_file(2026, 1, data_dir=data_dir)


def test_build_reports_a_missing_dataset(tmp_path: Path) -> None:
    """Without the ML dataset there is nothing to build from."""
    with pytest.raises(FileNotFoundError, match="all_data_ml.csv"):
        week_builder.build_week_file(2026, 3, data_dir=tmp_path)


def test_available_weeks_lists_the_seasons_unplayed_weeks(data_dir: Path) -> None:
    """The API needs to know which weeks could be generated."""
    assert week_builder.available_weeks(2026, data_dir=data_dir) == [3]
    assert week_builder.available_weeks(2025, data_dir=data_dir) == [3]
    assert week_builder.available_weeks(2024, data_dir=data_dir) == []
    assert week_builder.available_weeks(2026, data_dir=Path("/nonexistent")) == []


def test_main_builds_the_requested_week(data_dir: Path) -> None:
    """The CLI writes the file and reports success."""
    exit_code = week_builder.main(["--season", "2026", "--week", "3", "--data-dir", str(data_dir)])

    assert exit_code == 0
    assert (data_dir / "predict" / "week_03_games_to_predict.csv").is_file()


def test_main_defaults_to_the_current_season_and_week(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without arguments the CLI uses the project's NFL calendar and data directory."""
    monkeypatch.setattr(week_builder.polars_utils, "get_current_nfl_week", lambda: (2026, 3))
    monkeypatch.setattr(week_builder, "DEFAULT_DATA_DIR", data_dir)

    assert week_builder.main([]) == 0
    assert (data_dir / "predict" / "week_03_games_to_predict.csv").is_file()
