"""Tests for the stadium table and the stadium columns the schedule loader derives from it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _raw_schedule(stadium_id: str) -> pl.DataFrame:
    """Return one raw nflreadpy schedule row played at ``stadium_id``."""
    return pl.DataFrame(
        {
            "game_id": ["2020_01_AAA_BBB"],
            "season": [2020],
            "game_type": ["REG"],
            "week": [1],
            "gameday": ["2020-09-13"],
            "gametime": ["13:00"],
            "stadium_id": [stadium_id],
            "stadium": ["Test Field"],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "away_team": ["AAA"],
            "home_team": ["BBB"],
            "location": ["Home"],
        }
    )


def _fail_load_schedules(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """Fail if nflreadpy is called."""
    msg = "nflreadpy schedule load should not be called"
    raise AssertionError(msg)


def test_load_schedule_locates_a_cached_venue_the_table_gained_later(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cached schedule takes a venue's city and state from the table as it is now."""
    # Arrange
    monkeypatch.setattr(loaders.nfl, "load_schedules", lambda seasons: _raw_schedule("ZZZ00"))
    loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)
    monkeypatch.setitem(
        constants.STADIUMS,
        "ZZZ00",
        {"name": "Test Field", "city": "Testville", "state": "TS", "elevation_ft": 1234.0},
    )
    monkeypatch.setattr(loaders.nfl, "load_schedules", _fail_load_schedules)

    # Act
    schedule = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert schedule.select("stadium_city", "stadium_state", "stadium_elevation").row(0) == (
        "Testville",
        "TS",
        1234.0,
    )
