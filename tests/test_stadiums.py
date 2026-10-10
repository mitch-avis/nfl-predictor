"""Tests for the stadium table and the stadium columns the schedule loader derives from it."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders

if TYPE_CHECKING:
    import pytest

# Every venue id the cached nflverse schedules use, written by
# `.agents/sos_port/decisions/stadium_venues.py` from the schedule cache.
VENUE_FIXTURE = Path(__file__).parent / "fixtures" / "schedule_venue_ids.json"
_TABLE_TEXT_FIELDS = ("name", "city", "state")


def _scheduled_venue_ids() -> list[str]:
    """Return the venue ids of the fixture."""
    payload = json.loads(VENUE_FIXTURE.read_text(encoding="utf-8"))
    return [venue["stadium_id"] for venue in payload["venues"]]


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


def test_scheduled_venue_fixture_covers_the_built_seasons() -> None:
    """Guard the fixture: it lists the venues of every season the ETL builds."""
    # Act
    payload = json.loads(VENUE_FIXTURE.read_text(encoding="utf-8"))

    # Assert
    assert payload["seasons"][0] <= constants.NFLREADPY_MIN_SEASON
    assert payload["seasons"][1] >= 2025
    assert len(payload["venues"]) > len(constants.TEAM_TO_DIVISION)


def test_every_scheduled_venue_id_has_a_stadium_entry() -> None:
    """Each venue id the schedules use has an entry in the stadium table."""
    # Arrange
    venue_ids = _scheduled_venue_ids()

    # Act
    missing = [venue for venue in venue_ids if venue not in constants.STADIUMS]

    # Assert
    assert missing == []


def test_every_stadium_entry_names_its_place_and_elevation() -> None:
    """Each entry has a name, a city, a state or country and an elevation in feet."""
    # Act
    incomplete = [
        venue
        for venue, entry in constants.STADIUMS.items()
        if not all(
            isinstance(entry.get(field), str) and entry[field] for field in _TABLE_TEXT_FIELDS
        )
        or not isinstance(entry.get("elevation_ft"), float)
    ]

    # Assert
    assert incomplete == []


def test_add_stadium_features_gives_every_scheduled_venue_its_elevation() -> None:
    """No scheduled venue falls back to the unknown venue's elevation of 0."""
    # Arrange
    venue_ids = _scheduled_venue_ids()
    games = pl.DataFrame({"stadium_id": venue_ids})

    # Act
    elevations = loaders._add_stadium_features(games).get_column("stadium_elevation").to_list()

    # Assert
    assert [venue for venue, feet in zip(venue_ids, elevations, strict=True) if feet == 0] == []


def test_add_stadium_features_puts_mile_high_stadium_at_denver_altitude() -> None:
    """Denver's home games of 1999-2000 were played a mile up, not at sea level."""
    # Arrange
    games = pl.DataFrame({"stadium_id": ["DEN99"]})

    # Act
    elevation = loaders._add_stadium_features(games).get_column("stadium_elevation")[0]

    # Assert
    assert elevation > 5000


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the messages of the warnings captured so far."""
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


def test_add_stadium_features_names_venues_the_table_lacks_in_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Venues without an entry fall back to 0 ft, so one warning lists each of them once."""
    # Arrange
    caplog.set_level(logging.WARNING)
    games = pl.DataFrame({"stadium_id": ["ZZZ01", "DEN00", "ZZZ02", "ZZZ01"]})

    # Act
    loaders._add_stadium_features(games)

    # Assert
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "['ZZZ01', 'ZZZ02'] (3 games)" in warnings[0]
    assert "DEN00" not in warnings[0]


def test_add_stadium_features_counts_games_without_a_venue_id_in_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A game with no venue id also falls back to 0 ft, so the warning counts it."""
    # Arrange
    caplog.set_level(logging.WARNING)
    games = pl.DataFrame({"stadium_id": ["DEN00", None]})

    # Act
    loaders._add_stadium_features(games)

    # Assert
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "1 game without a venue id" in warnings[0]


def test_add_stadium_features_is_silent_when_every_venue_has_an_entry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Known venues raise no warning."""
    # Arrange
    caplog.set_level(logging.WARNING)
    games = pl.DataFrame({"stadium_id": ["DEN00", "DEN99"]})

    # Act
    loaders._add_stadium_features(games)

    # Assert
    assert _warnings(caplog) == []


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
