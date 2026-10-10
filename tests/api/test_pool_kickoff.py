"""Tests for reading a game's kickoff as US Eastern time."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from nfl_predictor.api.pool.kickoff import kickoff_eastern


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"game_datetime": "2026-10-11T13:00:00.000000"}, "2026-10-11T13:00:00-04:00"),
        ({"game_datetime": "2026-12-27 20:20:00"}, "2026-12-27T20:20:00-05:00"),
        ({"game_datetime": datetime(2026, 10, 11, 13, 0)}, "2026-10-11T13:00:00-04:00"),
        (
            {"game_datetime": datetime(2026, 10, 11, 17, 0, tzinfo=UTC)},
            "2026-10-11T13:00:00-04:00",
        ),
        ({"date": "2026-10-11", "gametime": "09:30"}, "2026-10-11T09:30:00-04:00"),
        ({"date": date(2026, 10, 11), "gametime": "16:25"}, "2026-10-11T16:25:00-04:00"),
        (
            {"game_datetime": "not a time", "date": "2026-10-11", "gametime": "13:00"},
            "2026-10-11T13:00:00-04:00",
        ),
    ],
    ids=[
        "iso-text",
        "space-separated-in-winter",
        "naive-datetime",
        "aware-datetime",
        "date-and-time",
        "date-object-and-time",
        "unreadable-datetime-falls-back",
    ],
)
def test_kickoff_eastern_reads_each_form(row: dict[str, Any], expected: str) -> None:
    """Every way a predictions file can carry the kickoff reads as the same Eastern time."""
    # Act
    kickoff = kickoff_eastern(row)

    # Assert
    assert kickoff is not None
    assert kickoff.isoformat() == expected


@pytest.mark.parametrize(
    "row",
    [{}, {"game_datetime": None, "date": None}, {"date": "2026-10-11"}, {"game_datetime": 7}],
    ids=["nothing", "nulls", "date-without-time", "number"],
)
def test_kickoff_eastern_without_a_kickoff_is_none(row: dict[str, Any]) -> None:
    """A row without a readable kickoff has none."""
    # Act
    kickoff = kickoff_eastern(row)

    # Assert
    assert kickoff is None
