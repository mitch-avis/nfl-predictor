"""Tests for the local calendar date helper."""

from __future__ import annotations

from datetime import datetime

from nfl_predictor.utils import clock


def test_local_today_is_the_local_calendar_date() -> None:
    """The helper reads the machine's local date, as season and week detection expect."""
    assert clock.local_today() == datetime.now().astimezone().date()
