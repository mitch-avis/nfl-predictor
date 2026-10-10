"""Kickoff times from a predictions row, as aware US Eastern datetimes.

nflverse schedules give ``gametime`` in US Eastern time, and the ETL's ``game_datetime`` joins it
to the game date without a time zone, so both read as Eastern.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

EASTERN = ZoneInfo("America/New_York")


def _parse(value: object) -> datetime | None:
    """Return ``value`` as a datetime when it is one or an ISO-8601 string, else ``None``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def kickoff_eastern(row: Mapping[str, object]) -> datetime | None:
    """Return the row's kickoff in US Eastern time, from ``game_datetime`` or date and time."""
    kickoff = _parse(row.get("game_datetime"))
    if kickoff is None:
        day, time_of_day = row.get("date"), row.get("gametime")
        if isinstance(day, date):
            day = day.isoformat()[:10]
        if isinstance(day, str) and isinstance(time_of_day, str):
            kickoff = _parse(f"{day}T{time_of_day}")
    if kickoff is None:
        return None
    if kickoff.tzinfo is None:
        return kickoff.replace(tzinfo=EASTERN)
    return kickoff.astimezone(EASTERN)


def first_kickoff(rows: Iterable[Mapping[str, object]]) -> datetime | None:
    """Return the earliest kickoff among ``rows``: when a week's single-shot pool picks lock."""
    kickoffs = [kickoff for row in rows if (kickoff := kickoff_eastern(row)) is not None]
    return min(kickoffs) if kickoffs else None
