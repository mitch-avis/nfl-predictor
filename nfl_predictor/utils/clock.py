"""The calendar date, NFL season and NFL week that season and week detection read."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from nfl_predictor import constants

# Playoff weeks after the regular season: wild card, divisional, conference, Super Bowl.
PLAYOFF_WEEKS = 4


def local_today() -> date:
    """Return today's date in the machine's local time zone."""
    return datetime.now().astimezone().date()


def nfl_season(given_date: date) -> int:
    """Return the NFL season a date belongs to; January and February close the previous one."""
    return given_date.year if given_date.month > constants.SEASON_END_MONTH else given_date.year - 1


def _kickoff_week_tuesday(year: int) -> date:
    """Return the Tuesday that starts a season's week 1 (kickoff follows Labor Day)."""
    september_first = date(year, 9, 1)
    labor_day = september_first + timedelta((7 - september_first.weekday()) % 7)
    kickoff = labor_day + timedelta(days=3)
    return kickoff - timedelta(days=(kickoff.weekday() - 1) % 7)


def nfl_week(given_date: date) -> int:
    """Determine the current NFL week for a given date.

    Args:
        given_date: Date to check

    Returns:
        Week number (1..regular season weeks + playoff weeks).

        For in-season dates in January/February, this can return playoff weeks
        (e.g., 19-22 for seasons with an 18-week regular season).

    """
    season_start = _kickoff_week_tuesday(given_date.year)
    max_week = constants.get_regular_season_weeks(nfl_season(given_date)) + PLAYOFF_WEEKS

    if given_date < season_start:
        previous_season_start = _kickoff_week_tuesday(given_date.year - 1)
        if given_date.month <= constants.SEASON_END_MONTH:
            week_number = ((given_date - previous_season_start).days // 7) + 1
            return max(1, min(week_number, max_week))
        if given_date >= previous_season_start:
            return 1
        return 0

    week_number = ((given_date - season_start).days // 7) + 1
    return max(1, min(week_number, max_week))
