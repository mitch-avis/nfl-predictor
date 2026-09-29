"""Tests for current season/week detection in scraping utilities."""

from __future__ import annotations

import datetime

from nfl_predictor.utils import clock, scraping_utils


def test_get_current_nfl_week_resolves_pre_kickoff_to_week_one(monkeypatch) -> None:
    """Dates immediately before kickoff should resolve to the new season's week 1."""

    # Today is the Monday before the 2026 opener.
    monkeypatch.setattr(clock, "local_today", lambda: datetime.date(2026, 9, 7))
    season, week = scraping_utils.get_current_nfl_week()

    assert season == 2026
    assert week == 1


def test_get_current_nfl_week_handles_playoffs(monkeypatch) -> None:
    """January dates resolve to the prior season and a playoff week (not week 22 blindly)."""

    # Today is a date in January 2026 (2025 season playoffs).
    monkeypatch.setattr(clock, "local_today", lambda: datetime.date(2026, 1, 8))
    season, week = scraping_utils.get_current_nfl_week()

    assert season == 2025
    assert week == 19
