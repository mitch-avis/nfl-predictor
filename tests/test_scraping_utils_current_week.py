"""Tests for current season/week detection in scraping utilities."""

from __future__ import annotations

import datetime
from typing import Self

from nfl_predictor.utils import scraping_utils


def test_get_current_nfl_week_resolves_pre_kickoff_to_week_one(monkeypatch) -> None:
    """Dates immediately before kickoff should resolve to the new season's week 1."""

    class _FakeDate(datetime.date):
        @classmethod
        def today(cls) -> Self:
            """Return the Monday before the 2026 opener."""
            return cls(2026, 9, 7)

    monkeypatch.setattr(scraping_utils, "date", _FakeDate)
    season, week = scraping_utils.get_current_nfl_week()

    assert season == 2026
    assert week == 1


def test_get_current_nfl_week_handles_playoffs(monkeypatch) -> None:
    """January dates resolve to the prior season and a playoff week (not week 22 blindly)."""

    class _FakeDate(datetime.date):
        @classmethod
        def today(cls) -> Self:
            """Return a date in January 2026 (2025 season playoffs)."""
            return cls(2026, 1, 8)

    monkeypatch.setattr(scraping_utils, "date", _FakeDate)
    season, week = scraping_utils.get_current_nfl_week()

    assert season == 2025
    assert week == 19
