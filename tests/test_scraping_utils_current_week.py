"""Tests for current season/week detection in scraping utilities."""

from __future__ import annotations

import datetime

from nfl_predictor.utils import scraping_utils


def test_get_current_nfl_week_handles_playoffs(monkeypatch) -> None:
    """January dates resolve to the prior season and a playoff week (not week 22 blindly)."""

    class _FakeDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            return cls(2026, 1, 8)

    monkeypatch.setattr(scraping_utils, "date", _FakeDate)
    season, week = scraping_utils.get_current_nfl_week()

    assert season == 2025
    assert week == 19
