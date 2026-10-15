"""Tests for current season/week detection in scraping utilities."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from nfl_predictor.utils import clock, scraping_utils

if TYPE_CHECKING:
    import pytest


def test_get_current_nfl_week_resolves_pre_kickoff_to_week_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dates immediately before kickoff should resolve to the new season's week 1."""
    # Arrange
    # Today is the Monday before the 2026 opener.
    monkeypatch.setattr(clock, "local_today", lambda: dt.date(2026, 9, 7))

    # Act
    season, week = scraping_utils.get_current_nfl_week()

    # Assert
    assert season == 2026
    assert week == 1


def test_get_current_nfl_week_handles_playoffs(monkeypatch: pytest.MonkeyPatch) -> None:
    """January dates resolve to the prior season and a playoff week (not week 22 blindly)."""
    # Arrange
    # Today is a date in January 2026 (2025 season playoffs).
    monkeypatch.setattr(clock, "local_today", lambda: dt.date(2026, 1, 8))

    # Act
    season, week = scraping_utils.get_current_nfl_week()

    # Assert
    assert season == 2025
    assert week == 19
