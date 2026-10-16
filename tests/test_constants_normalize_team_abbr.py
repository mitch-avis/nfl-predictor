"""Characterization tests for ``constants.normalize_team_abbr``."""

from __future__ import annotations

from nfl_predictor import constants


def test_normalize_team_abbr_none_returns_none() -> None:
    # Act
    result = constants.normalize_team_abbr(None)

    # Assert
    assert result is None


def test_normalize_team_abbr_lowercase_alias_returns_canonical() -> None:
    # Act
    result = constants.normalize_team_abbr("kan")

    # Assert
    assert result == "KC"
