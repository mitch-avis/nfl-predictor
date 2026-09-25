"""Tests for the option helpers shared by the command-line entry points."""

from __future__ import annotations

import pytest

from nfl_predictor.cli import options


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("features", [("features", True, False)]),
        ("anchor", [("anchor", False, True)]),
        ("hybrid", [("hybrid", True, True)]),
        (
            "all",
            [("features", True, False), ("anchor", False, True), ("hybrid", True, True)],
        ),
    ],
)
def test_market_modes_map_each_mode_to_its_market_flags(
    mode: str, expected: list[tuple[str, bool, bool]]
) -> None:
    """Each market mode names whether market features and market anchoring are on."""
    assert options.market_modes(mode) == expected


def test_parse_feature_groups_strips_names_and_drops_empty_entries() -> None:
    """A comma-separated list becomes a tuple of clean names; nothing gives an empty tuple."""
    assert options.parse_feature_groups(" pbp , other ,") == ("pbp", "other")
    assert options.parse_feature_groups(None) == ()
    assert options.parse_feature_groups("") == ()
