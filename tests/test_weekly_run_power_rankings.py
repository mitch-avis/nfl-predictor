"""Tests for the power-ranking options the weekly runner passes to the ranking script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import power_rankings, weekly_run


def _options(argv: list[str]) -> power_rankings.RankingOptions:
    """Parse weekly-run arguments and resolve the ranking options they describe."""
    args = weekly_run._build_parser().parse_args(argv)
    return weekly_run._power_ranking_options(args)


def test_weekly_rankings_default_to_the_composite() -> None:
    """With no ranking flags, the weekly run ranks on the adjusted composite."""
    options = _options([])

    assert options.method == "composite"
    assert options.strength_snapshots == power_rankings.DEFAULT_STRENGTH_SNAPSHOTS
    assert options.window_seasons == power_rankings.DEFAULT_RATINGS_WINDOW_SEASONS
    assert options.prior_season_weight == pytest.approx(power_rankings.DEFAULT_PRIOR_SEASON_WEIGHT)
    assert options.target == "margin"
    assert options.include_future is False


def test_weekly_ranking_flags_reach_the_ranking_options() -> None:
    """Every ranking flag set on the weekly run arrives unchanged."""
    options = _options(
        [
            "--power-rankings-method",
            "bradley_terry",
            "--ratings-window-seasons",
            "3",
            "--ratings-prior-season-weight",
            "0.5",
            "--ratings-target",
            "binary",
            "--ratings-include-future",
            "--ratings-min-season",
            "2015",
            "--power-rankings-strength-snapshots",
            "snapshots.csv",
        ]
    )

    assert options.method == "bradley_terry"
    assert options.window_seasons == 3
    assert options.prior_season_weight == pytest.approx(0.5)
    assert options.target == "binary"
    assert options.include_future is True
    assert options.ratings_min_season == 2015
    assert options.strength_snapshots == Path("snapshots.csv")


def test_weekly_legacy_flag_selects_the_franchise_fit() -> None:
    """The legacy flag restores the old all-seasons Bradley-Terry ranking."""
    options = _options(["--legacy-franchise-fit"])

    assert options.method == "bradley_terry"
    assert options.window_seasons == 0
    assert options.prior_season_weight == pytest.approx(1.0)
    assert options.target == "binary"
    assert options.include_future is True


def test_weekly_ranking_flags_are_valid_config_keys() -> None:
    """A config file can set the ranking options the command line can."""
    allowed = weekly_run._allowed_config_keys(weekly_run._build_parser())

    assert {
        "power_rankings_method",
        "power_rankings_strength_snapshots",
        "ratings_window_seasons",
        "ratings_prior_season_weight",
        "ratings_target",
        "ratings_include_future",
        "legacy_franchise_fit",
    } <= allowed


def test_weekly_ranking_options_change_the_report_config() -> None:
    """Changing the ranking method invalidates a reused reports stage."""
    composite = weekly_run._power_rankings_report_config(weekly_run._build_parser().parse_args([]))
    bradley_terry = weekly_run._power_rankings_report_config(
        weekly_run._build_parser().parse_args(["--power-rankings-method", "bradley_terry"])
    )

    assert composite != bradley_terry
    assert composite["power_rankings_method"] == "composite"
