"""Tests for the option helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable

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


def _parser_with(*builders: Callable[[argparse.ArgumentParser], None]) -> argparse.ArgumentParser:
    """Return a parser with the given option builders applied."""
    parser = argparse.ArgumentParser()
    for builder in builders:
        builder(parser)
    return parser


def test_market_prob_options_accept_the_old_blend_spelling() -> None:
    """``--market-prob-blend`` and ``--market-prob-weight`` set the same value."""
    parser = _parser_with(options.add_market_prob_options)

    assert parser.parse_args([]).market_prob_weight == 0.0
    assert parser.parse_args(["--market-prob-blend", "0.2"]).market_prob_weight == 0.2
    assert parser.parse_args(["--market-prob-weight", "0.3"]).market_prob_weight == 0.3


@pytest.mark.parametrize(
    ("argv", "dest", "value"),
    [
        (["--wf-eval-last-n-seasons", "6"], "eval_last_n_seasons", 6),
        (["--eval-last-n-seasons", "6"], "eval_last_n_seasons", 6),
        (["--wf-calibration-weeks", "2"], "wf_calibration_weeks", 2),
        (["--calibration-weeks", "2"], "wf_calibration_weeks", 2),
        (["--wf-exclude-incomplete-seasons"], "exclude_incomplete_seasons", True),
        (["--exclude-incomplete-seasons"], "exclude_incomplete_seasons", True),
        (["--no-exclude-incomplete-seasons"], "exclude_incomplete_seasons", False),
        (["--disable-feature-groups", "pbp"], "disable_feature_groups", "pbp"),
    ],
)
def test_window_options_accept_both_spellings(argv: list[str], dest: str, value: object) -> None:
    """Each walk-forward window option parses under its new and its old spelling."""
    parser = _parser_with(options.add_wf_window_options, options.add_feature_group_option)

    assert getattr(parser.parse_args(argv), dest) == value


def test_window_option_defaults_match_the_walk_forward_defaults() -> None:
    """With no options the window is three seasons from week 3 with four calibration weeks."""
    args = _parser_with(options.add_wf_window_options).parse_args([])

    assert (args.eval_last_n_seasons, args.wf_start_week, args.wf_calibration_weeks) == (3, 3, 4)
    assert args.exclude_incomplete_seasons is False
