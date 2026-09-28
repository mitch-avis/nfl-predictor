"""Tests for the option helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from nfl_predictor.cli import options


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


@pytest.mark.parametrize(
    ("argv", "dest", "value"),
    [
        (["--wf-eval-last-n-seasons", "6"], "eval_last_n_seasons", 6),
        (["--eval-last-n-seasons", "6"], "eval_last_n_seasons", 6),
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
    """With no options the window is three seasons from week 3."""
    args = _parser_with(options.add_wf_window_options).parse_args([])

    assert (args.eval_last_n_seasons, args.wf_start_week) == (3, 3)
    assert args.exclude_incomplete_seasons is False
