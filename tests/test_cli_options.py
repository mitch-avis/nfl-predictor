"""Tests for the option helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.cli import options

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" pbp , other ,", ("pbp", "other")),
        (None, ()),
        ("", ()),
    ],
)
def test_parse_feature_groups_strips_names_and_drops_empty_entries(
    raw: str | None, expected: tuple[str, ...]
) -> None:
    """A comma-separated list becomes a tuple of clean names; nothing gives an empty tuple."""
    # Act
    groups = options.parse_feature_groups(raw)

    # Assert
    assert groups == expected


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
    # Arrange
    parser = _parser_with(options.add_wf_window_options, options.add_feature_group_option)

    # Act
    args = parser.parse_args(argv)

    # Assert
    assert getattr(args, dest) == value


def test_window_option_defaults_match_the_walk_forward_defaults() -> None:
    """With no options the window is three seasons from week 3."""
    # Arrange
    parser = _parser_with(options.add_wf_window_options)

    # Act
    args = parser.parse_args([])

    # Assert
    assert (args.eval_last_n_seasons, args.wf_start_week) == (3, 3)
    assert args.exclude_incomplete_seasons is False
