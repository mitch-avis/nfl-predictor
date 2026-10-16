"""Option handling shared by the command-line entry points."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from nfl_predictor import constants

if TYPE_CHECKING:
    from collections.abc import Sequence

# The model kinds a run records in its metadata.
MODEL_KINDS = ("margin_total",)
# Kinds older runs recorded that can no longer be trained or loaded.
RETIRED_MODEL_KINDS = frozenset({"blend", "blended_margin_total"})
RETIRED_BLEND_MESSAGE = (
    "the blend model kind was retired (production submits the deterministic floor of a "
    "margin_total model); train a margin_total model instead"
)


def model_kind(value: str) -> str:
    """Return ``value``, or refuse a retired kind with the reason.

    Raises:
        argparse.ArgumentTypeError: If ``value`` names the retired blend kind.

    """
    if value in RETIRED_MODEL_KINDS:
        raise argparse.ArgumentTypeError(RETIRED_BLEND_MESSAGE)
    return value


def add_model_kind_option(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add ``--model-kind`` with the shared vocabulary (default ``margin_total``)."""
    parser.add_argument(
        "--model-kind",
        type=model_kind,
        choices=MODEL_KINDS,
        default="margin_total",
        help=f"{help_text} (margin_total, the only kind).",
    )


def parse_feature_groups(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature group list into a tuple of stripped, non-empty names."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def add_wf_window_options(parser: argparse.ArgumentParser) -> None:
    """Add the walk-forward evaluation window options; the bare spellings stay as aliases."""
    parser.add_argument(
        "--wf-eval-last-n-seasons",
        "--eval-last-n-seasons",
        dest="eval_last_n_seasons",
        type=int,
        default=3,
        help=(
            "Evaluate the last N seasons in the dataset (regular season only). The count "
            "includes a current season that has no completed week yet."
        ),
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--wf-exclude-incomplete-seasons",
        "--exclude-incomplete-seasons",
        dest="exclude_incomplete_seasons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude seasons whose regular season is incomplete in the dataset (useful when "
            "the current season is partial)."
        ),
    )


# What ``--disable-feature-groups`` means, wherever a command takes it.
FEATURE_GROUPS_HELP = (
    "Comma-separated feature groups to leave out, every column of each (for example "
    "'strength,own_elo,own_elo_wt,qb_adj,pbp'); the list replaces the default rather than adding "
    "to it. "
    "A column belongs to a group when one of the group's markers in "
    "constants.FEATURE_GROUP_COLUMN_MARKERS is part of its name. An unknown group is an error."
)


# The default list of ``--disable-feature-groups``, which depends on whether the run uses game
# lines; every command that takes the option or its config key says so with this text.
FEATURE_GROUPS_DEFAULT_HELP = (
    "Default: "
    f"{','.join(constants.PRODUCTION_DISABLED_FEATURE_GROUPS)} (production's groups), or "
    f"{','.join(constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS)} for a run without game lines "
    "(no market feature and no anchor), which keeps the own Elo with no win-total prior; an "
    "empty value leaves no group out."
)
# What a run without game lines leaves out, for the options that make one.
NO_GAME_LINES_GROUPS_HELP = (
    "A run without game lines leaves out the feature groups "
    f"{','.join(constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS)} by default, so it trains on "
    "no market information and keeps the own Elo with no win-total prior; a list given with "
    "--disable-feature-groups (or disable_feature_groups in the weekly config) replaces it (for "
    "example strength,elo,own_elo,own_elo_wt,qb_adj for no Elo at all)."
)


def is_run_without_game_lines(*, include_market: bool, market_anchor: bool) -> bool:
    """Return whether a run uses no game lines: no market feature and no anchor.

    The market transform only derives market features, so without them it adds no line
    column whatever its setting.
    """
    return not include_market and not market_anchor


def default_disabled_feature_groups(*, without_game_lines: bool) -> tuple[str, ...]:
    """Return the groups a run leaves out when no list is given, by whether it uses game lines."""
    if without_game_lines:
        return constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS
    return constants.PRODUCTION_DISABLED_FEATURE_GROUPS


def resolve_disabled_feature_groups(
    explicit: Sequence[str] | None, *, without_game_lines: bool
) -> tuple[str, ...]:
    """Return the groups a run leaves out: ``explicit`` when given (even empty), else the default.

    ``explicit`` is the list from the command line or a config file, ``None`` when neither gives
    one; the default is ``default_disabled_feature_groups``.
    """
    if explicit is not None:
        return tuple(explicit)
    return default_disabled_feature_groups(without_game_lines=without_game_lines)


def add_feature_group_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--disable-feature-groups``, which ``backtest`` and ``train`` share.

    The value is the raw comma-separated string, ``None`` when the option is not given;
    ``parse_feature_groups`` splits it and ``resolve_disabled_feature_groups`` applies the
    default. An empty value on the command line leaves no group out.
    """
    parser.add_argument(
        "--disable-feature-groups",
        type=str,
        default=None,
        help=f"{FEATURE_GROUPS_HELP} {FEATURE_GROUPS_DEFAULT_HELP}",
    )
