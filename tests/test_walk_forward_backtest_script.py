"""Tests for walk_forward_backtest script helpers."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from scripts import walk_forward_backtest


def test_trend_feature_columns_collects_trend_and_phase_fields() -> None:
    """Trend ablation drops trend and season-phase columns only."""
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "week_in_season_norm": [0.1],
            "season_phase_early": [1],
            "season_phase_mid": [0],
            "season_phase_late": [0],
            "away_elo_4wk_trend": [0.0],
            "home_qb_value_4wk_trend": [0.0],
            "last_5_games_rating_trend_diff": [0.0],
            "away_total_yards": [300.0],
            "home_scoring_margin": [7.0],
        }
    )

    dropped = walk_forward_backtest._trend_feature_columns(df)

    assert set(dropped) == {
        "week_in_season_norm",
        "season_phase_early",
        "season_phase_mid",
        "season_phase_late",
        "away_elo_4wk_trend",
        "home_qb_value_4wk_trend",
        "last_5_games_rating_trend_diff",
    }


def test_disable_feature_groups_arg_parses_comma_separated_list() -> None:
    """`--disable-feature-groups pbp,other` parses into the expected stripped tuple."""
    old_argv = sys.argv
    try:
        sys.argv = [
            "walk_forward_backtest.py",
            "--disable-feature-groups",
            "pbp,other",
        ]
        args = walk_forward_backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.disable_feature_groups == "pbp,other"
    assert walk_forward_backtest._parse_feature_groups(args.disable_feature_groups) == (
        "pbp",
        "other",
    )


def test_parse_feature_groups_strips_whitespace_and_drops_empty_entries() -> None:
    """Parsing tolerates surrounding whitespace, empty segments, and a missing/empty value."""
    assert walk_forward_backtest._parse_feature_groups(" pbp , other ,") == ("pbp", "other")
    assert walk_forward_backtest._parse_feature_groups(None) == ()
    assert walk_forward_backtest._parse_feature_groups("") == ()


def test_disable_feature_groups_drops_resolved_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved feature-group columns are the ones dropped from the loaded dataframe."""
    monkeypatch.setattr(
        walk_forward.constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"pbp": ("epa_per_play",)},
    )
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "away_epa_per_play": [0.1],
            "home_epa_per_play": [0.2],
            "epa_per_play_diff": [-0.1],
            "away_total_yards": [300.0],
        }
    )
    groups = walk_forward_backtest._parse_feature_groups("pbp")

    dropped = walk_forward.resolve_feature_group_columns(list(df.columns), groups)
    remaining = df.drop(columns=dropped)

    assert dropped == [
        "away_epa_per_play",
        "epa_per_play_diff",
        "home_epa_per_play",
    ]
    assert set(remaining.columns) == {"game_id", "season", "week", "away_total_yards"}


def test_disable_feature_groups_unknown_group_raises_from_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown feature group name raises a clear error, not an obscure library traceback."""
    monkeypatch.setattr(walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})
    groups = walk_forward_backtest._parse_feature_groups("not_a_real_group")

    with pytest.raises(ValueError, match="not_a_real_group"):
        walk_forward.resolve_feature_group_columns(["away_epa"], groups)
