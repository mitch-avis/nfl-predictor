"""Tests for the ablation group of this repository's own Elo columns."""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.cli import backtest
from nfl_predictor.ml import walk_forward
from nfl_predictor.utils.polars.finalize import build_final_column_order

GROUP = "own_elo"
WIN_TOTAL_GROUP = "own_elo_wt"
# Every column each own Elo variant publishes, written out by hand so a marker that drifts onto
# another column, or misses one, fails here.
OWN_ELO_FAMILY = sorted(["away_own_elo", "home_own_elo", "own_elo_diff", "own_elo_home_win_prob"])
OWN_WT_ELO_FAMILY = sorted(
    ["away_own_wt_elo", "home_own_wt_elo", "own_wt_elo_diff", "own_wt_elo_home_win_prob"]
)
# The `data/qb_elos.csv` family of the `elo` group, which a dataset can carry beside the own Elo.
QB_ELOS_FAMILY = sorted(
    [
        "away_elo_pre",
        "home_elo_pre",
        "elo_pre_diff",
        "away_qb_elo_pre",
        "home_qb_elo_pre",
        "qb_elo_pre_diff",
        "away_qb_value_pre",
        "home_qb_value_pre",
        "qb_value_pre_diff",
        "away_elo_4wk_trend",
        "home_elo_4wk_trend",
        "elo_4wk_trend_diff",
        "away_qb_elo_4wk_trend",
        "home_qb_elo_4wk_trend",
        "qb_elo_4wk_trend_diff",
        "away_qb_value_4wk_trend",
        "home_qb_value_4wk_trend",
        "qb_value_4wk_trend_diff",
    ]
)


def _both_families() -> list[str]:
    """Return the published schema, which carries both own Elo families."""
    return build_final_column_order()


def test_own_elo_columns_are_the_four_published_names() -> None:
    # Act
    columns = sorted(constants.OWN_ELO_COLUMNS)

    # Assert
    assert columns == OWN_ELO_FAMILY


def test_own_elo_group_resolves_to_exactly_the_own_elo_columns() -> None:
    # Arrange
    columns = _both_families()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(columns, [GROUP])

    # Assert
    assert resolved == OWN_ELO_FAMILY


def test_own_elo_group_is_disjoint_from_every_other_group() -> None:
    # Arrange
    columns = _both_families()
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(columns, [other]))
        & set(OWN_ELO_FAMILY)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_disable_feature_groups_own_elo_drops_exactly_the_own_elo_columns() -> None:
    # Arrange
    frame = pd.DataFrame(columns=_both_families())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups=GROUP)

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert sorted(set(frame.columns) - set(ablated.columns)) == OWN_ELO_FAMILY
    assert ablation.group_columns == OWN_ELO_FAMILY


def test_disable_feature_groups_elo_keeps_every_own_elo_column() -> None:
    # Arrange
    frame = pd.DataFrame(columns=_both_families())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups="elo")

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert ablation.group_columns == QB_ELOS_FAMILY
    assert set(OWN_ELO_FAMILY) <= set(ablated.columns)


def test_own_wt_elo_columns_are_the_four_published_names() -> None:
    # Act
    columns = sorted(constants.OWN_WT_ELO_COLUMNS)

    # Assert
    assert columns == OWN_WT_ELO_FAMILY


def test_own_elo_wt_group_resolves_to_exactly_the_win_total_columns() -> None:
    # Arrange
    columns = _both_families()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(columns, [WIN_TOTAL_GROUP])

    # Assert
    assert resolved == OWN_WT_ELO_FAMILY


def test_own_elo_wt_group_is_disjoint_from_every_other_group() -> None:
    # Arrange
    columns = _both_families()
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != WIN_TOTAL_GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(columns, [other]))
        & set(OWN_WT_ELO_FAMILY)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_a_market_free_run_drops_the_win_total_columns_with_the_elo_group() -> None:
    # Arrange
    frame = pd.DataFrame(columns=_both_families())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups="elo,own_elo_wt")

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert ablation.group_columns == sorted([*QB_ELOS_FAMILY, *OWN_WT_ELO_FAMILY])
    assert set(OWN_ELO_FAMILY) <= set(ablated.columns)
