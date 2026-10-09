"""Tests for the ablation group that drops every column derived from the Elo file."""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.cli import backtest
from nfl_predictor.ml import walk_forward
from nfl_predictor.utils.polars.finalize import build_final_column_order

GROUP = "elo"
# Every published column built from `data/qb_elos.csv`: the team and quarterback Elo values and
# the quarterback value, each side and the away-minus-home diff, and their four-week trends.
# Written out by hand so a marker that drifts onto another column, or misses one, fails here.
ELO_FAMILY = sorted(
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


def test_elo_group_resolves_to_exactly_the_elo_family_of_the_schema() -> None:
    """On the published schema the group names the 18 Elo-derived columns and nothing else."""
    # Arrange
    order = build_final_column_order()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(order, [GROUP])

    # Assert
    assert resolved == ELO_FAMILY


def test_elo_group_is_disjoint_from_every_other_group() -> None:
    """No other ablation group drops an Elo column, and the Elo group drops no other group's."""
    # Arrange
    order = build_final_column_order()
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(order, [other])) & set(ELO_FAMILY)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_disable_feature_groups_elo_drops_exactly_the_elo_family() -> None:
    """``--disable-feature-groups elo`` takes the 18 Elo columns out and leaves the rest."""
    # Arrange
    frame = pd.DataFrame(columns=build_final_column_order())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups=GROUP)

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert sorted(set(frame.columns) - set(ablated.columns)) == ELO_FAMILY
    assert ablation.group_columns == ELO_FAMILY
    assert list(ablated.columns) == [c for c in frame.columns if c not in set(ELO_FAMILY)]
