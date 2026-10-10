"""Tests for the ablation group of the pre-week rating features."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.cli import backtest
from nfl_predictor.ml import walk_forward
from nfl_predictor.utils.polars.finalize import build_final_column_order

GROUP = "ratings"
# Every column the group publishes, written out by hand so a marker that drifts onto another
# column, or misses one, fails here.
_STATS = (
    "ratings_offense",
    "ratings_defense",
    "ratings_special_teams",
    "ratings_team",
    "ratings_sos",
    "ratings_qb",
    "ratings_qb_dropbacks",
)
RATINGS_FAMILY = sorted(
    column for stat in _STATS for column in (f"away_{stat}", f"home_{stat}", f"{stat}_diff")
)


def _constant_names() -> set[str]:
    """Return every name held by a list, tuple, set or mapping of names in ``constants``."""
    names: set[str] = set()
    for value in vars(constants).values():
        items: list[object] = []
        if isinstance(value, list | tuple | set | frozenset):
            items = list(value)
        elif isinstance(value, Mapping):
            items = [*value.keys(), *value.values()]
        names.update(item for item in items if isinstance(item, str))
    return names


def test_ratings_feature_stats_are_the_seven_published_stats() -> None:
    # Act
    stats = tuple(constants.RATINGS_FEATURE_STATS)

    # Assert
    assert stats == _STATS


def test_ratings_columns_join_the_final_schema() -> None:
    # Act
    order = build_final_column_order()

    # Assert
    assert set(RATINGS_FAMILY) <= set(order)


def test_ratings_group_resolves_to_exactly_the_rating_columns() -> None:
    # Arrange
    columns = build_final_column_order()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(columns, [GROUP])

    # Assert
    assert resolved == RATINGS_FAMILY


def test_ratings_group_is_disjoint_from_every_other_group() -> None:
    # Arrange
    columns = [*build_final_column_order(), *constants.OWN_ELO_COLUMNS]
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(columns, [other]))
        & set(RATINGS_FAMILY)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_ratings_markers_catch_no_other_name_in_the_schema_or_the_constants() -> None:
    # Arrange
    names = (_constant_names() | set(build_final_column_order())) - set(RATINGS_FAMILY)
    names -= set(_STATS)
    markers = constants.FEATURE_GROUP_COLUMN_MARKERS[GROUP]

    # Act
    caught = sorted(name for name in names if any(marker in name for marker in markers))

    # Assert
    assert caught == []


def test_disable_feature_groups_ratings_drops_exactly_the_rating_columns() -> None:
    # Arrange
    frame = pd.DataFrame(columns=build_final_column_order())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups=GROUP)

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert sorted(set(frame.columns) - set(ablated.columns)) == RATINGS_FAMILY
    assert ablation.group_columns == RATINGS_FAMILY


def test_disable_feature_groups_strength_keeps_every_rating_column() -> None:
    # Arrange
    frame = pd.DataFrame(columns=build_final_column_order())
    args = argparse.Namespace(disable_trend_features=False, disable_feature_groups="strength")

    # Act
    ablated, _ = backtest._apply_ablations(frame, args)

    # Assert
    assert set(RATINGS_FAMILY) <= set(ablated.columns)


def test_ratings_directory_is_named_for_the_published_tables() -> None:
    # Act
    directory = constants.RATINGS_DIRNAME

    # Assert
    assert directory == "ratings"
