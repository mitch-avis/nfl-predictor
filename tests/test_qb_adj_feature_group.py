"""The published quarterback adjustment from ``data/qb_elos.csv`` and its ``qb_adj`` group.

``qb1_adj``/``qb2_adj`` in the file (``team1`` is home) are the starter's quarterback value minus
the team's usual one, in Elo points, from box-score quarterback values with no betting data. The
ETL publishes them per side and as an away-minus-home ``_diff``; the ``qb_adj`` feature group owns
those three columns, and production leaves the group out by default.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from nfl_predictor import constants, data_collection, ml_model
from nfl_predictor.cli import backtest, options, train
from nfl_predictor.ml import walk_forward
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.polars import loaders, week_rows
from nfl_predictor.utils.polars.finalize import build_final_column_order
from nfl_predictor.weekly_run import config as weekly_config

if TYPE_CHECKING:
    import pytest

GROUP = "qb_adj"
# The three published columns, written out by hand so a marker that drifts onto another column,
# or misses one, fails here.
QB_ADJ_FAMILY = sorted(["away_qb_adj", "home_qb_adj", "qb_adj_diff"])


def _elo_rows(**adjustments: list[float | None]) -> pl.DataFrame:
    """Return ``load_elo_ratings``-shaped rows of AAA at BBB, 2025 weeks 1 and 2."""
    return pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "home_abbr": ["BBB", "BBB"],
            "away_abbr": ["AAA", "AAA"],
            "home_elo_pre": [1510.0, 1512.0],
            "away_elo_pre": [1490.0, 1488.0],
            "home_qb": ["QB B", "QB B2"],
            "away_qb": ["QB A", "QB A"],
            "home_qb_value_pre": [80.0, 60.0],
            "away_qb_value_pre": [70.0, 71.0],
            "home_qb_elo_pre": [1500.0, 1450.0],
            "away_qb_elo_pre": [1480.0, 1481.0],
            **adjustments,
        },
        schema_overrides={"home_qb_adj": pl.Float64, "away_qb_adj": pl.Float64},
    )


def _week(week: int, elo_df: pl.DataFrame) -> week_rows._Week:
    return week_rows._Week(
        2025,
        week,
        pl.DataFrame(),
        pl.DataFrame(),
        data_collection.SeasonInputs(min_season=2024, elo_df=elo_df),
    )


def _game() -> pl.DataFrame:
    return pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})


# Columns and the diff ------------------------------------------------------------------------


def test_qb_adj_columns_name_the_per_team_stat() -> None:
    # Act
    columns = constants.QB_ADJ_COLUMNS

    # Assert
    assert columns == ["qb_adj"]


def test_the_final_schema_carries_each_qb_adj_column_once() -> None:
    # Act
    order = build_final_column_order()

    # Assert
    assert sorted(column for column in order if "qb_adj" in column) == QB_ADJ_FAMILY
    assert all(order.count(column) == 1 for column in QB_ADJ_FAMILY)


def test_the_diff_is_away_minus_home_like_the_qb_value_diff() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "away_qb_adj": [-30.0, None],
            "home_qb_adj": [12.5, 4.0],
            "away_qb_value_pre": [70.0, 71.0],
            "home_qb_value_pre": [80.0, 60.0],
        }
    )

    # Act
    out = polars_utils.calculate_stat_differentials(games, polars_utils.get_stats_for_diff())

    # Assert
    assert out.get_column("qb_adj_diff").to_list() == [-42.5, None]
    assert out.get_column("qb_value_pre_diff").to_list() == [-10.0, 11.0]


# Upcoming weeks ------------------------------------------------------------------------------


def test_the_exact_week_merge_carries_the_files_values_for_that_week() -> None:
    # Arrange
    elo_df = _elo_rows(home_qb_adj=[0.0, -25.0], away_qb_adj=[3.0, 4.0])

    # Act
    merged = week_rows._merge_elo(_game(), _week(2, elo_df))

    # Assert
    assert merged.select("away_qb_adj", "home_qb_adj").row(0) == (4.0, -25.0)


def test_latest_elo_by_team_carries_each_teams_latest_qb_adj() -> None:
    # Arrange
    elo_df = _elo_rows(home_qb_adj=[0.0, -25.0], away_qb_adj=[3.0, 4.0])

    # Act
    latest = loaders.get_latest_elo_by_team(elo_df, 2025)

    # Assert
    assert dict(latest.select("team_abbr", "qb_adj").iter_rows()) == {"AAA": 4.0, "BBB": -25.0}


def test_latest_elo_by_team_without_qb_adj_reads_null() -> None:
    # Arrange
    elo_df = _elo_rows()

    # Act
    latest = loaders.get_latest_elo_by_team(elo_df, 2025)

    # Assert
    assert latest.get_column("qb_adj").to_list() == [None, None]
    assert latest.get_column("qb_value_pre").null_count() == 0


def test_the_fallback_merge_carries_each_teams_latest_qb_adj() -> None:
    # Arrange
    elo_df = _elo_rows(home_qb_adj=[0.0, -25.0], away_qb_adj=[3.0, 4.0])

    # Act
    merged = week_rows._merge_elo(_game(), _week(3, elo_df))

    # Assert
    assert merged.select("away_qb_adj", "home_qb_adj").row(0) == (4.0, -25.0)
    assert merged.select("away_qb_value_pre", "home_qb_value_pre").row(0) == (71.0, 60.0)


def _raw_elo(**adjustments: list[float | None]) -> pl.DataFrame:
    """Return ``load_raw_elo_data``-shaped rows: BUF at KC, then KC at BUF a week later."""
    return pl.DataFrame(
        {
            "date": ["2024-09-01", "2024-09-08"],
            "team1": ["KC", "BUF"],
            "team2": ["BUF", "KC"],
            "qb1": ["P. Mahomes", "J. Allen"],
            "qb2": ["J. Allen", "C. Henne"],
            "qb1_value_pre": [3.0, 2.9],
            "qb2_value_pre": [2.8, 0.5],
            "qbelo1_pre": [200.0, 181.0],
            "qbelo2_pre": [180.0, 90.0],
            **adjustments,
        }
    )


def _future_game(**columns: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_qb": [None],
            "home_qb": [None],
            "away_qb_value_pre": [None],
            "home_qb_value_pre": [None],
            **columns,
        },
        schema_overrides={
            "away_qb": pl.String,
            "home_qb": pl.String,
            "away_qb_value_pre": pl.Float64,
            "home_qb_value_pre": pl.Float64,
            "away_qb_adj": pl.Float64,
            "home_qb_adj": pl.Float64,
        },
    )


def test_fill_future_qb_data_fills_a_null_qb_adj_from_the_latest_starter_row() -> None:
    # Arrange
    games = _future_game(away_qb_adj=[None], home_qb_adj=[None])
    raw = _raw_elo(qb1_adj=[0.0, 1.5], qb2_adj=[-2.0, -40.0])

    # Act
    filled = game_utils.fill_future_qb_data(games, raw)

    # Assert
    row = filled.row(0, named=True)
    assert (row["away_qb"], row["away_qb_value_pre"], row["away_qb_adj"]) == ("J. Allen", 2.9, 1.5)
    assert (row["home_qb"], row["home_qb_value_pre"], row["home_qb_adj"]) == (
        "C. Henne",
        0.5,
        -40.0,
    )


def test_fill_future_qb_data_keeps_a_qb_adj_the_row_already_has() -> None:
    # Arrange
    games = _future_game(away_qb_adj=[7.0], home_qb_adj=[None])
    raw = _raw_elo(qb1_adj=[0.0, 1.5], qb2_adj=[-2.0, -40.0])

    # Act
    filled = game_utils.fill_future_qb_data(games, raw)

    # Assert
    assert filled.select("away_qb_adj", "home_qb_adj").row(0) == (7.0, -40.0)


def test_fill_future_qb_data_without_qb_adj_in_the_file_leaves_it_null() -> None:
    # Arrange
    games = _future_game(away_qb_adj=[None], home_qb_adj=[None])
    raw = _raw_elo()

    # Act
    filled = game_utils.fill_future_qb_data(games, raw)

    # Assert
    assert filled.select("away_qb_adj", "home_qb_adj").row(0) == (None, None)
    assert filled.select("away_qb_value_pre", "home_qb_value_pre").row(0) == (2.9, 0.5)


# The feature group ---------------------------------------------------------------------------


def test_qb_adj_group_resolves_to_exactly_the_three_columns() -> None:
    # Arrange
    order = build_final_column_order()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(order, [GROUP])

    # Assert
    assert resolved == QB_ADJ_FAMILY


def test_qb_adj_group_is_disjoint_from_every_other_group() -> None:
    # Arrange
    order = build_final_column_order()
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(order, [other])) & set(QB_ADJ_FAMILY)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_qb_adj_group_is_not_market_informed() -> None:
    # Act
    market_informed = constants.MARKET_INFORMED_FEATURE_GROUPS

    # Assert
    assert GROUP not in market_informed


def test_production_leaves_the_qb_adj_group_out() -> None:
    # Act
    disabled = constants.PRODUCTION_DISABLED_FEATURE_GROUPS

    # Assert
    assert GROUP in disabled


def test_the_weekly_run_leaves_the_qb_adj_group_out_by_default() -> None:
    # Act
    args = weekly_config.parse_args([])

    # Assert
    assert GROUP in args.disable_feature_groups


def test_train_leaves_the_qb_adj_group_out_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train"])

    # Act
    args = train._parse_args()

    # Assert
    assert GROUP in args.disable_feature_groups


def test_backtest_leaves_the_qb_adj_group_out_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert GROUP in options.parse_feature_groups(args.disable_feature_groups)


def test_a_group_list_without_qb_adj_keeps_the_three_columns() -> None:
    # Arrange
    frame = pd.DataFrame(columns=build_final_column_order())
    args = argparse.Namespace(
        disable_trend_features=False, disable_feature_groups="strength,own_elo,own_elo_wt"
    )

    # Act
    ablated, ablation = backtest._apply_ablations(frame, args)

    # Assert
    assert set(QB_ADJ_FAMILY) <= set(ablated.columns)
    assert not set(QB_ADJ_FAMILY) & set(ablation.group_columns)


def test_the_feature_spec_keeps_the_three_columns_when_the_list_omits_qb_adj() -> None:
    # Arrange
    order = build_final_column_order()
    rng = np.random.default_rng(0)
    games = pd.DataFrame(rng.normal(size=(40, len(order))), columns=order)
    kept = backtest._apply_ablations(
        games,
        argparse.Namespace(
            disable_trend_features=False, disable_feature_groups="strength,own_elo,own_elo_wt"
        ),
    )[0]

    # Act
    spec = ml_model.build_feature_spec(
        kept, ml_model.FeatureSelection(include_market=True, max_cardinality_ratio=0.5)
    )

    # Assert
    assert set(QB_ADJ_FAMILY) <= set(spec.feature_columns)
