"""Tests for deterministic helper functions in ``ml_model_core``.

These tests cover leakage-sensitive split helpers and missing-data summaries.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core as core

if TYPE_CHECKING:
    from pathlib import Path


def test_get_target_columns_returns_away_home_scores() -> None:
    """Returns the canonical away/home score columns when present."""
    # Arrange
    df = pd.DataFrame(
        {
            "away_score": [17.0],
            "home_score": [24.0],
            "spread_line": [-3.5],
        }
    )

    # Act
    columns = core._get_target_columns(df)

    # Assert
    assert columns == ("away_score", "home_score")


def test_get_target_columns_raises_without_both_scores() -> None:
    """Raises when the expected away/home score targets are unavailable."""
    # Arrange
    df = pd.DataFrame({"away_score": [17.0], "home_team": ["KC"]})

    # Act & Assert
    with pytest.raises(ValueError, match="Expected away/home score columns"):
        core._get_target_columns(df)


def test_load_games_reads_csv(tmp_path: Path) -> None:
    """Loads a CSV file into a DataFrame without changing its rows."""
    # Arrange
    csv_path = tmp_path / "games.csv"
    expected = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "away_score": [17, 14],
            "home_score": [24, 20],
        }
    )
    expected.to_csv(csv_path, index=False)

    # Act
    actual = core.load_games(csv_path)

    # Assert
    pdt.assert_frame_equal(actual, expected)


def test_summarize_missing_data_counts_nulls_and_missing_columns() -> None:
    """Counts nulls for present groups and reports missing groups without crashing."""
    # Arrange
    line_columns = list(constants.LINES_COLUMNS)
    record_columns = list(constants.RECORD_FEATURE_COLUMNS)

    df = pd.DataFrame(
        {
            line_columns[0]: [1.5, np.nan],
            line_columns[1]: [41.5, 42.0],
            record_columns[0]: [np.nan, np.nan],
        }
    )

    # Act
    summary = core.summarize_missing_data(df)

    # Assert
    assert summary["total_rows"] == 2

    lines = summary["groups"]["lines"]
    assert lines == {
        "present_columns": 2,
        "missing_columns": len(line_columns) - 2,
        "null_cells": 1,
        "rows_with_any_null": 1,
        "columns_all_null": 0,
    }

    records = summary["groups"]["records"]
    assert records == {
        "present_columns": 1,
        "missing_columns": len(record_columns) - 1,
        "null_cells": 2,
        "rows_with_any_null": 2,
        "columns_all_null": 1,
    }

    for group_name in ("lookahead", "motivation"):
        assert summary["groups"][group_name]["present_columns"] == 0
        assert summary["groups"][group_name]["null_cells"] is None
        assert summary["groups"][group_name]["rows_with_any_null"] is None
        assert summary["groups"][group_name]["columns_all_null"] is None


def test_filter_season_bounds_returns_same_frame_without_season_column() -> None:
    """Leaves frames without a season column unchanged."""
    # Arrange
    df = pd.DataFrame({"week": [1, 2], "value": [10, 20]})

    # Act
    filtered = core.filter_season_bounds(df, min_season=2020, max_season=2024)

    # Assert
    assert filtered is df


def test_filter_season_bounds_filters_min_and_max_seasons() -> None:
    """Filters rows to the requested inclusive season window."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2021, 2022, 2023, 2024],
            "week": [1, 1, 1, 1],
        }
    )

    # Act
    filtered = core.filter_season_bounds(df, min_season=2022, max_season=2023)

    # Assert
    assert filtered["season"].tolist() == [2022, 2023]


def test_split_by_season_partitions_latest_holdout_seasons() -> None:
    """Uses the newest seasons as holdout data and keeps earlier rows for training."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2021, 2022, 2023, 2024],
            "week": [1, 1, 1, 1],
        }
    )

    # Act
    train_df, holdout_df, holdout = core.split_by_season(df, holdout_seasons=2)

    # Assert
    assert holdout == [2023, 2024]
    assert train_df["season"].tolist() == [2021, 2022]
    assert holdout_df["season"].tolist() == [2023, 2024]


def test_split_by_season_zero_holdout_and_invalid_inputs() -> None:
    """Handles zero-holdout mode by keeping every row for training."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "week": [1, 2],
        }
    )

    # Act
    train_df, holdout_df, holdout = core.split_by_season(df, holdout_seasons=0)

    # Assert
    assert holdout == []
    pdt.assert_frame_equal(train_df, df)
    assert holdout_df.empty


@pytest.mark.parametrize(
    ("df", "holdout_seasons", "match"),
    [
        pytest.param(pd.DataFrame({"week": [1]}), 1, "Expected a season column", id="no_season"),
        pytest.param(
            pd.DataFrame({"season": [2022, 2023], "week": [1, 2]}),
            2,
            "Not enough seasons",
            id="too_few_seasons",
        ),
    ],
)
def test_split_by_season_rejects_invalid_inputs(
    df: pd.DataFrame, holdout_seasons: int, match: str
) -> None:
    """Rejects frames without a season column or with too few seasons for the holdout."""
    # Act & Assert
    with pytest.raises(ValueError, match=match):
        core.split_by_season(df, holdout_seasons=holdout_seasons)


def _estimator_reporting(rounds: object) -> SimpleNamespace:
    booster = SimpleNamespace(num_boosted_rounds=lambda: rounds)
    return SimpleNamespace(get_booster=lambda: booster)


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [
        pytest.param(7, 7, id="python-int"),
        pytest.param(np.int64(7), 7, id="numpy-int64"),
        pytest.param(np.int32(7), 7, id="numpy-int32"),
        pytest.param(None, None, id="none"),
        pytest.param(7.0, None, id="python-float"),
        pytest.param(np.float64(7.0), None, id="numpy-float"),
        pytest.param("7", None, id="string"),
        pytest.param(np.array(7), None, id="zero-d-array"),
    ],
)
def test_boosted_rounds_reads_only_integer_round_counts(
    rounds: object, expected: int | None
) -> None:
    """Only a Python or numpy integer is a round count; anything else is unknown."""
    # Arrange
    estimator = _estimator_reporting(rounds)

    # Act
    result = core._boosted_rounds(estimator)

    # Assert
    assert result == expected
