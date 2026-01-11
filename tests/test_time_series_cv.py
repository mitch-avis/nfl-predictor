"""Tests for time-series cross-validation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.ml import ml_model_core as core


def test_build_time_series_folds_ordered_and_deterministic() -> None:
    """Time-series folds are built in order and deterministic manner."""

    seasons = [2021, 2019, 2020, 2022, 2018, 2023]

    folds = core._build_time_series_folds(
        seasons,
        n_splits=2,
        val_window=1,
        min_train_seasons=2,
    )

    expected = [
        ([2018, 2019, 2020, 2021], [2022]),
        ([2018, 2019, 2020, 2021, 2022], [2023]),
    ]
    assert folds == expected
    for train, val in folds:
        assert max(train) < min(val)

    assert (
        core._build_time_series_folds(
            seasons,
            n_splits=2,
            val_window=1,
            min_train_seasons=2,
        )
        == folds
    )


def test_build_time_series_folds_invalid() -> None:
    """Invalid time-series fold parameters raise errors."""

    seasons = [2020, 2021]
    with pytest.raises(ValueError):
        core._build_time_series_folds(seasons, n_splits=0)
    with pytest.raises(ValueError):
        core._build_time_series_folds(seasons, n_splits=1, min_train_seasons=3)


def test_build_season_week_timepoints_sorted_unique() -> None:
    """Season-week timepoints are built, sorted, and unique."""

    df = pd.DataFrame(
        {
            "season": [2023, "2022", None, 2023],
            "week": [1, "18", 5, None],
        }
    )

    timepoints = core._build_season_week_timepoints(df)

    assert timepoints == [202218, 202301]


def test_build_blocked_timepoint_folds_ordered_and_deterministic() -> None:
    """Blocked timepoint folds are built in order and deterministic manner."""

    timepoints = list(range(1, 13))
    folds = core._build_blocked_timepoint_folds(
        timepoints,
        n_splits=3,
        min_train_timepoints=2,
    )

    expected = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 2, 3, 4, 5, 6], [7, 8, 9]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]),
    ]
    assert folds == expected
    for train, val in folds:
        assert max(train) < min(val)

    assert (
        core._build_blocked_timepoint_folds(
            timepoints,
            n_splits=3,
            min_train_timepoints=2,
        )
        == folds
    )


def test_build_blocked_timepoint_folds_invalid() -> None:
    """Invalid blocked timepoint fold parameters raise errors."""

    with pytest.raises(ValueError):
        core._build_blocked_timepoint_folds([1, 2, 3], n_splits=0)
    with pytest.raises(ValueError):
        core._build_blocked_timepoint_folds([1, 2, 3], n_splits=2, min_train_timepoints=2)
