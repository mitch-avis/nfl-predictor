"""Tests for time-series cross-validation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.ml import ml_model_core as core


def test_build_season_week_timepoints_sorted_unique() -> None:
    """Season-week timepoints are built, sorted, and unique."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2023, "2022", None, 2023],
            "week": [1, "18", 5, None],
        }
    )

    # Act
    timepoints = core._build_season_week_timepoints(df)

    # Assert
    assert timepoints == [202218, 202301]


def test_build_blocked_timepoint_folds_ordered_and_deterministic() -> None:
    """Blocked timepoint folds are built in order."""
    # Arrange
    timepoints = list(range(1, 13))

    # Act
    folds = core._build_blocked_timepoint_folds(
        timepoints,
        n_splits=3,
        min_train_timepoints=2,
    )

    # Assert
    expected = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 2, 3, 4, 5, 6], [7, 8, 9]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]),
    ]
    assert folds == expected
    for train, val in folds:
        assert max(train) < min(val)


def test_build_blocked_timepoint_folds_is_deterministic() -> None:
    """Building blocked timepoint folds twice gives the same folds."""
    # Arrange
    timepoints = list(range(1, 13))
    folds = core._build_blocked_timepoint_folds(
        timepoints,
        n_splits=3,
        min_train_timepoints=2,
    )

    # Act
    again = core._build_blocked_timepoint_folds(
        timepoints,
        n_splits=3,
        min_train_timepoints=2,
    )

    # Assert
    assert again == folds


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_splits": 0}, "n_splits must be positive"),
        (
            {"n_splits": 2, "min_train_timepoints": 2},
            "Not enough timepoints to create requested CV folds",
        ),
    ],
)
def test_build_blocked_timepoint_folds_invalid(kwargs: dict[str, int], match: str) -> None:
    """Invalid blocked timepoint fold parameters raise errors."""
    # Act & Assert
    with pytest.raises(ValueError, match=match):
        core._build_blocked_timepoint_folds([1, 2, 3], **kwargs)
