"""Tests for deterministic helper functions in ``ml_model_core``.

These tests cover leakage-sensitive split helpers and missing-data summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core as core


def test_get_target_columns_returns_away_home_scores() -> None:
    """Returns the canonical away/home score columns when present."""
    df = pd.DataFrame(
        {
            "away_score": [17.0],
            "home_score": [24.0],
            "spread_line": [-3.5],
        }
    )

    assert core._get_target_columns(df) == ("away_score", "home_score")


def test_get_target_columns_raises_without_both_scores() -> None:
    """Raises when the expected away/home score targets are unavailable."""
    df = pd.DataFrame({"away_score": [17.0], "home_team": ["KC"]})

    with pytest.raises(ValueError, match="Expected away/home score columns"):
        core._get_target_columns(df)


def test_load_games_reads_csv(tmp_path: Path) -> None:
    """Loads a CSV file into a DataFrame without changing its rows."""
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

    actual = core._load_games(csv_path)

    pdt.assert_frame_equal(actual, expected)


def test_summarize_missing_data_counts_nulls_and_missing_columns() -> None:
    """Counts nulls for present groups and reports missing groups without crashing."""
    line_columns = list(constants.LINES_COLUMNS)
    record_columns = list(constants.RECORD_FEATURE_COLUMNS)

    df = pd.DataFrame(
        {
            line_columns[0]: [1.5, np.nan],
            line_columns[1]: [41.5, 42.0],
            record_columns[0]: [np.nan, np.nan],
        }
    )

    summary = core._summarize_missing_data(df)

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
    df = pd.DataFrame({"week": [1, 2], "value": [10, 20]})

    assert core._filter_season_bounds(df, min_season=2020, max_season=2024) is df


def test_filter_season_bounds_filters_min_and_max_seasons() -> None:
    """Filters rows to the requested inclusive season window."""
    df = pd.DataFrame(
        {
            "season": [2021, 2022, 2023, 2024],
            "week": [1, 1, 1, 1],
        }
    )

    filtered = core._filter_season_bounds(df, min_season=2022, max_season=2023)

    assert filtered["season"].tolist() == [2022, 2023]


def test_split_by_season_partitions_latest_holdout_seasons() -> None:
    """Uses the newest seasons as holdout data and keeps earlier rows for training."""
    df = pd.DataFrame(
        {
            "season": [2021, 2022, 2023, 2024],
            "week": [1, 1, 1, 1],
        }
    )

    train_df, holdout_df, holdout = core._split_by_season(df, holdout_seasons=2)

    assert holdout == [2023, 2024]
    assert train_df["season"].tolist() == [2021, 2022]
    assert holdout_df["season"].tolist() == [2023, 2024]


def test_split_by_season_zero_holdout_and_invalid_inputs() -> None:
    """Handles zero-holdout mode and rejects invalid season inputs."""
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "week": [1, 2],
        }
    )

    train_df, holdout_df, holdout = core._split_by_season(df, holdout_seasons=0)

    assert holdout == []
    pdt.assert_frame_equal(train_df, df)
    assert holdout_df.empty

    with pytest.raises(ValueError, match="Expected a season column"):
        core._split_by_season(pd.DataFrame({"week": [1]}), holdout_seasons=1)

    with pytest.raises(ValueError, match="Not enough seasons"):
        core._split_by_season(df, holdout_seasons=2)


def test_split_train_calibration_holdout_supports_season_and_week_calibration() -> None:
    """Separates holdout, calibration seasons, and in-season calibration weeks."""
    df = pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022, 2023, 2023, 2023, 2023, 2024, 2024],
            "week": [1, 2, 1, 2, 1, 2, 3, 4, 1, 2],
            "away_score": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "home_score": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    )

    (
        train_df,
        calibration_df,
        holdout_df,
        train_seasons,
        calibration_seasons,
        holdout_seasons,
        inseason_calibration_season,
        inseason_calibration_weeks,
    ) = core._split_train_calibration_holdout(
        df,
        holdout_seasons=1,
        calibration_seasons=1,
        calibration_weeks=2,
    )

    assert train_seasons == [2021, 2023]
    assert calibration_seasons == [2022]
    assert holdout_seasons == [2024]
    assert inseason_calibration_season == 2023
    assert inseason_calibration_weeks == [3, 4]

    assert sorted(train_df["season"].unique().tolist()) == [2021, 2023]
    assert train_df.loc[train_df["season"] == 2023, "week"].tolist() == [1, 2]
    assert calibration_df["season"].tolist() == [2022, 2022, 2023, 2023]
    assert calibration_df["week"].tolist() == [1, 2, 3, 4]
    assert holdout_df["season"].tolist() == [2024, 2024]


def test_split_train_calibration_holdout_rejects_invalid_calibration_requests() -> None:
    """Raises for negative counts, missing week data, and too-short calibration windows."""
    df = pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "away_score": [10, 11, 12, 13],
            "home_score": [20, 21, 22, 23],
        }
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        core._split_train_calibration_holdout(
            df,
            holdout_seasons=-1,
            calibration_seasons=0,
            calibration_weeks=0,
        )

    with pytest.raises(ValueError, match="Expected a week column"):
        core._split_train_calibration_holdout(
            df.drop(columns=["week"]),
            holdout_seasons=0,
            calibration_seasons=0,
            calibration_weeks=1,
        )

    with pytest.raises(ValueError, match="Not enough weeks in the training pool"):
        core._split_train_calibration_holdout(
            df,
            holdout_seasons=0,
            calibration_seasons=0,
            calibration_weeks=5,
        )


def _season_weeks_frame(season_weeks: dict[int, int]) -> pd.DataFrame:
    """Build one game row per ``(season, week)`` for weeks ``1..n`` of each season."""
    rows = [
        {"season": season, "week": week, "away_score": 10, "home_score": 20}
        for season, n_weeks in season_weeks.items()
        for week in range(1, n_weeks + 1)
    ]
    return pd.DataFrame(rows)


def _pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Return the sorted ``(season, week)`` pairs present in ``frame``."""
    return sorted({(int(s), int(w)) for s, w in zip(frame["season"], frame["week"], strict=True)})


def test_split_rolls_calibration_window_back_across_the_season_boundary() -> None:
    """Uses the newest completed weeks across seasons when the newest season is short.

    Week 2 of a season has one completed week; four requested weeks are that week plus
    the previous season's last three, and exactly those pairs leave the training rows.
    """
    df = _season_weeks_frame({2024: 18, 2025: 18, 2026: 1})

    (
        train_df,
        calibration_df,
        holdout_df,
        train_seasons,
        calibration_seasons,
        holdout_seasons,
        inseason_calibration_season,
        inseason_calibration_weeks,
    ) = core._split_train_calibration_holdout(
        df,
        holdout_seasons=0,
        calibration_seasons=0,
        calibration_weeks=4,
    )

    window = [(2025, 16), (2025, 17), (2025, 18), (2026, 1)]
    assert _pairs(calibration_df) == window
    assert inseason_calibration_season == 2026
    assert inseason_calibration_weeks == [1]
    assert calibration_seasons == []
    assert holdout_seasons == []
    assert holdout_df.empty
    assert train_seasons == [2024, 2025, 2026]
    assert set(_pairs(train_df)) == set(_pairs(df)) - set(window)
    assert len(train_df) + len(calibration_df) == len(df)


def test_split_keeps_the_window_inside_a_season_with_enough_weeks() -> None:
    """Leaves the split unchanged when the newest season already has enough weeks."""
    df = _season_weeks_frame({2024: 18, 2025: 18, 2026: 4})

    split = core._split_train_calibration_holdout(
        df,
        holdout_seasons=0,
        calibration_seasons=1,
        calibration_weeks=4,
    )
    train_df, calibration_df = split[0], split[1]

    assert split[3] == [2024, 2026]
    assert split[4] == [2025]
    assert split[6] == 2026
    assert split[7] == [1, 2, 3, 4]
    assert _pairs(calibration_df) == [(2025, week) for week in range(1, 19)] + [
        (2026, week) for week in range(1, 5)
    ]
    assert _pairs(train_df) == [(2024, week) for week in range(1, 19)]


def test_split_whole_season_calibration_skips_seasons_the_window_touched() -> None:
    """Chooses calibration seasons only from seasons outside the rolling window."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})

    split = core._split_train_calibration_holdout(
        df,
        holdout_seasons=0,
        calibration_seasons=1,
        calibration_weeks=4,
    )
    train_df, calibration_df = split[0], split[1]

    assert split[4] == [2024]
    assert split[3] == [2023, 2025, 2026]
    window = [(2025, 16), (2025, 17), (2025, 18), (2026, 1)]
    assert _pairs(calibration_df) == [(2024, week) for week in range(1, 19)] + window
    assert _pairs(train_df) == [(2023, week) for week in range(1, 19)] + [
        (2025, week) for week in range(1, 16)
    ]


def test_inseason_calibration_pairs_lists_the_rolling_window() -> None:
    """Lists the in-season window's pairs and ignores whole calibration seasons."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    split = core._split_train_calibration_holdout(
        df,
        holdout_seasons=0,
        calibration_seasons=1,
        calibration_weeks=4,
    )

    assert core._inseason_calibration_pairs(split[1], split[4]) == [
        [2025, 16],
        [2025, 17],
        [2025, 18],
        [2026, 1],
    ]
    assert core._inseason_calibration_pairs(df.iloc[0:0], []) == []
