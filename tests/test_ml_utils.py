"""Tests for ML utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_predictor.ml import ml_utils
from nfl_predictor.utils import ml_utils as compat_ml_utils


def _capture_logs(messages: list[str]) -> Any:
    def _log(fmt: str, *args: object) -> None:
        messages.append(fmt % args if args else fmt)

    return _log


def test_display_predictions_logs(monkeypatch) -> None:
    """Predictions are logged in expected format."""
    messages: list[str] = []
    monkeypatch.setattr(ml_utils.log, "info", _capture_logs(messages))

    y_pred = np.array([0.7, 0.25])
    x_test = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "away_name": ["Away A", "Away B"],
            "home_name": ["Home A", "Home B"],
        }
    )

    ml_utils.display_predictions(y_pred, x_test)

    assert len(messages) == 2
    assert "Season:" in messages[0]


def test_display_weekly_predictions_empty(monkeypatch) -> None:
    """Empty predictions DataFrame logs appropriate message."""
    messages: list[str] = []
    monkeypatch.setattr(ml_utils.log, "info", _capture_logs(messages))

    ml_utils.display_weekly_predictions(pd.DataFrame())

    assert messages == ["No predictions to display."]


def test_display_weekly_predictions_missing_team_columns(monkeypatch) -> None:
    """Missing team columns log appropriate message."""
    messages: list[str] = []
    monkeypatch.setattr(ml_utils.log, "info", _capture_logs(messages))

    df = pd.DataFrame({"season": [2023], "week": [1]})
    ml_utils.display_weekly_predictions(df)

    assert messages == ["Missing team columns for pretty output."]


def test_display_weekly_predictions_formats(monkeypatch) -> None:
    """Weekly predictions are logged in expected format."""
    messages: list[str] = []
    monkeypatch.setattr(ml_utils.log, "info", _capture_logs(messages))

    df = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [2, 2],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["CCC", "DDD"],
            "predicted_away_score": [17.5, 20.0],
            "predicted_home_score": [21.0, 14.0],
            "home_win_prob": [0.6, 0.4],
            "away_win_prob": [0.4, 0.6],
            "confidence_rank": [2, 1],
        }
    )

    ml_utils.display_weekly_predictions(df)

    assert any("Weekly predictions" in msg for msg in messages)
    assert any(msg.startswith("#") for msg in messages if msg.startswith("#"))


def test_flatten_and_nested_dict_to_df() -> None:
    """Nested dicts are flattened and converted to DataFrame correctly."""
    nested = {"alpha": {"x": 1, "y": 2}, "beta": {"z": 3}}
    flat = ml_utils.flatten_dict(nested)

    assert flat[("alpha", "x")] == 1
    assert flat[("beta", "z")] == 3

    df = ml_utils.nested_dict_to_df(nested)
    assert set(df.columns) == {"x", "y", "z"}
    assert df.loc["alpha", "x"] == 1


def test_utils_ml_utils_facade() -> None:
    """Compatibility ml_utils functions are correctly mapped."""
    assert compat_ml_utils.flatten_dict is ml_utils.flatten_dict
    assert "display_predictions" in dir(compat_ml_utils)
