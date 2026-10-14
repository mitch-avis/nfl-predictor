"""The weekly run hands the predictions' floor sigma to the betting report."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    from pathlib import Path


def _sigmas_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predictions: pd.DataFrame
) -> list[float | None]:
    """Run the report stage on ``predictions`` and return each sigma the report received."""
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    args = run_config._build_parser().parse_args(["--skip-power-rankings"])
    context = pipeline._RunContext(
        args=args,
        run_id="run",
        run_dir=tmp_path,
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={},
        config_payload={},
    )
    received: list[float | None] = []

    def record(frame: pd.DataFrame, *, floor_sigma: float | None = None) -> pd.DataFrame:
        received.append(floor_sigma)
        return frame

    monkeypatch.setattr(pipeline, "build_betting_report", record)
    pipeline._run_stage4(context, predictions_path, "model-hash", tmp_path / "model.joblib")
    return received


def test_weekly_run_passes_the_predictions_floor_sigma_to_the_betting_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    predictions = pd.DataFrame({"season": [2026, 2026], "week": [3, 3], "floor_sigma": [13.25] * 2})

    # Act
    received = _sigmas_received(tmp_path, monkeypatch, predictions)

    # Assert
    assert received == [13.25]


def test_weekly_run_without_a_floor_sigma_column_passes_no_sigma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    predictions = pd.DataFrame({"season": [2026, 2026], "week": [3, 3]})

    # Act
    received = _sigmas_received(tmp_path, monkeypatch, predictions)

    # Assert
    assert received == [None]


def test_weekly_run_with_several_floor_sigmas_passes_no_sigma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    predictions = pd.DataFrame(
        {"season": [2026, 2026], "week": [3, 3], "floor_sigma": [13.25, 12.0]}
    )

    # Act
    received = _sigmas_received(tmp_path, monkeypatch, predictions)

    # Assert
    assert received == [None]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 0.0], None),
        ([-1.0, -1.0], None),
        ([math.nan, math.nan], None),
        ([13.5, 13.6], None),
        ([13.5, math.nan], 13.5),
    ],
)
def test_report_floor_sigma_keeps_exactly_one_positive_value(
    values: list[float], expected: float | None
) -> None:
    # Arrange
    predictions = pd.DataFrame({"floor_sigma": values})

    # Act
    sigma = pipeline._report_floor_sigma(predictions)

    # Assert
    assert sigma == expected
