"""Tests for the canonical margin/total modeling path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor import ml_model


def test_margin_total_round_trip() -> None:
    """Derived scores round-trip back to the original margin/total."""
    # Arrange
    margins = np.array([3.5, -7.0, 0.0, 10.25])
    totals = np.array([45.5, 38.0, 41.0, 52.25])

    # Act
    pred_away, pred_home = ml_model.derive_scores_from_margin_total(margins, totals)

    # Assert
    roundtrip_margin = pred_home - pred_away
    roundtrip_total = pred_home + pred_away

    assert np.allclose(roundtrip_margin, margins)
    assert np.allclose(roundtrip_total, totals)


def test_prediction_output_schema() -> None:
    """Prediction output includes required derived fields for pools."""
    # Arrange
    games_df = pd.DataFrame(
        {
            "game_id": ["2024_01_ARI_ATL", "2024_01_BAL_CIN"],
            "away_abbr": ["ARI", "BAL"],
            "home_abbr": ["ATL", "CIN"],
        }
    )
    pred_away = np.array([21.4, 17.9])
    pred_home = np.array([24.6, 20.2])
    home_win_prob = np.array([0.62, 0.55])

    # Act
    output_df = ml_model.build_prediction_output(games_df, pred_away, pred_home, home_win_prob)

    # Assert
    required_cols = {
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "predicted_margin",
        "home_win_prob",
        "away_win_prob",
        "confidence_rank",
    }
    assert required_cols.issubset(output_df.columns)
    assert np.allclose(output_df["predicted_margin"], np.round(pred_home - pred_away, 1))


@pytest.mark.parametrize(
    ("score_rounding", "expected"),
    [
        pytest.param("int", (21.0, 25.0, 46.0, 4.0), id="int"),
        pytest.param("half", (21.5, 24.5, 46.0, 3.0), id="half"),
        pytest.param("nfl", (21.0, 24.0, 45.0, 3.0), id="nfl"),
    ],
)
def test_prediction_score_rounding_modes(
    score_rounding: str, expected: tuple[float, float, float, float]
) -> None:
    """Optional score rounding snaps output scores and derived total/margin."""
    # Arrange
    games_df = pd.DataFrame(
        {
            "game_id": ["2024_01_ARI_ATL"],
            "away_abbr": ["ARI"],
            "home_abbr": ["ATL"],
        }
    )
    pred_away = np.array([21.4])
    pred_home = np.array([24.6])
    home_win_prob = np.array([0.62])

    # Act
    out = ml_model.build_prediction_output(
        games_df, pred_away, pred_home, home_win_prob, score_rounding=score_rounding
    )

    # Assert
    assert float(out["predicted_away_score_raw"].to_numpy(dtype=float)[0]) == 21.4
    assert float(out["predicted_home_score_raw"].to_numpy(dtype=float)[0]) == 24.6
    assert float(out["predicted_total_raw"].to_numpy(dtype=float)[0]) == 46.0
    assert float(out["predicted_margin_raw"].to_numpy(dtype=float)[0]) == 3.2
    rounded_away = float(out["predicted_away_score"].to_numpy(dtype=float)[0])
    rounded_home = float(out["predicted_home_score"].to_numpy(dtype=float)[0])
    rounded_total = float(out["predicted_total"].to_numpy(dtype=float)[0])
    rounded_margin = float(out["predicted_margin"].to_numpy(dtype=float)[0])
    assert (rounded_away, rounded_home, rounded_total, rounded_margin) == expected
