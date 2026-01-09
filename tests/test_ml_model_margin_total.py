"""Tests for the canonical margin/total modeling path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor import ml_model


def test_margin_total_round_trip() -> None:
    """Derived scores round-trip back to the original margin/total."""
    margins = np.array([3.5, -7.0, 0.0, 10.25])
    totals = np.array([45.5, 38.0, 41.0, 52.25])

    pred_away, pred_home = ml_model.derive_scores_from_margin_total(margins, totals)

    roundtrip_margin = pred_home - pred_away
    roundtrip_total = pred_home + pred_away

    assert np.allclose(roundtrip_margin, margins)
    assert np.allclose(roundtrip_total, totals)


def test_prediction_output_schema() -> None:
    """Prediction output includes required derived fields for pools."""
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

    output_df = ml_model.build_prediction_output(games_df, pred_away, pred_home, home_win_prob)

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


def test_prediction_score_rounding_modes() -> None:
    """Optional score rounding snaps output scores and derived total/margin."""
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

    out_int = ml_model.build_prediction_output(
        games_df, pred_away, pred_home, home_win_prob, score_rounding="int"
    )
    int_away = float(out_int["predicted_away_score"].to_numpy(dtype=float)[0])
    int_home = float(out_int["predicted_home_score"].to_numpy(dtype=float)[0])
    int_total = float(out_int["predicted_total"].to_numpy(dtype=float)[0])
    int_margin = float(out_int["predicted_margin"].to_numpy(dtype=float)[0])

    assert int_away == 21.0
    assert int_home == 25.0
    assert int_total == 46.0
    assert int_margin == 4.0

    out_half = ml_model.build_prediction_output(
        games_df, pred_away, pred_home, home_win_prob, score_rounding="half"
    )
    half_away = float(out_half["predicted_away_score"].to_numpy(dtype=float)[0])
    half_home = float(out_half["predicted_home_score"].to_numpy(dtype=float)[0])
    half_total = float(out_half["predicted_total"].to_numpy(dtype=float)[0])
    half_margin = float(out_half["predicted_margin"].to_numpy(dtype=float)[0])

    assert half_away == 21.5
    assert half_home == 24.5
    assert half_total == 46.0
    assert half_margin == 3.0

    out_nfl = ml_model.build_prediction_output(
        games_df, pred_away, pred_home, home_win_prob, score_rounding="nfl"
    )
    nfl_away = float(out_nfl["predicted_away_score"].to_numpy(dtype=float)[0])
    nfl_home = float(out_nfl["predicted_home_score"].to_numpy(dtype=float)[0])
    nfl_total = float(out_nfl["predicted_total"].to_numpy(dtype=float)[0])
    nfl_margin = float(out_nfl["predicted_margin"].to_numpy(dtype=float)[0])

    assert nfl_away == 21.0
    assert nfl_home == 24.0
    assert nfl_total == 45.0
    assert nfl_margin == 3.0
