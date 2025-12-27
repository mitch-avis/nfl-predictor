from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor import ml_model


def test_margin_total_round_trip() -> None:
    margins = np.array([3.5, -7.0, 0.0, 10.25])
    totals = np.array([45.5, 38.0, 41.0, 52.25])

    pred_away, pred_home = ml_model.derive_scores_from_margin_total(margins, totals)

    roundtrip_margin = pred_home - pred_away
    roundtrip_total = pred_home + pred_away

    assert np.allclose(roundtrip_margin, margins)
    assert np.allclose(roundtrip_total, totals)


def test_prediction_output_schema() -> None:
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
    assert np.allclose(output_df["predicted_total"], np.round(pred_home + pred_away, 1))
