"""Prediction routines for NFL models.

This module hosts the prediction entrypoints that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predictor.ml import ml_utils
from nfl_predictor.ml.ml_model_core import (
    MarginTotalModel,
    _build_prediction_output,
    _derive_scores_from_margin_total,
    _load_games,
    _margin_to_home_win_prob,
    _predict_margin_total_from_model,
    _predict_margin_total_quantiles_from_model,
)
from nfl_predictor.utils.logger import log


def predict_week_margin_total(
    model: MarginTotalModel,
    games_path: Path,
    output_path: Path | None = None,
    pretty_output: bool = True,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Generate weekly predictions from a margin/total model.

    Win probabilities are the deterministic floor of the predicted margins; the margin and total
    quantiles, when the model has them, are added as interval columns.
    """
    games_df = _load_games(games_path)
    pred_margin, pred_total = _predict_margin_total_from_model(model, games_df)
    margin_quantiles, total_quantiles = _predict_margin_total_quantiles_from_model(model, games_df)
    pred_away, pred_home = _derive_scores_from_margin_total(pred_margin, pred_total)
    home_win_prob = _margin_to_home_win_prob(pred_margin)
    output_df = _build_prediction_output(
        games_df,
        pred_away,
        pred_home,
        home_win_prob,
        score_rounding=score_rounding,
    )

    for q in sorted(margin_quantiles.keys()):
        output_df[f"predicted_margin_p{int(round(q * 100)):02d}"] = np.round(margin_quantiles[q], 1)
    for q in sorted(total_quantiles.keys()):
        output_df[f"predicted_total_p{int(round(q * 100)):02d}"] = np.round(total_quantiles[q], 1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df
