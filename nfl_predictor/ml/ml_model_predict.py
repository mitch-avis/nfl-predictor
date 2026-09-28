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
    model_floor_sigma,
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

    Win probabilities are the deterministic floor of the predicted margins, through the sigma
    the model recorded at its final fit (the constant for a model saved without one); the
    ``floor_sigma`` and ``floor_sigma_fallback`` columns say which. The margin and total
    quantiles, when the model has them, are added as interval columns.
    """
    games_df = _load_games(games_path)
    pred_margin, pred_total = _predict_margin_total_from_model(model, games_df)
    margin_quantiles, total_quantiles = _predict_margin_total_quantiles_from_model(model, games_df)
    pred_away, pred_home = _derive_scores_from_margin_total(pred_margin, pred_total)
    sigma, fallback = model_floor_sigma(model)
    _log_floor_sigma(model, games_df, sigma)
    home_win_prob = _margin_to_home_win_prob(pred_margin, sigma)
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
    output_df["floor_sigma"] = sigma
    output_df["floor_sigma_fallback"] = fallback

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df


def _log_floor_sigma(model: MarginTotalModel, games_df: pd.DataFrame, sigma: float) -> None:
    """Log the sigma the probabilities use, and warn when it was estimated for another week."""
    record = getattr(model, "floor_sigma", None)
    if record is None:
        log.info("The model records no floor sigma; using the constant %.4f.", sigma)
        return
    log.info(
        "Floor sigma %.4f%s for season %d week %d, from %d earlier games.",
        sigma,
        " (the fallback constant)" if record.fallback else "",
        record.season,
        record.week,
        record.pool_games,
    )
    if {"season", "week"}.issubset(games_df.columns):
        weeks = {
            (int(season), int(week))
            for season, week in zip(games_df["season"], games_df["week"], strict=True)
        }
        if weeks != {(record.season, record.week)}:
            log.warning(
                "The model's floor sigma was estimated for season %d week %d, but these games "
                "are in %s; it is used as recorded.",
                record.season,
                record.week,
                sorted(weeks),
            )
