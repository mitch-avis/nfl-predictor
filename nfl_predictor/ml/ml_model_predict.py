"""Prediction routines for NFL models.

This module hosts the prediction entrypoints that were historically defined in
`nfl_predictor/ml_model.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nfl_predictor.ml import ml_utils
from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    MarginTotalModel,
    ScoreModel,
    _adjust_home_win_prob,
    _apply_feature_spec,
    _build_prediction_output,
    _derive_scores_from_margin_total,
    _load_games,
    _margin_to_home_win_prob,
    _predict_home_win_prob,
    _predict_margin_total_from_model,
    _predict_margin_total_quantiles_from_model,
    _predict_xgb,
    get_market_baseline,
)
from nfl_predictor.utils.logger import log


def predict_week(
    model: ScoreModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Generate weekly predictions and optional confidence ranks."""

    games_df = _load_games(games_path)
    feature_df = _apply_feature_spec(games_df, model.feature_spec)
    log.debug("Prediction feature matrix: %d rows x %d columns", *feature_df.shape)
    x_games = model.preprocessor.transform(feature_df)

    pred_away = _predict_xgb(model.away_model, x_games)
    pred_home = _predict_xgb(model.home_model, x_games)

    home_win_prob = _margin_to_home_win_prob(pred_home - pred_away)
    home_win_prob = _adjust_home_win_prob(
        games_df, home_win_prob, getattr(model, "market_prob_config", None)
    )
    output_df = _build_prediction_output(
        games_df,
        pred_away,
        pred_home,
        home_win_prob,
        score_rounding=score_rounding,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df


def predict_week_margin_total(
    model: MarginTotalModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Generate weekly predictions from a margin/total model."""

    games_df = _load_games(games_path)
    pred_margin, pred_total = _predict_margin_total_from_model(model, games_df)
    margin_quantiles, total_quantiles = _predict_margin_total_quantiles_from_model(model, games_df)
    pred_away, pred_home = _derive_scores_from_margin_total(pred_margin, pred_total)
    home_win_prob = _predict_home_win_prob(pred_margin, model.calibrator)
    home_win_prob = _adjust_home_win_prob(
        games_df, home_win_prob, getattr(model, "market_prob_config", None)
    )
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


def predict_week_blended(
    model: BlendedMarginTotalModel,
    games_path: Path,
    output_path: Optional[Path] = None,
    pretty_output: bool = True,
    score_rounding: str = "none",
) -> pd.DataFrame:
    """Generate weekly predictions from a blended margin/total model."""

    games_df = _load_games(games_path)

    team_margin, team_total = _predict_margin_total_from_model(model.team_model, games_df)
    if model.market_model is None:
        market_margin, market_total = get_market_baseline(games_df)
    else:
        market_margin, market_total = _predict_margin_total_from_model(model.market_model, games_df)

    blended_margin = model.blend_layer.margin_model.predict(
        np.column_stack([team_margin, market_margin])
    )
    blended_total = model.blend_layer.total_model.predict(
        np.column_stack([team_total, market_total])
    )
    pred_away, pred_home = _derive_scores_from_margin_total(blended_margin, blended_total)
    home_win_prob = _predict_home_win_prob(blended_margin, model.calibrator)
    home_win_prob = _adjust_home_win_prob(
        games_df, home_win_prob, getattr(model, "market_prob_config", None)
    )

    output_df = _build_prediction_output(
        games_df,
        pred_away,
        pred_home,
        home_win_prob,
        score_rounding=score_rounding,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        log.info("Saved predictions to %s", output_path)

    if pretty_output:
        ml_utils.display_weekly_predictions(output_df)

    return output_df
