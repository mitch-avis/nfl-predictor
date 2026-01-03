"""Tests for prediction entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from nfl_predictor.ml import ml_model_predict
from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    ScoreModel,
)


class _DummyPreprocessor:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a stable dummy feature matrix for tests."""
        return np.zeros((len(df), 1))


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_columns=["feat1"],
        categorical_columns=[],
        numeric_columns=[],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="feat1",
        feature_end="feat1",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def test_predict_week_writes_output_and_pretty(monkeypatch, tmp_path: Path) -> None:
    """Predict week generates output file and pretty-prints when requested."""
    games_df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2023, 2023],
            "week": [1, 1],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["CCC", "DDD"],
        }
    )

    away_model = cast(xgb.XGBRegressor, object())
    home_model = cast(xgb.XGBRegressor, object())
    predictions = {
        away_model: np.array([10.2, 14.7]),
        home_model: np.array([20.4, 17.2]),
    }

    monkeypatch.setattr(ml_model_predict, "_load_games", lambda _: games_df)
    monkeypatch.setattr(ml_model_predict, "_apply_feature_spec", lambda df, spec: df)

    def fake_predict_xgb(model: Any, features: np.ndarray) -> np.ndarray:
        """Return canned predictions keyed by model identity."""
        _ = features
        return predictions[model]

    monkeypatch.setattr(ml_model_predict, "_predict_xgb", fake_predict_xgb)

    displayed: dict[str, pd.DataFrame] = {}

    def fake_display(df: pd.DataFrame) -> None:
        displayed["df"] = df

    monkeypatch.setattr(ml_model_predict.ml_utils, "display_weekly_predictions", fake_display)

    model = ScoreModel(
        preprocessor=cast(ColumnTransformer, _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        away_model=away_model,
        home_model=home_model,
        target_columns=("away_score", "home_score"),
        market_prob_config=None,
        xgb_params=None,
    )

    output_path = tmp_path / "preds.csv"
    output_df = ml_model_predict.predict_week(
        model,
        tmp_path / "games.csv",
        output_path=output_path,
        pretty_output=True,
        score_rounding="half",
    )

    assert output_path.exists()
    assert "predicted_away_score" in output_df.columns
    assert "predicted_home_score" in output_df.columns
    assert "predicted_winner" in output_df.columns
    assert "confidence_rank" in output_df.columns
    assert displayed["df"].equals(output_df)

    assert np.allclose(output_df["predicted_away_score"].to_numpy(), [10.0, 14.5])
    assert np.allclose(output_df["predicted_home_score"].to_numpy(), [20.5, 17.0])


def test_predict_week_margin_total_adds_quantiles(monkeypatch, tmp_path: Path) -> None:
    """Predict week margin/total adds quantile predictions when models are present."""
    games_df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )

    monkeypatch.setattr(ml_model_predict, "_load_games", lambda _: games_df)
    monkeypatch.setattr(
        ml_model_predict,
        "_predict_margin_total_from_model",
        lambda _model, _df: (np.array([3.5]), np.array([44.2])),
    )
    monkeypatch.setattr(
        ml_model_predict,
        "_predict_margin_total_quantiles_from_model",
        lambda _model, _df: (
            {0.25: np.array([2.0]), 0.75: np.array([5.0])},
            {0.25: np.array([40.0]), 0.75: np.array([48.0])},
        ),
    )
    monkeypatch.setattr(
        ml_model_predict,
        "_derive_scores_from_margin_total",
        lambda _m, _t: (np.array([20.0]), np.array([23.5])),
    )
    monkeypatch.setattr(ml_model_predict, "_predict_home_win_prob", lambda _m, _c: np.array([0.6]))

    model = MarginTotalModel(
        preprocessor=cast(ColumnTransformer, _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        margin_model=cast(xgb.XGBRegressor, object()),
        total_model=cast(xgb.XGBRegressor, object()),
        target_columns=("away_score", "home_score"),
        calibrator=None,
        margin_quantile_models=None,
        total_quantile_models=None,
        quantiles=None,
        market_anchor=False,
        market_prob_config=None,
        xgb_params=None,
        tuned_params=None,
        tuned_cv_summary=None,
    )

    output_path = tmp_path / "mt_preds.csv"
    output_df = ml_model_predict.predict_week_margin_total(
        model,
        tmp_path / "games.csv",
        output_path=output_path,
        pretty_output=False,
    )

    assert output_path.exists()
    assert "predicted_margin_p25" in output_df.columns
    assert "predicted_margin_p75" in output_df.columns
    assert "predicted_total_p25" in output_df.columns
    assert "predicted_total_p75" in output_df.columns
    margin_p25 = pd.to_numeric(output_df["predicted_margin_p25"]).iloc[0]
    total_p75 = pd.to_numeric(output_df["predicted_total_p75"]).iloc[0]
    assert float(margin_p25) == 2.0
    assert float(total_p75) == 48.0


def test_predict_week_blended_uses_market_baseline(monkeypatch, tmp_path: Path) -> None:
    """Predict week blended model uses market baseline predictions."""
    games_df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )

    monkeypatch.setattr(ml_model_predict, "_load_games", lambda _: games_df)
    monkeypatch.setattr(
        ml_model_predict,
        "_predict_margin_total_from_model",
        lambda _model, _df: (np.array([4.0]), np.array([41.0])),
    )
    monkeypatch.setattr(
        ml_model_predict,
        "get_market_baseline",
        lambda _df: (np.array([1.0]), np.array([44.0])),
    )
    monkeypatch.setattr(ml_model_predict, "_predict_home_win_prob", lambda _m, _c: np.array([0.7]))

    class _BlendModel:
        def __init__(self, weights: np.ndarray) -> None:
            """Minimal sklearn-like model with a predict method."""
            self.weights = weights

        def predict(self, x: np.ndarray) -> np.ndarray:
            """Apply a fixed linear transformation."""
            return x @ self.weights

    blend_layer = BlendLayer(
        margin_model=cast(Ridge, _BlendModel(np.array([0.6, 0.4]))),
        total_model=cast(Ridge, _BlendModel(np.array([0.5, 0.5]))),
    )

    team_model = MarginTotalModel(
        preprocessor=cast(ColumnTransformer, _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        margin_model=cast(xgb.XGBRegressor, object()),
        total_model=cast(xgb.XGBRegressor, object()),
        target_columns=("away_score", "home_score"),
        calibrator=None,
        margin_quantile_models=None,
        total_quantile_models=None,
        quantiles=None,
        market_anchor=False,
        market_prob_config=None,
        xgb_params=None,
        tuned_params=None,
        tuned_cv_summary=None,
    )

    blended_model = BlendedMarginTotalModel(
        team_model=team_model,
        market_model=None,
        blend_layer=blend_layer,
        calibrator=None,
        target_columns=("away_score", "home_score"),
        market_prob_config=None,
        xgb_params=None,
        tuned_params=None,
        tuned_cv_summary=None,
    )

    output_df = ml_model_predict.predict_week_blended(
        blended_model,
        tmp_path / "games.csv",
        output_path=None,
        pretty_output=False,
    )

    assert "predicted_margin" in output_df.columns
    assert "predicted_total" in output_df.columns
    assert "home_win_prob" in output_df.columns
