"""Additional tests for training orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    ScoreModel,
)

# pylint: disable=protected-access


class _DummyPreprocessor:
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a stable dummy feature matrix for tests."""
        return np.zeros((len(df), 1))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a stable dummy feature matrix for tests."""
        return np.zeros((len(df), 1))


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_columns=["feat1"],
        categorical_columns=[],
        numeric_columns=["feat1"],
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


def _import_ml_model_training(monkeypatch):
    stub = types.SimpleNamespace(
        train_score_model=lambda **_kwargs: None,
        train_margin_total_model=lambda **_kwargs: None,
        train_blended_margin_total_model=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "nfl_predictor.ml_model", stub)
    monkeypatch.delitem(sys.modules, "nfl_predictor.ml.ml_model_training", raising=False)
    return importlib.import_module("nfl_predictor.ml.ml_model_training")


def test_train_score_model_minimal(monkeypatch) -> None:
    """Score model training works with minimal data and settings."""
    ml_model_training = _import_ml_model_training(monkeypatch)
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "away_score": [10, 14],
            "home_score": [20, 17],
            "feat1": [1.0, 2.0],
            "away_injury_burden_total": [0.2, 0.3],
        }
    )

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training, "_build_feature_spec", lambda *_args, **_kwargs: _feature_spec()
    )
    monkeypatch.setattr(
        ml_model_training, "_apply_feature_spec", lambda frame, _spec: frame[["feat1"]]
    )
    monkeypatch.setattr(
        ml_model_training, "_build_preprocessor", lambda *_args, **_kwargs: _DummyPreprocessor()
    )
    monkeypatch.setattr(
        ml_model_training, "_fit_models", lambda *_args, **_kwargs: ("away_model", "home_model")
    )
    monkeypatch.setattr(ml_model_training, "_predict_xgb", lambda _model, _x: np.array([10.0]))
    monkeypatch.setattr(
        ml_model_training, "_evaluate_predictions", lambda *_args, **_kwargs: {"mae": 1.0}
    )

    model = ml_model_training.train_score_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        include_market=True,
        include_injuries=False,
        max_cardinality_ratio=0.5,
        market_prob_config=None,
    )

    assert isinstance(model, ScoreModel)
    assert model.target_columns == ("away_score", "home_score")


def test_train_score_model_with_report(monkeypatch) -> None:
    """Score model training with report works as expected."""
    ml_model_training = _import_ml_model_training(monkeypatch)
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "away_score": [10, 14],
            "home_score": [20, 17],
            "feat1": [1.0, 2.0],
        }
    )

    model = ScoreModel(
        preprocessor=cast(ColumnTransformer, _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        away_model=cast(xgb.XGBRegressor, object()),
        home_model=cast(xgb.XGBRegressor, object()),
        target_columns=("away_score", "home_score"),
        market_prob_config=None,
        xgb_params={"n_estimators": 1},
    )

    monkeypatch.setattr(ml_model_training, "train_score_model", lambda **_kwargs: model)
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(ml_model_training, "_predict_xgb", lambda _model, _x: np.array([10.0]))
    monkeypatch.setattr(
        ml_model_training,
        "_evaluate_predictions",
        lambda *_args, **_kwargs: {"away_mae": 1.0},
    )

    result = ml_model_training.train_score_model_with_report(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        include_market=True,
        include_injuries=True,
        max_cardinality_ratio=0.5,
        market_prob_config=None,
    )

    assert result.metrics_report["model_kind"] == "score"
    assert result.metrics_report["metrics"]["holdout"] == {"away_mae": 1.0}


def test_train_blended_margin_total_model_with_report(monkeypatch) -> None:
    """Blended margin/total model training with report works as expected."""
    ml_model_training = _import_ml_model_training(monkeypatch)
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022, 2022],
            "week": [1, 2, 1, 2, 1, 2],
            "away_score": [10, 12, 14, 9, 7, 17],
            "home_score": [20, 18, 21, 16, 10, 24],
        }
    )

    class _BlendModel:
        def __init__(self, weights: np.ndarray) -> None:
            self.weights = weights

        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predict by applying weights."""

            return x @ self.weights

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

    model = BlendedMarginTotalModel(
        team_model=team_model,
        market_model=None,
        blend_layer=BlendLayer(
            margin_model=cast(Ridge, _BlendModel(np.array([0.6, 0.4]))),
            total_model=cast(Ridge, _BlendModel(np.array([0.5, 0.5]))),
        ),
        calibrator=None,
        target_columns=("away_score", "home_score"),
        market_prob_config=None,
        xgb_params={"team": {"n_estimators": 1}},
        tuned_params=None,
        tuned_cv_summary={"team": {"cv_splits": 2}},
    )

    monkeypatch.setattr(
        ml_model_training,
        "train_blended_margin_total_model",
        lambda **_kwargs: model,
    )
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training,
        "_predict_margin_total_from_model",
        lambda _model, _df: (np.array([3.0, -2.0]), np.array([40.0, 35.0])),
    )
    monkeypatch.setattr(
        ml_model_training,
        "get_market_baseline",
        lambda _df: (np.array([1.0, -1.0]), np.array([42.0, 33.0])),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_predict_home_win_prob",
        lambda _margin, _cal: np.array([0.6, 0.4]),
    )

    result = ml_model_training.train_blended_margin_total_model_with_report(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        calibration_seasons=1,
        calibration_weeks=0,
        max_cardinality_ratio=0.5,
        win_prob_calibration="none",
        optuna_config=None,
        market_transform=False,
        market_anchor=False,
        market_prob_config=None,
    )

    assert result.metrics_report["model_kind"] == "blend"
    assert result.metrics_report["metrics"]["holdout"] is not None
