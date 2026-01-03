"""Tests for margin/total training orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predictor.ml.ml_model_core import FeatureSpec, MarketProbConfig, OptunaConfig


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


def test_train_margin_total_model_full_path(monkeypatch) -> None:
    """Margin/total model training works end to end."""
    ml_model_training = _import_ml_model_training(monkeypatch)

    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022, 2022],
            "week": [1, 2, 1, 2, 1, 2],
            "away_score": [10, 12, 14, 9, 7, 17],
            "home_score": [20, 18, 21, 16, 10, 24],
            "feat1": [0.1, 0.2, 0.4, 0.3, 0.5, 0.6],
            "market_home_margin": [1.0, -1.0, 2.0, -2.0, 1.5, -1.5],
            "market_total_line": [40.0, 38.0, 42.0, 39.0, 41.0, 37.0],
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
        ml_model_training,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: ("margin_model", "total_model"),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_quantile_models",
        lambda *_args, **_kwargs: {0.1: "q_model", 0.9: "q_model"},
    )
    monkeypatch.setattr(ml_model_training, "_validate_quantiles", lambda _q: (0.1, 0.9))
    monkeypatch.setattr(
        ml_model_training,
        "_run_optuna_search",
        lambda *_args, **_kwargs: ({"max_depth": 2}, {"cv_splits": 2}),
    )

    def fake_predict_xgb(model: object, x: np.ndarray) -> np.ndarray:
        """Return canned predictions for margin vs total models."""
        if model == "margin_model":
            return np.array([1.0] * len(x))
        return np.array([40.0] * len(x))

    monkeypatch.setattr(ml_model_training, "_predict_xgb", fake_predict_xgb)

    class DummyCalibrator:
        """Dummy calibrator for testing."""

        method = "isotonic"

        def __init__(self) -> None:
            self.model = self

        def predict(self, margin: np.ndarray) -> np.ndarray:
            """Predict dummy probabilities."""
            return np.full_like(margin, 0.6, dtype=float)

    monkeypatch.setattr(
        ml_model_training,
        "_fit_win_prob_calibrator",
        lambda *_args, **_kwargs: DummyCalibrator(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "get_market_baseline",
        lambda _df: (np.zeros(len(_df)), np.zeros(len(_df))),
    )

    optuna_config = OptunaConfig(
        enabled=True,
        timeout_seconds=1,
        n_trials=1,
        cv_splits=2,
        objective="combined_mae",
        early_stopping_rounds=5,
        tree_method="auto",
        device="cpu",
        tune_scope="both",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=1,
    )

    model = ml_model_training.train_margin_total_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        calibration_seasons=1,
        calibration_weeks=0,
        include_market=True,
        include_injuries=True,
        max_cardinality_ratio=0.5,
        win_prob_calibration="isotonic",
        optuna_config=optuna_config,
        market_transform=True,
        market_anchor=True,
        market_prob_config=MarketProbConfig(blend_weight=0.2, clamp_delta=0.1),
    )

    assert model.margin_model == "margin_model"
    assert model.total_model == "total_model"
    assert model.tuned_params == {"max_depth": 2}
