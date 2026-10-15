"""Tests for margin/total training orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nfl_predictor.ml.ml_model_core import FeatureSpec, OptunaConfig
from tests.fakes import returning

if TYPE_CHECKING:
    import pytest


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


def _import_ml_model_training(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stub = types.SimpleNamespace(
        train_score_model=returning(None),
        train_margin_total_model=returning(None),
        train_blended_margin_total_model=returning(None),
    )
    monkeypatch.setitem(sys.modules, "nfl_predictor.ml_model", stub)
    monkeypatch.delitem(sys.modules, "nfl_predictor.ml.ml_model_training", raising=False)
    return importlib.import_module("nfl_predictor.ml.ml_model_training")


def test_train_margin_total_model_full_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Margin/total model training works end to end."""
    # Arrange
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

    monkeypatch.setattr(ml_model_training, "load_games", returning(df))
    monkeypatch.setattr(
        ml_model_training,
        "build_feature_spec",
        returning(_feature_spec()),
    )

    def select_feature(frame: pd.DataFrame, _spec: FeatureSpec) -> pd.DataFrame:
        return frame[["feat1"]]

    monkeypatch.setattr(ml_model_training, "apply_feature_spec", select_feature)
    monkeypatch.setattr(
        ml_model_training,
        "build_preprocessor",
        returning(_DummyPreprocessor()),
    )
    monkeypatch.setattr(
        ml_model_training,
        "fit_margin_total_models",
        returning(("margin_model", "total_model")),
    )
    monkeypatch.setattr(
        ml_model_training,
        "fit_quantile_models",
        returning({0.1: "q_model", 0.9: "q_model"}),
    )
    monkeypatch.setattr(ml_model_training, "validate_quantiles", returning((0.1, 0.9)))
    monkeypatch.setattr(
        ml_model_training,
        "run_optuna_search",
        returning(
            (
                {"max_depth": 2},
                {"cv_splits": 2},
                {"best_params": {"max_depth": 2}},
            )
        ),
    )

    def fake_predict_xgb(model: object, x: np.ndarray) -> np.ndarray:
        """Return canned predictions for margin vs total models."""
        if model == "margin_model":
            return np.array([1.0] * len(x))
        return np.array([40.0] * len(x))

    monkeypatch.setattr(ml_model_training, "predict_xgb", fake_predict_xgb)

    def zero_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(df)), np.zeros(len(df))

    monkeypatch.setattr(ml_model_training, "get_market_baseline", zero_baseline)

    optuna_config = OptunaConfig(
        enabled=True,
        timeout_seconds=1,
        n_trials=1,
        cv_splits=2,
        objective="combined_mae",
        early_stopping_rounds=5,
        tree_method="auto",
        device="cpu",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=1,
    )

    # Act
    model = ml_model_training.train_margin_total_model(
        ml_model_training.TrainingOptions(
            data_path=Path("dummy.csv"),
            holdout_seasons=1,
            include_market=True,
            max_cardinality_ratio=0.5,
            optuna_config=optuna_config,
            market_transform=True,
            market_anchor=True,
        )
    )

    # Assert
    assert model.margin_model == "margin_model"
    assert model.total_model == "total_model"
    assert model.tuned_params == {"max_depth": 2}
