"""Tests for training report assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import ml_model_training
from nfl_predictor.ml.ml_model_core import FeatureSpec, MarginTotalModel, OptunaConfig


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


def test_train_margin_total_model_with_report_collects_metrics(monkeypatch) -> None:
    """Training with report collects expected metrics and splits."""

    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022, 2022],
            "week": [1, 2, 1, 2, 1, 2],
            "away_score": [10, 20, 11, 21, 14, 17],
            "home_score": [17, 13, 20, 24, 21, 10],
        }
    )

    margin_sentinel = cast(xgb.XGBRegressor, object())
    total_sentinel = cast(xgb.XGBRegressor, object())

    model = MarginTotalModel(
        preprocessor=cast(ColumnTransformer, _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        margin_model=margin_sentinel,
        total_model=total_sentinel,
        target_columns=("away_score", "home_score"),
        calibrator=None,
        margin_quantile_models=None,
        total_quantile_models=None,
        quantiles=None,
        market_anchor=False,
        market_prob_config=None,
        xgb_params={"n_estimators": 1},
        tuned_params={"max_depth": 2},
        tuned_cv_summary={"cv_splits": 2},
    )

    def fake_train_margin_total_model(**_kwargs: object) -> MarginTotalModel:
        """Return a prebuilt model without training."""

        return model

    def fake_predict_xgb(model_name: Any, features: Any) -> np.ndarray:
        """Return canned predictions for margin vs total models."""

        _ = features
        if model_name is margin_sentinel:
            return np.array([3.0, -4.0])
        return np.array([40.0, 35.0])

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training,
        "train_margin_total_model",
        fake_train_margin_total_model,
    )
    monkeypatch.setattr(ml_model_training, "_predict_xgb", fake_predict_xgb)

    result = ml_model_training.train_margin_total_model_with_report(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        calibration_seasons=1,
        calibration_weeks=0,
        include_market=True,
        max_cardinality_ratio=0.5,
        win_prob_calibration="none",
        optuna_config=OptunaConfig(
            enabled=False,
            timeout_seconds=1,
            n_trials=None,
            cv_splits=2,
            objective="combined_mae",
            early_stopping_rounds=10,
            tree_method="auto",
            device="cpu",
            tune_scope="both",
            storage=None,
            study_name=None,
            best_params_out=None,
            xgb_n_jobs=1,
        ),
        market_transform=False,
        market_anchor=False,
        market_prob_config=None,
    )

    metrics = result.metrics_report["metrics"]["holdout"]
    assert metrics is not None
    assert metrics["margin_mae"] == 3.5
    assert metrics["total_mae"] == 6.5
    assert metrics["winner_accuracy"] == 1.0
    assert "brier" in metrics

    pool_summary = result.metrics_report["pool"]
    assert pool_summary is not None
    assert pool_summary["weeks"] == 2.0

    splits = result.splits
    assert splits["train_seasons"] == [2020]
    assert splits["calibration_seasons"] == [2021]
    assert splits["holdout_seasons"] == [2022]
    assert result.metrics_report["tuning_cv"] == {"cv_splits": 2}
