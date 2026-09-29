"""Tests for training report assembly."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from tests.weekly_fixture import build_fixture

from nfl_predictor.ml import ml_model_training
from nfl_predictor.ml.ml_model_core import FeatureSpec, MarginTotalModel, OptunaConfig

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer

    from nfl_predictor.ml.ml_model_training import TrainingOptions

xgb.set_config(verbosity=0)


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

    margin_sentinel = cast("xgb.XGBRegressor", object())
    total_sentinel = cast("xgb.XGBRegressor", object())

    model = MarginTotalModel(
        preprocessor=cast("ColumnTransformer", _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        margin_model=margin_sentinel,
        total_model=total_sentinel,
        target_columns=("away_score", "home_score"),
        margin_quantile_models=None,
        total_quantile_models=None,
        quantiles=None,
        market_anchor=False,
        xgb_params={"n_estimators": 1},
        tuned_params={"max_depth": 2},
        tuned_cv_summary={"cv_splits": 2},
    )

    def fake_train_margin_total_model(options: TrainingOptions) -> MarginTotalModel:
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
    monkeypatch.setattr(ml_model_training, "predict_xgb", fake_predict_xgb)

    result = ml_model_training.train_margin_total_model_with_report(
        ml_model_training.TrainingOptions(
            data_path=Path("dummy.csv"),
            holdout_seasons=1,
            include_market=True,
            max_cardinality_ratio=0.5,
            optuna_config=OptunaConfig(
                enabled=False,
                timeout_seconds=1,
                n_trials=None,
                cv_splits=2,
                objective="combined_mae",
                early_stopping_rounds=10,
                tree_method="auto",
                device="cpu",
                storage=None,
                study_name=None,
                best_params_out=None,
                xgb_n_jobs=1,
            ),
            market_transform=False,
            market_anchor=False,
        )
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

    assert result.splits == {"train_seasons": [2020, 2021], "holdout_seasons": [2022]}
    assert result.metrics_report["tuning_cv"] == {"cv_splits": 2}


def test_train_margin_total_model_with_report_explains_every_training_row(
    monkeypatch,
) -> None:
    """SHAP explains every row the trees trained on, the newest completed week included."""
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021],
            "week": [1, 2, 1],
            "away_score": [10, 20, 11],
            "home_score": [17, 13, 20],
        }
    )
    model = MarginTotalModel(
        preprocessor=cast("ColumnTransformer", _DummyPreprocessor()),
        feature_spec=_feature_spec(),
        margin_model=cast("xgb.XGBRegressor", object()),
        total_model=cast("xgb.XGBRegressor", object()),
        target_columns=("away_score", "home_score"),
        margin_quantile_models=None,
        total_quantile_models=None,
        quantiles=None,
        market_anchor=False,
        xgb_params={"n_estimators": 1},
        tuned_params=None,
        tuned_cv_summary=None,
    )
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(ml_model_training, "train_margin_total_model", lambda _options: model)
    explained: list[pd.DataFrame] = []

    def fake_report(_model: object, shap_rows: pd.DataFrame | None = None) -> dict[str, Any]:
        """Record the rows the report was asked to explain."""
        assert shap_rows is not None
        explained.append(shap_rows)
        return {"feature_names": []}

    monkeypatch.setattr(
        ml_model_training.feature_importance, "build_feature_importance_report", fake_report
    )

    result = ml_model_training.train_margin_total_model_with_report(
        ml_model_training.TrainingOptions(
            data_path=Path("dummy.csv"),
            holdout_seasons=0,
            include_market=True,
            max_cardinality_ratio=0.5,
            optuna_config=OptunaConfig(
                enabled=False,
                timeout_seconds=1,
                n_trials=None,
                cv_splits=2,
                objective="combined_mae",
                early_stopping_rounds=10,
                tree_method="auto",
                device="cpu",
                storage=None,
                study_name=None,
                best_params_out=None,
                xgb_n_jobs=1,
            ),
            market_transform=False,
            market_anchor=False,
        )
    )

    assert result.splits == {"train_seasons": [2020, 2021], "holdout_seasons": []}
    assert [list(rows[["season", "week"]].itertuples(index=False)) for rows in explained] == [
        [(2020, 1), (2020, 2), (2021, 1)]
    ]


def test_trained_report_ranks_base_features_by_shap_over_the_training_rows(
    tmp_path: Path,
) -> None:
    """A real fit records mean |SHAP| for every base feature, over its training rows only."""
    completed = build_fixture(tmp_path)["completed"]
    games = pd.read_csv(completed)

    result = ml_model_training.train_margin_total_model_with_report(
        ml_model_training.TrainingOptions(
            data_path=completed,
            holdout_seasons=0,
            include_market=True,
            max_cardinality_ratio=0.5,
            optuna_config=OptunaConfig(
                enabled=False,
                timeout_seconds=0,
                n_trials=None,
                cv_splits=2,
                objective="mae",
                early_stopping_rounds=50,
                tree_method=None,
                device="cpu",
                storage=None,
                study_name=None,
                best_params_out=None,
                xgb_n_jobs=1,
            ),
            market_transform=True,
            market_anchor=True,
        )
    )

    report = result.feature_importance
    assert report is not None
    completed_games = games.dropna(subset=["away_score", "home_score"])
    assert report["shap"] == {
        "rows": "train",
        "row_count": len(completed_games),
        "seasons": [2021, 2024],
    }
    base = report["base_features"]
    shap_by_head = {head: base[head]["mean_abs_shap"] for head in ("margin", "total")}
    assert len(base["combined"]["mean_abs_shap"]) == len(base["feature_names"])
    assert base["combined"]["mean_abs_shap"] == pytest.approx(
        [m + t for m, t in zip(shap_by_head["margin"], shap_by_head["total"], strict=True)]
    )
    assert max(base["combined"]["mean_abs_shap"]) > 0.0
