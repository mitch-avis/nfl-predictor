"""Additional branch-coverage tests for ``ml_model_training``.

These tests target deterministic orchestration branches that do not require real model fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from nfl_predictor.ml import ml_model_training
from nfl_predictor.ml.ml_model_core import (
    DEFAULT_XGB_PARAMS,
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    MarketProbConfig,
    OptunaConfig,
    ScoreModel,
)

xgb.set_config(verbosity=0)


class _DummyPreprocessor:
    """Return stable feature matrices for orchestration tests."""

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a deterministic feature matrix for fitting."""
        return np.zeros((len(df), 1), dtype=float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a deterministic feature matrix for inference."""
        return np.zeros((len(df), 1), dtype=float)


class _BlendModel:
    """Simple linear predictor used to stub blend layers."""

    def __init__(self, weights: np.ndarray) -> None:
        """Store fixed weights for later predictions."""
        self.weights = weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the fixed linear weights to the input matrix."""
        return x @ self.weights


def _feature_spec(*, high_cardinality: bool = False) -> FeatureSpec:
    """Build a stable feature specification for training tests."""
    return FeatureSpec(
        feature_columns=["feat1"],
        categorical_columns=[],
        numeric_columns=["feat1"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=["team_code"] if high_cardinality else [],
        feature_start="feat1",
        feature_end="feat1",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def _disabled_optuna() -> OptunaConfig:
    """Return a disabled Optuna configuration for fast orchestration tests."""
    return OptunaConfig(
        enabled=False,
        timeout_seconds=1,
        n_trials=None,
        cv_splits=2,
        objective="combined_mae",
        early_stopping_rounds=5,
        tree_method="hist",
        device="cpu",
        tune_scope="both",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=1,
    )


def test_filter_to_regular_season_for_training_drops_non_regular_rows() -> None:
    """Drops non-regular-season rows when postseason training is disabled."""
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "game_type": ["REG", "POST", "reg"],
        }
    )

    filtered = ml_model_training._filter_to_regular_season_for_training(
        df,
        include_postseason=False,
    )

    assert filtered["week"].tolist() == [1, 3]


def test_filter_to_regular_season_for_training_returns_input_when_not_applicable() -> None:
    """Leaves the input unchanged when postseason rows are allowed or missing."""
    with_game_type = pd.DataFrame({"game_type": ["POST"]})
    without_game_type = pd.DataFrame({"week": [1]})

    assert (
        ml_model_training._filter_to_regular_season_for_training(
            with_game_type,
            include_postseason=True,
        )
        is with_game_type
    )
    assert (
        ml_model_training._filter_to_regular_season_for_training(
            without_game_type,
            include_postseason=False,
        )
        is without_game_type
    )


def test_train_score_model_without_holdout_skips_holdout_evaluation(monkeypatch) -> None:
    """Skips holdout inference when no holdout seasons are configured."""
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "away_score": [10, 14],
            "home_score": [20, 17],
            "feat1": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training,
        "_build_feature_spec",
        lambda *_args, **_kwargs: _feature_spec(high_cardinality=True),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_apply_feature_spec",
        lambda frame, _spec: frame[["feat1"]],
    )
    monkeypatch.setattr(
        ml_model_training,
        "_build_preprocessor",
        lambda *_args, **_kwargs: _DummyPreprocessor(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_models",
        lambda *_args, **_kwargs: ("away_model", "home_model"),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_postseason_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_recency_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "combine_sample_weights",
        lambda postseason, _recency: postseason,
    )
    monkeypatch.setattr(
        ml_model_training,
        "_transform_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("holdout transform should be skipped")
        ),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_evaluate_predictions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("holdout evaluation should be skipped")
        ),
    )

    model = ml_model_training.train_score_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=0,
        include_market=True,
        max_cardinality_ratio=0.5,
        market_prob_config=None,
        xgb_n_jobs=7,
    )

    assert isinstance(model, ScoreModel)
    assert model.xgb_params is not None
    assert model.xgb_params["n_jobs"] == 7
    assert model.feature_spec.high_cardinality_columns == ["team_code"]


def test_train_score_model_with_report_without_holdout_uses_default_params(monkeypatch) -> None:
    """Reports `None` holdout metrics and falls back to default XGBoost params."""
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
        xgb_params=None,
    )

    monkeypatch.setattr(ml_model_training, "train_score_model", lambda **_kwargs: model)
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training.feature_importance,
        "build_feature_importance_report",
        lambda _model: {"feature_names": []},
    )
    monkeypatch.setattr(
        ml_model_training,
        "_predict_xgb",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("holdout prediction should be skipped")
        ),
    )

    result = ml_model_training.train_score_model_with_report(
        data_path=Path("dummy.csv"),
        holdout_seasons=0,
        include_market=True,
        max_cardinality_ratio=0.5,
        market_prob_config=None,
    )

    assert result.metrics_report["metrics"]["holdout"] is None
    assert result.splits["train_seasons"] == [2022, 2023]
    assert result.splits["holdout_seasons"] == []
    assert result.params == DEFAULT_XGB_PARAMS


def test_train_margin_total_model_uncertainty_elo_falls_back_to_none(monkeypatch) -> None:
    """Skips Elo calibration when uncertainty-aware probabilities are enabled."""
    df = pd.DataFrame(
        {
            "season": [2022, 2023],
            "week": [1, 1],
            "away_score": [10, 14],
            "home_score": [20, 17],
            "feat1": [1.0, 2.0],
        }
    )
    train_df = df.iloc[[0]].copy()
    calibration_df = df.iloc[[1]].copy()
    holdout_df = df.iloc[0:0].copy()

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training,
        "_split_train_calibration_holdout",
        lambda *_args, **_kwargs: (
            train_df,
            calibration_df,
            holdout_df,
            [2022],
            [2023],
            [],
            2023,
            [1],
        ),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_build_feature_spec",
        lambda *_args, **_kwargs: _feature_spec(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_apply_feature_spec",
        lambda frame, _spec: frame[["feat1"]],
    )
    monkeypatch.setattr(
        ml_model_training,
        "_build_preprocessor",
        lambda *_args, **_kwargs: _DummyPreprocessor(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_transform_matrix",
        lambda _preprocessor, frame: np.zeros((len(frame), 1), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_transform_matrix",
        lambda _preprocessor, frame: np.zeros((len(frame), 1), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_postseason_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_recency_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "combine_sample_weights",
        lambda postseason, _recency: postseason,
    )

    def fake_targets(frame: pd.DataFrame, _targets: tuple[str, str], _anchor: bool):
        rows = len(frame)
        return (
            np.zeros(rows, dtype=float),
            np.full(rows, 40.0, dtype=float),
            np.zeros(rows, dtype=float),
            np.zeros(rows, dtype=float),
        )

    monkeypatch.setattr(
        ml_model_training,
        "_prepare_margin_total_targets_with_anchor",
        fake_targets,
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: ("margin_model", "total_model"),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_quantile_models",
        lambda *_args, **_kwargs: {0.1: "q10", 0.9: "q90"},
    )
    monkeypatch.setattr(ml_model_training, "_validate_quantiles", lambda _q: (0.1, 0.9))
    monkeypatch.setattr(
        ml_model_training,
        "resolve_win_prob_calibration_method",
        lambda _method, _count: "elo",
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_win_prob_calibrator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Elo calibration should be skipped when uncertainty is enabled")
        ),
    )

    model = ml_model_training.train_margin_total_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=0,
        calibration_seasons=1,
        calibration_weeks=0,
        include_market=False,
        max_cardinality_ratio=0.5,
        win_prob_calibration="elo",
        optuna_config=_disabled_optuna(),
        market_transform=False,
        market_anchor=False,
        market_prob_config=None,
        win_prob_use_uncertainty=True,
    )

    assert model.calibrator is None
    assert model.win_prob_use_uncertainty is True


def test_train_margin_total_model_raises_without_calibration_rows(monkeypatch) -> None:
    """Raises when fitted calibration is requested but no calibration rows are available."""
    df = pd.DataFrame(
        {
            "season": [2022],
            "week": [1],
            "away_score": [10],
            "home_score": [20],
            "feat1": [1.0],
        }
    )
    train_df = df.copy()
    calibration_df = df.iloc[0:0].copy()

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training,
        "_split_train_calibration_holdout",
        lambda *_args, **_kwargs: (
            train_df,
            calibration_df,
            calibration_df,
            [2022],
            [],
            [],
            None,
            [],
        ),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_build_feature_spec",
        lambda *_args, **_kwargs: _feature_spec(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_apply_feature_spec",
        lambda frame, _spec: frame[["feat1"]],
    )
    monkeypatch.setattr(
        ml_model_training,
        "_build_preprocessor",
        lambda *_args, **_kwargs: _DummyPreprocessor(),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_transform_matrix",
        lambda _preprocessor, frame: np.zeros((len(frame), 1), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_postseason_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "compute_recency_sample_weight",
        lambda frame, **_kwargs: np.ones(len(frame), dtype=float),
    )
    monkeypatch.setattr(
        ml_model_training,
        "combine_sample_weights",
        lambda postseason, _recency: postseason,
    )
    monkeypatch.setattr(
        ml_model_training,
        "_prepare_margin_total_targets_with_anchor",
        lambda frame, _targets, _anchor: (
            np.zeros(len(frame), dtype=float),
            np.full(len(frame), 40.0, dtype=float),
            np.zeros(len(frame), dtype=float),
            np.zeros(len(frame), dtype=float),
        ),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: ("margin_model", "total_model"),
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_quantile_models",
        lambda *_args, **_kwargs: {0.1: "q10", 0.9: "q90"},
    )
    monkeypatch.setattr(ml_model_training, "_validate_quantiles", lambda _q: (0.1, 0.9))
    monkeypatch.setattr(
        ml_model_training,
        "resolve_win_prob_calibration_method",
        lambda _method, _count: "platt",
    )

    with pytest.raises(ValueError, match="Calibration requested but no calibration seasons"):
        ml_model_training.train_margin_total_model(
            data_path=Path("dummy.csv"),
            holdout_seasons=0,
            calibration_seasons=0,
            calibration_weeks=0,
            include_market=False,
            max_cardinality_ratio=0.5,
            win_prob_calibration="platt",
            optuna_config=_disabled_optuna(),
            market_transform=False,
            market_anchor=False,
            market_prob_config=None,
        )


@pytest.mark.parametrize(
    ("calibration_seasons", "calibration_weeks", "market_anchor", "match"),
    [
        (0, 0, False, "require calibration seasons or calibration weeks"),
        (1, 0, True, "only supported for margin_total models"),
    ],
)
def test_train_blended_margin_total_model_rejects_invalid_inputs(
    calibration_seasons: int,
    calibration_weeks: int,
    market_anchor: bool,
    match: str,
) -> None:
    """Validates the blended-model guard rails before any training work starts."""
    with pytest.raises(ValueError, match=match):
        ml_model_training.train_blended_margin_total_model(
            data_path=Path("dummy.csv"),
            holdout_seasons=1,
            calibration_seasons=calibration_seasons,
            calibration_weeks=calibration_weeks,
            max_cardinality_ratio=0.5,
            win_prob_calibration="none",
            optuna_config=_disabled_optuna(),
            market_transform=False,
            market_anchor=market_anchor,
            market_prob_config=MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        )


def test_train_blended_margin_total_model_with_report_uses_explicit_market_model(
    monkeypatch,
) -> None:
    """Uses the model-backed market branch when a separate market model is present."""
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "week": [1, 2, 1, 2],
            "away_score": [10, 12, 14, 9],
            "home_score": [20, 18, 21, 16],
            "feat1": [0.1, 0.2, 0.4, 0.3],
        }
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
    market_model = MarginTotalModel(
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
        market_model=market_model,
        blend_layer=BlendLayer(
            margin_model=cast(Ridge, _BlendModel(np.array([0.6, 0.4]))),
            total_model=cast(Ridge, _BlendModel(np.array([0.5, 0.5]))),
        ),
        calibrator=None,
        target_columns=("away_score", "home_score"),
        market_prob_config=None,
        xgb_params=None,
        tuned_params=None,
        tuned_cv_summary=None,
        optuna_summary=None,
    )

    monkeypatch.setattr(
        ml_model_training,
        "train_blended_margin_total_model",
        lambda **_kwargs: model,
    )
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(
        ml_model_training.feature_importance,
        "build_feature_importance_report",
        lambda _model: {"feature_names": []},
    )

    def fake_predict_margin_total_from_model(current_model: object, _df: pd.DataFrame):
        """Return distinct team vs market predictions to exercise both branches."""
        if current_model is team_model:
            return np.array([3.0, -2.0]), np.array([40.0, 35.0])
        return np.array([1.0, -1.0]), np.array([42.0, 33.0])

    monkeypatch.setattr(
        ml_model_training,
        "_predict_margin_total_from_model",
        fake_predict_margin_total_from_model,
    )
    monkeypatch.setattr(
        ml_model_training,
        "_predict_home_win_prob",
        lambda _margin, _calibrator: np.array([0.6, 0.4]),
    )

    result = ml_model_training.train_blended_margin_total_model_with_report(
        data_path=Path("dummy.csv"),
        holdout_seasons=1,
        calibration_seasons=0,
        calibration_weeks=0,
        max_cardinality_ratio=0.5,
        win_prob_calibration="none",
        optuna_config=None,
        market_transform=False,
        market_anchor=False,
        market_prob_config=None,
    )

    assert result.metrics_report["metrics"]["holdout"] is not None
    assert result.splits["holdout_seasons"] == [2021]
    assert result.params == DEFAULT_XGB_PARAMS
