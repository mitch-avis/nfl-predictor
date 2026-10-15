"""Unit tests for CV/objective helpers in ml_model_core.

These tests stub out model training so they remain fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import ml_model_core as core
from tests.fakes import returning


def test_score_margin_total_fold_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scores a fold end-to-end using stubbed preprocess/training/prediction."""

    # Arrange
    class _DummyPreprocessor:
        def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones((len(x), 2), dtype=float)

        def transform(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones((len(x), 2), dtype=float)

    class _DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name

    def passthrough_feature_spec(df: pd.DataFrame, _spec: object) -> pd.DataFrame:
        return df

    monkeypatch.setattr(core, "build_feature_spec", returning(object()))
    monkeypatch.setattr(core, "build_preprocessor", returning(_DummyPreprocessor()))
    monkeypatch.setattr(core, "apply_feature_spec", passthrough_feature_spec)

    def _targets(
        df: pd.DataFrame, *_args: object, **_kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        baseline_margin = np.ones(n, dtype=float)
        baseline_total = np.ones(n, dtype=float) * 40.0
        return np.zeros(n), np.zeros(n), baseline_margin, baseline_total

    monkeypatch.setattr(core, "prepare_margin_total_targets_with_anchor", _targets)
    monkeypatch.setattr(
        core,
        "fit_margin_total_models",
        returning((_DummyModel("margin"), _DummyModel("total"))),
    )

    def _predict(model: _DummyModel, x: np.ndarray) -> np.ndarray:
        if model.name == "margin":
            return np.full(len(x), 3.0)
        return np.full(len(x), 44.0)

    monkeypatch.setattr(core, "predict_xgb", _predict)
    monkeypatch.setattr(core, "margin_to_home_win_prob", returning(np.array([0.7])))
    monkeypatch.setattr(
        core,
        "evaluate_margin_total_predictions",
        returning(
            {
                "margin_mae": 3.0,
                "total_mae": 7.0,
                "winner_accuracy": 0.5,
                "brier": 0.2,
            }
        ),
    )
    monkeypatch.setattr(
        core,
        "summarize_confidence_pool",
        returning({"weekly_expected_points_avg": 10.0}),
    )

    df_train = pd.DataFrame(
        {
            "feat1": [1.0, 2.0],
            "away_score": [10.0, 14.0],
            "home_score": [13.0, 10.0],
        }
    )
    df_val = pd.DataFrame({"feat1": [3.0], "away_score": [17.0], "home_score": [21.0]})

    # Act
    score = core._score_margin_total_fold(
        df_train,
        df_val,
        _fold_setup(market_anchor=True),
        {"n_estimators": 1},
        _optuna_config(objective="combined_mae"),
    )

    # Assert
    assert score == 5.0


def test_evaluate_margin_total_cv_summary_aggregates_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregates fold scores and returns mean/std summary."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "away_score": [10.0, 10.0, 10.0, 10.0],
            "home_score": [13.0, 13.0, 13.0, 13.0],
        }
    )

    def two_folds(_tp: object, n_splits: int) -> list[tuple[list[int], list[int]]]:
        del n_splits
        return [
            ([202401, 202402], [202403]),
            ([202401, 202402, 202403], [202404]),
        ]

    # Return different scores per fold so std is non-zero.
    scores = iter([1.0, 3.0])

    def next_score(*_args: object) -> float:
        return next(scores)

    monkeypatch.setattr(
        core,
        "_build_season_week_timepoints",
        returning([202401, 202402, 202403, 202404]),
    )
    monkeypatch.setattr(core, "_build_blocked_timepoint_folds", two_folds)
    monkeypatch.setattr(core, "_score_margin_total_fold", next_score)

    # Act
    summary = core._evaluate_margin_total_cv_summary(
        df, _fold_setup(), {"n_estimators": 1}, _optuna_config(objective="margin_mae")
    )

    # Assert
    assert summary["cv_splits"] == 2
    assert summary["fold_scores"] == [1.0, 3.0]
    assert summary["mean"] == 2.0
    assert summary["std"] == 1.0


def test_evaluate_margin_total_cv_returns_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns the mean of fold scores for the objective."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "away_score": [10.0, 10.0, 10.0],
            "home_score": [13.0, 13.0, 13.0],
        }
    )

    def two_folds(_tp: object, n_splits: int) -> list[tuple[list[int], list[int]]]:
        del n_splits
        return [([202401], [202402]), ([202401, 202402], [202403])]

    scores = iter([2.0, 4.0])

    def next_score(*_args: object) -> float:
        return next(scores)

    monkeypatch.setattr(core, "_build_season_week_timepoints", returning([202401, 202402, 202403]))
    monkeypatch.setattr(core, "_build_blocked_timepoint_folds", two_folds)
    monkeypatch.setattr(core, "_score_margin_total_fold", next_score)

    # Act
    mean = core._evaluate_margin_total_cv(
        df, _fold_setup(), {"n_estimators": 1}, _optuna_config(objective="margin_mae")
    )

    # Assert
    assert mean == 3.0


def test_run_optuna_search_rejects_holdout_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject Optuna tuning when holdout seasons are present."""
    # Arrange
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "away_score": [10.0, 10.0],
            "home_score": [13.0, 13.0],
        }
    )

    optuna_config = core.OptunaConfig(
        enabled=True,
        timeout_seconds=1,
        n_trials=1,
        cv_splits=2,
        objective="margin_mae",
        early_stopping_rounds=5,
        tree_method=None,
        device=None,
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=None,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="holdout seasons"):
        core.run_optuna_search(df, _fold_setup(), optuna_config, holdout_seasons=[2024])


def _fold_setup(*, market_anchor: bool = False) -> core.FoldSetup:
    """Return a fold setup on the single ``feat1`` feature."""
    return core.FoldSetup(
        target_columns=("away_score", "home_score"),
        selection=core.FeatureSelection(
            include_market=False,
            max_cardinality_ratio=0.5,
            feature_start="feat1",
            feature_end="feat1",
        ),
        market_anchor=market_anchor,
    )


def _optuna_config(*, objective: str) -> core.OptunaConfig:
    """Return a two-split tuning config with the given objective."""
    return core.OptunaConfig(
        enabled=True,
        timeout_seconds=1,
        n_trials=1,
        cv_splits=2,
        objective=objective,
        early_stopping_rounds=5,
        tree_method=None,
        device=None,
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=None,
    )
