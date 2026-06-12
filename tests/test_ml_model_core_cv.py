"""Unit tests for CV/objective helpers in ml_model_core.

These tests stub out model training so they remain fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from nfl_predictor.ml import ml_model_core as core


def test_score_margin_total_fold_with_stubs(monkeypatch: MonkeyPatch) -> None:
    """Scores a fold end-to-end using stubbed preprocess/training/prediction."""

    class _DummyPreprocessor:
        def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones((len(x), 2), dtype=float)

        def transform(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones((len(x), 2), dtype=float)

    class _DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(core, "_build_feature_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(core, "_build_preprocessor", lambda *_args, **_kwargs: _DummyPreprocessor())
    monkeypatch.setattr(core, "_apply_feature_spec", lambda df, _spec: df)

    def _targets(df: pd.DataFrame, *_args: object, **_kwargs: object):
        n = len(df)
        baseline_margin = np.ones(n, dtype=float)
        baseline_total = np.ones(n, dtype=float) * 40.0
        return np.zeros(n), np.zeros(n), baseline_margin, baseline_total

    monkeypatch.setattr(core, "_prepare_margin_total_targets_with_anchor", _targets)
    monkeypatch.setattr(
        core,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: (_DummyModel("margin"), _DummyModel("total")),
    )

    def _predict(model: _DummyModel, x: np.ndarray) -> np.ndarray:
        if model.name == "margin":
            return np.full(len(x), 3.0)
        return np.full(len(x), 44.0)

    monkeypatch.setattr(core, "_predict_xgb", _predict)
    monkeypatch.setattr(core, "_margin_to_home_win_prob", lambda _m: np.array([0.7]))
    monkeypatch.setattr(core, "_adjust_home_win_prob", lambda _df, probs, _cfg: probs)
    monkeypatch.setattr(
        core,
        "_evaluate_margin_total_predictions",
        lambda *_args, **_kwargs: {
            "margin_mae": 3.0,
            "total_mae": 7.0,
            "winner_accuracy": 0.5,
            "brier": 0.2,
        },
    )
    monkeypatch.setattr(
        core,
        "_summarize_confidence_pool",
        lambda *_args, **_kwargs: {"weekly_expected_points_avg": 10.0},
    )

    df_train = pd.DataFrame(
        {
            "feat1": [1.0, 2.0],
            "away_score": [10.0, 14.0],
            "home_score": [13.0, 10.0],
        }
    )
    df_val = pd.DataFrame({"feat1": [3.0], "away_score": [17.0], "home_score": [21.0]})

    score = core._score_margin_total_fold(
        train_df=df_train,
        val_df=df_val,
        target_columns=("away_score", "home_score"),
        include_market=False,
        max_cardinality_ratio=0.5,
        feature_start="feat1",
        feature_end="feat1",
        params={"n_estimators": 1},
        early_stopping_rounds=5,
        objective="combined_mae",
        market_anchor=True,
        market_prob_config=core.MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
    )

    assert score == 5.0


def test_evaluate_margin_total_cv_summary_aggregates_folds(monkeypatch: MonkeyPatch) -> None:
    """Aggregates fold scores and returns mean/std summary."""
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "away_score": [10.0, 10.0, 10.0, 10.0],
            "home_score": [13.0, 13.0, 13.0, 13.0],
        }
    )

    monkeypatch.setattr(
        core,
        "_build_season_week_timepoints",
        lambda _df: [202401, 202402, 202403, 202404],
    )
    monkeypatch.setattr(
        core,
        "_build_blocked_timepoint_folds",
        lambda _tp, n_splits: [
            ([202401, 202402], [202403]),
            ([202401, 202402, 202403], [202404]),
        ],
    )

    # Return different scores per fold so std is non-zero.
    scores = iter([1.0, 3.0])
    monkeypatch.setattr(core, "_score_margin_total_fold", lambda **_kwargs: next(scores))

    summary = core._evaluate_margin_total_cv_summary(
        df,
        target_columns=("away_score", "home_score"),
        include_market=False,
        max_cardinality_ratio=0.5,
        feature_start="feat1",
        feature_end="feat1",
        params={"n_estimators": 1},
        cv_splits=2,
        early_stopping_rounds=5,
        objective="margin_mae",
    )

    assert summary["cv_splits"] == 2
    assert summary["fold_scores"] == [1.0, 3.0]
    assert summary["mean"] == 2.0
    assert summary["std"] == 1.0


def test_evaluate_margin_total_cv_returns_mean(monkeypatch: MonkeyPatch) -> None:
    """Returns the mean of fold scores for the objective."""
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "away_score": [10.0, 10.0, 10.0],
            "home_score": [13.0, 13.0, 13.0],
        }
    )

    monkeypatch.setattr(core, "_build_season_week_timepoints", lambda _df: [202401, 202402, 202403])
    monkeypatch.setattr(
        core,
        "_build_blocked_timepoint_folds",
        lambda _tp, n_splits: [([202401], [202402]), ([202401, 202402], [202403])],
    )

    scores = iter([2.0, 4.0])
    monkeypatch.setattr(core, "_score_margin_total_fold", lambda **_kwargs: next(scores))

    mean = core._evaluate_margin_total_cv(
        df,
        target_columns=("away_score", "home_score"),
        include_market=False,
        max_cardinality_ratio=0.5,
        feature_start="feat1",
        feature_end="feat1",
        params={"n_estimators": 1},
        cv_splits=2,
        early_stopping_rounds=5,
        objective="margin_mae",
    )

    assert mean == 3.0


def test_run_optuna_search_rejects_holdout_seasons(monkeypatch: MonkeyPatch) -> None:
    """Reject Optuna tuning when holdout seasons are present."""
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "away_score": [10.0, 10.0],
            "home_score": [13.0, 13.0],
        }
    )

    monkeypatch.setattr(core, "optuna", object())

    optuna_config = core.OptunaConfig(
        enabled=True,
        timeout_seconds=1,
        n_trials=1,
        cv_splits=2,
        objective="margin_mae",
        early_stopping_rounds=5,
        tree_method=None,
        device=None,
        tune_scope="both",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=None,
    )

    with pytest.raises(ValueError, match="holdout seasons"):
        core._run_optuna_search(
            df,
            target_columns=("away_score", "home_score"),
            include_market=False,
            max_cardinality_ratio=0.5,
            feature_start="away_score",
            feature_end="home_score",
            optuna_config=optuna_config,
            holdout_seasons=[2024],
        )
