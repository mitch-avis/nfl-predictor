"""Pin which rows production training fits its trees on.

The final fit trains on every eligible completed game, exactly as each walk-forward fold does:
nothing newer is held out of the trees, and no frame is handed to XGBoost as an eval set. Only
the evaluation holdout (the newest ``holdout_seasons`` whole seasons) stays out of training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from nfl_predictor.ml import ml_model_training
from nfl_predictor.ml.ml_model_core import (
    FeatureSpec,
    MarginTotalModel,
    OptunaConfig,
)

xgb.set_config(verbosity=0)


class _ZeroPreprocessor:
    """Return all-zero feature matrices of the right height."""

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a zero matrix for fitting."""
        return np.zeros((len(df), 1), dtype=float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return a zero matrix for inference."""
        return np.zeros((len(df), 1), dtype=float)


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


def _disabled_optuna() -> OptunaConfig:
    return OptunaConfig(
        enabled=False,
        timeout_seconds=1,
        n_trials=None,
        cv_splits=2,
        objective="combined_mae",
        early_stopping_rounds=5,
        tree_method="hist",
        device="cpu",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=1,
    )


def _season_weeks_frame(season_weeks: dict[int, int]) -> pd.DataFrame:
    """Build one completed game per ``(season, week)`` for weeks ``1..n`` of each season."""
    rows = [
        {"season": season, "week": week, "away_score": 10, "home_score": 20, "feat1": 1.0}
        for season, n_weeks in season_weeks.items()
        for week in range(1, n_weeks + 1)
    ]
    return pd.DataFrame(rows)


def _pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted({(int(s), int(w)) for s, w in zip(frame["season"], frame["week"], strict=True)})


def _stub_fitting(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> dict[str, Any]:
    """Replace every fit with a stub and record the frames production hands to it."""
    recorded: dict[str, Any] = {"fit_kwargs": [], "target_frames": []}

    def record_spec(frame: pd.DataFrame, **_kwargs: Any) -> FeatureSpec:
        recorded["train"] = frame.copy()
        return _feature_spec()

    def record_targets(
        frame: pd.DataFrame, _targets: tuple[str, str], _anchor: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        recorded["target_frames"].append(frame.copy())
        rows = len(frame)
        return (
            np.zeros(rows, dtype=float),
            np.full(rows, 40.0, dtype=float),
            np.zeros(rows, dtype=float),
            np.zeros(rows, dtype=float),
        )

    def record_heads(*_args: Any, **kwargs: Any) -> tuple[str, str]:
        recorded["fit_kwargs"].append(kwargs)
        return "margin_model", "total_model"

    def record_quantiles(*_args: Any, **kwargs: Any) -> dict[float, Any]:
        recorded["fit_kwargs"].append(kwargs)
        return {}

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(ml_model_training, "_build_feature_spec", record_spec)
    monkeypatch.setattr(ml_model_training, "_apply_feature_spec", lambda frame, _spec: frame)
    monkeypatch.setattr(
        ml_model_training, "_build_preprocessor", lambda *_a, **_k: _ZeroPreprocessor()
    )
    monkeypatch.setattr(
        ml_model_training, "_prepare_margin_total_targets_with_anchor", record_targets
    )
    monkeypatch.setattr(ml_model_training, "_fit_margin_total_models", record_heads)
    monkeypatch.setattr(ml_model_training, "_fit_quantile_models", record_quantiles)
    monkeypatch.setattr(
        ml_model_training, "_predict_xgb", lambda _model, x: np.zeros(x.shape[0], dtype=float)
    )
    return recorded


def _train(holdout_seasons: int = 0) -> MarginTotalModel:
    return ml_model_training.train_margin_total_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=holdout_seasons,
        include_market=False,
        max_cardinality_ratio=0.5,
        optuna_config=_disabled_optuna(),
        market_transform=False,
        market_anchor=False,
    )


def test_final_fit_trains_on_every_completed_week(monkeypatch: pytest.MonkeyPatch) -> None:
    """At week 2 the trees see 2026 week 1 and the end of 2025, as a walk-forward fold does."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    recorded = _stub_fitting(monkeypatch, df)

    _train()

    assert _pairs(recorded["train"]) == _pairs(df)
    (train_targets,) = recorded["target_frames"]
    assert _pairs(train_targets) == _pairs(df)


def test_final_fit_hands_xgboost_no_eval_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """No head receives an eval frame, so none can early-stop on one or report metrics for it."""
    df = _season_weeks_frame({2025: 18, 2026: 1})
    recorded = _stub_fitting(monkeypatch, df)

    _train()

    assert len(recorded["fit_kwargs"]) == 3
    for kwargs in recorded["fit_kwargs"]:
        assert kwargs.get("x_eval") is None


def test_final_fit_leaves_only_the_evaluation_holdout_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The newest whole season is the evaluation holdout; every other game trains."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18})
    recorded = _stub_fitting(monkeypatch, df)
    caplog.set_level(logging.INFO)

    _train(holdout_seasons=1)

    assert _pairs(recorded["train"]) == _pairs(df[df["season"] < 2025])
    assert "Training seasons: [2023, 2024]" in caplog.messages
    assert "Holdout seasons: [2025]" in caplog.messages
    assert not [message for message in caplog.messages if "alibration" in message]


def test_a_negative_holdout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative season count is an input error, not an empty holdout."""
    _stub_fitting(monkeypatch, _season_weeks_frame({2025: 18, 2026: 1}))

    with pytest.raises(ValueError, match="non-negative"):
        _train(holdout_seasons=-1)
