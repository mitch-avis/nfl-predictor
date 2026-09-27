"""Pin which rows production training fits on, holds out and calibrates on.

The final fit trains its trees on the pool minus the in-season calibration window (the newest
completed ``(season, week)`` pairs, rolling back across the season boundary) and minus any whole
calibration seasons; fitted calibrators use the pooled calibrator frame. These tests pin those
row sets, the window record in the training report and the window log line.
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

WINDOW_2026_WEEK_2 = [(2025, 16), (2025, 17), (2025, 18), (2026, 1)]


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


def _season_pairs(season: int, weeks: range) -> list[tuple[int, int]]:
    return [(season, week) for week in weeks]


def _stub_fitting(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> dict[str, Any]:
    """Replace every fit with a stub and record the frames production hands to it."""
    recorded: dict[str, Any] = {}

    def record_spec(frame: pd.DataFrame, **_kwargs: Any) -> FeatureSpec:
        recorded["train"] = frame.copy()
        return _feature_spec()

    def record_targets(
        frame: pd.DataFrame, _targets: tuple[str, str], _anchor: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        recorded.setdefault("target_frames", []).append(frame.copy())
        rows = len(frame)
        return (
            np.zeros(rows, dtype=float),
            np.full(rows, 40.0, dtype=float),
            np.zeros(rows, dtype=float),
            np.zeros(rows, dtype=float),
        )

    real_pooled = ml_model_training._pooled_calibration_frame

    def record_pooled(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        pooled = real_pooled(frame, **kwargs)
        recorded["calibrator_frame"] = pooled.copy()
        return pooled

    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: df)
    monkeypatch.setattr(ml_model_training, "_build_feature_spec", record_spec)
    monkeypatch.setattr(ml_model_training, "_apply_feature_spec", lambda frame, _spec: frame)
    monkeypatch.setattr(
        ml_model_training, "_build_preprocessor", lambda *_a, **_k: _ZeroPreprocessor()
    )
    monkeypatch.setattr(
        ml_model_training, "_prepare_margin_total_targets_with_anchor", record_targets
    )
    monkeypatch.setattr(
        ml_model_training,
        "_fit_margin_total_models",
        lambda *_a, **_k: ("margin_model", "total_model"),
    )
    monkeypatch.setattr(ml_model_training, "_fit_quantile_models", lambda *_a, **_k: {})
    monkeypatch.setattr(ml_model_training, "_pooled_calibration_frame", record_pooled)
    return recorded


def _train(calibration_seasons: int, calibration_weeks: int) -> MarginTotalModel:
    return ml_model_training.train_margin_total_model(
        data_path=Path("dummy.csv"),
        holdout_seasons=0,
        calibration_seasons=calibration_seasons,
        calibration_weeks=calibration_weeks,
        include_market=False,
        max_cardinality_ratio=0.5,
        win_prob_calibration="none",
        optuna_config=_disabled_optuna(),
        market_transform=False,
        market_anchor=False,
    )


def test_final_fit_holds_the_rolled_back_window_out_of_the_trees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """At week 2 the four held-out weeks are 2026 week 1 plus 2025 weeks 16-18."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    recorded = _stub_fitting(monkeypatch, df)
    caplog.set_level(logging.INFO)

    _train(calibration_seasons=0, calibration_weeks=4)

    assert _pairs(recorded["train"]) == sorted(set(_pairs(df)) - set(WINDOW_2026_WEEK_2))
    train_targets, calibration_targets = recorded["target_frames"]
    assert _pairs(train_targets) == _pairs(recorded["train"])
    assert _pairs(calibration_targets) == WINDOW_2026_WEEK_2
    assert _pairs(recorded["calibrator_frame"]) == (
        _season_pairs(2024, range(1, 19)) + _season_pairs(2025, range(1, 19)) + [(2026, 1)]
    )
    assert (
        "Calibration weeks: season 2026 weeks [1] "
        "(window pairs [[2025, 16], [2025, 17], [2025, 18], [2026, 1]])"
    ) in caplog.messages


def test_final_fit_holds_the_window_and_a_whole_season_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole calibration season is the newest one the window leaves untouched."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    recorded = _stub_fitting(monkeypatch, df)

    _train(calibration_seasons=1, calibration_weeks=4)

    assert _pairs(recorded["train"]) == (
        _season_pairs(2023, range(1, 19)) + _season_pairs(2025, range(1, 16))
    )
    _, calibration_targets = recorded["target_frames"]
    assert _pairs(calibration_targets) == _season_pairs(2024, range(1, 19)) + WINDOW_2026_WEEK_2


def test_week_one_calibrator_frame_pools_three_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before week 1 the newest pool season is last season, so the frame reaches back three."""
    df = _season_weeks_frame({2022: 18, 2023: 18, 2024: 18, 2025: 18})
    recorded = _stub_fitting(monkeypatch, df)

    _train(calibration_seasons=0, calibration_weeks=4)

    assert _pairs(recorded["train"]) == sorted(
        set(_pairs(df)) - set(_season_pairs(2025, range(15, 19)))
    )
    assert _pairs(recorded["calibrator_frame"]) == (
        _season_pairs(2023, range(1, 19))
        + _season_pairs(2024, range(1, 19))
        + _season_pairs(2025, range(1, 19))
    )


def test_final_fit_logs_why_the_whole_calibration_season_moved(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The log names the seasons the window touches, which whole seasons must skip."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    _stub_fitting(monkeypatch, df)
    caplog.set_level(logging.INFO)

    _train(calibration_seasons=1, calibration_weeks=4)

    assert (
        "Calibration seasons: [2024] (the newest seasons the in-season window does not "
        "touch; it touches [2025, 2026])"
    ) in caplog.messages


def test_final_fit_logs_the_calibrator_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The pooled calibrator frame is logged as seasons and week ranges."""
    df = _season_weeks_frame({2023: 18, 2024: 18, 2025: 18, 2026: 1})
    _stub_fitting(monkeypatch, df)
    caplog.set_level(logging.INFO)

    _train(calibration_seasons=0, calibration_weeks=4)

    assert "Calibration seasons: []" in caplog.messages
    assert (
        "Calibrator frame: 2024 weeks 1-18, 2025 weeks 1-18, 2026 week 1 (37 rows)"
        in caplog.messages
    )
