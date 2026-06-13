"""Tests for the power rankings script helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import ml_model_core
from scripts import power_rankings


def test_load_current_records_filters_to_reg_only(tmp_path) -> None:
    """Records should exclude postseason games when computing current standings."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 5,
                "game_type": "WC",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    records = power_rankings._load_current_records(
        schedule_path, season=2024, through_week=5, include_postseason=False
    )
    records = records.set_index("team_abbr")

    assert records.loc["AAA", "wins"] == 0
    assert records.loc["AAA", "losses"] == 1
    assert records.loc["BBB", "wins"] == 1
    assert records.loc["BBB", "losses"] == 0


def test_load_current_records_can_include_postseason(tmp_path) -> None:
    """Records should include postseason games when requested."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 19,
                "game_type": "DIV",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    records = power_rankings._load_current_records(
        schedule_path, season=2024, through_week=19, include_postseason=True
    )
    records = records.set_index("team_abbr")

    assert records.loc["AAA", "wins"] == 1
    assert records.loc["AAA", "losses"] == 1
    assert records.loc["BBB", "wins"] == 1
    assert records.loc["BBB", "losses"] == 1


def test_predict_future_games_requires_feature_columns(tmp_path) -> None:
    """Missing feature columns should raise a clear validation error."""
    data_path = tmp_path / "ml.csv"
    pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
            }
        ]
    ).to_csv(data_path, index=False)

    model = SimpleNamespace(feature_spec=SimpleNamespace(feature_columns=["feat_required"]))

    with pytest.raises(ValueError, match="Missing required feature columns \\(1\\): feat_required"):
        power_rankings._predict_future_games(
            model,
            model_kind="margin_total",
            data_ml=data_path,
            season=2025,
            through_week=1,
        )


def test_score_model_calibration_applied_for_win_prob(tmp_path, monkeypatch) -> None:
    """ScoreModel paths should apply a calibrator when present."""
    data_path = tmp_path / "ml.csv"
    pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "feat1": 1.0,
            }
        ]
    ).to_csv(data_path, index=False)

    class DummyPreprocessor:
        """Minimal preprocessor stub for score model predictions."""

        def transform(self, _df: pd.DataFrame) -> np.ndarray:
            """Return a deterministic feature matrix."""
            return np.zeros((len(_df), 1), dtype=float)

    class DummyModel:
        """Labelled dummy model for predict_xgb."""

        def __init__(self, name: str) -> None:
            self.name = name

    class DummyCalibrator:
        """Predictor stub returning fixed probabilities."""

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            """Return a fixed 0.9 home win probability."""
            return np.tile(np.array([0.1, 0.9], dtype=float), (len(x), 1))

    def fake_predict_xgb(model: DummyModel, x: np.ndarray) -> np.ndarray:
        """Return deterministic away/home scores."""
        if model.name == "away":
            return np.full(len(x), 10.0)
        return np.full(len(x), 20.0)

    monkeypatch.setattr(ml_model_core, "predict_xgb", fake_predict_xgb)

    spec = SimpleNamespace(feature_columns=["feat1"])
    calibrator = ml_model_core.WinProbCalibrator(method="platt", model=DummyCalibrator())
    model = SimpleNamespace(
        feature_spec=spec,
        preprocessor=DummyPreprocessor(),
        away_model=DummyModel("away"),
        home_model=DummyModel("home"),
        market_prob_config=None,
        calibrator=calibrator,
    )

    out = power_rankings._predict_future_games(
        model,
        model_kind="score",
        data_ml=data_path,
        season=2025,
        through_week=1,
    )
    assert np.allclose(out["home_win_prob"].to_numpy(dtype=float), 0.9)


def test_build_games_for_ratings_logs_diagnostics(tmp_path, caplog) -> None:
    """Ratings diagnostics should log once per invocation."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            }
        ]
    ).to_csv(schedule_path, index=False)

    future_games = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "home_win_prob": 0.6,
            }
        ]
    )

    caplog.set_level(logging.INFO)
    power_rankings._build_games_for_ratings(
        schedule_path=schedule_path,
        season=2024,
        through_week=1,
        ratings_min_season=None,
        future_games_with_probs=future_games,
        include_postseason=False,
    )

    messages = [record.message for record in caplog.records]
    diag = [msg for msg in messages if msg.startswith("Ratings fit diagnostics:")]
    assert len(diag) == 1


def test_build_games_for_ratings_includes_postseason(tmp_path) -> None:
    """Ratings fit should include postseason games when requested."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 19,
                "game_type": "CON",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    games = power_rankings._build_games_for_ratings(
        schedule_path=schedule_path,
        season=2024,
        through_week=19,
        ratings_min_season=None,
        future_games_with_probs=pd.DataFrame(),
        include_postseason=True,
    )

    assert len(games) == 2
    assert set(games["week"]) == {1, 19}
