"""Models saved before the probability paths were retired still predict, with the floor.

Every run type now submits the deterministic floor, ``Phi(margin / SCORE_DIFF_STD_DEV)``. A
checkpoint saved earlier can still carry a market blend or clamp; loading it logs what is
ignored and drops it, and its predictions are the floor of its own predicted margins.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core, ml_model_predict, ml_model_training
from nfl_predictor.ml.ml_model_core import MarginTotalModel, OptunaConfig
from tests.weekly_fixture import build_fixture

OPTUNA_OFF = OptunaConfig(
    enabled=False,
    timeout_seconds=0,
    n_trials=None,
    cv_splits=2,
    objective="brier",
    early_stopping_rounds=50,
    tree_method=None,
    device="cpu",
    storage=None,
    study_name=None,
    best_params_out=None,
    xgb_n_jobs=1,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarginTotalModel, dict[str, Path]]:
    """Train a small market-anchored model on the fixture seasons."""
    paths = build_fixture(tmp_path_factory.mktemp("retired_paths"))
    model = ml_model_training.train_margin_total_model(
        data_path=paths["completed"],
        holdout_seasons=0,
        calibration_seasons=0,
        calibration_weeks=4,
        include_market=True,
        max_cardinality_ratio=0.5,
        win_prob_calibration="none",
        optuna_config=OPTUNA_OFF,
        market_transform=True,
        market_anchor=True,
    )
    return model, paths


def _floor(model: MarginTotalModel, games_path: Path) -> np.ndarray:
    """Return the deterministic floor of the model's margins, as the output rounds it."""
    games = pd.read_csv(games_path)
    margin, _total = ml_model_core.predict_margin_total_from_model(model, games)
    floor = norm.cdf(margin / constants.SCORE_DIFF_STD_DEV)
    return np.clip(np.round(floor, 4), 0.0001, 0.9999)


def test_a_saved_market_blend_is_ignored_and_the_floor_is_predicted(
    trained: tuple[MarginTotalModel, dict[str, Path]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A checkpoint with a market blend and clamp loads, says so, and predicts the floor."""
    model, paths = trained
    model = copy.copy(model)
    object.__setattr__(
        model,
        "market_prob_config",
        ml_model_core.MarketProbConfig(blend_weight=0.5, clamp_delta=0.02),
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    with caplog.at_level(logging.WARNING):
        loaded = ml_model_core.load_model_checkpoint(model_path, "margin_total")
    predictions = ml_model_predict.predict_week_margin_total(
        loaded, paths["predict"], pretty_output=False
    )

    assert "market blend" in caplog.text
    assert not hasattr(loaded, "market_prob_config")
    np.testing.assert_allclose(
        predictions["home_win_prob"].to_numpy(), _floor(loaded, paths["predict"])
    )
