"""The deterministic floor is the one probability path, and old models predict with it.

Every run type now submits the deterministic floor, ``Phi(margin / sigma)``; a checkpoint saved
before the sigma was recorded predicts with ``SCORE_DIFF_STD_DEV``. A checkpoint saved earlier
can still carry a fitted or Elo calibrator or a market blend or clamp;
loading it logs what is ignored and drops it, and its predictions are the floor of its own
predicted margins. Asking a command or a walk-forward for a retired calibrator fails with the
reason.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_predictor import constants
from nfl_predictor.api.jobs import catalog
from nfl_predictor.cli import backtest, train
from nfl_predictor.ml import ml_model_core, ml_model_predict, ml_model_training, walk_forward
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
RETIRED_CALIBRATORS = ["platt", "isotonic", "sigma", "logistic", "elo"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarginTotalModel, dict[str, Path]]:
    """Train a small market-anchored model on the fixture seasons."""
    paths = build_fixture(tmp_path_factory.mktemp("retired_paths"))
    model = ml_model_training.train_margin_total_model(
        data_path=paths["completed"],
        holdout_seasons=0,
        include_market=True,
        max_cardinality_ratio=0.5,
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


def _load_legacy(
    model: MarginTotalModel,
    attribute: str,
    value: object,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> MarginTotalModel:
    """Save ``model`` with a retired ``attribute`` set, then load it back."""
    legacy = copy.copy(model)
    object.__setattr__(legacy, attribute, value)
    model_path = tmp_path / "model.joblib"
    joblib.dump(legacy, model_path)
    with caplog.at_level(logging.WARNING):
        return ml_model_core.load_model_checkpoint(model_path, "margin_total")


def test_a_saved_market_blend_is_ignored_and_the_floor_is_predicted(
    trained: tuple[MarginTotalModel, dict[str, Path]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A checkpoint with a market blend and clamp loads, says so, and predicts the floor."""
    model, paths = trained
    config = ml_model_core.MarketProbConfig(blend_weight=0.5, clamp_delta=0.02)
    loaded = _load_legacy(model, "market_prob_config", config, tmp_path, caplog)
    predictions = ml_model_predict.predict_week_margin_total(
        loaded, paths["predict"], pretty_output=False
    )

    assert "market blend" in caplog.text
    assert not hasattr(loaded, "market_prob_config")
    np.testing.assert_allclose(
        predictions["home_win_prob"].to_numpy(), _floor(loaded, paths["predict"])
    )


@pytest.mark.parametrize("method", RETIRED_CALIBRATORS)
def test_a_saved_calibrator_is_ignored_and_the_floor_is_predicted(
    trained: tuple[MarginTotalModel, dict[str, Path]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    method: str,
) -> None:
    """A checkpoint with a retired calibrator loads, names it, and predicts the floor."""
    model, paths = trained
    calibrator = ml_model_core.WinProbCalibrator(method=method, model=None)
    loaded = _load_legacy(model, "calibrator", calibrator, tmp_path, caplog)
    predictions = ml_model_predict.predict_week_margin_total(
        loaded, paths["predict"], pretty_output=False
    )

    assert f"'{method}' calibrator" in caplog.text
    assert not hasattr(loaded, "calibrator")
    np.testing.assert_allclose(
        predictions["home_win_prob"].to_numpy(), _floor(loaded, paths["predict"])
    )


@pytest.mark.parametrize("method", ["auto", "none"])
def test_both_spellings_of_the_floor_resolve_to_auto(method: str) -> None:
    """``auto`` is the documented value; ``none`` is accepted and means the same."""
    assert ml_model_core.resolve_calibration(method) == "auto"
    assert walk_forward.WalkForwardConfig(calibration=method).calibration == "auto"
    assert walk_forward.WalkForwardConfig().calibration == "auto"


@pytest.mark.parametrize("method", RETIRED_CALIBRATORS)
def test_a_retired_calibrator_is_refused_with_a_reason(method: str) -> None:
    """Asking a run for a retired calibrator fails and says the floor replaced it."""
    with pytest.raises(ValueError, match="retired"):
        ml_model_core.resolve_calibration(method)
    with pytest.raises(ValueError, match="retired"):
        walk_forward.WalkForwardConfig(calibration=method)


@pytest.mark.parametrize(
    ("module", "flag", "dest"),
    [
        (backtest, "--calibration", "calibration"),
        (train, "--win-prob-calibration", "win_prob_calibration"),
    ],
)
def test_the_commands_accept_only_the_floor(
    module: object, flag: str, dest: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backtest`` and ``train`` default to ``auto``, accept ``none`` and refuse the rest."""
    parse = backtest._parse_args if module is backtest else train._parse_args
    monkeypatch.setattr(sys, "argv", ["prog"])
    assert getattr(parse(), dest) == "auto"
    monkeypatch.setattr(sys, "argv", ["prog", flag, "none"])
    assert getattr(parse(), dest) == "none"
    monkeypatch.setattr(sys, "argv", ["prog", flag, "platt"])
    with pytest.raises(SystemExit):
        parse()


def test_the_train_job_form_offers_no_calibration_choice() -> None:
    """The web train job has one calibration, so its form does not ask."""
    names = {spec.name for spec in catalog.get_template("train").params}
    assert "win_prob_calibration" not in names


def test_a_saved_uncertainty_flag_is_ignored_and_the_floor_is_predicted(
    trained: tuple[MarginTotalModel, dict[str, Path]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A checkpoint saved with quantile-spread probabilities loads, says so, predicts the floor."""
    model, paths = trained
    loaded = _load_legacy(model, "win_prob_use_uncertainty", True, tmp_path, caplog)
    predictions = ml_model_predict.predict_week_margin_total(
        loaded, paths["predict"], pretty_output=False
    )

    assert "uncertainty-aware" in caplog.text
    assert not hasattr(loaded, "win_prob_use_uncertainty")
    np.testing.assert_allclose(
        predictions["home_win_prob"].to_numpy(), _floor(loaded, paths["predict"])
    )


@pytest.mark.parametrize("module", [backtest, train])
def test_the_uncertainty_option_is_gone(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--win-prob-uncertainty`` no longer parses; the floor uses one fixed sigma."""
    parse = backtest._parse_args if module is backtest else train._parse_args
    monkeypatch.setattr(sys, "argv", ["prog", "--win-prob-uncertainty"])
    with pytest.raises(SystemExit):
        parse()


def test_the_walk_forward_config_has_no_uncertainty_setting() -> None:
    """The walk-forward config and its record carry no probability alternative."""
    config = walk_forward.WalkForwardConfig()
    assert not hasattr(config, "win_prob_use_uncertainty")
    assert "win_prob_use_uncertainty" not in config.to_dict()
