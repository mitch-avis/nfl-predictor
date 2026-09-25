"""Tests for the blended margin/total model's helper paths.

A blended model is the team model plus the market line, combined by a ridge blend layer. These
tests pin how a market-probability config reaches it, how an older checkpoint is patched on
load, how the blend trainer runs Optuna, and what the power rankings do with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from nfl_predictor.ml import ml_model_core, ml_model_training
from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    MarketProbConfig,
    OptunaConfig,
)
from nfl_predictor.reporting import power_rankings


def _feature_spec() -> FeatureSpec:
    """Return a one-column feature spec."""
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


def _team_model() -> MarginTotalModel:
    """Return a team model whose estimators are never called."""
    return MarginTotalModel(
        preprocessor=cast(ColumnTransformer, object()),
        feature_spec=_feature_spec(),
        margin_model=cast(xgb.XGBRegressor, object()),
        total_model=cast(xgb.XGBRegressor, object()),
        target_columns=("away_score", "home_score"),
        calibrator=None,
    )


def _blended_model(**overrides: Any) -> BlendedMarginTotalModel:
    """Return a blended model built from ``_team_model``."""
    fields: dict[str, Any] = {
        "team_model": _team_model(),
        "blend_layer": BlendLayer(
            margin_model=cast(Ridge, object()), total_model=cast(Ridge, object())
        ),
        "calibrator": None,
        "target_columns": ("away_score", "home_score"),
    }
    fields.update(overrides)
    return BlendedMarginTotalModel(**fields)


def test_a_market_probability_config_reaches_the_blend_and_its_team_model() -> None:
    """The config is set on the blended model and on the team model inside it."""
    config = MarketProbConfig(blend_weight=0.2, clamp_delta=0.1)

    updated = ml_model_core._with_market_prob_config(_blended_model(), config)

    assert isinstance(updated, BlendedMarginTotalModel)
    assert updated.market_prob_config == config
    assert updated.team_model.market_prob_config == config


def test_a_market_probability_config_reaches_a_margin_total_model() -> None:
    """A plain margin/total model takes the config directly."""
    config = MarketProbConfig(blend_weight=0.2, clamp_delta=0.1)

    updated = ml_model_core._with_market_prob_config(_team_model(), config)

    assert updated.market_prob_config == config


def test_no_market_probability_config_leaves_the_model_unchanged() -> None:
    """Without a config the model is returned as it is."""
    model = _blended_model()

    assert ml_model_core._with_market_prob_config(model, None) is model


def test_an_older_blended_checkpoint_gains_the_newer_fields() -> None:
    """Fields added after a checkpoint was saved are filled in with ``None`` on load."""
    model = _blended_model()
    team = model.team_model
    for name in ("margin_quantile_models", "total_quantile_models", "quantiles"):
        object.__setattr__(team, name, "stale")
        object.__delattr__(team, name)
    object.__setattr__(model, "optuna_summary", "stale")
    object.__delattr__(model, "optuna_summary")

    patched = ml_model_core._ensure_backward_compatible_model(model)

    assert patched is model
    assert patched.team_model.quantiles is None
    assert patched.optuna_summary is None


def test_the_blend_trainer_tunes_the_team_model_once_with_the_whole_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuning a blend runs one Optuna study with the full timeout and the given study name."""

    class _StopAfterTuningError(Exception):
        """Stop training once the tuning call has been captured."""

    calls: list[OptunaConfig] = []

    def _fake_search(*_args: object, **kwargs: Any) -> tuple[dict, dict, dict]:
        """Capture the tuning config, then stop."""
        calls.append(kwargs["optuna_config"])
        raise _StopAfterTuningError()

    games = pd.DataFrame(
        {
            "season": [2021] * 4 + [2022] * 4,
            "week": [1, 2, 3, 4] * 2,
            "away_score": [10, 12, 14, 9, 7, 17, 20, 13],
            "home_score": [20, 18, 21, 16, 10, 24, 17, 14],
        }
    )
    monkeypatch.setattr(ml_model_training, "_load_games", lambda _path: games)
    monkeypatch.setattr(ml_model_training, "_run_optuna_search", _fake_search)

    optuna_config = OptunaConfig(
        enabled=True,
        timeout_seconds=600,
        n_trials=None,
        cv_splits=2,
        objective="brier",
        early_stopping_rounds=50,
        tree_method=None,
        device=None,
        storage="sqlite:///optuna.db",
        study_name="study",
        best_params_out=None,
    )
    with pytest.raises(_StopAfterTuningError):
        ml_model_training.train_blended_margin_total_model(
            data_path=tmp_path / "games.csv",
            holdout_seasons=0,
            calibration_seasons=1,
            calibration_weeks=0,
            max_cardinality_ratio=0.5,
            win_prob_calibration="none",
            optuna_config=optuna_config,
            market_transform=False,
            market_anchor=False,
            market_prob_config=None,
        )

    assert len(calls) == 1
    assert calls[0].timeout_seconds == 600
    assert calls[0].study_name == "study"


def test_power_rankings_cannot_predict_with_a_blended_model(tmp_path: Path) -> None:
    """A blended model keeps its feature spec on the team model, so rankings reject it.

    This pins today's behavior; ranking with a blended model has never worked.
    """
    data_ml = tmp_path / "all_data_ml.csv"
    pd.DataFrame({"season": [2024], "week": [5], "away_abbr": ["A"], "home_abbr": ["B"]}).to_csv(
        data_ml, index=False
    )

    with pytest.raises(ValueError, match="missing feature_spec"):
        power_rankings._predict_future_games(
            _blended_model(),
            model_kind="blend",
            data_ml=data_ml,
            season=2024,
            through_week=4,
        )
