"""The final fit records its floor sigma, and prediction uses the recorded value."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_predictor import constants
from nfl_predictor.ml import artifacts, floor_sigma, ml_model_core
from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
from nfl_predictor.ml.ml_model_training import (
    train_margin_total_model,
    train_margin_total_model_with_report,
)

SEASONS = (2021, 2022, 2023)


def _games() -> pd.DataFrame:
    """Four games a week, weeks 1-4, over three seasons, with seeded scores."""
    rng = np.random.default_rng(5)
    rows = []
    for season in SEASONS:
        for week in (1, 2, 3, 4):
            for game in range(4):
                edge = float(rng.normal(0, 4))
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "game_id": f"{season}_{week:02d}_{game}",
                        "away_abbr": f"A{game}",
                        "home_abbr": f"H{game}",
                        "feat1": edge,
                        "feat2": float(rng.normal(0, 1)),
                        "away_score": 20,
                        "home_score": 20 + round(edge + float(rng.normal(0, 10))),
                    }
                )
    return pd.DataFrame(rows)


def _pool() -> floor_sigma.ErrorPool:
    """Return reference errors: 2016-2023, one game a week, 10 before 2022 and 20 after."""
    rows = [
        {
            "game_id": f"r{season}_{week}",
            "season": season,
            "week": week,
            "squared_error": 100.0 if season < 2022 else 400.0,
        }
        for season in range(2016, 2024)
        for week in (1, 2, 3, 4)
    ]
    return floor_sigma.ErrorPool(pd.DataFrame(rows), ("models/reference_a",))


def _train_kwargs(data_path: Path, **overrides: Any) -> dict[str, Any]:
    """Return small, fast final-fit options on the fixture dataset."""
    return {
        "data_path": data_path,
        "holdout_seasons": 0,
        "include_market": False,
        "max_cardinality_ratio": 0.5,
        "optuna_config": ml_model_core.OptunaConfig(
            enabled=False,
            timeout_seconds=1,
            n_trials=None,
            cv_splits=2,
            objective="brier",
            early_stopping_rounds=5,
            tree_method=None,
            device="cpu",
            storage=None,
            study_name=None,
            best_params_out=None,
            xgb_n_jobs=1,
        ),
        "market_transform": False,
        "market_anchor": False,
        "feature_start": "feat1",
        "feature_end": "feat2",
        "xgb_params_overrides": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        **overrides,
    }


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    """Write the fixture's completed games."""
    path = tmp_path / "completed_games_ml.csv"
    _games().to_csv(path, index=False)
    return path


def test_the_final_fit_records_the_sigma_for_the_week_it_predicts(data_path: Path) -> None:
    """The model carries the pool's sigma for the named week, and where it came from."""
    pool = _pool()

    model = train_margin_total_model(
        **_train_kwargs(data_path, floor_sigma_pool=pool, floor_sigma_week=(2023, 3))
    )

    assert model.floor_sigma is not None
    assert model.floor_sigma == floor_sigma.estimate(pool.errors, 2023, 3, sources=pool.sources)
    assert model.floor_sigma.fallback is False
    assert model.floor_sigma.sources == ("models/reference_a",)


def test_without_a_named_week_the_sigma_is_for_the_week_after_the_newest_game(
    data_path: Path,
) -> None:
    """The newest completed game is 2023 week 4, so the record is for 2023 week 5."""
    model = train_margin_total_model(**_train_kwargs(data_path, floor_sigma_pool=_pool()))

    assert model.floor_sigma is not None
    assert (model.floor_sigma.season, model.floor_sigma.week) == (2023, 5)
    assert model.floor_sigma.pool_games == len(_pool().errors)


def test_a_short_pool_records_the_fallback(data_path: Path) -> None:
    """A pool with fewer than three earlier seasons records the constant as a fallback."""
    short = floor_sigma.ErrorPool(_pool().errors.query("season >= 2022"), ("ref",))

    model = train_margin_total_model(
        **_train_kwargs(data_path, floor_sigma_pool=short, floor_sigma_week=(2024, 1))
    )

    assert model.floor_sigma is not None
    assert model.floor_sigma.fallback is True
    assert model.floor_sigma.sigma == constants.SCORE_DIFF_STD_DEV


def test_holdout_probabilities_use_each_weeks_sigma_from_the_pool(data_path: Path) -> None:
    """Holdout games are scored with the sigma the pool gives strictly before their week."""
    with_pool = train_margin_total_model_with_report(
        **_train_kwargs(data_path, holdout_seasons=1, floor_sigma_pool=_pool())
    )
    constant = train_margin_total_model_with_report(**_train_kwargs(data_path, holdout_seasons=1))

    pooled_brier = with_pool.metrics_report["metrics"]["holdout"]["brier"]
    constant_brier = constant.metrics_report["metrics"]["holdout"]["brier"]
    assert pooled_brier != constant_brier
    assert with_pool.metrics_report["floor_sigma"] == with_pool.model.floor_sigma.to_dict()


def test_weekly_probabilities_use_each_games_week() -> None:
    """Each game maps through the sigma of its own week."""
    errors = _pool().errors
    margin = np.array([3.0, 3.0])

    probs = floor_sigma.weekly_home_win_prob(
        margin, np.array([2022, 2023]), np.array([1, 1]), errors
    )

    assert probs[0] == norm.cdf(3.0 / 10.0)
    assert probs[1] == norm.cdf(3.0 / floor_sigma.estimate(errors, 2023, 1).sigma)


def _write_predict_rows(tmp_path: Path) -> Path:
    """Write the games of 2023 week 3 as the week to predict."""
    path = tmp_path / "week_03_games_to_predict.csv"
    games = _games()
    games[(games["season"] == 2023) & (games["week"] == 3)].to_csv(path, index=False)
    return path


def _raw_margin(model: ml_model_core.MarginTotalModel, predict_path: Path) -> np.ndarray:
    """Return the model's unrounded predicted margins for the week's games."""
    margin, _total = ml_model_core.predict_margin_total_from_model(model, pd.read_csv(predict_path))
    return margin


def test_prediction_uses_the_recorded_sigma_and_says_so(data_path: Path, tmp_path: Path) -> None:
    """A saved model predicts with its own sigma; margins and ranks match the constant path."""
    pool = _pool()
    model = train_margin_total_model(
        **_train_kwargs(data_path, floor_sigma_pool=pool, floor_sigma_week=(2023, 3))
    )
    assert model.floor_sigma is not None
    predict_path = _write_predict_rows(tmp_path)

    output = predict_week_margin_total(model, predict_path, pretty_output=False)
    without = predict_week_margin_total(
        replace(model, floor_sigma=None), predict_path, pretty_output=False
    )

    margin = _raw_margin(model, predict_path)
    np.testing.assert_array_equal(
        output["home_win_prob"], np.round(norm.cdf(margin / model.floor_sigma.sigma), 4)
    )
    assert (output["floor_sigma"] == model.floor_sigma.sigma).all()
    assert not output["floor_sigma_fallback"].any()
    for column in ("predicted_winner", "confidence_rank", "predicted_home_score"):
        assert output[column].tolist() == without[column].tolist()
    assert not np.allclose(output["home_win_prob"], without["home_win_prob"])


def test_a_model_saved_before_the_sigma_was_recorded_predicts_with_the_constant(
    data_path: Path, tmp_path: Path
) -> None:
    """An old checkpoint has no ``floor_sigma``: it loads and uses the constant, flagged."""
    model = train_margin_total_model(**_train_kwargs(data_path))
    checkpoint = tmp_path / "model.joblib"
    artifacts.save_model(checkpoint, model)
    loaded = ml_model_core.load_model_checkpoint(checkpoint, "margin_total")
    object.__delattr__(loaded, "floor_sigma")
    artifacts.save_model(checkpoint, loaded)

    old = ml_model_core.load_model_checkpoint(checkpoint, "margin_total")
    predict_path = _write_predict_rows(tmp_path)
    output = predict_week_margin_total(old, predict_path, pretty_output=False)

    assert old.floor_sigma is None
    margin = _raw_margin(old, predict_path)
    np.testing.assert_array_equal(
        output["home_win_prob"], np.round(norm.cdf(margin / constants.SCORE_DIFF_STD_DEV), 4)
    )
    assert (output["floor_sigma"] == constants.SCORE_DIFF_STD_DEV).all()
    assert output["floor_sigma_fallback"].all()


def test_the_metadata_records_the_models_sigma(tmp_path: Path) -> None:
    """The artifact contract: the saved model's metadata names its sigma and its sources."""
    record = floor_sigma.estimate(_pool().errors, 2023, 3, sources=("models/reference_a",))

    metadata = artifacts.build_metadata(
        created_at="2026-09-28T00:00:00+00:00",
        run_id="run",
        dataset_hash="hash",
        config={},
        floor_sigma=record.to_dict(),
    )
    path = tmp_path / "metadata.json"
    artifacts.write_json(path, metadata)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert floor_sigma.FloorSigma.from_dict(payload["floor_sigma"]) == record
