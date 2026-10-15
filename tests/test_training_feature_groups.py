"""The final fit leaves out the columns of disabled feature groups, and its model says so.

``TrainingOptions.disabled_feature_groups`` names groups of
``constants.FEATURE_GROUP_COLUMN_MARKERS``; training drops their columns right after loading the
games, through the resolver a walk-forward uses, and the result and the saved model's
``metadata.json`` record the groups and the columns they removed. The model's feature spec holds
the columns it trained on, so a saved model predicts without knowing the setting. Every test
runs on the weekly fixture with four groups' columns added (``tests/feature_group_fixture.py``),
and the ported ratings beside them wherever the strength snapshot is left out, with a ten-tree
budget to stay fast.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nfl_predictor.ml import ml_model_core, ml_model_training
from nfl_predictor.ml.ml_model_core import OptunaConfig
from nfl_predictor.ml.ml_model_predict import predict_week_margin_total
from tests.feature_group_fixture import GROUP_COLUMNS, build_group_fixture
from tests.weekly_fixture import build_fixture

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.ml.ml_model_core import TrainingResult

PREDICTION_COLUMNS = ("predicted_margin_raw", "predicted_total_raw", "home_win_prob")


def _options(data_path: Path, groups: tuple[str, ...]) -> ml_model_training.TrainingOptions:
    """Return production-like training options on the fixture, leaving out ``groups``."""
    return ml_model_training.TrainingOptions(
        data_path=data_path,
        holdout_seasons=0,
        include_market=True,
        max_cardinality_ratio=0.5,
        optuna_config=OptunaConfig(
            enabled=False,
            timeout_seconds=1,
            n_trials=None,
            cv_splits=2,
            objective="brier",
            early_stopping_rounds=10,
            tree_method=None,
            device="cpu",
            storage=None,
            study_name=None,
            best_params_out=None,
            xgb_n_jobs=1,
        ),
        market_transform=True,
        market_anchor=True,
        xgb_params_overrides={"n_estimators": 10},
        disabled_feature_groups=groups,
    )


def _train(data_path: Path, groups: tuple[str, ...]) -> TrainingResult:
    """Train on the fixture without ``groups`` and return the result."""
    return ml_model_training.train_margin_total_model_with_report(_options(data_path, groups))


def _columns(*groups: str) -> list[str]:
    """Return the fixture's columns of ``groups``, sorted."""
    return sorted(column for group in groups for column in GROUP_COLUMNS[group])


def test_a_disabled_group_leaves_none_of_its_columns_in_the_features(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)

    # Act
    result = _train(paths["completed"], ("strength",))

    # Assert
    assert not set(result.feature_list) & set(GROUP_COLUMNS["strength"])
    assert set(result.feature_list) >= set(_columns("elo", "own_elo", "own_elo_wt"))


def test_the_result_records_the_groups_and_the_columns_they_removed(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)

    # Act
    result = _train(paths["completed"], ("elo", "own_elo_wt"))

    # Assert
    assert result.disabled_feature_groups == ("elo", "own_elo_wt")
    assert result.dropped_feature_group_columns == tuple(_columns("elo", "own_elo_wt"))


def test_no_disabled_group_drops_nothing_and_records_empty_lists(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)

    # Act
    result = _train(paths["completed"], ())

    # Assert
    assert result.disabled_feature_groups == ()
    assert result.dropped_feature_group_columns == ()
    assert set(result.feature_list) >= set(_columns(*GROUP_COLUMNS))


def test_an_unknown_group_stops_training_and_names_the_group(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)

    # Act & Assert
    with pytest.raises(ValueError, match="not_a_group"):
        _train(paths["completed"], ("elo", "not_a_group"))


def test_the_saved_metadata_names_the_groups_and_their_columns(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)
    result = _train(paths["completed"], ("own_elo", "strength"))
    record = ml_model_training.TrainingRecord(
        run_id="groups", created_at="now", dataset_hash="hash", config_payload={}
    )

    # Act
    run_paths = ml_model_training.write_training_artifacts(result, record, tmp_path / "run")

    # Assert
    metadata = json.loads(run_paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["disabled_feature_groups"] == ["own_elo", "strength"]
    assert metadata["dropped_feature_group_columns"] == _columns("own_elo", "strength")
    assert not set(metadata["feature_list"]) & set(_columns("own_elo", "strength"))


def test_a_saved_model_predicts_without_reading_the_dropped_columns(tmp_path: Path) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    result = _train(paths["completed"], ("elo",))
    record = ml_model_training.TrainingRecord(
        run_id="groups", created_at="now", dataset_hash="hash", config_payload={}
    )
    run_paths = ml_model_training.write_training_artifacts(result, record, tmp_path / "run")
    model = ml_model_core.load_model_checkpoint(run_paths.model_path, "margin_total")
    expected = predict_week_margin_total(model, paths["predict"], pretty_output=False)
    games = pd.read_csv(paths["predict"])
    games[list(GROUP_COLUMNS["elo"])] = -999.0
    scrambled = tmp_path / "scrambled_week.csv"
    games.to_csv(scrambled, index=False)

    # Act
    predictions = predict_week_margin_total(model, scrambled, pretty_output=False)

    # Assert
    pd.testing.assert_frame_equal(
        predictions[list(PREDICTION_COLUMNS)], expected[list(PREDICTION_COLUMNS)]
    )


def test_a_group_the_data_lacks_drops_nothing_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    paths = build_fixture(tmp_path)

    # Act
    result = _train(paths["completed"], ("own_elo",))

    # Assert
    assert result.dropped_feature_group_columns == ()
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("own_elo" in message and "match no column" in message for message in warnings)
