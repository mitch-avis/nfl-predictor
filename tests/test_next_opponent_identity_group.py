"""Tests for the next opponent's team identity: left out of the model, and its ablation group."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core, ml_model_training, ml_model_xgb_utils, walk_forward
from nfl_predictor.ml.feature_spec import (
    FeatureSelection,
    apply_feature_spec,
    build_feature_spec,
    build_preprocessor,
)
from nfl_predictor.utils.polars.finalize import build_final_column_order
from nfl_predictor.weekly_run import stage1

if TYPE_CHECKING:
    from pathlib import Path

GROUP = "next_opponent_identity"
PAIR = ["away_next_opponent_abbr", "home_next_opponent_abbr"]
ROWS = 256
TEAMS = sorted(set(constants.ALIAS_TO_CANONICAL.values()))
STRING_FEATURES = {
    "stadium_surface": ("grass", "fieldturf", "a_turf"),
    "stadium_type": ("outdoors", "dome", "retractable"),
}


def _schema_frame() -> pd.DataFrame:
    """Return a frame with every published column, typed the way the real dataset loads.

    Inside the model's feature range the real dataset has four string columns: the stadium
    surface and type, and the next-opponent pair. Everything else there is numeric.
    """
    rng = np.random.default_rng(7)
    data: dict[str, object] = {}
    for column in build_final_column_order():
        if column in PAIR:
            data[column] = [TEAMS[int(i) % len(TEAMS)] for i in rng.integers(0, 10**6, ROWS)]
        elif column in STRING_FEATURES:
            choices = STRING_FEATURES[column]
            data[column] = [choices[int(i)] for i in rng.integers(0, len(choices), ROWS)]
        else:
            data[column] = rng.normal(size=ROWS)
    frame = pd.DataFrame(data)
    frame["season"] = 2024
    frame["week"] = 3
    frame["game_type"] = "REG"
    return frame


def _frame_after_group_drop(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, groups: tuple[str, ...]
) -> pd.DataFrame:
    """Return the games a walk-forward run trains on after it drops ``groups``."""
    captured: dict[str, pd.DataFrame] = {}

    def capture(df: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
        captured["df"] = df
        msg = "stop after the drop"
        raise RuntimeError(msg)

    monkeypatch.setattr(walk_forward, "filter_regular_season", capture)
    config = walk_forward.WalkForwardConfig(
        disabled_feature_groups=groups, xgb_params_overrides={"device": "cpu"}
    )
    with pytest.raises(RuntimeError, match="stop after the drop"):
        walk_forward.run_walk_forward_backtest(frame, config)
    return captured["df"]


def _feature_matrix(
    frame: pd.DataFrame, *, disable_pruning: bool = walk_forward.WalkForwardConfig().disable_pruning
) -> pd.DataFrame:
    """Return the encoded matrix a walk-forward week fits on, with its column names.

    Pruning follows the walk-forward default unless ``disable_pruning`` says otherwise.
    """
    spec = build_feature_spec(
        frame,
        FeatureSelection(
            include_market=True,
            max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
            market_transform=True,
            disable_pruning=disable_pruning,
        ),
    )
    preprocessor = build_preprocessor(spec, for_tree=True)
    matrix = ml_model_xgb_utils.fit_transform_matrix(preprocessor, apply_feature_spec(frame, spec))
    dense = matrix.toarray() if isinstance(matrix, sparse.csr_matrix) else np.asarray(matrix)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(dense, columns=names)


def test_group_resolves_to_exactly_the_next_opponent_pair() -> None:
    """The group names the two identity columns of the published schema and nothing else."""
    # Arrange
    order = build_final_column_order()

    # Act
    resolved = walk_forward.resolve_feature_group_columns(order, [GROUP])

    # Assert
    assert resolved == PAIR
    for column in ("away_next_opponent_win_pct", "home_next_opponent_win_pct"):
        assert column in order


def test_group_is_disjoint_from_every_other_group() -> None:
    """No other ablation group drops either identity column."""
    # Arrange
    order = build_final_column_order()
    others = [group for group in constants.FEATURE_GROUP_COLUMN_MARKERS if group != GROUP]

    # Act
    overlaps = {
        other: set(walk_forward.resolve_feature_group_columns(order, [other])) & set(PAIR)
        for other in others
    }

    # Assert
    assert {other: overlap for other, overlap in overlaps.items() if overlap} == {}


def test_feature_matrix_is_unchanged_when_the_group_is_not_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no group disabled, the run fits on exactly the matrix the full frame gives."""
    # Arrange
    frame = _schema_frame()

    # Act
    after = _frame_after_group_drop(monkeypatch, frame, ())

    # Assert
    pd.testing.assert_frame_equal(after, frame)
    pd.testing.assert_frame_equal(_feature_matrix(after), _feature_matrix(frame))


def test_disabling_the_group_removes_only_the_pair_and_its_one_hot_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With pruning off, the group takes out the pair's encoded columns and nothing else."""
    # Arrange
    frame = _schema_frame()
    full = _feature_matrix(frame, disable_pruning=True)

    # Act
    after = _frame_after_group_drop(monkeypatch, frame, (GROUP,))
    ablated = _feature_matrix(after, disable_pruning=True)

    # Assert
    assert set(frame.columns) - set(after.columns) == set(PAIR)
    removed = [name for name in full.columns if name not in set(ablated.columns)]
    pair_encodings = [name for name in full.columns if any(f"__{c}_" in name for c in PAIR)]
    assert removed == pair_encodings
    assert len(pair_encodings) == 2 * len(TEAMS)
    assert "num__away_next_opponent_win_pct" in ablated.columns
    pd.testing.assert_frame_equal(ablated, full[list(ablated.columns)])


def test_default_feature_matrix_equals_the_matrix_with_the_group_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By default a week fits on exactly the matrix that disabling the group gives."""
    # Arrange
    frame = _schema_frame()
    group_disabled = _feature_matrix(_frame_after_group_drop(monkeypatch, frame, (GROUP,)))

    # Act
    default = _feature_matrix(frame)

    # Assert
    pd.testing.assert_frame_equal(default, group_disabled)


def test_final_fit_trains_without_the_pair(tmp_path: Path) -> None:
    """The final fit (train and the weekly run) leaves the pair out and keeps the win share."""
    # Arrange
    data_path = tmp_path / "completed_games_ml.csv"
    _schema_frame().to_csv(data_path, index=False)
    options = ml_model_training.TrainingOptions(
        data_path=data_path,
        holdout_seasons=0,
        include_market=True,
        max_cardinality_ratio=0.5,
        optuna_config=ml_model_core.OptunaConfig(
            enabled=False,
            timeout_seconds=0,
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
        market_anchor=False,
        xgb_params_overrides={"n_estimators": 2, "max_depth": 2},
    )

    # Act
    model = ml_model_training.train_margin_total_model(options)

    # Assert
    features = model.feature_spec.feature_columns
    assert set(PAIR) & set(features) == set()
    assert {"away_next_opponent_win_pct", "home_next_opponent_win_pct"} <= set(features)


def test_weekly_walk_forward_prunes_and_disables_no_group() -> None:
    """The weekly run's stage 1 keeps the default pruning, so it leaves the pair out too."""
    # Arrange
    options = stage1.ProductionOptions(
        eval_last_n_seasons=3,
        wf_start_week=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_seasons=None,
        market_mode="hybrid",
        xgb_params_overrides={"device": "cpu"},
        include_quantiles=False,
    )

    # Act
    config = stage1.production_walk_forward_config(options)

    # Assert
    assert config.disable_pruning is False
    assert config.disabled_feature_groups == ()
