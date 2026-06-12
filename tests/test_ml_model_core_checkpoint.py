"""Unit tests for model checkpoint loading utilities.

These tests avoid training by using minimal, picklable placeholder objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import ml_model_core as core


def _dummy_feature_spec() -> core.FeatureSpec:
    """Build a minimal FeatureSpec suitable for unit tests."""
    cols = ["feat1", "feat2"]
    return core.FeatureSpec(
        feature_columns=cols,
        categorical_columns=[],
        numeric_columns=cols,
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="feat1",
        feature_end="feat2",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def test_load_model_checkpoint_loads_expected_kind(tmp_path: Path) -> None:
    """Loads a MarginTotalModel checkpoint when model_kind matches."""
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metadata.json"

    model = core.MarginTotalModel(
        preprocessor=cast(ColumnTransformer, None),
        feature_spec=_dummy_feature_spec(),
        margin_model=cast(xgb.XGBRegressor, None),
        total_model=cast(xgb.XGBRegressor, None),
        target_columns=("away_score", "home_score"),
        calibrator=None,
        market_anchor=False,
        market_prob_config=core.MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        xgb_params={"n_estimators": 1},
    )
    joblib.dump(model, model_path)

    # Include a metadata file to exercise the version-mismatch check path.
    meta_path.write_text(
        '{"library_versions": {"xgboost": "0.0.0"}}',
        encoding="utf-8",
    )

    loaded = core.load_model_checkpoint(model_path, "margin_total")
    assert isinstance(loaded, core.MarginTotalModel)


def test_load_model_checkpoint_type_mismatch_raises(tmp_path: Path) -> None:
    """Raises when a checkpoint type doesn't match the requested model_kind."""
    model_path = tmp_path / "model.joblib"

    score_model = core.ScoreModel(
        preprocessor=cast(ColumnTransformer, None),
        feature_spec=_dummy_feature_spec(),
        away_model=cast(xgb.XGBRegressor, None),
        home_model=cast(xgb.XGBRegressor, None),
        target_columns=("away_score", "home_score"),
    )
    joblib.dump(score_model, model_path)

    with pytest.raises(ValueError, match=r"type mismatch"):
        core.load_model_checkpoint(model_path, "margin_total")


def test_load_model_checkpoint_unknown_kind_raises(tmp_path: Path) -> None:
    """Raises ValueError for unknown model_kind values."""
    model_path = tmp_path / "model.joblib"
    joblib.dump({"x": 1}, model_path)

    with pytest.raises(ValueError, match=r"Unknown model kind"):
        core.load_model_checkpoint(model_path, "not_a_kind")
