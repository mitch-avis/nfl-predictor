"""Tests for the tree-friendly preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from nfl_predictor import ml_model
from nfl_predictor.ml.ml_model_xgb_utils import fit_transform_matrix


def _make_feature_spec() -> ml_model.FeatureSpec:
    """Build a minimal FeatureSpec with one numeric and one categorical feature."""
    return ml_model.FeatureSpec(
        feature_columns=["num_feature", "cat_feature"],
        categorical_columns=["cat_feature"],
        numeric_columns=["num_feature"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="num_feature",
        feature_end="cat_feature",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def _make_numeric_feature_spec() -> ml_model.FeatureSpec:
    """Build a minimal FeatureSpec with one numeric feature."""
    return ml_model.FeatureSpec(
        feature_columns=["num_feature"],
        categorical_columns=[],
        numeric_columns=["num_feature"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="num_feature",
        feature_end="num_feature",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def test_preprocessor_sparse_with_categorical() -> None:
    """One-hot categoricals should keep the transformed matrix sparse for trees."""
    # Arrange
    df = pd.DataFrame({"num_feature": [1.0, 2.0], "cat_feature": ["A", "B"]})
    preprocessor = ml_model.build_preprocessor(_make_feature_spec(), for_tree=True)

    # Act
    features = preprocessor.fit_transform(df)

    # Assert
    assert sparse.issparse(features)


def test_preprocessor_handles_missing_numeric() -> None:
    """Numeric NaNs are imputed so downstream training can proceed."""
    # Arrange
    df = pd.DataFrame({"num_feature": [1.0, np.nan, 3.0, 4.0]})
    preprocessor = ml_model.build_preprocessor(_make_numeric_feature_spec(), for_tree=True)

    # Act
    features = fit_transform_matrix(preprocessor, df)

    # Assert
    if isinstance(features, sparse.csr_matrix):
        assert not np.isnan(np.asarray(features.data, dtype=float)).any()
    else:
        dense = np.asarray(features, dtype=float)
        assert not np.isnan(dense).any()


def test_models_train_on_the_imputed_matrix() -> None:
    """The margin and total models fit on a matrix whose numeric NaNs were imputed."""
    # Arrange
    df = pd.DataFrame({"num_feature": [1.0, np.nan, 3.0, 4.0]})
    preprocessor = ml_model.build_preprocessor(_make_numeric_feature_spec(), for_tree=True)
    features = fit_transform_matrix(preprocessor, df)
    params = ml_model.resolve_xgb_params(
        ml_model.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
        },
    )
    y_margin = np.array([1.0, -2.0, 0.5, 3.0])
    y_total = np.array([40.0, 38.0, 42.0, 44.0])

    # Act
    margin_model, total_model = ml_model.fit_margin_total_models(
        ml_model.FitData(features, y_margin, y_total), params
    )

    # Assert
    assert hasattr(margin_model, "feature_importances_")
    assert hasattr(total_model, "feature_importances_")
