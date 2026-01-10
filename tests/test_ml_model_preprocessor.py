"""Tests for the tree-friendly preprocessing pipeline."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from nfl_predictor import ml_model

# pylint: disable=protected-access


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


def test_preprocessor_sparse_with_categorical() -> None:
    """One-hot categoricals should keep the transformed matrix sparse for trees."""
    df = pd.DataFrame({"num_feature": [1.0, 2.0], "cat_feature": ["A", "B"]})
    spec = _make_feature_spec()

    preprocessor = ml_model._build_preprocessor(spec, for_tree=True)
    features = preprocessor.fit_transform(df)

    assert sparse.issparse(features)


def test_preprocessor_handles_missing_numeric() -> None:
    """Numeric NaNs are imputed so downstream training can proceed."""
    df = pd.DataFrame({"num_feature": [1.0, np.nan, 3.0, 4.0]})
    spec = ml_model.FeatureSpec(
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

    preprocessor = ml_model._build_preprocessor(spec, for_tree=True)
    features = preprocessor.fit_transform(df)

    if sparse.issparse(features):
        to_csr = getattr(features, "tocsr", None)
        csr_matrix = cast(Any, to_csr)() if to_csr is not None else features
        data = csr_matrix.data
        assert not np.isnan(np.asarray(data, dtype=float)).any()
    else:
        dense = np.asarray(features, dtype=float)
        assert not np.isnan(dense).any()
    params = ml_model._resolve_xgb_params(
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
    margin_model, total_model = ml_model._fit_margin_total_models(
        features,
        y_margin,
        y_total,
        params,
    )

    assert hasattr(margin_model, "feature_importances_")
    assert hasattr(total_model, "feature_importances_")
    assert hasattr(total_model, "feature_importances_")
