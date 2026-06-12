"""Tests for feature importance reporting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.ml import feature_importance, ml_model_core


def test_feature_importance_report_margin_total() -> None:
    """Feature importance report returns aligned gain/weight arrays."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.normal(size=40),
        }
    )
    spec = ml_model_core.FeatureSpec(
        feature_columns=["f1", "f2"],
        categorical_columns=[],
        numeric_columns=["f1", "f2"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="f1",
        feature_end="f2",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )
    preprocessor = ml_model_core._build_preprocessor(spec, for_tree=True)
    x_matrix = preprocessor.fit_transform(df)
    y_margin = rng.normal(size=40)
    y_total = rng.normal(size=40)
    params = ml_model_core._resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 15,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )
    margin_model, total_model = ml_model_core._fit_margin_total_models(
        x_matrix,
        y_margin,
        y_total,
        params,
    )
    model = ml_model_core.MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=("away_score", "home_score"),
        calibrator=None,
    )

    report = feature_importance.build_feature_importance_report(model)
    assert report is not None
    assert report["model_kind"] == "margin_total"
    assert "feature_names" in report
    assert "models" in report
    assert set(report["models"].keys()) == {"margin", "total"}
    feature_names = report["feature_names"]
    for key in ("margin", "total"):
        assert len(report["models"][key]["gain"]) == len(feature_names)
        assert len(report["models"][key]["weight"]) == len(feature_names)
