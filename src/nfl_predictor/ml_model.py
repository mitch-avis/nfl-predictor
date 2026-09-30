"""Compatibility facade for ML training, prediction, and CLI.

The implementation lives in the modules under `nfl_predictor.ml`; this module re-exports the
names scripts and tests import from here and keeps the ``python -m nfl_predictor.ml_model``
form of the training CLI.
"""

from __future__ import annotations

import importlib

from nfl_predictor.ml.ml_model_core import (
    CALIBRATION_FLOOR,
    DEFAULT_FEATURE_END_COLUMN,
    DEFAULT_FEATURE_START_COLUMN,
    DEFAULT_QUANTILES,
    DEFAULT_XGB_PARAMS,
    FeatureSelection,
    FeatureSpec,
    FitData,
    OptunaConfig,
    apply_feature_spec,
    build_feature_spec,
    build_prediction_output,
    build_preprocessor,
    derive_scores_from_margin_total,
    fit_margin_total_models,
    fit_quantile_models,
    get_market_baseline,
    get_target_columns,
    margin_to_home_win_prob,
    normalize_no_vig,
    predict_xgb,
    prepare_margin_total_targets_with_anchor,
    resolve_calibration,
    resolve_xgb_params,
    validate_quantiles,
)
from nfl_predictor.ml.ml_model_training import (
    train_margin_total_model_with_report,
)

__all__ = [
    "CALIBRATION_FLOOR",
    "DEFAULT_FEATURE_END_COLUMN",
    "DEFAULT_FEATURE_START_COLUMN",
    "DEFAULT_QUANTILES",
    "DEFAULT_XGB_PARAMS",
    "FeatureSelection",
    "FeatureSpec",
    "FitData",
    "OptunaConfig",
    "apply_feature_spec",
    "build_feature_spec",
    "build_prediction_output",
    "build_preprocessor",
    "derive_scores_from_margin_total",
    "fit_margin_total_models",
    "fit_quantile_models",
    "get_market_baseline",
    "get_target_columns",
    "main",
    "margin_to_home_win_prob",
    "normalize_no_vig",
    "predict_xgb",
    "prepare_margin_total_targets_with_anchor",
    "resolve_calibration",
    "resolve_xgb_params",
    "train_margin_total_model_with_report",
    "validate_quantiles",
]


def main() -> None:
    """CLI entrypoint wrapper for the legacy `nfl_predictor.ml_model` path."""
    _module = importlib.import_module("nfl_predictor.cli.train")
    _module.main()


if __name__ == "__main__":
    main()
