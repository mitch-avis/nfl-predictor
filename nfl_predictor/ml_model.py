"""Compatibility facade for ML training, prediction, and CLI.

Historically, `nfl_predictor.ml_model` was a single large module. It has been split into smaller
modules under `nfl_predictor.ml` to keep files under the pylint `max-module-lines` limit while
preserving the public API expected by scripts and unit tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfl_predictor.ml import ml_model_core as _core
from nfl_predictor.ml import ml_model_predict as _predict
from nfl_predictor.ml import ml_model_training as _training
from nfl_predictor.ml.ml_model_cli import main


def __getattr__(name: str) -> Any:
    """Forward attribute access to the split implementation modules."""

    for module in (_core, _training, _predict):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names: set[str] = set(globals().keys())
    for module in (_core, _training, _predict):
        names.update(dir(module))
    return sorted(names)


if TYPE_CHECKING:
    # Import key symbols for type-checkers and IDEs.
    from nfl_predictor.ml.ml_model_core import (  # noqa: F401
        BlendedMarginTotalModel,
        BlendLayer,
        FeatureSpec,
        MarginTotalModel,
        MarketProbConfig,
        OptunaConfig,
        ScoreModel,
        TrainingResult,
        WinProbCalibrator,
        _build_preprocessor,
        _drop_injury_feature_columns,
        _fit_blend_ridge_constrained,
        _fit_margin_total_models,
        _prepare_margin_total_targets_with_anchor,
        _resolve_xgb_params,
        _should_enable_injury_features,
    )


if __name__ == "__main__":
    main()
