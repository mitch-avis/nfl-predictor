"""Compatibility facade for ML utility helpers.

The implementation now lives in `nfl_predictor.ml.ml_utils`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfl_predictor.ml import ml_utils as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(dir(_impl)))


if TYPE_CHECKING:
    from nfl_predictor.ml.ml_utils import (  # noqa: F401
        display_predictions,
        display_weekly_predictions,
        flatten_dict,
        nested_dict_to_df,
    )
