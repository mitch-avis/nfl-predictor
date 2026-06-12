"""Compatibility facade for Polars-first ETL helpers.

Historically, this project stored most Polars ETL and feature engineering utilities in this
single module. It has been split into smaller modules under `nfl_predictor.utils.polars` to keep
files under the pylint `max-module-lines` limit while preserving the import path expected by the
pipeline and tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfl_predictor.utils.polars import features as _features
from nfl_predictor.utils.polars import finalize as _finalize
from nfl_predictor.utils.polars import loaders as _loaders
from nfl_predictor.utils.polars import teamrankings as _teamrankings


def __getattr__(name: str) -> Any:
    """Forward attribute access to the split implementation modules."""
    for module in (_features, _loaders, _teamrankings, _finalize):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names: set[str] = set(globals().keys())
    for module in (_features, _loaders, _teamrankings, _finalize):
        names.update(dir(module))
    return sorted(names)


if TYPE_CHECKING:
    # Import frequently used symbols for IDEs/type-checkers.
    pass
