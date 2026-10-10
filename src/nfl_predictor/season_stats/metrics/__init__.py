"""Metric registry package — the single source of truth for published stats."""

from nfl_predictor.season_stats.metrics.catalog import get_registry
from nfl_predictor.season_stats.metrics.registry import MetricRegistry, RegistryValidationError
from nfl_predictor.season_stats.metrics.schema import (
    CategoryDef,
    MetricDef,
    ResolvedColumn,
)

__all__ = [
    "CategoryDef",
    "MetricDef",
    "MetricRegistry",
    "RegistryValidationError",
    "ResolvedColumn",
    "get_registry",
]
