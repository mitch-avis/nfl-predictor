"""Column metadata registry: labels, descriptions, formats, and polarity for every column."""

from nfl_predictor.api.registry import betting, model, power, predictions  # noqa: F401 - registers
from nfl_predictor.api.registry.columns import REGISTRY, ColumnMeta, group_map, project

__all__ = ["REGISTRY", "ColumnMeta", "group_map", "project"]
