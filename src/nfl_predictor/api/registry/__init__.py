"""Column metadata registry: labels, descriptions, formats, and polarity for every column."""

# Importing each column module registers its columns in REGISTRY.
from nfl_predictor.api.registry import betting, model, power, predictions
from nfl_predictor.api.registry.columns import REGISTRY, ColumnMeta, group_map, project

__all__ = [
    "REGISTRY",
    "ColumnMeta",
    "betting",
    "group_map",
    "model",
    "power",
    "predictions",
    "project",
]
