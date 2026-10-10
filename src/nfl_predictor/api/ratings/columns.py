"""Column metadata for the ratings tables, from the metric registry.

Every column of a season table resolves against the metric registry
(:func:`nfl_predictor.season_stats.metrics.get_registry`), directly or through its prefix and
suffix rules. This module turns a resolved column into the app's column format, so the existing
table component labels, describes, sorts and shades it, and adds the registry's own fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from nfl_predictor.api.registry.columns import jsonable
from nfl_predictor.api.schemas.ratings import MetricColumnOut, MetricTableOut
from nfl_predictor.season_stats.metrics import get_registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from nfl_predictor.api.schemas.common import ColumnKind

STICKY_COLUMNS = frozenset({"team", "qb_name"})
"""The name column of each table, pinned while a wide table scrolls sideways."""
_IDENTITY_GROUP = "identity"
_UNSHADED_SHAPES = frozenset({"id", "flag"})


class ColumnOverride(TypedDict, total=False):
    """Registry text a table replaces where its column means something narrower."""

    label: str
    full_name: str
    description: str


def _kind(shape: str, dtype: pl.DataType, *, percent: bool) -> ColumnKind:
    """Return how the page formats a column: by its shape first, then by its Parquet type."""
    if shape == "id":
        return "text"
    if shape == "flag":
        return "bool"
    if percent:
        return "prob"
    if dtype.is_integer():
        return "int"
    return "float"


def metric_column(
    column: str,
    dtype: pl.DataType,
    group: str,
    override: ColumnOverride | None = None,
) -> MetricColumnOut | None:
    """Return one column's metadata, or ``None`` when the registry cannot resolve it.

    Args:
        column: The data column.
        dtype: Its Parquet type, which tells whole numbers from decimals.
        group: The payload group the column is shown in.
        override: Registry text to replace for this table.

    Returns:
        The column in the app's format: graded and contextual numbers carry a gradient (its
        direction from the registry's polarity), and the identity group's name column is pinned.

    """
    resolved = get_registry().resolve_column(column)
    if resolved is None:
        return None
    shape = resolved.base.shape
    kind = _kind(shape, dtype, percent=resolved.base.percent)
    numeric = kind in {"int", "float", "prob"}
    meta = MetricColumnOut(
        key=column,
        label=resolved.label,
        description=resolved.description,
        group=group,
        kind=kind,
        polarity=resolved.polarity,
        heatmap=numeric and resolved.polarity != "neutral" and shape not in _UNSHADED_SHAPES,
        sticky=group == _IDENTITY_GROUP and column in STICKY_COLUMNS,
        full_name=resolved.full_name,
        category=resolved.category,
        subcategory=resolved.subcategory,
        contextual=resolved.contextual,
        shape=shape,
        percent=resolved.base.percent,
    )
    return meta.model_copy(update=dict(override)) if override else meta


def _cell(value: object) -> object:
    """Return a JSON-safe cell, list cells element by element."""
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return jsonable(value)


def metric_table(
    frame: pl.DataFrame,
    groups: Mapping[str, Sequence[str]],
    overrides: Mapping[str, ColumnOverride] | None = None,
    references: Mapping[str, Sequence[str]] | None = None,
    local: Mapping[str, MetricColumnOut] | None = None,
) -> MetricTableOut:
    """Return ``frame``'s grouped columns, rows and registry metadata as a table payload.

    Args:
        frame: The season table.
        groups: The column groups in display order. A column appears once, in the first group that
            names it; one missing from ``frame`` or unknown to the registry is left out, and every
            declared group is kept, empty or not.
        overrides: Registry text to replace, by column.
        references: Groups that name columns already placed in ``groups`` (a quarterback's
            unadjusted rate beside his rating, for example) without moving them.
        local: Metadata for the table's own columns, which the registry does not define.

    """
    overrides = overrides or {}
    local = local or {}
    metadata: dict[str, MetricColumnOut] = {}
    column_groups: dict[str, list[str]] = {}
    for group, columns in groups.items():
        kept: list[str] = []
        for column in columns:
            if column in metadata or column not in frame.columns:
                continue
            meta = local.get(column) or metric_column(
                column, frame.schema[column], group, overrides.get(column)
            )
            if meta is not None:
                metadata[column] = meta
                kept.append(column)
        column_groups[group] = kept
    for group, columns in (references or {}).items():
        column_groups[group] = [column for column in columns if column in metadata]
    visible = list(metadata)
    rows = [
        {column: _cell(value) for column, value in row.items()}
        for row in frame.select(visible).to_dicts()
    ]
    return MetricTableOut(
        rows=rows,
        visible_columns=visible,
        column_groups=column_groups,
        column_metadata=metadata,
    )
