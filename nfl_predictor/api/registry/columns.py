"""The registry container and the projection helper that turns a frame into a table payload."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from nfl_predictor.api.schemas.common import ColumnKind, ColumnMetaOut, Polarity, TablePayload


@dataclass(frozen=True)
class ColumnMeta:
    """Display metadata for one column (see :class:`ColumnMetaOut` for field meanings)."""

    key: str
    label: str
    description: str
    group: str
    kind: ColumnKind = "text"
    polarity: Polarity = "neutral"
    heatmap: bool = False
    decimals: int | None = None
    actionable: bool = True
    sticky: bool = False

    def to_out(self) -> ColumnMetaOut:
        """Convert to the API model."""
        return ColumnMetaOut(**asdict(self))


REGISTRY: dict[str, ColumnMeta] = {}


def register(*metas: ColumnMeta) -> None:
    """Add ``metas`` to the registry; duplicate keys are a programming error."""
    for meta in metas:
        if meta.key in REGISTRY:
            raise ValueError(f"Column {meta.key!r} is already registered")
        REGISTRY[meta.key] = meta


def group_map(keys: Iterable[str]) -> dict[str, list[str]]:
    """Group registered ``keys`` by their group, preserving order of first appearance."""
    groups: dict[str, list[str]] = {}
    for key in keys:
        meta = REGISTRY.get(key)
        if meta is None:
            continue
        groups.setdefault(meta.group, []).append(key)
    return groups


def jsonable(value: Any) -> Any:
    """Convert a Polars cell to a JSON-safe value (NaN -> null, dates -> ISO strings)."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value


def project(df: pl.DataFrame, keys: Sequence[str]) -> TablePayload:
    """Select the registered ``keys`` present in ``df`` and build a :class:`TablePayload`.

    Unregistered or absent columns are dropped so the 500-column feature matrix never leaks
    through, and the order of ``keys`` is the display order.
    """
    visible = [key for key in keys if key in REGISTRY and key in df.columns]
    rows = [
        {key: jsonable(value) for key, value in row.items()}
        for row in df.select(visible).to_dicts()
    ]
    return TablePayload(
        rows=rows,
        visible_columns=visible,
        column_groups=group_map(visible),
        column_metadata={key: REGISTRY[key].to_out() for key in visible},
    )
