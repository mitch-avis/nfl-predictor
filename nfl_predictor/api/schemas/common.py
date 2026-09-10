"""Table payloads shared by every data route."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ColumnKind = Literal[
    "text",
    "int",
    "float",
    "pct",
    "prob",
    "money",
    "spread",
    "datetime",
    "date",
    "team",
    "action",
    "bool",
]
Polarity = Literal["higher", "lower", "neutral"]


class ColumnMetaOut(BaseModel):
    """How the frontend should label, format, and color one column."""

    key: str
    label: str
    description: str
    group: str
    kind: ColumnKind
    polarity: Polarity = "neutral"
    heatmap: bool = False
    decimals: int | None = None
    actionable: bool = True
    sticky: bool = False


class TablePayload(BaseModel):
    """Rows plus the metadata needed to render them without per-column frontend code."""

    rows: list[dict[str, Any]]
    visible_columns: list[str]
    column_groups: dict[str, list[str]]
    column_metadata: dict[str, ColumnMetaOut]


class RegistryOut(BaseModel):
    """The whole column registry, grouped by domain."""

    columns: dict[str, ColumnMetaOut]
    groups: dict[str, list[str]]
