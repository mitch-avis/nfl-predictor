"""Small helpers shared by the generated Markdown reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# dump_json puts one key per line in this many outer levels of objects.
ONE_KEY_PER_LINE_DEPTH = 2


def cell(value: object) -> str:
    """Render a table cell without breaking the Markdown table."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def season_ranges(seasons: Iterable[int]) -> str:
    """Render seasons as compact ranges: ``1999-2005, 2007``."""
    ordered = sorted(set(seasons))
    if not ordered:
        return "-"
    parts: list[str] = []
    start = previous = ordered[0]
    for season in ordered[1:]:
        if season == previous + 1:
            previous = season
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = season
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(parts)


def number(value: object) -> str:
    """Render a number compactly: integers as integers, floats to six decimals."""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6f}".rstrip("0")
    return str(value)


def dump_json(value: object, depth: int = 0) -> str:
    """Return compact, line-oriented JSON for a report.

    The first two levels of objects put one key per line, and a list of objects or lists puts one
    item per line in compact form, so a report diffs line by line without one value per line.
    """
    pad = " " * (depth + 1)
    if isinstance(value, dict) and value and depth < ONE_KEY_PER_LINE_DEPTH:
        items = [f"{pad}{json.dumps(str(k))}: {dump_json(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    if isinstance(value, list) and value and all(isinstance(v, dict | list) for v in value):
        items = [pad + json.dumps(v, default=str) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * depth + "]"
    return json.dumps(value, default=str)


def table(header: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    """Return the lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines
