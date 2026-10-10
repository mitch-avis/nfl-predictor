"""Count how the inventory's classes moved between the reports before and after the refresh.

Reads the inventory report as checked in at ``9bba627`` (before the refresh, through ``git show``)
and the one beside this directory (after it), and prints, for the output columns, the sources and
the web items, the count of each move from an old class to a new one, then each column rule's
moves (old rule and class to new rule and class). Read-only; it writes nothing. Run from the
repository root:

    .venv/bin/python .agents/sos_port/decisions/phase1_refresh_transitions.py

``decisions/phase1_refresh.md`` quotes its output (``phase1_refresh_transitions_output.txt``).
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
BEFORE = "9bba627"


def before() -> dict[str, Any]:
    """Return the inventory report as checked in before the refresh."""
    text = subprocess.run(
        ["git", "--no-optional-locks", "show", f"{BEFORE}:.agents/sos_port/inventory.json"],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(text)


def old_columns(report: dict[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    """Return (kind, column) -> (rule, class) from the report's one table of rows."""
    fields = report["columns"]["fields"]
    rows = [dict(zip(fields, row, strict=True)) for row in report["columns"]["rows"]]
    return {(r["kind"], r["column"]): (r["rule"], r["classification"]) for r in rows}


def new_columns(report: dict[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    """Return (kind, column) -> (rule, class) from the report's rows per file kind."""
    fields = report["column_fields"]
    found = {}
    for kind, rows in report["columns"].items():
        for row in rows:
            entry = dict(zip(fields, row, strict=True))
            found[(kind, entry["column"])] = (entry["rule"], entry["classification"])
    return found


def classes(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Return key -> class for classified source or web rows."""
    return {row["key"]: row["class"] for row in rows}


def moves(old: dict[Any, str], new: dict[Any, str]) -> Counter[str]:
    """Count the moves from each old class to each new class over the keys of both."""
    return Counter(f"{old.get(key, 'absent')} -> {new.get(key, 'absent')}" for key in old | new)


def main() -> None:
    """Print the class moves of columns, sources and web items, then each column rule's."""
    old, new = before(), json.loads((HERE / "inventory.json").read_text(encoding="utf-8"))
    old_cols, new_cols = old_columns(old), new_columns(new)
    sections = {
        "output columns": moves(
            {k: v[1] for k, v in old_cols.items()}, {k: v[1] for k, v in new_cols.items()}
        ),
        "data sources": moves(
            classes(old["sources"]["classified"]), classes(new["sources"]["classified"])
        ),
        "web items": moves(classes(old["web"]["classified"]), classes(new["web"]["classified"])),
    }
    for name, counter in sections.items():
        print(f"{name}:")
        for move, count in sorted(counter.items()):
            print(f"  {move}: {count}")
    print("column rule moves (old rule, old class -> new rule, new class: columns):")
    rule_moves = Counter(
        f"{old_cols[key][0]} {old_cols[key][1]} -> {new_cols[key][0]} {new_cols[key][1]}"
        for key in old_cols
        if key in new_cols and old_cols[key] != new_cols[key]
    )
    for move, count in sorted(rule_moves.items()):
        print(f"  {move}: {count}")


if __name__ == "__main__":
    main()
