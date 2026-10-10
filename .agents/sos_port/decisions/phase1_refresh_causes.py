"""Summarize a parity check's causes: what each explains and what its adjustment changes alone.

Reads a ``parity.json`` the parity check wrote and prints, for each judged pair with mismatching
team-games, how many it has, explains and leaves unexplained, then for each cause the team-games
it explains and the team-games its adjustment alone changes on each value it is judged by.
Read-only; it writes nothing. Run from the repository root, with the report as its argument:

    .venv/bin/python .agents/sos_port/decisions/phase1_refresh_causes.py <path>/parity.json

``decisions/phase1_refresh.md`` quotes it for two runs: with the causes as they stood before the
refresh (commit ``a9d4e6b``) and as refreshed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def summary(payload: dict[str, Any]) -> list[str]:
    """Return the summary lines for one parity payload."""
    lines = [f"generated_by: {payload['generated_by']}"]
    lines.append("pair: mismatched, explained, unexplained")
    for pair in payload["pairs"]:
        if pair["mismatched"]:
            counts = f"{pair['mismatched']}, {pair['explained']}, {pair['unexplained']}"
            lines.append(f"  {pair['id']}: {counts}")
    lines.append("cause: team-games explained; team-games its adjustment alone changes, per value")
    for cause, entry in payload["causes"].items():
        changed = {name: value["team_games_changed"] for name, value in entry["effect"].items()}
        lines.append(f"  {cause}: {entry['team_games']}; {changed or '-'}")
    return lines


if __name__ == "__main__":
    print("\n".join(summary(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))))
