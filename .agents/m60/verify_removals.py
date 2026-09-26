"""Check that every command-line option removed in Milestone 60 has a note in ``CHANGELOG.md``.

The parser surface of every entrypoint is pinned in ``tests/fixtures/cli_surface.json``. This
script compares the copy committed when the snapshot was created (``aefb287``, before any move)
with the current file, lists every option string that no longer exists on any parser, and checks
each one against the changelog: either the option is named there, or every parser that carried it
belongs to a script whose retirement the changelog records.

Run from the repository root:

    .venv/bin/python .agents/m60/verify_removals.py > .agents/m60/verify_removals_output.txt
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SURFACE = "tests/fixtures/cli_surface.json"
FIRST_SNAPSHOT = "aefb287"
# Scripts retired whole in 0.19.0; the changelog names each file.
RETIRED_SCRIPTS = (
    "golden_command",
    "betting_pipeline",
    "backtest_predictions",
    "objective_compare_models",
    "betting_report_excel",
)


def _options(surface: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    """Map every option string to the parsers that define it."""
    found: dict[str, set[str]] = {}
    for parser, actions in surface.items():
        for action in actions:
            for option in action["options"]:
                found.setdefault(str(option), set()).add(parser)
    return found


def main() -> int:
    """Print each removed option with how the changelog covers it; exit 1 if one is uncovered."""
    old = json.loads(
        subprocess.check_output(  # noqa: S603 - fixed argv built from this file
            ["git", "show", f"{FIRST_SNAPSHOT}:{SURFACE}"],  # noqa: S607 (git is fixed)
            cwd=ROOT,
        )
    )
    new = json.loads((ROOT / SURFACE).read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    retired = {f"scripts.{name}" for name in RETIRED_SCRIPTS}
    missing_scripts = [n for n in RETIRED_SCRIPTS if f"scripts/{n}.py" not in changelog]

    old_options, new_options = _options(old), _options(new)
    removed = sorted(set(old_options) - set(new_options) - {"-h", "--help"})
    uncovered: list[str] = []
    print(f"parsers: {len(old)} at {FIRST_SNAPSHOT}, {len(new)} now")
    print(f"options no longer on any parser: {len(removed)}")
    for option in removed:
        parsers = old_options[option]
        if option in changelog:
            how = "named in CHANGELOG.md"
        elif parsers <= retired:
            how = "its script's retirement is in CHANGELOG.md: " + ", ".join(sorted(parsers))
        else:
            how = "NOT COVERED: " + ", ".join(sorted(parsers))
            uncovered.append(option)
        print(f"  {option}: {how}")
    print(f"retired scripts missing from CHANGELOG.md: {len(missing_scripts)}")
    print(f"uncovered: {len(uncovered)}")
    return 1 if uncovered or missing_scripts else 0


if __name__ == "__main__":
    sys.exit(main())
