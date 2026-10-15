"""The gate's checks that pyproject.toml can't express on its own."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "gate.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_the_gate_holds_src_alone_to_the_coverage_bar() -> None:
    """tests/ counts toward the total, so src/ gets its own check at the same bar."""
    # Arrange
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    fail_under = pyproject["tool"]["coverage"]["report"]["fail_under"]

    # Act
    gate = GATE.read_text(encoding="utf-8")

    # Assert
    pytest_step = gate.index('run_step "pytest"')
    src_step = gate.index(f'coverage report --include="src/*" --fail-under={fail_under}')
    assert pytest_step < src_step
