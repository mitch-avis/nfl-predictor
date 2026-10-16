"""The gate's checks that pyproject.toml can't express on its own, and this repo's gate hooks."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "gate.sh"
HOOKS = REPO_ROOT / "scripts" / "gate.hooks.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"
BASH = shutil.which("bash") or "/bin/bash"


def extra_steps(*, run_tests: int, run_web: int) -> list[str]:
    """Return the names of the steps the hooks file adds, with `run_step` only recording them."""
    script = (
        f'source "{HOOKS}"; RUN_TESTS={run_tests}; RUN_WEB={run_web}; '
        'run_step() { echo "$1"; }; gate_extra_steps'
    )
    result = subprocess.run(
        [BASH, "-c", script], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    return result.stdout.splitlines()


def test_the_gate_holds_src_alone_to_the_coverage_bar() -> None:
    """tests/ counts toward the total, so src/ gets its own check at the configured bar."""
    # Arrange
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    # Act
    gate = GATE.read_text(encoding="utf-8")

    # Assert
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 90
    assert gate.index('run_step "pytest"') < gate.index('coverage report --include="src/*"')


def test_the_hooks_file_accepts_the_web_option() -> None:
    # Arrange
    script = f'source "{HOOKS}"; RUN_WEB=0; gate_option --web && echo "$RUN_WEB"'

    # Act
    result = subprocess.run([BASH, "-c", script], capture_output=True, text=True, check=False)

    # Assert
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_the_gate_refuses_an_option_neither_it_nor_the_hooks_know() -> None:
    # Act
    result = subprocess.run(
        [BASH, str(GATE), "--no-such-option"], capture_output=True, text=True, check=False
    )

    # Assert
    assert result.returncode == 2
    assert "unknown option: --no-such-option" in result.stderr


def test_every_front_door_command_is_checked_but_the_web_gate_waits_for_web() -> None:
    # Act
    steps = extra_steps(run_tests=1, run_web=0)

    # Assert
    assert steps == ["nfl-predictor commands --help"]


def test_the_web_option_adds_the_web_gate() -> None:
    # Act
    steps = extra_steps(run_tests=1, run_web=1)

    # Assert
    assert steps == ["nfl-predictor commands --help", "web gate"]


def test_quick_skips_the_web_gate_even_with_the_web_option() -> None:
    # Act
    steps = extra_steps(run_tests=0, run_web=1)

    # Assert
    assert steps == ["nfl-predictor commands --help"]
