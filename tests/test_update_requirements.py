"""Behavior of update_requirements.sh's environment health checks, with uv stubbed out."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "update_requirements.sh"

BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

# Sources the script's functions, then replaces uv, the prompt and the venv's interpreter with
# stubs. The fake interpreter passes a check only while its state file says "ok"; the uv stub
# logs every call and heals the interpreter when the test asks a given command to.
HARNESS = r"""
set -Eeuo pipefail
source "$SCRIPT"
VENV_PATH="$WORK/.venv"
PYTHON_REQUEST=""
cd "$WORK"
confirm() { return 0; }
uv() {
	echo "uv $*" >>"$WORK/calls.log"
	if [[ "$1 $2" == "python install" && -n "${HEAL_ON_REINSTALL:-}" ]]; then
		echo ok >"$WORK/stdlib"
	fi
	if [[ "$1" == venv && "$*" == *--no-managed-python* ]]; then
		echo ok >"$WORK/stdlib"
	fi
}
"$@"
"""

FAKE_PYTHON = """#!/usr/bin/env bash
if [[ "$2" == *numpy* ]]; then
	[[ "$(cat "{work}/packages")" == ok ]]
else
	[[ "$(cat "{work}/stdlib")" == ok ]]
fi
"""


def _run(
    tmp_path: Path,
    *command: str,
    stdlib: str = "ok",
    packages: str = "ok",
    heal_on_reinstall: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("version_info = 3.14\n", encoding="utf-8")
    python = venv_bin / "python"
    python.write_text(FAKE_PYTHON.format(work=tmp_path), encoding="utf-8")
    python.chmod(0o755)
    (tmp_path / "stdlib").write_text(stdlib, encoding="utf-8")
    (tmp_path / "packages").write_text(packages, encoding="utf-8")
    env = {
        "PATH": "/usr/bin:/bin",
        "SCRIPT": str(SCRIPT),
        "WORK": str(tmp_path),
        "HEAL_ON_REINSTALL": "1" if heal_on_reinstall else "",
    }
    result = subprocess.run(
        [str(BASH), "-c", HARNESS, "harness", *command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    log = tmp_path / "calls.log"
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_a_healthy_interpreter_is_left_alone(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "ensure_venv_python_is_healthy")
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_damaged_interpreter_is_repaired_by_reinstalling_its_managed_python(
    tmp_path: Path,
) -> None:
    result, calls = _run(
        tmp_path, "ensure_venv_python_is_healthy", stdlib="bad", heal_on_reinstall=True
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
    ]


def test_a_system_python_is_the_last_resort(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "ensure_venv_python_is_healthy", stdlib="bad")
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
        "uv venv .venv --clear --no-managed-python --python 3.14",
    ]


def test_packages_that_import_pass_the_check(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "ensure_venv_packages_import")
    assert result.returncode == 0, result.stderr


def test_packages_that_fail_to_import_stop_with_the_cache_fix(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "ensure_venv_packages_import", packages="bad")
    assert result.returncode == 1
    assert "uv cache clean" in result.stderr
    assert "uv sync --active --reinstall" in result.stderr


def test_the_package_check_runs_after_the_sync() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    main = script[script.index("main() {") :]
    assert main.index("uv sync --active") < main.index("ensure_venv_packages_import")
