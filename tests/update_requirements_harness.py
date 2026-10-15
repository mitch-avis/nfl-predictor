"""Runs update_requirements.sh functions under bash with uv, the venv and its tools stubbed out."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "update_requirements.sh"
HOOKS_FILE_NAME = "update_requirements.hooks.sh"

BASH = shutil.which("bash")

# Sources the script's functions, points them at the work directory, accepts every prompt, and
# loads the work directory's hooks file when the test put one there.
HARNESS = r"""
set -Eeuo pipefail
source "$SCRIPT"
REPO_ROOT="$WORK"
VENV_PATH="$WORK/.venv"
PYTHON_REQUEST=""
cd "$WORK"
confirm() { return 0; }
load_hooks
"$@"
"""

# An executable on PATH, as uv is: it logs every call, answers `--version` from a file, and fails
# or changes state on the commands the test asks it to.
FAKE_UV = r"""#!/usr/bin/env bash
echo "uv $*" >>"$WORK/calls.log"
case "$1" in
--version) cat "$WORK/uv_version" ;;
self)
	[[ -z "${UV_SELF_UPDATE_FAILS:-}" ]] || exit 1
	[[ -z "${UV_UPDATE_TO:-}" ]] || echo "$UV_UPDATE_TO" >"$WORK/uv_version"
	;;
lock) [[ -z "${UV_LOCK_FAILS:-}" ]] || exit 1 ;;
python) [[ -z "${HEAL_ON_REINSTALL:-}" ]] || echo ok >"$WORK/stdlib" ;;
venv)
	mkdir -p "$WORK/.venv/bin"
	cp "$WORK/fake_python" "$WORK/.venv/bin/python"
	[[ "$*" != *--no-managed-python* ]] || echo ok >"$WORK/stdlib"
	;;
esac
"""

# Passes the script's stdlib check while the stdlib state file says "ok", and any other `-c`
# code (a hooks file's import check) while the packages state file does.
FAKE_PYTHON = """#!/usr/bin/env bash
if [[ "$2" == "import dis; import json" ]]; then
	[[ "$(cat "{work}/stdlib")" == ok ]]
else
	echo "python import check" >>"{work}/calls.log"
	[[ "$(cat "{work}/packages")" == ok ]]
fi
"""

# A venv tool that logs its calls and exits with the given status on `install`.
LOGGING_TOOL = """#!/usr/bin/env bash
echo "{name} $*" >>"{work}/calls.log"
[[ "$1" != install ]] || exit {install_exit}
"""


@dataclass(frozen=True)
class Stubs:
    """The state the stubbed tools start from.

    Attributes:
        stdlib: "ok" while the venv interpreter's standard library imports.
        packages: "ok" while a hooks file's import check passes.
        heal_on_reinstall: Whether `uv python install --reinstall` repairs the standard library.
        venv: Whether .venv exists before the run.
        hooks: The contents of the project's hooks file, or None for no hooks file.
        venv_tools: Logging tools to put in .venv/bin, by name, with the exit status of their
            `install` command.
        env: Extra environment variables for the fake uv.
    """

    stdlib: str = "ok"
    packages: str = "ok"
    heal_on_reinstall: bool = False
    venv: bool = True
    hooks: str | None = None
    venv_tools: dict[str, int] = field(default_factory=dict[str, int])
    env: dict[str, str] = field(default_factory=dict[str, str])


def write_tool(path: Path, work: Path, install_exit: int = 0) -> None:
    """Write a logging tool to `path` that records its calls in the work directory's log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        LOGGING_TOOL.format(name=path.name, work=work, install_exit=install_exit),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def run(
    tmp_path: Path, *command: str, stubs: Stubs | None = None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run `command` (a script function and its arguments) with the stubs in `tmp_path`.

    Returns:
        The finished bash process and the calls the stubbed tools logged, in order.
    """
    stubs = stubs or Stubs()
    fake_python = tmp_path / "fake_python"
    _write_executable(fake_python, FAKE_PYTHON.format(work=tmp_path))
    fake_uv = tmp_path / "bin" / "uv"
    _write_executable(fake_uv, FAKE_UV)
    if stubs.venv:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        (tmp_path / ".venv" / "pyvenv.cfg").write_text("version_info = 3.14\n", encoding="utf-8")
        shutil.copy(fake_python, venv_bin / "python")
        for name, install_exit in stubs.venv_tools.items():
            write_tool(venv_bin / name, tmp_path, install_exit)
    if stubs.hooks is not None:
        (tmp_path / HOOKS_FILE_NAME).write_text(stubs.hooks, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'stub'\n", encoding="utf-8")
    (tmp_path / "uv_version").write_text("uv 0.13.1\n", encoding="utf-8")
    (tmp_path / "stdlib").write_text(stubs.stdlib, encoding="utf-8")
    (tmp_path / "packages").write_text(stubs.packages, encoding="utf-8")
    result = subprocess.run(
        [str(BASH), "-c", HARNESS, "harness", *command],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{fake_uv.parent}:/usr/bin:/bin",
            "SCRIPT": str(SCRIPT),
            "WORK": str(tmp_path),
            "HEAL_ON_REINSTALL": "1" if stubs.heal_on_reinstall else "",
            **stubs.env,
        },
        check=False,
    )
    log = tmp_path / "calls.log"
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls
