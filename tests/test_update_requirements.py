"""Behavior of update_requirements.sh and this repo's hooks for it, with uv stubbed out."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "update_requirements.sh"
HOOKS = REPO_ROOT / "update_requirements.hooks.sh"

BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

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

# Passes the stdlib check or the package import check only while its state file says "ok".
FAKE_PYTHON = """#!/usr/bin/env bash
if [[ "$2" == *numpy* ]]; then
	echo "python import check" >>"{work}/calls.log"
	[[ "$(cat "{work}/packages")" == ok ]]
else
	[[ "$(cat "{work}/stdlib")" == ok ]]
fi
"""

LOGGING_TOOL = """#!/usr/bin/env bash
echo "{name} $*" >>"{work}/calls.log"
[[ "$1" != install ]] || exit {install_exit}
"""


@dataclass(frozen=True)
class Stubs:
    """The state the stubbed tools start from."""

    stdlib: str = "ok"
    packages: str = "ok"
    heal_on_reinstall: bool = False
    cuda_install_exit: int = 0
    venv: bool = True
    hooks: bool = False
    env: dict[str, str] = field(default_factory=dict[str, str])


def _write_tool(path: Path, work: Path, install_exit: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        LOGGING_TOOL.format(name=path.name, work=work, install_exit=install_exit),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run(
    tmp_path: Path, *command: str, stubs: Stubs | None = None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    stubs = stubs or Stubs()
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(FAKE_PYTHON.format(work=tmp_path), encoding="utf-8")
    fake_python.chmod(0o755)
    fake_uv = tmp_path / "bin" / "uv"
    fake_uv.parent.mkdir()
    fake_uv.write_text(FAKE_UV, encoding="utf-8")
    fake_uv.chmod(0o755)
    if stubs.venv:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        (tmp_path / ".venv" / "pyvenv.cfg").write_text("version_info = 3.14\n", encoding="utf-8")
        shutil.copy(fake_python, venv_bin / "python")
        _write_tool(venv_bin / "nfl-lightgbm-cuda-install", tmp_path, stubs.cuda_install_exit)
    if stubs.hooks:
        shutil.copy(HOOKS, tmp_path / HOOKS.name)
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


def test_a_healthy_interpreter_is_left_alone(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "ensure_venv_python_is_healthy")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_damaged_interpreter_is_repaired_by_reinstalling_its_managed_python(
    tmp_path: Path,
) -> None:
    # Arrange
    stubs = Stubs(stdlib="bad", heal_on_reinstall=True)

    # Act
    result, calls = _run(tmp_path, "ensure_venv_python_is_healthy", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
    ]


def test_a_system_python_is_the_last_resort(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "ensure_venv_python_is_healthy", stubs=Stubs(stdlib="bad"))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
        "uv venv .venv --clear --no-managed-python --python 3.14",
    ]


def test_packages_that_import_pass_the_check(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "ensure_venv_packages_import", stubs=Stubs(hooks=True))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == ["python import check"]


def test_packages_that_fail_to_import_stop_with_the_cache_fix(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(packages="bad", hooks=True)

    # Act
    result, _ = _run(tmp_path, "ensure_venv_packages_import", stubs=stubs)

    # Assert
    assert result.returncode == 1
    assert "uv cache clean && uv sync --reinstall" in result.stderr


def test_without_hooks_there_is_no_import_check(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "ensure_venv_packages_import", stubs=Stubs(packages="bad"))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_cuda_build_that_installs_passes_quietly(tmp_path: Path) -> None:
    # Act
    result, _ = _run(tmp_path, "install_lightgbm_cuda_build", stubs=Stubs(hooks=True))

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr


def test_a_failed_cuda_build_warns_and_continues(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(cuda_install_exit=1, hooks=True)

    # Act
    result, _ = _run(tmp_path, "install_lightgbm_cuda_build", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Warning" in result.stderr
    assert "CPU build" in result.stderr


def test_a_refresh_needs_no_activated_venv(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "main")

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv lock --upgrade" in calls
    assert "uv sync" in calls
    assert not [call for call in calls if "--active" in call]


def test_this_repos_hooks_build_lightgbm_after_the_sync_then_check_imports(
    tmp_path: Path,
) -> None:
    # Act
    result, calls = _run(tmp_path, "main", stubs=Stubs(hooks=True))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls[calls.index("uv lock --upgrade") :] == [
        "uv lock --upgrade",
        "nfl-lightgbm-cuda-install uv-args",
        "uv sync",
        "nfl-lightgbm-cuda-install install",
        "python import check",
    ]


def test_hooks_add_sync_arguments_and_run_after_the_sync(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / HOOKS.name).write_text(
        'hook_sync_args() { printf "%s\\n" --extra "two words"; }\n'
        'hook_after_sync() { echo "after sync" >>"$WORK/calls.log"; }\n',
        encoding="utf-8",
    )

    # Act
    result, calls = _run(tmp_path, "main")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls[-2:] == ["uv sync --extra two words", "after sync"]


def test_a_missing_venv_is_created_and_the_refresh_continues(tmp_path: Path) -> None:
    # Act
    result, calls = _run(tmp_path, "main", stubs=Stubs(venv=False))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls.index("uv venv .venv") < calls.index("uv lock --upgrade")


def test_a_uv_update_reports_the_old_and_new_versions(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_UPDATE_TO": "uv 0.13.2"})

    # Act
    result, _ = _run(tmp_path, "update_uv", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv 0.13.1 -> uv 0.13.2" in result.stdout


def test_a_failed_uv_update_says_so_and_keeps_going(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_SELF_UPDATE_FAILS": "1"})

    # Act
    result, _ = _run(tmp_path, "update_uv", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "keeping uv 0.13.1" in result.stdout


def test_pre_commit_hook_versions_are_refreshed_when_the_project_uses_pre_commit(
    tmp_path: Path,
) -> None:
    # Arrange
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    _write_tool(tmp_path / ".venv" / "bin" / "pre-commit", tmp_path)

    # Act
    result, calls = _run(tmp_path, "refresh_pre_commit_hooks")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == ["pre-commit autoupdate"]


def test_pre_commit_is_left_alone_without_its_config(tmp_path: Path) -> None:
    # Arrange
    _write_tool(tmp_path / ".venv" / "bin" / "pre-commit", tmp_path)

    # Act
    result, calls = _run(tmp_path, "refresh_pre_commit_hooks")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_failed_step_names_its_command(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_LOCK_FAILS": "1"})

    # Act
    result, _ = _run(tmp_path, "main", stubs=stubs)

    # Assert
    assert result.returncode != 0
    assert "uv lock --upgrade" in result.stderr
