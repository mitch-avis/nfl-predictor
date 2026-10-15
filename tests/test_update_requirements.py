"""Behavior of update_requirements.sh, the same script in every project, with uv stubbed out."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.update_requirements_harness import BASH, Stubs, run, write_tool

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

SMOKE_IMPORTS_HOOK = 'hook_smoke_imports() { echo "import example_package"; }\n'


def test_a_healthy_interpreter_is_left_alone(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "ensure_venv_python_is_healthy")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_damaged_interpreter_is_repaired_by_reinstalling_its_managed_python(
    tmp_path: Path,
) -> None:
    # Arrange
    stubs = Stubs(stdlib="bad", heal_on_reinstall=True)

    # Act
    result, calls = run(tmp_path, "ensure_venv_python_is_healthy", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
    ]


def test_a_system_python_is_the_last_resort(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "ensure_venv_python_is_healthy", stubs=Stubs(stdlib="bad"))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == [
        "uv python install --reinstall 3.14",
        "uv venv .venv --clear --managed-python --python 3.14",
        "uv venv .venv --clear --no-managed-python --python 3.14",
    ]


def test_packages_that_import_pass_the_check(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(hooks=SMOKE_IMPORTS_HOOK)

    # Act
    result, calls = run(tmp_path, "ensure_venv_packages_import", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == ["python import check"]


def test_packages_that_fail_to_import_stop_with_the_cache_fix(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(packages="bad", hooks=SMOKE_IMPORTS_HOOK)

    # Act
    result, _ = run(tmp_path, "ensure_venv_packages_import", stubs=stubs)

    # Assert
    assert result.returncode == 1
    assert "import example_package" in result.stderr
    assert "uv cache clean && uv sync --reinstall" in result.stderr


def test_without_hooks_there_is_no_import_check(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "ensure_venv_packages_import", stubs=Stubs(packages="bad"))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_refresh_needs_no_activated_venv(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "main")

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv lock --upgrade" in calls
    assert "uv sync" in calls
    assert not [call for call in calls if "--active" in call]


def test_hooks_add_sync_arguments_and_run_after_the_sync(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(
        hooks=(
            'hook_sync_args() { printf "%s\\n" --extra "two words"; }\n'
            'hook_after_sync() { echo "after sync" >>"$WORK/calls.log"; }\n'
            f"{SMOKE_IMPORTS_HOOK}"
        )
    )

    # Act
    result, calls = run(tmp_path, "main", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls[-3:] == ["uv sync --extra two words", "after sync", "python import check"]


def test_a_missing_venv_is_created_and_the_refresh_continues(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "main", stubs=Stubs(venv=False))

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls.index("uv venv .venv") < calls.index("uv lock --upgrade")


def test_a_python_request_reaches_the_new_venv(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "main", "--python", "3.13", stubs=Stubs(venv=False))

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv venv .venv --python 3.13" in calls


def test_an_unknown_option_stops_before_any_change(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "main", "--frobnicate")

    # Assert
    assert result.returncode == 1
    assert "Unknown option: --frobnicate" in result.stderr
    assert calls == []


def test_a_uv_update_reports_the_old_and_new_versions(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_UPDATE_TO": "uv 0.13.2"})

    # Act
    result, _ = run(tmp_path, "update_uv", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv 0.13.1 -> uv 0.13.2" in result.stdout


def test_an_unchanged_uv_says_it_is_up_to_date(tmp_path: Path) -> None:
    # Act
    result, _ = run(tmp_path, "update_uv")

    # Assert
    assert result.returncode == 0, result.stderr
    assert "uv is up to date (uv 0.13.1)" in result.stdout


def test_a_failed_uv_update_says_so_and_keeps_going(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_SELF_UPDATE_FAILS": "1"})

    # Act
    result, _ = run(tmp_path, "update_uv", stubs=stubs)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "keeping uv 0.13.1" in result.stdout


def test_pre_commit_hook_versions_are_refreshed_when_the_project_uses_pre_commit(
    tmp_path: Path,
) -> None:
    # Arrange
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    write_tool(tmp_path / ".venv" / "bin" / "pre-commit", tmp_path)

    # Act
    result, calls = run(tmp_path, "refresh_pre_commit_hooks")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == ["pre-commit autoupdate"]


def test_pre_commit_is_left_alone_without_its_config(tmp_path: Path) -> None:
    # Arrange
    write_tool(tmp_path / ".venv" / "bin" / "pre-commit", tmp_path)

    # Act
    result, calls = run(tmp_path, "refresh_pre_commit_hooks")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_requirements_files_are_exported_from_the_lockfile(tmp_path: Path) -> None:
    # Arrange
    for name in ("requirements.txt", "requirements-dev.in", "requirements-dev.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")

    # Act
    result, calls = run(tmp_path, "refresh_compatibility_requirements")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == [
        (
            "uv export --format requirements.txt --no-default-groups --no-emit-project"
            " --no-hashes --output-file requirements.txt"
        ),
        (
            "uv export --format requirements.txt --only-group dev --no-emit-project"
            " --no-hashes --output-file requirements-dev.txt"
        ),
    ]


def test_without_requirements_files_nothing_is_exported(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "refresh_compatibility_requirements")

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_a_failed_step_names_its_command(tmp_path: Path) -> None:
    # Arrange
    stubs = Stubs(env={"UV_LOCK_FAILS": "1"})

    # Act
    result, _ = run(tmp_path, "main", stubs=stubs)

    # Assert
    assert result.returncode != 0
    assert "uv lock --upgrade" in result.stderr
