"""This repo's update_requirements.hooks.sh: the LightGBM CUDA build and the core import check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.update_requirements_harness import BASH, HOOKS_FILE_NAME, REPO_ROOT, Stubs, run

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

HOOKS = (REPO_ROOT / HOOKS_FILE_NAME).read_text(encoding="utf-8")
CUDA_INSTALLER = "nfl-lightgbm-cuda-install"


def _stubs(*, cuda_install_exit: int = 0, packages: str = "ok") -> Stubs:
    return Stubs(packages=packages, hooks=HOOKS, venv_tools={CUDA_INSTALLER: cuda_install_exit})


def test_the_core_packages_that_import_pass_the_check(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "ensure_venv_packages_import", stubs=_stubs())

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls == ["python import check"]


def test_core_packages_that_fail_to_import_are_named(tmp_path: Path) -> None:
    # Act
    result, _ = run(tmp_path, "ensure_venv_packages_import", stubs=_stubs(packages="bad"))

    # Assert
    assert result.returncode == 1
    assert "import numpy, pandas, polars, scipy, sklearn, xgboost" in result.stderr


def test_a_cuda_build_that_installs_passes_quietly(tmp_path: Path) -> None:
    # Act
    result, _ = run(tmp_path, "install_lightgbm_cuda_build", stubs=_stubs())

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr


def test_a_failed_cuda_build_warns_and_continues(tmp_path: Path) -> None:
    # Act
    result, _ = run(tmp_path, "install_lightgbm_cuda_build", stubs=_stubs(cuda_install_exit=1))

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Warning" in result.stderr
    assert "CPU build" in result.stderr


def test_lightgbm_is_built_after_the_sync_then_the_imports_are_checked(tmp_path: Path) -> None:
    # Act
    result, calls = run(tmp_path, "main", stubs=_stubs())

    # Assert
    assert result.returncode == 0, result.stderr
    assert calls[calls.index("uv lock --upgrade") :] == [
        "uv lock --upgrade",
        f"{CUDA_INSTALLER} uv-args",
        "uv sync",
        f"{CUDA_INSTALLER} install",
        "python import check",
    ]
