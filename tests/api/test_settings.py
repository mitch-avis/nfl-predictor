"""Tests for API settings resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nfl_predictor.api.settings import SIGNING_KEY_FILENAME, Settings


def test_paths_default_relative_to_root(tmp_path: Path) -> None:
    """Every unset directory resolves under ``root_dir``."""
    # Act
    settings = Settings(root_dir=tmp_path)

    # Assert
    assert settings.data_path == tmp_path / "data"
    assert settings.models_path == tmp_path / "models"
    assert settings.reports_path == tmp_path / "reports"
    assert settings.state_path == tmp_path / "data" / "web"
    assert settings.database_path == tmp_path / "data" / "web" / "app.db"
    assert settings.web_dist_path == tmp_path / "web" / "dist"
    assert settings.ratings_path == tmp_path / "data" / "ratings"
    assert settings.python_path == tmp_path / ".venv" / "bin" / Path(sys.executable).name


def test_explicit_paths_win(tmp_path: Path) -> None:
    """Explicit directories are kept and the state dir follows the data dir."""
    # Act
    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / "elsewhere")

    # Assert
    assert settings.data_path == tmp_path / "elsewhere"
    assert settings.state_path == tmp_path / "elsewhere" / "web"
    assert settings.ratings_path == tmp_path / "elsewhere" / "ratings"


def test_environment_overrides(tmp_path: Path, monkeypatch) -> None:
    """``NFLP_``-prefixed variables override defaults."""
    # Arrange
    monkeypatch.setenv("NFLP_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("NFLP_PORT", "9999")

    # Act
    settings = Settings()

    # Assert
    assert settings.root_dir == tmp_path.resolve()
    assert settings.port == 9999


def test_jwt_secret_generated_once(tmp_path: Path) -> None:
    """A missing secret is generated, persisted with 0600, and reused."""
    # Arrange
    settings = Settings(root_dir=tmp_path)
    first = settings.resolve_jwt_secret()
    secret_path = settings.state_path / SIGNING_KEY_FILENAME

    # Act
    again = Settings(root_dir=tmp_path).resolve_jwt_secret()

    # Assert
    assert secret_path.read_text(encoding="utf-8") == first
    assert oct(secret_path.stat().st_mode & 0o777) == "0o600"
    assert again == first


def test_jwt_secret_explicit(tmp_path: Path) -> None:
    """An explicit secret is returned without touching disk."""
    # Arrange
    settings = Settings(root_dir=tmp_path, jwt_secret="abc" * 11)

    # Act
    secret = settings.resolve_jwt_secret()

    # Assert
    assert secret == "abc" * 11
    assert not (settings.state_path / SIGNING_KEY_FILENAME).exists()


def test_python_executable_keeps_the_venv_symlink(tmp_path: Path) -> None:
    """The interpreter path is not resolved: the venv python is a symlink to the base Python."""
    # Arrange
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    real = tmp_path / "base-python"
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    (venv_bin / Path(sys.executable).name).symlink_to(real)

    # Act
    settings = Settings(root_dir=tmp_path)

    # Assert
    assert settings.python_path == venv_bin / Path(sys.executable).name
    assert settings.python_path.is_symlink()


def test_python_executable_can_be_given_relative_to_the_root(tmp_path: Path) -> None:
    """A relative override is anchored at the repository root."""
    # Act
    settings = Settings(root_dir=tmp_path, python_executable=Path(".venv/bin/python"))

    # Assert
    assert settings.python_path == tmp_path / ".venv" / "bin" / "python"


def test_an_unfilled_path_is_an_error_not_none(tmp_path: Path) -> None:
    """A path that is somehow unset raises instead of handing out ``None``."""
    # Arrange
    settings = Settings(root_dir=tmp_path)
    settings.data_dir = None

    # Act & Assert
    with pytest.raises(RuntimeError, match="data_dir is unset"):
        _ = settings.data_path
