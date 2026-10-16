"""Tests for the uvicorn entrypoint and the user CLI."""

from __future__ import annotations

import getpass
import logging
from typing import TYPE_CHECKING

import pytest
import uvicorn

from nfl_predictor.api import __main__ as entry
from nfl_predictor.api.auth import cli
from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.db import Database
from tests.fakes import returning

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.api.settings import Settings


def test_main_runs_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The entrypoint passes host/port through and builds an app."""
    # Arrange
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kw: object) -> None:
        calls.append((app, kw))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("NFLP_ROOT_DIR", str(tmp_path))

    # Act
    entry.main(["--host", "192.0.2.10", "--port", "1234"])

    # Assert
    assert calls[0][1] == {"host": "192.0.2.10", "port": 1234}


def test_main_reload_runs_uvicorn_with_app_factory_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With ``--reload`` the entrypoint passes the app factory import path to uvicorn."""
    # Arrange
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kw: object) -> None:
        calls.append((app, kw))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("NFLP_ROOT_DIR", str(tmp_path))
    entry.main(["--host", "192.0.2.10", "--port", "1234"])

    # Act
    entry.main(["--reload"])

    # Assert
    assert calls[1][0] == "nfl_predictor.api:create_app"
    assert calls[1][1]["reload"] is True


def test_cli_create_user_with_password_flag(settings: Settings) -> None:
    """The CLI creates a user from ``--password`` and that password authenticates."""
    # Act
    code = cli.main(
        ["create-user", "mitch", "--role", "admin", "--password", "password123"], settings
    )

    # Assert
    assert code == 0
    db = Database(settings.database_path)
    assert user_store.authenticate(db, "mitch", "password123") is not None


def test_cli_create_user_with_password_env(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI creates a user from the password environment variable."""
    # Arrange
    cli.main(["create-user", "mitch", "--role", "admin", "--password", "password123"], settings)
    monkeypatch.setenv(cli.PASSWORD_ENV, "envpassword")

    # Act
    code = cli.main(["create-user", "viewer"], settings)

    # Assert
    assert code == 0


def test_cli_set_password_from_prompt(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI resets a password from the prompt when no env value is set."""
    # Arrange
    cli.main(["create-user", "mitch", "--role", "admin", "--password", "password123"], settings)
    monkeypatch.setenv(cli.PASSWORD_ENV, "envpassword")
    cli.main(["create-user", "viewer"], settings)
    monkeypatch.delenv(cli.PASSWORD_ENV)
    monkeypatch.setattr(getpass, "getpass", returning("promptpassword"))

    # Act
    code = cli.main(["set-password", "viewer"], settings)

    # Assert
    assert code == 0
    db = Database(settings.database_path)
    assert user_store.authenticate(db, "mitch", "password123") is not None
    assert user_store.authenticate(db, "viewer", "promptpassword") is not None


def test_cli_set_password_unknown_user_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resetting the password of a user that does not exist exits with 1."""
    # Arrange
    cli.main(["create-user", "mitch", "--role", "admin", "--password", "password123"], settings)
    monkeypatch.setattr(getpass, "getpass", returning("promptpassword"))

    # Act
    code = cli.main(["set-password", "nobody"], settings)

    # Assert
    assert code == 1


def test_cli_list_users_logs_usernames(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Listing users succeeds and logs the created usernames."""
    # Arrange
    caplog.set_level(logging.INFO)
    cli.main(["create-user", "mitch", "--role", "admin", "--password", "password123"], settings)

    # Act
    code = cli.main(["list-users"], settings)

    # Assert
    assert code == 0
    assert any("mitch" in record.getMessage() for record in caplog.records)


def test_cli_requires_command(settings: Settings) -> None:
    """Argparse exits when no sub-command is given."""
    # Act & Assert
    with pytest.raises(SystemExit):
        cli.main([], settings)
