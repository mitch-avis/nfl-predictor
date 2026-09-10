"""Tests for the uvicorn entrypoint and the user CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nfl_predictor.api import __main__ as entry
from nfl_predictor.api.auth import cli
from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.db import Database
from nfl_predictor.api.settings import Settings


def test_main_runs_uvicorn(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    """The entrypoint passes host/port through and builds an app."""
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    monkeypatch.setenv("NFLP_ROOT_DIR", str(tmp_path))
    entry.main(["--host", "0.0.0.0", "--port", "1234"])  # noqa: S104 - test value
    assert calls[0][1] == {"host": "0.0.0.0", "port": 1234}  # noqa: S104
    entry.main(["--reload"])
    assert calls[1][0] == "nfl_predictor.api:create_app"
    assert calls[1][1]["reload"] is True


def test_cli_create_list_and_reset(settings: Settings, monkeypatch, caplog) -> None:  # noqa: ANN001
    """The CLI creates users, lists them, and resets passwords via env or prompt."""
    caplog.set_level(logging.INFO)
    assert (
        cli.main(["create-user", "mitch", "--role", "admin", "--password", "password123"], settings)
        == 0
    )
    monkeypatch.setenv(cli.PASSWORD_ENV, "envpassword")
    assert cli.main(["create-user", "viewer"], settings) == 0
    monkeypatch.delenv(cli.PASSWORD_ENV)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "promptpassword")
    assert cli.main(["set-password", "viewer"], settings) == 0
    assert cli.main(["set-password", "nobody"], settings) == 1
    assert cli.main(["list-users"], settings) == 0
    db = Database(settings.database_path)
    assert user_store.authenticate(db, "mitch", "password123") is not None
    assert user_store.authenticate(db, "viewer", "promptpassword") is not None
    assert any("mitch" in record.getMessage() for record in caplog.records)


def test_cli_requires_command(settings: Settings) -> None:
    """Argparse exits when no sub-command is given."""
    with pytest.raises(SystemExit):
        cli.main([], settings)
