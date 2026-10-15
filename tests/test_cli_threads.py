"""Tests for the front door's Polars thread default."""

from __future__ import annotations

import os

import pytest

from nfl_predictor.cli import main as front_door
from nfl_predictor.cli import threads
from tests.fakes import returning


def test_prefer_one_polars_thread_defaults_to_one() -> None:
    # Arrange
    environ: dict[str, str] = {}

    # Act
    value = threads.prefer_one_polars_thread(environ)

    # Assert
    assert value == "1"
    assert environ == {"POLARS_MAX_THREADS": "1"}


def test_prefer_one_polars_thread_keeps_an_explicit_setting() -> None:
    # Arrange
    environ = {"POLARS_MAX_THREADS": "12"}

    # Act
    value = threads.prefer_one_polars_thread(environ)

    # Assert
    assert value == "12"
    assert environ == {"POLARS_MAX_THREADS": "12"}


@pytest.mark.parametrize("command", ["data", "weekly", "validate", "web"])
def test_every_command_runs_polars_on_one_thread_when_unset(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    # Arrange
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    seen: dict[str, str | None] = {}

    def resolve(_command: front_door.Command) -> object:
        """Record the setting when the command module would load, and return a no-op entry."""
        seen["threads"] = os.environ.get("POLARS_MAX_THREADS")
        return returning(0)

    monkeypatch.setattr(front_door, "_resolve", resolve)

    # Act
    front_door.main([command])

    # Assert
    assert seen["threads"] == "1"


def test_the_front_door_keeps_an_explicit_polars_thread_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("POLARS_MAX_THREADS", "24")
    monkeypatch.setattr(front_door, "_resolve", returning(returning(0)))

    # Act
    front_door.main(["data"])

    # Assert
    assert os.environ["POLARS_MAX_THREADS"] == "24"
