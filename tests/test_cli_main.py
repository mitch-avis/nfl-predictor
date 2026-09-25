"""Tests for the ``nfl-predictor`` front door."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nfl_predictor.cli import main as front_door

ROOT = Path(__file__).resolve().parents[1]


def test_every_command_resolves_to_a_callable_entry_point() -> None:
    """Each command's module imports and exposes the function the front door calls."""
    for command in front_door.COMMANDS:
        entry = getattr(importlib.import_module(command.module), command.function)
        assert callable(entry), command.name


def test_help_lists_every_command_under_its_group(capsys: pytest.CaptureFixture[str]) -> None:
    """``nfl-predictor --help`` shows every command, grouped, and exits cleanly."""
    with pytest.raises(SystemExit) as exit_info:
        front_door.main(["--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    for command in front_door.COMMANDS:
        assert f"  {command.name} " in output
    for title in front_door.GROUP_TITLES.values():
        assert f"{title}:" in output


def test_an_unknown_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A command that does not exist exits with argparse's usage status."""
    with pytest.raises(SystemExit) as exit_info:
        front_door.main(["no-such-command"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def _fake_command(monkeypatch: pytest.MonkeyPatch, *, takes_argv: bool) -> SimpleNamespace:
    """Point the ``validate`` command at a recorder and return what it records."""
    seen = SimpleNamespace(args=None, argv=None)

    def entry(*args: list[str]) -> int:
        """Record the arguments and ``sys.argv`` the front door passed, then exit 3."""
        seen.args = args
        seen.argv = list(sys.argv)
        return 3

    command = front_door.Command(
        "validate", "data", "test", "unused", takes_argv=takes_argv, function="entry"
    )
    monkeypatch.setitem(front_door.COMMANDS_BY_NAME, "validate", command)
    monkeypatch.setattr(front_door, "_resolve", lambda _command: entry)
    return seen


@pytest.mark.parametrize("takes_argv", [True, False])
def test_dispatch_passes_the_remaining_arguments(
    monkeypatch: pytest.MonkeyPatch, takes_argv: bool
) -> None:
    """Options after the command reach it unchanged, and its exit code is returned."""
    monkeypatch.setattr(sys, "argv", ["original"])
    seen = _fake_command(monkeypatch, takes_argv=takes_argv)

    code = front_door.main(["validate", "--live", "--data-dir", "x"])

    assert code == 3
    assert seen.argv == ["nfl-predictor validate", "--live", "--data-dir", "x"]
    assert seen.args == ((["--live", "--data-dir", "x"],) if takes_argv else ())
    assert sys.argv == ["original"]


def test_sys_argv_is_restored_when_the_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside a command still restores the caller's ``sys.argv``."""
    monkeypatch.setattr(sys, "argv", ["original"])

    def boom() -> None:
        """Fail inside the command."""
        raise RuntimeError("boom")

    monkeypatch.setattr(front_door, "_resolve", lambda _command: boom)

    with pytest.raises(RuntimeError, match="boom"):
        front_door.main(["sweep"])
    assert sys.argv == ["original"]


def test_predict_requires_a_saved_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``predict`` without ``--model-in`` is refused before any model code runs."""
    monkeypatch.setattr(front_door, "_resolve", lambda _command: pytest.fail)

    with pytest.raises(SystemExit) as exit_info:
        front_door.main(["predict", "--predict-path", "week.csv"])

    assert exit_info.value.code == 2
    assert "predict needs --model-in" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [["predict", "--model-in", "m.joblib"], ["predict", "--model-in=m.joblib"], ["predict", "-h"]],
)
def test_predict_accepts_a_saved_model_or_a_help_request(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """Either spelling of ``--model-in``, or a help request, reaches the model command."""
    monkeypatch.setattr(front_door, "_resolve", lambda _command: lambda: None)

    assert front_door.main(args) == 0


def test_python_dash_m_runs_the_front_door() -> None:
    """``python -m nfl_predictor --help`` is the same front door."""
    result = subprocess.run(
        [sys.executable, "-m", "nfl_predictor", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0
    assert "nfl-predictor" in result.stdout
    assert "leakage-audit" in result.stdout
