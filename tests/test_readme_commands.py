"""Every ``nfl-predictor`` command shown in ``README.md`` parses with that command's parser.

The README's code blocks are the commands people copy. This test joins each block's
backslash-continued lines, takes every line that starts with ``nfl-predictor``, drops a trailing
``#`` comment, and parses it with the command's own parser, so a removed or renamed option or
command in an example fails here instead of in someone's terminal. Options are checked, not run;
the weekly example's config file is not read.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from nfl_predictor import constants
from nfl_predictor.cli import main as front_door
from tests.cli_parsing import parse_command

README = Path(constants.ROOT_DIR) / "README.md"
FENCE = re.compile(r"^\s*```(?:bash|sh)?\s*$\n(.*?)^\s*```\s*$", re.MULTILINE | re.DOTALL)


def readme_commands() -> list[str]:
    """Return every ``nfl-predictor`` command line in the README's code blocks."""
    commands: list[str] = []
    for block in FENCE.findall(README.read_text(encoding="utf-8")):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if line.startswith("nfl-predictor"):
                commands.append(line)
    return commands


def test_the_readme_shows_front_door_commands() -> None:
    """The extraction finds the README's examples (guards against a silent empty list)."""
    names = {shlex.split(line, comments=True)[1] for line in readme_commands()}
    assert {"data", "train", "backtest", "sweep", "weekly", "validate", "web"} <= names


@pytest.mark.parametrize("line", readme_commands())
def test_every_readme_command_parses(line: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The example's command exists and its parser accepts every option shown."""
    words = shlex.split(line, comments=True)
    assert words[0] == "nfl-predictor"
    if words[1:] == ["--help"]:
        with pytest.raises(SystemExit) as exit_info:
            front_door.main(["--help"])
        assert exit_info.value.code == 0
        return
    name, args = words[1], words[2:]
    assert name in front_door.COMMANDS_BY_NAME, line
    if "--help" in args:
        with pytest.raises(SystemExit) as exit_info:
            parse_command(name, args, monkeypatch, read_config=False)
        assert exit_info.value.code == 0
        return
    parse_command(name, args, monkeypatch, read_config=False)
