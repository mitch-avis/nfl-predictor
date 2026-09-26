"""Parse an ``nfl-predictor <command>`` argument list with that command's own parser.

Shared by the tests that check command lines written elsewhere: the web job templates and the
examples in ``README.md``. Parsing never runs the command.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys

import pytest

from nfl_predictor.cli import main as front_door

# Commands whose parser lives in a different module from the one the front door runs.
_PARSER_MODULES = {"weekly": "nfl_predictor.weekly_run.config"}


def parse_command(
    name: str,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    read_config: bool = True,
) -> argparse.Namespace:
    """Parse ``nfl-predictor <name> <args>`` with the command's own parser.

    Args:
        name: Front-door command name.
        args: The command's arguments.
        monkeypatch: Used to set ``sys.argv`` for parsers that read it.
        read_config: For ``weekly``, whether to read the ``--config`` file (and validate its
            keys) as a real run does; ``False`` parses the options alone.

    Returns:
        The parsed namespace.

    """
    command = front_door.COMMANDS_BY_NAME[name]
    if command.requires is not None and not {"-h", "--help"} & set(args):
        assert command.requires in args, (name, args)
    module = importlib.import_module(_PARSER_MODULES.get(name, command.module))
    if name == "weekly" and not read_config:
        return module._build_parser().parse_args(args)
    if not hasattr(module, "_parse_args"):
        return module._build_parser().parse_args(args)
    parse = module._parse_args
    if inspect.signature(parse).parameters:
        return parse(args)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [name, *args])
        return parse()
