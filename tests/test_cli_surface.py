"""Snapshot of every command-line entrypoint's parser surface.

Each entrypoint's parser is built through its own ``_build_parser`` or ``_parse_args``
(``parse_args`` is intercepted before it reads anything), and every action is recorded:
option strings, dest, default, choices, nargs, type, whether it is required, and its help
text. The result must equal ``tests/fixtures/cli_surface.json``. A flag that appears,
disappears, is renamed or changes its default fails here, so an intended change updates the
snapshot in the same commit (run with ``NFLP_UPDATE_SNAPSHOTS=1``) and an unintended one is
caught. Absolute repository paths in defaults are written as ``<repo>``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nfl_predictor import constants

ROOT = Path(constants.ROOT_DIR)
SNAPSHOT = Path(__file__).parent / "fixtures" / "cli_surface.json"
UPDATE_ENV = "NFLP_UPDATE_SNAPSHOTS"


class _ParserCapturedError(Exception):
    """Carries the parser out of an entrypoint's parse function."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        """Keep the captured parser."""
        super().__init__("parser captured")
        self.parser = parser


def _entrypoint_modules() -> list[tuple[str, str]]:
    """Return ``(module, parse function)`` for every module that builds a parser."""
    candidates = sorted((ROOT / "nfl_predictor").rglob("*.py"))
    found: list[tuple[str, str]] = []
    for path in candidates:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        builds_parser = any(
            isinstance(node, ast.Call)
            and (
                getattr(node.func, "attr", None) == "ArgumentParser"
                or getattr(node.func, "id", None) == "ArgumentParser"
            )
            for node in ast.walk(tree)
        )
        if not builds_parser:
            continue
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        parse_function = "_build_parser" if "_build_parser" in functions else "_parse_args"
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        found.append((module, parse_function))
    return found


def _capture(
    module_name: str, function_name: str, monkeypatch: pytest.MonkeyPatch
) -> argparse.ArgumentParser:
    """Build one entrypoint's parser without parsing anything."""
    function = getattr(importlib.import_module(module_name), function_name)
    if function_name == "_build_parser":
        return function()

    def _intercept(self: argparse.ArgumentParser, *_args: object, **_kwargs: object) -> None:
        raise _ParserCapturedError(self)

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", _intercept)
        patch.setattr(argparse.ArgumentParser, "parse_known_args", _intercept)
        patch.setattr("sys.argv", [module_name])
        try:
            if inspect.signature(function).parameters:
                function([])
            else:
                function()
        except _ParserCapturedError as captured:
            return captured.parser
    raise AssertionError(f"{module_name}.{function_name} returned without parsing")


def _portable(value: Any) -> Any:
    """Return a JSON value with absolute repository paths made relative."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [_portable(item) for item in value]
    return str(value).replace(str(ROOT), "<repo>")


def _surface(parser: argparse.ArgumentParser, prefix: str = "") -> list[dict[str, Any]]:
    """Describe every action of ``parser``, recursing into subcommands."""
    actions: list[dict[str, Any]] = []
    for action in parser._actions:  # noqa: SLF001 - argparse has no public action list
        if isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for name, child in action.choices.items():
                actions.extend(_surface(child, prefix=f"{prefix}{name} "))
            continue
        actions.append(
            {
                "subcommand": prefix.strip() or None,
                "options": list(action.option_strings),
                "dest": action.dest,
                "default": _portable(action.default),
                "choices": _portable(list(action.choices)) if action.choices else None,
                "nargs": _portable(action.nargs),
                "type": getattr(action.type, "__name__", None) if action.type else None,
                "action": type(action).__name__,
                "required": bool(action.required),
                "help": action.help,
            }
        )
    return actions


def test_every_entrypoint_parser_matches_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flags, defaults and help of every entrypoint equal the committed snapshot."""
    surface = {
        module: _surface(_capture(module, function, monkeypatch))
        for module, function in _entrypoint_modules()
    }
    assert surface, "no entrypoint parsers found"

    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(surface, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert sorted(surface) == sorted(expected), "the set of entrypoints changed"
    for module, actions in expected.items():
        assert surface[module] == actions, f"{module}: parser surface changed"
