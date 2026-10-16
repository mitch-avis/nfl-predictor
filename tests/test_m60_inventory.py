"""Characterization tests for the CLI flag inventory script under ``.agents/m60/``.

The script is a checked-in record run from the repository root, so the tests import it by path.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

INVENTORY_PATH = Path(__file__).resolve().parents[1] / ".agents" / "m60" / "inventory.py"


@pytest.fixture(scope="module")
def inventory() -> ModuleType:
    """Return the inventory script, imported from its file."""
    spec = importlib.util.spec_from_file_location("m60_inventory", INVENTORY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parse_args_entrypoint(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Return an entrypoint whose ``_parse_args`` builds a parser and parses ``sys.argv``."""
    module_name = "m60_inventory_fake_cli"
    path = tmp_path / f"{module_name}.py"
    path.write_text(
        "import argparse\n"
        "def _parse_args():\n"
        "    parser = argparse.ArgumentParser(prog='fake')\n"
        "    parser.add_argument('--season', type=int, default=2025)\n"
        "    return parser.parse_args()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return inventory.Entrypoint(
        name="fake", module=module_name, path=path, invocations=[], capture_fn="_parse_args"
    )


def _sink_of(inventory: ModuleType, source: str) -> str:
    """Return the sink the inventory describes for the ``args.x`` read in ``source``."""
    tree = ast.parse(source)
    read = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "x"
    )
    return inventory._describe_sink(read, inventory._parents(tree))


def test_capture_parser_returns_the_parser_parse_args_was_called_on(
    inventory: ModuleType, parse_args_entrypoint: object
) -> None:
    # Act
    parser = inventory.capture_parser(parse_args_entrypoint)

    # Assert
    assert [action.dest for action in parser._actions] == ["help", "season"]


def test_capture_parser_restores_argparse_parsing_afterwards(
    inventory: ModuleType, parse_args_entrypoint: object
) -> None:
    # Arrange
    saved = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    # Act
    inventory.capture_parser(parse_args_entrypoint)

    # Assert
    assert (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args) == saved


def test_describe_sink_names_the_key_of_a_dict_value(inventory: ModuleType) -> None:
    # Act
    sink = _sink_of(inventory, "payload = {'season': args.x}")

    # Assert
    assert sink == "dict key payload.'season'"


def test_describe_sink_names_the_keys_of_a_nested_dict_value(inventory: ModuleType) -> None:
    # Act
    sink = _sink_of(inventory, "payload = {'outer': {'inner': args.x}}")

    # Assert
    assert sink == "dict key payload.'outer'.'inner'"


def test_describe_sink_skips_the_key_of_an_unpacked_outer_dict(inventory: ModuleType) -> None:
    # Act
    sink = _sink_of(inventory, "def f():\n    return {**{'inner': args.x}}")

    # Assert
    assert sink == "dict key return.'inner'"
