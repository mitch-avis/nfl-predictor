"""Tests that the local type stubs under ``typings/`` match the installed libraries.

The stubs are hand-written for third-party libraries that ship no type information. These
tests fail when a library upgrade drops or renames a name a stub declares, or stops
accepting a parameter a stub promises, so a stale stub cannot hide a real break.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

TYPINGS_DIR = Path(__file__).resolve().parents[1] / "typings"

STUBBED_MODULES = frozenset(
    {
        "joblib",
        "nflreadpy",
        "nflreadpy.downloader",
        "sklearn.compose",
        "sklearn.impute",
        "sklearn.linear_model",
        "sklearn.metrics",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "threadpoolctl",
    }
)

# Parameters a runtime signature never lists for a bound or class call.
IMPLICIT_PARAMETERS = frozenset({"self", "cls"})


@dataclass(frozen=True)
class StubFunction:
    """One function or method a stub declares.

    Attributes:
        qualname: Dotted name inside the module, such as ``Pipeline.fit``.
        parameters: The keyword-addressable parameter names the stub declares.
    """

    qualname: str
    parameters: tuple[str, ...]


def _module_name(stub: Path) -> str:
    """Return the dotted module name a stub file under ``typings/`` stands for."""
    parts = stub.relative_to(TYPINGS_DIR).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _stub_files() -> dict[str, Path]:
    """Map each stubbed module name to its stub file."""
    return {_module_name(stub): stub for stub in sorted(TYPINGS_DIR.rglob("*.pyi"))}


def _keyword_parameters(node: ast.FunctionDef) -> tuple[str, ...]:
    """Return a stub function's parameter names a caller can pass by keyword."""
    arguments = node.args
    names = [arg.arg for arg in [*arguments.args, *arguments.kwonlyargs]]
    return tuple(name for name in names if name not in IMPLICIT_PARAMETERS)


def _public_nodes(body: list[ast.stmt]) -> Iterator[tuple[str, ast.stmt]]:
    """Yield each public name a stub body declares with the node declaring it.

    Underscore-prefixed names are stub-only helper types (Protocols, TypedDicts) that the
    runtime module does not need to define.
    """
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        if not name.startswith("_"):
            yield name, node


def _declared_names(stub: Path | None) -> list[str]:
    """Return the public top-level functions, classes and constants a stub declares."""
    if stub is None:
        return []
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    return sorted({name for name, _node in _public_nodes(tree.body)})


def _declared_functions(stub: Path) -> list[StubFunction]:
    """Return the public functions and class methods a stub declares, constructors included."""
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    functions: list[StubFunction] = []
    for name, node in _public_nodes(tree.body):
        if isinstance(node, ast.FunctionDef):
            functions.append(StubFunction(name, _keyword_parameters(node)))
        elif isinstance(node, ast.ClassDef):
            functions.extend(
                StubFunction(f"{name}.{member.name}", _keyword_parameters(member))
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and (member.name == "__init__" or not member.name.startswith("_"))
            )
    return functions


def _runtime_callable(module_name: str, qualname: str) -> Callable[..., object]:
    """Return the runtime object whose signature a stubbed function must match.

    A constructor is checked through the class itself, so ``self`` drops out of the signature.
    """
    target: object = importlib.import_module(module_name)
    parts = qualname.split(".")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for part in parts:
        target = getattr(target, part)
    assert callable(target)
    return target


def _unaccepted(function: StubFunction, signature: inspect.Signature) -> list[str]:
    """Return the stub's parameter names the runtime signature would reject by keyword."""
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    keyword_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    return [
        name
        for name in function.parameters
        if name not in parameters or parameters[name].kind not in keyword_kinds
    ]


def _stub_functions() -> list[tuple[str, StubFunction]]:
    """Return every (module, function) pair the stubs declare, for parametrizing."""
    return [
        (module_name, function)
        for module_name, stub in _stub_files().items()
        for function in _declared_functions(stub)
    ]


def test_typings_stub_the_untyped_libraries_exactly() -> None:
    # Act
    stubbed = set(_stub_files())

    # Assert
    assert stubbed == STUBBED_MODULES


@pytest.mark.parametrize("module_name", sorted(STUBBED_MODULES))
def test_typings_declared_names_exist_at_runtime(module_name: str) -> None:
    # Arrange
    declared = _declared_names(_stub_files().get(module_name))
    module = importlib.import_module(module_name)

    # Act
    missing = [name for name in declared if not hasattr(module, name)]

    # Assert
    assert declared
    assert missing == []


@pytest.mark.parametrize(
    ("module_name", "function"),
    _stub_functions(),
    ids=lambda value: value if isinstance(value, str) else value.qualname,
)
def test_typings_function_parameters_accepted_at_runtime(
    module_name: str, function: StubFunction
) -> None:
    # Arrange
    signature = inspect.signature(_runtime_callable(module_name, function.qualname))

    # Act
    rejected = _unaccepted(function, signature)

    # Assert
    assert rejected == []
