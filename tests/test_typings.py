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


def _is_property(node: ast.FunctionDef) -> bool:
    """Return True when a stub method is declared as a property."""
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in node.decorator_list
    )


def _public_methods(node: ast.ClassDef) -> Iterator[ast.FunctionDef]:
    """Yield a stub class's public methods and properties, its constructor included."""
    for member in node.body:
        if isinstance(member, ast.FunctionDef) and (
            member.name == "__init__" or not member.name.startswith("_")
        ):
            yield member


def _declared_names(stub: Path | None) -> list[str]:
    """Return the public names a stub declares that the runtime module must define.

    These are the top-level functions, classes and constants, plus each class's public
    methods and properties as ``Class.member``. Annotated instance attributes are left out:
    sklearn sets its fitted attributes (``transformers_``) only after ``fit``.
    """
    if stub is None:
        return []
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    names: set[str] = set()
    for name, node in _public_nodes(tree.body):
        names.add(name)
        if isinstance(node, ast.ClassDef):
            names.update(
                f"{name}.{member.name}"
                for member in _public_methods(node)
                if member.name != "__init__"
            )
    return sorted(names)


def _declared_functions(stub: Path) -> list[StubFunction]:
    """Return the public functions and class methods a stub declares, constructors included.

    Overloads of one function merge into one entry; properties take no parameters and are
    checked by name only.
    """
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    parameters: dict[str, list[str]] = {}
    for name, node in _public_nodes(tree.body):
        if isinstance(node, ast.FunctionDef):
            members = [(name, node)]
        elif isinstance(node, ast.ClassDef):
            members = [
                (f"{name}.{member.name}", member)
                for member in _public_methods(node)
                if not _is_property(member)
            ]
        else:
            continue
        for qualname, function in members:
            merged = parameters.setdefault(qualname, [])
            merged.extend(p for p in _keyword_parameters(function) if p not in merged)
    return [StubFunction(qualname, tuple(names)) for qualname, names in parameters.items()]


def _resolve(module_name: str, qualname: str) -> object | None:
    """Return the runtime object a dotted stub name stands for, or None when it is missing."""
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        if not hasattr(target, part):
            return None
        target = getattr(target, part)
    return target


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

    # Act
    missing = [name for name in declared if _resolve(module_name, name) is None]

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
