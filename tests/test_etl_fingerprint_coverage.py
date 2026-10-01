"""Guard: the season-cache code fingerprint covers every module a season build imports.

The incremental ETL reuses a cached season only while ``season_cache.etl_code_fingerprint``
is unchanged, so an edit to any module the build runs has to change it. This test follows
the imports of ``nfl_predictor.data_collection`` through the package (relative imports
resolved against the importing module's package, imports that only run for type checkers
excluded), edits each module it reaches in a copy of the package, and checks that every edit
moves the fingerprint.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

import nfl_predictor
from nfl_predictor.utils import season_cache

_PACKAGE = Path(nfl_predictor.__file__).parent
_ROOT = "nfl_predictor"
_ENTRY = f"{_ROOT}.data_collection"


def _module_file(package: Path, name: str) -> Path | None:
    """Return the source file of a module in the package, or None outside it."""
    parts = name.split(".")
    if parts[0] != _ROOT:
        return None
    base = package.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _type_checking_only(tree: ast.Module) -> set[int]:
    """Return the ids of every node under an ``if TYPE_CHECKING:`` block."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                skipped.update(id(child) for child in ast.walk(statement))
    return skipped


def _imported_from(name: str, path: Path, node: ast.ImportFrom) -> str:
    """Return the absolute name of the module a ``from ... import`` statement reads.

    A relative import counts its dots from the importing module's package, which for a
    package's ``__init__`` is the package itself.
    """
    if node.level == 0:
        return node.module or ""
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    base = package.rsplit(".", node.level - 1)[0]
    return f"{base}.{node.module}" if node.module else base


def _imports(package: Path, name: str, path: Path) -> set[str]:
    """Return the package modules a source file imports at run time."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = _type_checking_only(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _imported_from(name, path, node)
            for alias in node.names:
                submodule = f"{module}.{alias.name}"
                names.add(submodule if _module_file(package, submodule) else module)
    return {name for name in names if _module_file(package, name) is not None}


def _etl_modules(package: Path) -> dict[str, Path]:
    """Return every package module the ETL entry module runs, by name, packages included."""
    reached: dict[str, Path] = {}
    pending = [_ENTRY]
    while pending:
        name = pending.pop()
        path = _module_file(package, name)
        if name in reached or path is None:
            continue
        reached[name] = path
        pending.extend(_imports(package, name, path))
        # Importing a module first runs every package above it.
        pending.extend(name.rsplit(".", depth)[0] for depth in range(1, name.count(".") + 1))
    return reached


def _is_docstring_only(path: Path) -> bool:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    return all(
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        for statement in body
    )


def _uncovered_modules(package: Path) -> list[str]:
    """Return the ETL modules whose edit leaves the code fingerprint unchanged, by name."""
    baseline = season_cache.etl_code_fingerprint(package)
    uncovered: list[str] = []
    for name, path in sorted(_etl_modules(package).items()):
        if name == _ROOT:
            continue  # holds no code; see test_the_root_package_runs_no_code
        original = path.read_bytes()
        path.write_bytes(original + b"\n# edited\n")
        try:
            if season_cache.etl_code_fingerprint(package) == baseline:
                uncovered.append(name)
        finally:
            path.write_bytes(original)
    assert season_cache.etl_code_fingerprint(package) == baseline
    return uncovered


def _copy_package(directory: Path) -> Path:
    copy = directory / _ROOT
    shutil.copytree(_PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _copy_package(tmp_path_factory.mktemp("package"))


def test_the_import_walk_reaches_the_etl_modules() -> None:
    reached = _etl_modules(_PACKAGE)

    assert {
        _ENTRY,
        f"{_ROOT}.constants",
        f"{_ROOT}.utils.polars_utils",
        f"{_ROOT}.utils.polars.features",
        f"{_ROOT}.utils.polars.strength_snapshot",
    } <= set(reached)


def test_the_root_package_runs_no_code() -> None:
    """Every ETL import runs the root ``__init__``, which the fingerprint leaves out."""
    assert _is_docstring_only(_PACKAGE / "__init__.py")


def test_every_module_the_etl_imports_is_in_the_code_fingerprint(package_copy: Path) -> None:
    uncovered = _uncovered_modules(package_copy)

    assert uncovered == [], f"ETL modules outside the code fingerprint: {uncovered}"


@pytest.mark.parametrize(
    ("importer", "statement"),
    [
        ("data_collection.py", f"from {_ROOT} import stray"),
        ("data_collection.py", "from . import stray"),
        ("data_collection.py", "from .stray import VALUE"),
        ("utils/polars/week_rows.py", "from ...stray import VALUE"),
        ("utils/polars/__init__.py", "from ... import stray"),
    ],
)
def test_a_relatively_imported_module_outside_the_fingerprint_is_flagged(
    tmp_path: Path, importer: str, statement: str
) -> None:
    package = _copy_package(tmp_path)
    (package / "stray.py").write_text('"""Outside the fingerprint."""\n\nVALUE = 1\n')
    importing_file = package / importer
    importing_file.write_text(importing_file.read_text(encoding="utf-8") + f"\n{statement}\n")

    assert _uncovered_modules(package) == [f"{_ROOT}.stray"]
