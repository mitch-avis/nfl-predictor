"""Mechanical scans for the inventory: data loads, network sources, API routes and web pages.

Every scan reads source text (this checkout's files, or `nfl-sos-ratings` at the frozen commit
through ``git show``) and returns what it found with the file and line, so the inventory never
lists a source, an endpoint or a page by hand.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from port_audit import ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

    from port_audit import SosCode

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
# A web address with a real host name (a regular expression of local hosts is not one).
URL = re.compile(r"https?://[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[a-z]{2,}(?:/[^\s\"'<>()\\]*)?")
DATA_ROOTS = {"DATA_PATH", "DATA_DIR"}
DATA_SUFFIXES = (".csv", ".parquet", ".json")


@dataclass(frozen=True)
class Load:
    """One call that loads data from outside the repository."""

    repo: str
    source: str
    path: str
    line: int
    function: str


def _enclosing(tree: ast.Module) -> dict[int, str]:
    """Map every line inside a function to the function's qualified name."""
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                for line in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    names[line] = name
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def scan_loads(repo: str, files: dict[str, str]) -> list[Load]:
    """Return every nflreadpy load call, network fetch and URL literal in ``files``.

    ``files`` maps a repository path to its text. A call counts as an nflreadpy load when its
    callee is ``load_*`` reached through a name bound to ``nflreadpy`` (``import nflreadpy as
    nfl`` then ``nfl.load_pbp``, or ``from nflreadpy import load_pbp``).
    """
    loads: list[Load] = []
    for path, text in sorted(files.items()):
        tree = ast.parse(text)
        where = _enclosing(tree)
        modules: set[str] = set()
        functions: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.asname or a.name for a in node.names if a.name == "nflreadpy"}
            elif isinstance(node, ast.ImportFrom) and node.module == "nflreadpy":
                functions |= {a.asname or a.name: a.name for a in node.names}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _load_name(node.func, modules, functions)
                if name is not None:
                    loads.append(
                        Load(
                            repo,
                            f"nflreadpy.{name}",
                            path,
                            node.lineno,
                            where.get(node.lineno, "-"),
                        )
                    )
                elif _is_fetch(node.func):
                    loads.append(
                        Load(repo, "network fetch", path, node.lineno, where.get(node.lineno, "-"))
                    )
            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and node.right.value.endswith(DATA_SUFFIXES)
                and {getattr(n, "attr", getattr(n, "id", "")) for n in ast.walk(node.left)}
                & DATA_ROOTS
            ):
                loads.append(
                    Load(
                        repo,
                        f"local file {node.right.value}",
                        path,
                        node.lineno,
                        where.get(node.lineno, "-"),
                    )
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                function = where.get(node.lineno, "-")
                loads.extend(
                    Load(repo, f"url {url}", path, node.lineno, function)
                    for url in URL.findall(node.value)
                )
    return loads


def _load_name(func: ast.expr, modules: set[str], functions: dict[str, str]) -> str | None:
    """Return the nflreadpy loader a callee names, or None."""
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in modules
        and func.attr.startswith("load_")
    ):
        return func.attr
    if isinstance(func, ast.Name) and func.id in functions:
        original = functions[func.id]
        return original if original.startswith("load_") else None
    return None


def _is_fetch(func: ast.expr) -> bool:
    """Return whether a callee fetches over the network (``urlopen``, ``requests.get``...)."""
    if isinstance(func, ast.Name):
        return func.id == "urlopen"
    if isinstance(func, ast.Attribute):
        owner = func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
        if func.attr == "urlopen":
            return True
        return owner_name in {"requests", "httpx", "session", "_session"} and (
            func.attr in HTTP_METHODS or func.attr == "request"
        )
    return False


def tracked(pattern: str) -> list[str]:
    """Return this checkout's tracked files matching a git pathspec."""
    out = subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "ls-files", pattern],  # noqa: S607 - git from PATH
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(out.split())


def here_python_files() -> dict[str, str]:
    """Return this repository's tracked Python source under ``src/`` and ``scripts/``."""
    paths = [p for p in tracked("*.py") if p.startswith(("src/", "scripts/"))]
    return {path: (ROOT / path).read_text(encoding="utf-8") for path in paths}


def sos_python_files(code: SosCode) -> dict[str, str]:
    """Return `nfl-sos-ratings`'s package source at the frozen commit."""
    return {path: code.show(path) for path in code.glob(r"nfl_sos_ratings/.*\.py")}


# ---------------------------------------------------------------------------------------------
# API routes and web pages
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """One API endpoint or one page route."""

    repo: str
    kind: str
    method: str
    path: str
    file: str
    line: int


def _router_prefixes(tree: ast.Module) -> dict[str, str]:
    """Return each module-level ``APIRouter(prefix=...)`` name with its prefix."""
    prefixes: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", getattr(node.value.func, "attr", "")) == "APIRouter"
        ):
            prefix = next(
                (
                    kw.value.value
                    for kw in node.value.keywords
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant)
                ),
                "",
            )
            for target in node.targets:
                if isinstance(target, ast.Name):
                    prefixes[target.id] = str(prefix)
    return prefixes


def _decorated_routes(
    repo: str, path: str, text: str, mount: Callable[[str, str], str | None]
) -> list[Route]:
    """Return the routes declared by ``@<router>.<method>("...")`` decorators in one file.

    ``mount`` maps (router variable, decorator path) to the served path, or None to skip it.
    """
    tree = ast.parse(text)
    routes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in HTTP_METHODS
                and isinstance(decorator.func.value, ast.Name)
                and decorator.args
            ):
                continue
            first = decorator.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                route_path = first.value
            elif isinstance(first, ast.JoinedStr):
                route_path = "".join(
                    str(part.value)
                    if isinstance(part, ast.Constant)
                    else "{" + ast.unparse(part.value) + "}"
                    if isinstance(part, ast.FormattedValue)
                    else ast.unparse(part)
                    for part in first.values
                )
            else:
                continue
            served = mount(decorator.func.value.id, route_path)
            if served is not None:
                routes.append(
                    Route(repo, "api", decorator.func.attr.upper(), served, path, decorator.lineno)
                )
    return routes


def here_api_routes() -> list[Route]:
    """Return this app's endpoints: router prefixes, decorator paths and the app's ``/api``."""
    factory = (ROOT / "src/nfl_predictor/api/__init__.py").read_text(encoding="utf-8")
    api_prefix = re.search(r'^API_PREFIX = "([^"]+)"', factory, re.MULTILINE)
    prefix = api_prefix.group(1) if api_prefix else ""
    included = set(re.findall(r"app\.include_router\((\w+), prefix=API_PREFIX\)", factory))
    router_import = re.compile(
        r"^from (nfl_predictor\.api[\w.]*) import router as (\w+)$", re.MULTILINE
    )
    imports = dict(router_import.findall(factory))
    routes: list[Route] = []
    for module, alias in sorted(imports.items()):
        if alias not in included:
            continue
        rel = "src/" + module.replace(".", "/") + ".py"
        text = (ROOT / rel).read_text(encoding="utf-8")
        prefixes = _router_prefixes(ast.parse(text))
        routes += _decorated_routes(
            "here", rel, text, lambda var, p, ps=prefixes: f"{prefix}{ps.get(var, '')}{p}"
        )
    routes += _decorated_routes(
        "here",
        "src/nfl_predictor/api/__init__.py",
        factory,
        lambda var, p: p.replace("{API_PREFIX}", prefix) if var == "app" else None,
    )
    return routes


def sos_api_routes(code: SosCode) -> list[Route]:
    """Return its endpoints: every decorator path in ``ui_api.py`` outside the SPA fallback."""
    path = "nfl_sos_ratings/ui_api.py"
    return _decorated_routes(
        "sos", path, code.show(path), lambda _var, p: None if "{path:path}" in p else p
    )


PAGE_ROUTE = re.compile(r"path:\s*'([^']*)'")


def page_routes(repo: str, path: str, text: str) -> list[Route]:
    """Return the page paths a React Router route table declares (``path: '...'``)."""
    routes = []
    for number, line in enumerate(text.splitlines(), start=1):
        for found in PAGE_ROUTE.findall(line):
            if found == "*":
                continue
            served = found if found.startswith("/") else f"/{found}"
            routes.append(Route(repo, "page", "GET", served, path, number))
        if "index: true" in line:
            routes.append(Route(repo, "page", "GET", "/ (index)", path, number))
    return routes


def components(repo: str, paths: list[str]) -> list[Route]:
    """Return one entry per page or entity component file (a view the app can show)."""
    return [
        Route(repo, "component", "-", path.rsplit("/", 1)[-1].removesuffix(".tsx"), path, 1)
        for path in paths
        if path.endswith(".tsx") and not path.endswith(".test.tsx")
    ]
