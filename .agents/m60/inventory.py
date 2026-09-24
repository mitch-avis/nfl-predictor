"""Generate the CLI flag and ``scripts/`` file inventory from the code itself.

Run from the repository root:

    .venv/bin/python .agents/m60/inventory.py

It writes ``INVENTORY.md`` and ``inventory.json`` beside this file and exits non-zero when any
search that must find something comes back empty, or when a hand-written judgment in
``annotations.yaml`` no longer matches its evidence. Nothing in the output is typed by hand:

- every parser is captured by intercepting ``ArgumentParser.parse_args`` while the entrypoint's
  own ``_build_parser``/``_parse_args`` runs, and every action is read from the parser object;
- read sites and their sinks come from the entrypoint's AST;
- web job flags come from calling each job template's build function with every parameter set;
- shell, workflow and Markdown commands are tokenized and attributed to the entrypoint they call;
- the judgments in ``annotations.yaml`` are applied only when each evidence pattern still holds.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUT_MD = HERE / "INVENTORY.md"
OUT_JSON = HERE / "inventory.json"
ANNOTATIONS = HERE / "annotations.yaml"

CAST_CALLS = {"int", "float", "bool", "str", "Path", "list", "tuple", "set", "sorted"}
SHELL_SEPARATORS = {";", "&&", "||", "|", "&"}
# Package modules that are run through a thin facade module rather than directly.
FACADES = {"nfl_predictor.ml.ml_model_cli": "nfl_predictor.ml_model"}
# Sinks that only record a value (run metadata, resume fingerprints, logs), not use it.
RECORD_SINK = re.compile(
    r"^dict key (config_payload|wf_run_fingerprint|report_config|payload|metadata|meta|"
    r"src_metadata|fingerprint)\.|^assigned to (config_payload|src_metadata|metadata)\[|"
    r"^positional arg of log\.\w+\(\)"
)
TUNING_SINK = re.compile(r"^kw (\w+)= of (?:\w+\.)*OptunaConfig\(\)$")
WF_CONFIG_SINK = re.compile(r"^kw (\w+)= of (?:\w+\.)*WalkForwardConfig\(\)$")
TUNING_FUNCTIONS = {"_run_optuna_search", "objective_fn"}
# Config dataclasses whose per-field reads the inventory traces across the package.
CONFIG_CLASSES = {
    "WalkForwardConfig": "nfl_predictor/ml/walk_forward.py",
    "OptunaConfig": "nfl_predictor/ml/ml_model_core.py",
    "MarketProbConfig": "nfl_predictor/ml/ml_model_core.py",
}


class _CapturedParserError(Exception):
    """Raised by the patched ``parse_args`` to hand the parser back to the generator."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        """Keep the parser that was about to parse."""
        super().__init__("captured")
        self.parser = parser


@dataclass
class Failure:
    """A search or check that must have matched and did not."""

    what: str
    detail: str


FAILURES: list[Failure] = []


def fail(what: str, detail: str) -> None:
    """Record a failure; the run exits non-zero at the end."""
    FAILURES.append(Failure(what, detail))


def rel(path: Path) -> str:
    """Return ``path`` relative to the repository root."""
    return str(path.resolve().relative_to(ROOT))


def read(path: Path) -> str:
    """Read a text file, replacing undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def tracked_files(pattern: str) -> list[Path]:
    """Return tracked files matching a git pathspec."""
    out = subprocess.run(  # noqa: S603 - fixed argv; the pattern is a literal from this file
        ["git", "ls-files", pattern],  # noqa: S607 (git is fixed, not user-controlled)
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [ROOT / p for p in out]


# ---------------------------------------------------------------------------------------------
# Entrypoints and parser capture
# ---------------------------------------------------------------------------------------------


@dataclass
class Entrypoint:
    """One CLI: its source file, the module path and how it is invoked."""

    name: str
    module: str
    path: Path
    invocations: list[str]
    capture_fn: str


def _module_name(path: Path) -> str:
    """Return the dotted module name of a repository file."""
    parts = list(path.resolve().relative_to(ROOT).with_suffix("").parts)
    return ".".join(parts)


def _defines_argparse(tree: ast.Module) -> bool:
    """Return whether the module constructs an ``ArgumentParser``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "ArgumentParser":
                return True
    return False


def discover_entrypoints() -> list[Entrypoint]:
    """Find every module under ``nfl_predictor/`` and ``scripts/`` that builds a parser."""
    found: list[Entrypoint] = []
    candidates = sorted((ROOT / "nfl_predictor").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )
    for path in candidates:
        tree = ast.parse(read(path))
        if not _defines_argparse(tree):
            continue
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        capture = "_build_parser" if "_build_parser" in funcs else "_parse_args"
        if capture not in funcs:
            fail("entrypoint discovery", f"{rel(path)} builds a parser but has no {capture}")
            continue
        module = _module_name(path)
        if path.parts[-2] == "scripts":
            name = path.stem
            invocations = [f"scripts/{path.name}"]
        else:
            run_module = module.removesuffix(".__main__")
            run_module = FACADES.get(run_module, run_module)
            name = run_module.removeprefix("nfl_predictor.")
            invocations = [f"-m {run_module}"]
        found.append(Entrypoint(name, module, path, invocations, capture))
    if not found:
        fail("entrypoint discovery", "no ArgumentParser found under nfl_predictor/ or scripts/")
    for module, facade in FACADES.items():
        facade_path = ROOT / Path(*facade.split(".")).with_suffix(".py")
        if module.rsplit(".", 1)[-1] not in read(facade_path):
            fail("facade check", f"{rel(facade_path)} no longer references {module}")
    return found


def capture_parser(ep: Entrypoint) -> argparse.ArgumentParser:
    """Build the entrypoint's parser without parsing anything."""
    module = importlib.import_module(ep.module)
    fn = getattr(module, ep.capture_fn)
    if ep.capture_fn == "_build_parser":
        return fn()

    def _raise(self: argparse.ArgumentParser, *_a: object, **_k: object) -> None:
        raise _CapturedParserError(self)

    saved = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)
    saved_argv = sys.argv
    argparse.ArgumentParser.parse_args = _raise  # type: ignore[method-assign]
    argparse.ArgumentParser.parse_known_args = _raise  # type: ignore[method-assign]
    sys.argv = [ep.path.name]
    try:
        if inspect.signature(fn).parameters:
            fn([])
        else:
            fn()
    except _CapturedParserError as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = saved
        sys.argv = saved_argv
    raise RuntimeError(f"{ep.name}: {ep.capture_fn} returned without parsing")


def _jsonable(value: Any) -> Any:
    """Return a JSON-friendly rendering of an argparse default."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class Flag:
    """One parser action and everything the inventory learns about it."""

    entrypoint: str
    option_strings: list[str]
    dest: str
    default: Any
    type: str | None
    choices: list[Any] | None
    nargs: Any
    action: str
    required: bool
    help: str | None
    subcommand: str | None = None
    reads: list[dict[str, Any]] = field(default_factory=list)
    writes: list[int] = field(default_factory=list)
    weekly_yaml: Any = None
    web_templates: list[str] = field(default_factory=list)
    consumers: dict[str, list[str]] = field(default_factory=dict)
    auto_class: str = ""
    tags: list[str] = field(default_factory=list)
    annotation: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        """Return the display key: the first long option, or the dest for positionals."""
        longs = [o for o in self.option_strings if o.startswith("--")]
        return longs[0] if longs else (self.option_strings[0] if self.option_strings else self.dest)


def walk_actions(ep: Entrypoint, parser: argparse.ArgumentParser) -> list[Flag]:
    """Return every non-help action, recursing into subcommands."""
    flags: list[Flag] = []

    def _walk(p: argparse.ArgumentParser, sub: str | None) -> None:
        for action in p._actions:  # noqa: SLF001 - argparse exposes no public action list
            if isinstance(action, argparse._HelpAction):  # noqa: SLF001
                continue
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                for name, child in action.choices.items():
                    _walk(child, name)
                continue
            type_name = getattr(action.type, "__name__", None) if action.type else None
            flags.append(
                Flag(
                    entrypoint=ep.name,
                    option_strings=list(action.option_strings),
                    dest=action.dest,
                    default=_jsonable(action.default),
                    type=type_name,
                    choices=[_jsonable(c) for c in action.choices] if action.choices else None,
                    nargs=action.nargs,
                    action=type(action).__name__,
                    required=bool(action.required),
                    help=action.help,
                    subcommand=sub,
                )
            )

    _walk(parser, None)
    return flags


# ---------------------------------------------------------------------------------------------
# Read sites and sinks
# ---------------------------------------------------------------------------------------------


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Map every node to its parent."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _namespace_names(tree: ast.Module) -> set[str]:
    """Return the variable names that hold a parsed namespace in this module."""
    names = {"args"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.annotation is not None and "Namespace" in ast.unparse(arg.annotation):
                    names.add(arg.arg)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = ast.unparse(node.value.func)
            if callee.endswith(("_parse_args", "parse_args")):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Return the name of the function that contains ``node``."""
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
            return cur.name
        cur = parents.get(cur)
    return "<module>"


def _describe_sink(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Describe where a namespace value flows, one AST step past any casts."""
    cur = node
    parent = parents.get(cur)
    while (
        isinstance(parent, ast.Call)
        and cur in parent.args
        and isinstance(parent.func, ast.Name)
        and parent.func.id in CAST_CALLS
    ):
        cur = parent
        parent = parents.get(cur)
    if isinstance(parent, ast.keyword):
        call = parents.get(parent)
        callee = ast.unparse(call.func) if isinstance(call, ast.Call) else "?"
        return f"kw {parent.arg}= of {callee}()"
    if isinstance(parent, ast.Dict) and cur in parent.values:
        key = parent.keys[parent.values.index(cur)]  # type: ignore[arg-type]
        owner = parents.get(parent)
        where = ""
        while owner is not None and not isinstance(owner, ast.stmt):
            if isinstance(owner, ast.Dict):
                idx = owner.values.index(parent) if parent in owner.values else -1
                if idx >= 0 and owner.keys[idx] is not None:
                    where = f"{ast.unparse(owner.keys[idx])}." + where  # type: ignore[arg-type]
            parent, owner = owner, parents.get(owner)
        if isinstance(owner, ast.Assign):
            where = ast.unparse(owner.targets[0]) + "." + where
        elif isinstance(owner, ast.Return):
            where = "return." + where
        return f"dict key {where}{ast.unparse(key) if key is not None else '**'}"
    if isinstance(parent, ast.Call) and cur in parent.args:
        return f"positional arg of {ast.unparse(parent.func)}()"
    if isinstance(parent, ast.Assign | ast.AnnAssign):
        target = parent.targets[0] if isinstance(parent, ast.Assign) else parent.target
        return f"assigned to {ast.unparse(target)}"
    if isinstance(parent, ast.Attribute):
        return f"attribute .{parent.attr} of the value"
    if isinstance(parent, ast.Compare | ast.If | ast.BoolOp | ast.UnaryOp | ast.IfExp | ast.While):
        return "condition / control flow"
    if isinstance(parent, ast.FormattedValue):
        return "f-string"
    if isinstance(parent, ast.Return):
        return "returned"
    if isinstance(parent, ast.BinOp):
        return "arithmetic / path join"
    return type(parent).__name__ if parent is not None else "?"


@dataclass
class ReadScan:
    """Namespace reads and writes found in one entrypoint file."""

    reads: dict[str, list[dict[str, Any]]]
    writes: dict[str, list[int]]
    bulk: list[dict[str, Any]]
    dynamic: list[dict[str, Any]]


def scan_reads(path: Path) -> ReadScan:
    """Find every ``args.<dest>`` read, write, bulk ``vars(args)`` and dynamic ``getattr``."""
    source = read(path)
    tree = ast.parse(source)
    parents = _parents(tree)
    names = _namespace_names(tree)
    reads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    writes: dict[str, list[int]] = defaultdict(list)
    bulk: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
        ):
            if isinstance(node.ctx, ast.Store):
                writes[node.attr].append(node.lineno)
                continue
            sink = _describe_sink(node, parents)
            reads[node.attr].append(
                {
                    "line": node.lineno,
                    "function": _enclosing_function(node, parents),
                    "sink": sink,
                    "record_only": bool(RECORD_SINK.search(sink)),
                }
            )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (
                node.func.id == "vars"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in names
            ):
                bulk.append({"line": node.lineno, "function": _enclosing_function(node, parents)})
            if (
                node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in names
            ):
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                    sink = _describe_sink(node, parents)
                    reads[attr.value].append(
                        {
                            "line": node.lineno,
                            "function": _enclosing_function(node, parents),
                            "sink": "getattr " + sink,
                            "record_only": bool(RECORD_SINK.search(sink)),
                        }
                    )
                else:
                    dynamic.append(
                        {"line": node.lineno, "function": _enclosing_function(node, parents)}
                    )
    name_loads: dict[tuple[str, str], list[ast.Name]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name_loads[(_enclosing_function(node, parents), node.id)].append(node)
    for sites in reads.values():
        for site in sites:
            match = re.fullmatch(r"assigned to (\w+)", site["sink"])
            if not match:
                continue
            hops = [
                _describe_sink(load, parents)
                for load in name_loads.get((site["function"], match.group(1)), [])
                if load.lineno >= site["line"]
            ]
            site["hops"] = sorted(set(hops))
            if hops:
                site["record_only"] = all(RECORD_SINK.search(h) for h in hops)
    return ReadScan(dict(reads), dict(writes), bulk, dynamic)


# ---------------------------------------------------------------------------------------------
# Consumers: weekly config, web job templates, commands in text files, argv builders, tests
# ---------------------------------------------------------------------------------------------


def weekly_yaml() -> dict[str, Any]:
    """Load the production weekly config."""
    return yaml.safe_load(read(ROOT / "config" / "weekly_run.yaml")) or {}


def _sample_param(spec: Any) -> Any:
    """Return a value that makes a job template emit the parameter's flag."""
    if spec.kind == "bool":
        return True
    if spec.kind == "choice":
        return spec.choices[0]
    if spec.kind in {"int", "float"}:
        return int(spec.minimum) if spec.minimum is not None else 1
    return "x"


def web_templates() -> list[dict[str, Any]]:
    """Call every job template's build function with every parameter set."""
    from nfl_predictor.api.jobs import catalog
    from nfl_predictor.api.settings import Settings

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="inventory_") as tmp:
        settings = Settings(state_dir=Path(tmp) / "state", data_dir=Path(tmp) / "data")
        run_dir = Path(tmp) / "run"
        run = SimpleNamespace(
            run_dir=run_dir,
            model_kind="margin_total",
            run_files=SimpleNamespace(
                model=run_dir / "model.joblib", predictions=run_dir / "predictions.csv"
            ),
        )
        for template in catalog.TEMPLATES:
            params = {spec.name: _sample_param(spec) for spec in template.params}
            ctx = catalog.JobContext(
                settings=settings,
                job_id=f"inventory_{template.id}",
                params=params,
                run=run,  # type: ignore[arg-type]
            )
            argv = template.build(ctx)
            variants: list[dict[str, Any]] = []
            for spec in template.params:
                for choice in spec.choices:
                    vctx = catalog.JobContext(
                        settings=settings,
                        job_id=f"inventory_{template.id}_variant",
                        params={**params, spec.name: choice},
                        run=run,  # type: ignore[arg-type]
                    )
                    variants.append(
                        {"param": spec.name, "value": choice, "argv": template.build(vctx)}
                    )
            recorded = set(
                re.findall(
                    r'"model_kind": "(\w+)"',
                    read(ROOT / "nfl_predictor/ml/ml_model_training.py"),
                )
            )
            for kind in sorted(set(catalog.MODEL_KINDS) | recorded):
                vrun = SimpleNamespace(**{**run.__dict__, "model_kind": kind})
                vctx = catalog.JobContext(
                    settings=settings,
                    job_id=f"inventory_{template.id}_variant",
                    params=params,
                    run=vrun,  # type: ignore[arg-type]
                )
                vargv = template.build(vctx)
                if vargv != argv:
                    variants.append(
                        {"param": "active run model_kind", "value": kind, "argv": vargv}
                    )
            config_keys: list[str] = []
            config_file = Path(tmp) / "state" / "job_configs" / f"inventory_{template.id}.json"
            if config_file.exists():
                config_keys = sorted(json.loads(read(config_file)))
            target = argv[1] if argv[1] != "-m" else f"-m {argv[2]}"
            if target.endswith(".py"):
                target = "scripts/" + Path(target).name
            results.append(
                {
                    "id": template.id,
                    "target": target,
                    "flags": [a.split("=")[0] for a in argv if a.startswith("--")],
                    "config_keys": config_keys,
                    "params": [spec.name for spec in template.params],
                    "argv": argv,
                    "variants": variants,
                }
            )
    if not results:
        fail("web templates", "catalog.TEMPLATES built no commands")
    return results


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations; return (first line number, joined text)."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not buf:
            start = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(stripped)
        out.append((start, " ".join(buf)))
        buf = []
    if buf:
        out.append((start, " ".join(buf)))
    return out


def _tokens(line: str) -> list[str]:
    """Split a shell-ish line, falling back to whitespace when quoting is unbalanced."""
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.split()


def commands_in_text(path: Path, entrypoints: list[Entrypoint]) -> list[dict[str, Any]]:
    """Find invocations of each entrypoint in a text file and the flags each passes."""
    found: list[dict[str, Any]] = []
    for number, line in _logical_lines(read(path)):
        tokens = [t.strip("`") for t in _tokens(line)]
        for idx, token in enumerate(tokens):
            for ep in entrypoints:
                hit = False
                for inv in ep.invocations:
                    if inv.startswith("-m "):
                        hit = token == inv[3:] and idx > 0 and tokens[idx - 1] == "-m"
                    else:
                        hit = token == inv or token.endswith("/" + inv)
                    if hit:
                        break
                if not hit:
                    continue
                flags: list[str] = []
                for tok in tokens[idx + 1 :]:
                    if tok in SHELL_SEPARATORS or tok.endswith(";"):
                        break
                    if tok.startswith("--"):
                        option = tok.split("=")[0].rstrip("`'\".,);:")
                        if option != "--help":
                            flags.append(option)
                found.append({"entrypoint": ep.name, "line": number, "flags": flags})
    return found


def argv_literals(paths: list[Path]) -> dict[str, list[str]]:
    """Map each ``--option`` string literal in Python sources to its ``file:line`` sites."""
    sites: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        tree = ast.parse(read(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.fullmatch(r"--[a-z0-9][a-z0-9-]*(=.*)?", node.value)
            ):
                sites[node.value.split("=")[0]].append(f"{rel(path)}:{node.lineno}")
    return sites


def python_imports(path: Path) -> dict[str, set[str]]:
    """Return ``{module: {names used}}`` for ``scripts`` and package imports in a file."""
    tree = ast.parse(read(path))
    imported: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "scripts":
                for alias in node.names:
                    imported[f"scripts.{alias.name}"]
                    aliases[alias.asname or alias.name] = f"scripts.{alias.name}"
            else:
                for alias in node.names:
                    imported[node.module].add(alias.name)
                    full = f"{node.module}.{alias.name}"
                    aliases[alias.asname or alias.name] = full
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.name]
                if alias.asname:
                    aliases[alias.asname] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id)
            if module is not None and module.startswith("scripts."):
                imported[module].add(node.attr)
    return dict(imported)


def string_mentions(path: Path) -> list[str]:
    """Return string constants in a Python file (used to find paths like ``scripts/x.py``)."""
    tree = ast.parse(read(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


# ---------------------------------------------------------------------------------------------
# Annotations: hand judgments whose evidence the script re-checks
# ---------------------------------------------------------------------------------------------


def check_evidence(items: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify each ``{path, pattern, expect}`` item; return (all held, rendered lines)."""
    ok = True
    lines: list[str] = []
    for item in items:
        path = ROOT / item["path"]
        expect = item.get("expect", "present")
        if not path.exists():
            ok = False
            lines.append(f"MISSING FILE {item['path']}")
            continue
        text = read(path)
        hits = [
            n
            for n, line in enumerate(text.splitlines(), start=1)
            if re.search(item["pattern"], line)
        ]
        held = bool(hits) if expect == "present" else not hits
        ok = ok and held
        where = ", ".join(str(h) for h in hits[:8]) or "none"
        status = "holds" if held else "FAILS"
        lines.append(f"{status}: `{item['path']}` /{item['pattern']}/ {expect} (lines: {where})")
    return ok, lines


# ---------------------------------------------------------------------------------------------
# Fixed checks for the findings carried from the 2026-09-23 review and the step-2 follow-ups
# ---------------------------------------------------------------------------------------------


def _attr_loads(path: Path, attr: str) -> list[tuple[int, str, str]]:
    """Return (line, enclosing function, value expression) for loads of ``.attr``."""
    tree = ast.parse(read(path))
    parents = _parents(tree)
    return [
        (node.lineno, _enclosing_function(node, parents), ast.unparse(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Load)
    ]


def _flag(flags: list[Flag], entrypoint: str, option: str) -> Flag | None:
    """Return the flag of ``entrypoint`` carrying ``option``."""
    for flag in flags:
        if flag.entrypoint == entrypoint and option in flag.option_strings:
            return flag
    return None


def fixed_checks(flags: list[Flag], yaml_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the fixed, code-level checks and return one record per claim."""
    checks: list[dict[str, Any]] = []

    def add(claim: str, confirmed: bool, evidence: list[str]) -> None:
        checks.append({"claim": claim, "confirmed": confirmed, "evidence": evidence})

    wf_py = ROOT / "nfl_predictor/ml/walk_forward.py"
    es_loads = [
        load
        for load in _attr_loads(wf_py, "early_stopping_rounds")
        if load[2] in {"self", "config", "cfg", "wf_config"}
    ]
    es_funcs = {load[1] for load in es_loads}
    wf_flag = _flag(flags, "weekly_run", "--wf-early-stopping-rounds")
    evidence = [
        f"walk_forward.py:{ln} in {fn}() reads {val}.early_stopping_rounds"
        for ln, fn, val in es_loads
    ]
    if wf_flag:
        evidence += [f"weekly_run.py:{r['line']} {r['sink']}" for r in wf_flag.reads]
    none_fits = [
        ln
        for ln, text in enumerate(read(wf_py).splitlines(), 1)
        if "early_stopping_rounds=None" in text
    ]
    evidence.append(f"walk_forward.py passes early_stopping_rounds=None at lines {none_fits}")
    add(
        "`--wf-early-stopping-rounds` (weekly_run) is inert: it reaches "
        "WalkForwardConfig.early_stopping_rounds, which is read only by to_dict() "
        "(metadata and fingerprint); every in-season fit passes None",
        bool(wf_flag) and es_funcs == {"to_dict"} and bool(none_fits),
        evidence,
    )

    ml_dir = ROOT / "nfl_predictor/ml"
    optuna_loads = []
    for path in sorted(ml_dir.glob("*.py")):
        for ln, fn, val in _attr_loads(path, "early_stopping_rounds"):
            if "optuna" in val:
                optuna_loads.append(f"{path.name}:{ln} {fn}() reads {val}.early_stopping_rounds")
    sinks = []
    for ep, opt in [
        ("weekly_run", "--train-early-stopping-rounds"),
        ("golden_command", "--train-early-stopping-rounds"),
        ("ml_model", "--early-stopping-rounds"),
    ]:
        flag = _flag(flags, ep, opt)
        if flag is None:
            sinks.append(f"{ep} {opt}: NOT FOUND")
            continue
        sinks += [f"{ep} {opt} -> {r['function']}(): {r['sink']}" for r in flag.reads]
    golden = _flag(flags, "golden_command", "--train-early-stopping-rounds")
    help_says = bool(golden and golden.help and "production training" in golden.help)
    add(
        "`--train-early-stopping-rounds` (weekly_run, golden_command) and "
        "`--early-stopping-rounds` (ml_model) reach only OptunaConfig.early_stopping_rounds, "
        "which only the tuning code reads; golden_command's help still says "
        "'during production training'",
        help_says and all("NOT FOUND" not in s for s in sinks),
        [*sinks, *optuna_loads, f"golden help: {golden.help if golden else None!r}"],
    )

    postseason = yaml_cfg.get("postseason_weight")
    weights_src = read(ROOT / "nfl_predictor/ml/sample_weights.py")
    gate = re.search(r"if not include_postseason:\s*\n\s*return None", weights_src)
    add(
        "`postseason_weight: 1.3` in config/weekly_run.yaml is inert: include_postseason is "
        "false and compute_postseason_sample_weight returns None when it is",
        postseason == 1.3 and yaml_cfg.get("include_postseason") is False and bool(gate),
        [
            f"weekly_run.yaml postseason_weight={postseason!r}, "
            f"include_postseason={yaml_cfg.get('include_postseason')!r}",
            "sample_weights.py `if not include_postseason: return None` "
            + ("present" if gate else "ABSENT"),
        ],
    )

    inc = _flag(flags, "betting_pipeline", "--include-postseason")
    wt = _flag(flags, "betting_pipeline", "--postseason-weight")
    add(
        "betting_pipeline defaults `--include-postseason` to true with weight 1.15",
        bool(inc and inc.default is True and wt and wt.default == 1.15),
        [
            f"--include-postseason default={inc.default if inc else None!r}",
            f"--postseason-weight default={wt.default if wt else None!r}",
        ],
    )

    pair_eps = sorted(
        {f.entrypoint for f in flags if "--market-prob-weight" in f.option_strings}
        & {f.entrypoint for f in flags if "--market-prob-blend" in f.option_strings}
    )
    pair_lines = [
        f"{f.entrypoint}: {f.option_strings} dest={f.dest} default={f.default!r} help={f.help!r}"
        for f in flags
        if f.entrypoint in pair_eps
        and {"--market-prob-weight", "--market-prob-blend"} & set(f.option_strings)
    ]
    golden_mpw = sorted(
        o
        for f in flags
        if f.entrypoint == "golden_command"
        for o in f.option_strings
        if o.startswith("--market-prob-") and o in {"--market-prob-weight", "--market-prob-blend"}
    )
    wf_field = "market_prob_weight" in read(ROOT / "nfl_predictor/ml/walk_forward.py")
    add(
        "`--market-prob-weight` / `--market-prob-blend` are an alias pair; the internal "
        "field, the metadata key and golden_command's only flag all say `weight`",
        bool(pair_eps) and golden_mpw == ["--market-prob-weight"] and wf_field,
        [
            *pair_lines,
            f"golden_command options: {golden_mpw}",
            f"WalkForwardConfig/to_dict field market_prob_weight present: {wf_field}",
            "note: two separate argparse actions reconciled in code, not one action with two "
            "option strings",
        ],
    )
    return checks


def recency_weeks_usage() -> dict[str, Any]:
    """Search run artifacts and launchers for any use of week-based recency weighting."""
    launchers = [
        rel(p)
        for p in sorted((ROOT / "models").rglob("*.sh"))
        if "recency-half-life-weeks" in read(p)
    ]
    pattern = re.compile(r'"(?:wf_|train_)?recency_half_life_weeks"\s*:\s*(?!null)[^,}\s]+')
    nonnull: list[str] = []
    scanned = 0
    for path in sorted((ROOT / "models").rglob("*.json")):
        if "wf_checkpoints" in path.parts or path.stat().st_size > 50_000_000:
            continue
        scanned += 1
        match = pattern.search(read(path))
        if match:
            nonnull.append(f"{rel(path)}: {match.group(0)}")
    if scanned == 0:
        fail("recency weeks search", "no JSON artifacts found under models/")
    return {"launchers": launchers, "nonnull_json": nonnull, "json_scanned": scanned}


def followups(flags: list[Flag]) -> list[dict[str, Any]]:
    """Collect evidence for the step-2 follow-ups."""
    out: list[dict[str, Any]] = []
    wf_py = ROOT / "nfl_predictor/ml/walk_forward.py"
    loads = _attr_loads(wf_py, "early_stopping_rounds")
    out.append(
        {
            "item": "WalkForwardConfig.early_stopping_rounds in the config and fingerprint",
            "evidence": [
                f"walk_forward.py:{ln} {fn}() reads {v}.early_stopping_rounds"
                for ln, fn, v in loads
            ],
        }
    )
    fingerprint_sites = [
        f"{rel(p)}:{n}: {line.strip()}"
        for p in sorted((ROOT / "nfl_predictor").rglob("*.py"))
        for n, line in enumerate(read(p).splitlines(), 1)
        if "to_dict()" in line and ("fingerprint" in line or "config" in line)
    ]
    out[-1]["evidence"] += fingerprint_sites[:10]

    wfc = read(ROOT / "scripts/wf_compare.py")
    display = re.search(r"display_cols = \[(.*?)\]", wfc, re.S)
    cols = re.findall(r'"([a-z_]+)"', display.group(1)) if display else []
    if not display:
        fail("wf_compare display columns", "display_cols list not found in scripts/wf_compare.py")
    out.append(
        {
            "item": "wf_compare.py prints deterministic_pick_accuracy, not pick_accuracy",
            "confirmed": "deterministic_pick_accuracy" in cols and "pick_accuracy" not in cols,
            "evidence": [f"display_cols = {cols}"],
        }
    )

    code = sorted((ROOT / "nfl_predictor").rglob("*.py")) + sorted((ROOT / "scripts").glob("*"))
    omp = [
        f"{rel(p)}:{n}"
        for p in code
        if p.is_file()
        for n, line in enumerate(read(p).splitlines(), 1)
        if "OMP_WAIT_POLICY" in line
    ]
    omp_launch = [
        rel(p) for p in sorted((ROOT / "models").rglob("*.sh")) if "OMP_WAIT_POLICY" in read(p)
    ]
    out.append(
        {
            "item": "Automatic OpenMP wait policy in the walk-forward entry points",
            "confirmed": not omp,
            "evidence": [
                f"code references: {omp or 'none'}",
                f"{len(omp_launch)} launchers under models/ set it by hand",
            ],
        }
    )

    ckpt_code = [
        f"{rel(p)}:{n}: {line.strip()}"
        for p in code
        if p.is_file() and p.suffix == ".py"
        for n, line in enumerate(read(p).splitlines(), 1)
        if "wf_checkpoints" in line
    ]
    prune = [c for c in ckpt_code if re.search(r"rmtree|unlink|prune|cleanup", c)]
    ckpt_dir = ROOT / "models" / "wf_checkpoints"
    dirs = [d for d in ckpt_dir.iterdir() if d.is_dir()] if ckpt_dir.exists() else []
    size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
    out.append(
        {
            "item": "models/wf_checkpoints/ is never pruned",
            "confirmed": not prune,
            "evidence": [
                *ckpt_code,
                f"pruning code: {prune or 'none'}",
                f"{len(dirs)} checkpoint directories, {size / 1e6:.1f} MB",
            ],
        }
    )

    import xgboost

    fit_params = list(inspect.signature(xgboost.XGBRegressor.fit).parameters)
    helper = read(ROOT / "nfl_predictor/ml/ml_model_xgb_utils.py")
    guarded = 'callbacks and _xgb_fit_supports("callbacks")' in helper
    core = read(ROOT / "nfl_predictor/ml/ml_model_core.py")
    passes = [n for n, line in enumerate(core.splitlines(), 1) if "LogEvalCallback(log" in line]
    out.append(
        {
            "item": "_build_xgb_fit_kwargs drops LogEvalCallback on this XGBoost",
            "confirmed": guarded and "callbacks" not in fit_params,
            "evidence": [
                f"xgboost {xgboost.__version__} XGBRegressor.fit parameters: {fit_params}",
                f'guard `callbacks and _xgb_fit_supports("callbacks")` present: {guarded}',
                f"ml_model_core.py passes LogEvalCallback at lines {passes}",
            ],
        }
    )
    return out


def _tuning_gated(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Return whether ``node`` sits in tuning code or under an ``if <x>.enabled`` branch."""
    if _enclosing_function(node, parents) in TUNING_FUNCTIONS:
        return True
    cur: ast.AST | None = parents.get(node)
    while cur is not None and not isinstance(cur, ast.FunctionDef):
        if isinstance(cur, ast.If) and re.search(r"\.enabled\b|\btune\b", ast.unparse(cur.test)):
            return True
        cur = parents.get(cur)
    return False


def config_field_reads() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Map each config dataclass field to the functions that read it.

    A read counts when the value is ``self`` inside the class, a parameter annotated with the
    class, or a local assigned from the class constructor or ``dataclasses.replace``.
    """
    files = sorted((ROOT / "nfl_predictor").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cls, home in CONFIG_CLASSES.items():
        tree = ast.parse(read(ROOT / home))
        class_node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
        )
        fields = [
            n.target.id
            for n in class_node.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        ]
        reads: dict[str, list[dict[str, Any]]] = {f: [] for f in fields}
        for path in files:
            ftree = ast.parse(read(path))
            parents = _parents(ftree)
            for func in ast.walk(ftree):
                if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                names: set[str] = set()
                for arg in func.args.args + func.args.kwonlyargs:
                    if arg.annotation is not None and cls in ast.unparse(arg.annotation):
                        names.add(arg.arg)
                owner = parents.get(func)
                if isinstance(owner, ast.ClassDef) and owner.name == cls:
                    names.add("self")
                for node in ast.walk(func):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                        callee = ast.unparse(node.value.func)
                        if callee.endswith(cls) or (
                            callee.endswith("replace")
                            and node.value.args
                            and ast.unparse(node.value.args[0]) in names
                        ):
                            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
                if not names:
                    continue
                for node in ast.walk(func):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in names
                        and node.attr in reads
                        and _enclosing_function(node, parents) == func.name
                    ):
                        reads[node.attr].append(
                            {
                                "site": f"{rel(path)}:{node.lineno}",
                                "function": func.name,
                                "tuning": _tuning_gated(node, parents),
                            }
                        )
        result[cls] = reads
        if not any(result[cls].values()):
            fail("config field reads", f"no reads of any {cls} field found")
    return result


def score_model_evidence(flags: list[Flag]) -> dict[str, Any]:
    """Collect every use of ``ScoreModel`` and of the ``score`` model kind."""
    texts = (
        tracked_files("*.py")
        + tracked_files("*.md")
        + tracked_files("*.sh")
        + tracked_files("*.ts")
        + tracked_files("*.tsx")
        + tracked_files("*.yaml")
    )
    launchers = sorted((ROOT / "models").rglob("*.sh"))
    pattern = re.compile(
        r"\bScoreModel\b|model[_-]kind\W{0,6}score\b|[\"']score[\"']\s*[:,)\]]|\bscore_model\b"
    )
    hits: list[str] = []
    for path in texts + launchers:
        if not path.exists() or ".agents/m60/" in str(path):
            continue
        for n, line in enumerate(read(path).splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel(path)}:{n}: {line.strip()[:140]}")
    kinds = [
        f"{f.entrypoint} {f.key} choices={f.choices} default={f.default!r}"
        for f in flags
        if f.dest == "model_kind"
    ]
    meta_kinds: dict[str, int] = defaultdict(int)
    for meta in sorted((ROOT / "models").glob("*/metadata.json")):
        try:
            data = json.loads(read(meta))
        except json.JSONDecodeError:
            continue
        stack: list[tuple[str, Any]] = [("", data)]
        found_kind = False
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "model_kind":
                        meta_kinds[f"{prefix}{key}={value}"] += 1
                        found_kind = True
                    stack.append((f"{prefix}{key}.", value))
        if not found_kind:
            meta_kinds["(no model_kind key)"] += 1
    if not hits:
        fail("ScoreModel search", "no ScoreModel or score model-kind occurrences found")
    import warnings

    import joblib

    artifact_types: dict[str, int] = defaultdict(int)
    model_files = sorted(
        p for p in (ROOT / "models").rglob("model*.joblib") if "wf_checkpoints" not in p.parts
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in model_files:
            try:
                artifact_types[type(joblib.load(path)).__name__] += 1
            except Exception as exc:  # noqa: BLE001 - any unloadable artifact is reported
                artifact_types[f"unloadable ({type(exc).__name__})"] += 1
    if not model_files:
        fail("model artifact scan", "no model*.joblib files found under models/")
    return {
        "hits": hits,
        "model_kind_flags": kinds,
        "run_metadata_model_kinds": dict(meta_kinds),
        "saved_model_types": dict(artifact_types),
    }


# ---------------------------------------------------------------------------------------------
# Scripts file inventory
# ---------------------------------------------------------------------------------------------


def scripts_inventory(
    entrypoints: list[Entrypoint], templates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the per-file inventory of ``scripts/``."""
    py_files = tracked_files("*.py")
    md_files = tracked_files("*.md")
    ci_files = [ROOT / "scripts/gate.sh", *sorted((ROOT / ".github/workflows").glob("*.yml"))]
    launchers = sorted((ROOT / "models").rglob("*.sh"))
    imports = {rel(p): python_imports(p) for p in py_files}
    strings = {rel(p): string_mentions(p) for p in py_files}
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "scripts").iterdir()):
        if not path.is_file():
            continue
        name = path.name
        module = f"scripts.{path.stem}"
        text = read(path)
        own = imports.get(rel(path), {})
        importers = {
            f: sorted(names)
            for f, mods in imports.items()
            for m, names in mods.items()
            if m == module and f != rel(path)
        }
        row = {
            "file": rel(path),
            "lines": len(text.splitlines()),
            "imports_scripts": sorted(m for m in own if m.startswith("scripts.")),
            "imported_by_tests": sorted(f for f in importers if f.startswith("tests/")),
            "imported_by_code": {f: n for f, n in importers.items() if not f.startswith("tests/")},
            "names_used_by_importers": sorted({n for names in importers.values() for n in names}),
            "web_templates": sorted(t["id"] for t in templates if t["target"] == rel(path)),
            "ci_gate": [
                f"{rel(c)}:{n}"
                for c in ci_files
                for n, line in enumerate(read(c).splitlines(), 1)
                if f"scripts/{name}" in line
            ],
            "docs": {
                rel(m): c
                for m in md_files
                if (c := len(re.findall(rf"(?<![\w-]){re.escape(name)}\b", read(m))))
            },
            "launchers": sorted(
                rel(p) for p in launchers if re.search(rf"scripts/{re.escape(name)}\b", read(p))
            ),
            "python_path_strings": sorted(
                f for f, s in strings.items() if f != rel(path) and any(name in v for v in s)
            ),
            "has_main_guard": '__name__ == "__main__"' in text,
            "exclusive_package_modules": sorted(
                m
                for m in own
                if m.startswith("nfl_predictor.")
                and (ROOT / Path(*m.split("."))).with_suffix(".py").exists()
                and not any(
                    m in mods
                    for f, mods in imports.items()
                    if f != rel(path)
                    and not f.startswith("tests/")
                    and f != rel((ROOT / Path(*m.split("."))).with_suffix(".py"))
                )
            ),
        }
        rows.append(row)
    if not rows:
        fail("scripts inventory", "scripts/ has no files")
    return rows


def duplicate_functions() -> list[dict[str, Any]]:
    """Find top-level functions of ``scripts/`` defined again elsewhere in scripts or package."""
    files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "nfl_predictor").rglob("*.py"))
    defs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in files:
        tree = ast.parse(read(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                body = ast.Module(
                    body=node.body[1:] if ast.get_docstring(node) else node.body, type_ignores=[]
                )
                defs[node.name].append((f"{rel(path)}:{node.lineno}", ast.dump(body)))
    rows = []
    for name, sites in sorted(defs.items()):
        if len(sites) < 2 or not any(s.startswith("scripts/") for s, _ in sites):
            continue
        if name in {"main", "_parse_args", "_build_parser"}:
            continue
        rows.append(
            {
                "name": name,
                "sites": [s for s, _ in sites],
                "identical": len({d for _, d in sites}) == 1,
            }
        )
    return rows


def scripts_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the facts about ``scripts/`` already recorded in the worklist."""
    facts: list[dict[str, Any]] = []
    py_rows = [r for r in rows if r["file"].endswith(".py")]
    total = sum(r["lines"] for r in py_rows)
    facts.append(
        {
            "claim": "scripts/*.py holds 6,486 lines",
            "confirmed": total == 6486,
            "evidence": [f"measured {total} lines over {len(py_rows)} files"],
        }
    )
    pyproject = tomllib.loads(read(ROOT / "pyproject.toml"))
    source = pyproject["tool"]["coverage"]["run"]["source"]
    pythonpath = pyproject["tool"]["pytest"]["ini_options"]["pythonpath"]
    has_init = (ROOT / "scripts/__init__.py").exists()
    facts.append(
        {
            "claim": "coverage measures nfl_predictor only; scripts/ is not a package and imports "
            "only through pytest's pythonpath",
            "confirmed": source == ["nfl_predictor"] and not has_init and "." in pythonpath,
            "evidence": [
                f"coverage source={source}",
                f"scripts/__init__.py exists={has_init}",
                f"pytest pythonpath={pythonpath}",
            ],
        }
    )
    weekly = next(r for r in rows if r["file"] == "scripts/weekly_run.py")
    used = {
        name
        for r in rows
        for f, names in r["imported_by_code"].items()
        if f == "scripts/weekly_run.py"
        for name in names
    }
    want = {"compute_power_rankings", "_write_outputs", "build_betting_report"}
    facts.append(
        {
            "claim": "weekly_run imports betting_pipeline and power_rankings and calls "
            "compute_power_rankings, _write_outputs, build_betting_report",
            "confirmed": set(weekly["imports_scripts"])
            == {"scripts.betting_pipeline", "scripts.power_rankings"}
            and want <= used,
            "evidence": [f"imports: {weekly['imports_scripts']}", f"names used: {sorted(used)}"],
        }
    )
    test_modules = sorted({t for r in rows for t in r["imported_by_tests"]})
    facts.append(
        {
            "claim": "ten test modules import from scripts",
            "confirmed": len(test_modules) == 10,
            "evidence": [f"{len(test_modules)}: {test_modules}"],
        }
    )
    matrices: dict[str, Any] = {}
    for f in ("scripts/weekly_run.py", "scripts/betting_pipeline.py"):
        tree = ast.parse(read(ROOT / f))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.List) or not node.elts:
                continue
            first = node.elts[0]
            if not (
                isinstance(first, ast.Tuple)
                and isinstance(first.elts[0], ast.Constant)
                and str(first.elts[0].value).endswith("_base")
            ):
                continue
            with contextlib.suppress(ValueError):
                where = _enclosing_function(node, parents)
                owner = parents.get(node)
                if isinstance(owner, ast.Assign | ast.AnnAssign):
                    target = owner.targets[0] if isinstance(owner, ast.Assign) else owner.target
                    where = ast.unparse(target)
                matrices[f"{f}:{where}"] = ast.literal_eval(node)
    facts.append(
        {
            "claim": "betting_pipeline keeps its own copy of the walk-forward candidate matrix",
            "confirmed": sum(1 for k in matrices if "betting_pipeline" in k) >= 1,
            "evidence": [f"{k}: {len(v)} rows: {[r[0] for r in v]}" for k, v in matrices.items()]
            + [f"identical: {len({json.dumps(v) for v in matrices.values()}) == 1}"],
        }
    )
    web_scripts = sorted(r["file"] for r in rows if r["web_templates"])
    facts.append(
        {
            "claim": "the web API launches eight scripts by file path",
            "confirmed": len(web_scripts) == 8,
            "evidence": [f"{len(web_scripts)}: {web_scripts}"],
        }
    )
    return facts


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


def tuning_only_fields(field_reads: dict[str, dict[str, list[dict[str, Any]]]]) -> set[str]:
    """Return OptunaConfig fields that no package code reads outside tuning."""
    return {
        name
        for name, sites in field_reads["OptunaConfig"].items()
        if sites and all(s["tuning"] for s in sites if s["site"].startswith("nfl_predictor/"))
    }


def record_only_wf_fields(field_reads: dict[str, dict[str, list[dict[str, Any]]]]) -> set[str]:
    """Return WalkForwardConfig fields that package code reads only in ``to_dict``."""
    return {
        name
        for name, sites in field_reads["WalkForwardConfig"].items()
        if {x["function"] for x in sites if x["site"].startswith("nfl_predictor/")} == {"to_dict"}
    }


def classify(
    flag: Flag, ep_scan: ReadScan, tuning_fields: set[str], wf_record_fields: set[str]
) -> None:
    """Apply the mechanical classification and tags."""
    longs = [o for o in flag.option_strings if o.startswith("--")]
    if len(longs) > 1 and flag.action != "BooleanOptionalAction":
        flag.tags.append("alias")
    if flag.weekly_yaml is not None:
        flag.tags.append("weekly-config")
    if flag.web_templates:
        flag.tags.append("web")
    if flag.consumers.get("launchers"):
        flag.tags.append("launch.sh")
    behavioral = [r for r in flag.reads if not r["record_only"]]
    final_sinks = [
        h for r in behavioral for h in (r.get("hops") or [r["sink"]]) if not RECORD_SINK.search(h)
    ]
    tuning_sinks = [TUNING_SINK.search(sink) for sink in final_sinks]
    wf_sinks = [WF_CONFIG_SINK.search(sink) for sink in final_sinks]
    if final_sinks and all(m and m.group(1) in wf_record_fields for m in wf_sinks):
        flag.auto_class = "record-only (WalkForwardConfig field read only by to_dict)"
    elif final_sinks and all(m and m.group(1) in tuning_fields for m in tuning_sinks):
        flag.auto_class = "tuning-only (reaches tuning-only OptunaConfig fields)"
    elif behavioral:
        flag.auto_class = "active (read)"
    elif flag.reads:
        flag.auto_class = "record-only (metadata, fingerprint or log)"
    elif flag.writes:
        flag.auto_class = "written, never read"
    elif ep_scan.dynamic:
        flag.auto_class = "no direct read (dynamic getattr present)"
    else:
        flag.auto_class = "inert (no read site)"


def main() -> int:
    """Generate the inventory files."""
    entrypoints = discover_entrypoints()
    yaml_cfg = weekly_yaml()
    templates = web_templates()
    flags: list[Flag] = []
    scans: dict[str, ReadScan] = {}
    parsers: dict[str, argparse.ArgumentParser] = {}
    for ep in entrypoints:
        with contextlib.redirect_stderr(io.StringIO()):
            parser = capture_parser(ep)
        parsers[ep.name] = parser
        scan = scan_reads(ep.path)
        scans[ep.name] = scan
        for flag in walk_actions(ep, parser):
            flag.reads = scan.reads.get(flag.dest, [])
            flag.writes = scan.writes.get(flag.dest, [])
            if ep.name == "weekly_run" and flag.dest in yaml_cfg:
                flag.weekly_yaml = yaml_cfg[flag.dest]
            for template in templates:
                if template["target"] not in ep.invocations and (
                    template["target"] != f"-m {ep.module}"
                ):
                    continue
                if set(flag.option_strings) & set(template["flags"]) or (
                    flag.dest in template["config_keys"]
                ):
                    flag.web_templates.append(template["id"])
            flags.append(flag)
        if not [f for f in flags if f.entrypoint == ep.name]:
            fail("parser capture", f"{ep.name}: parser has no actions")

    known = {(f.entrypoint, o) for f in flags for o in f.option_strings}
    template_parse: list[dict[str, Any]] = []
    for template in templates:
        target = next((e for e in entrypoints if template["target"] in e.invocations), None)
        if target is None:
            fail("web template", f"{template['id']} launches unknown {template['target']}")
            continue
        runs = [{"param": "(sample)", "value": "", "argv": template["argv"]}]
        runs += template["variants"]
        for variant in runs:
            argv = variant["argv"]
            start = (
                argv.index(template["target"].split()[-1]) + 1
                if template["target"].startswith("-m ")
                else 2
            )
            tail = argv[start:]
            if target.name == "weekly_run":
                unknown_keys = sorted(
                    set(template["config_keys"])
                    - {f.dest for f in flags if f.entrypoint == "weekly_run"}
                )
                ok, message = not unknown_keys, f"unknown config keys {unknown_keys}"
            else:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    try:
                        parsers[target.name].parse_args(tail)
                        ok, message = True, ""
                    except SystemExit:
                        ok, message = False, err.getvalue().strip().splitlines()[-1]
            template_parse.append(
                {
                    "template": template["id"],
                    "param": variant["param"],
                    "value": variant["value"],
                    "ok": ok,
                    "error": message,
                }
            )
    by_ep = {ep.name: ep for ep in entrypoints}
    for key in yaml_cfg:
        if not any(f.entrypoint == "weekly_run" and f.dest == key for f in flags):
            fail("weekly config", f"config/weekly_run.yaml key {key!r} is not a weekly_run dest")
    for template in templates:
        for option in template["flags"]:
            target_eps = [e.name for e in entrypoints if template["target"] in e.invocations]
            if target_eps and (target_eps[0], option) not in known:
                fail("web template", f"{template['id']} passes unknown {option}")

    text_sources = {
        "tests": [p for p in tracked_files("*.py") if rel(p).startswith("tests/")],
        "ci_gate": [ROOT / "scripts/gate.sh", *sorted((ROOT / ".github/workflows").glob("*"))],
        "docs": [p for p in tracked_files("*.md") if not rel(p).startswith(".agents/m60/")],
        "launchers": sorted((ROOT / "models").rglob("*.sh")),
    }
    unknown_in_commands: list[str] = []
    consumer_hits: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for kind in ("ci_gate", "docs", "launchers"):
        for path in text_sources[kind]:
            for cmd in commands_in_text(path, entrypoints):
                for option in cmd["flags"]:
                    if (cmd["entrypoint"], option) in known:
                        consumer_hits[(cmd["entrypoint"], option)][kind].append(
                            f"{rel(path)}:{cmd['line']}"
                        )
                    else:
                        unknown_in_commands.append(
                            f"{rel(path)}:{cmd['line']} {cmd['entrypoint']} {option}"
                        )
    for path in text_sources["tests"]:
        mods = python_imports(path)
        strings = string_mentions(path)
        linked = [
            ep.name
            for ep in entrypoints
            if ep.module in mods
            or any(ep.module.startswith(m + ".") or m.startswith(ep.module) for m in mods)
            or any(inv.split()[-1] in s for s in strings for inv in ep.invocations)
        ]
        literals = argv_literals([path])
        for option, sites in literals.items():
            for ep_name in linked:
                if (ep_name, option) in known:
                    consumer_hits[(ep_name, option)]["tests"].extend(sites)
    field_reads = config_field_reads()
    tuning_fields = tuning_only_fields(field_reads)
    wf_record_fields = record_only_wf_fields(field_reads)
    code_files = [p for p in tracked_files("*.py") if not rel(p).startswith("tests/")]
    code_literals = argv_literals(code_files)
    linked_files: dict[str, set[str]] = {}
    for ep in entrypoints:
        needle = ep.path.name if ep.path.parts[-2] == "scripts" else ep.invocations[0][3:]
        linked_files[ep.name] = {
            rel(p)
            for p in code_files
            if p != ep.path
            and (any(needle in v for v in string_mentions(p)) or ep.module in python_imports(p))
        }
    for flag in flags:
        hits: dict[str, list[str]] = {}
        for option in flag.option_strings:
            for kind, sites in consumer_hits.get((flag.entrypoint, option), {}).items():
                hits.setdefault(kind, [])
                hits[kind] = sorted(set(hits[kind]) | set(sites))
            own = rel(by_ep[flag.entrypoint].path)
            external = [
                s
                for s in code_literals.get(option, [])
                if not s.startswith(own + ":") and s.split(":")[0] in linked_files[flag.entrypoint]
            ]
            if external:
                hits.setdefault("argv_builders", [])
                hits["argv_builders"] = sorted(set(hits["argv_builders"]) | set(external))
        flag.consumers = hits
        classify(flag, scans[flag.entrypoint], tuning_fields, wf_record_fields)

    annotations = yaml.safe_load(read(ANNOTATIONS)) if ANNOTATIONS.exists() else {}
    for note in annotations.get("flags", []):
        flag = _flag(flags, note["entrypoint"], note["flag"])
        if flag is None:
            fail("annotation", f"{note['entrypoint']} {note['flag']} does not exist")
            continue
        ok, lines = check_evidence(note.get("evidence", []))
        flag.annotation = {**note, "holds": ok, "evidence_lines": lines}
        if not ok:
            fail("annotation", f"{note['entrypoint']} {note['flag']}: evidence no longer holds")

    concept_rows = []
    for concept in annotations.get("concepts", []):
        spellings = []
        for option in concept["options"]:
            where = sorted({f.entrypoint for f in flags if option in f.option_strings})
            if not where:
                fail("concept", f"{concept['name']}: {option} exists in no entrypoint")
            spellings.append({"option": option, "entrypoints": where})
        concept_rows.append(
            {"name": concept["name"], "spellings": spellings, "note": concept.get("note", "")}
        )

    by_dest: dict[str, list[Flag]] = defaultdict(list)
    for flag in flags:
        if flag.subcommand is None:
            by_dest[flag.dest].append(flag)
    shared = {
        dest: group
        for dest, group in sorted(by_dest.items())
        if len({f.entrypoint for f in group}) > 1
    }

    checks = fixed_checks(flags, yaml_cfg)
    recency = recency_weeks_usage()
    checks.append(
        {
            "claim": "`--recency-half-life-weeks` has never been measured (no launcher passes it "
            "and no run artifact records a non-null value)",
            "confirmed": not recency["launchers"] and not recency["nonnull_json"],
            "evidence": [
                f"launchers passing it: {recency['launchers'] or 'none'}",
                f"JSON artifacts scanned: {recency['json_scanned']}",
                f"non-null values: {recency['nonnull_json'][:10] or 'none'}",
            ],
        }
    )
    rows = scripts_inventory(entrypoints, templates)
    sfacts = scripts_facts(rows)
    follow = followups(flags)
    dupes = duplicate_functions()
    score = score_model_evidence(flags)
    dispositions = annotations.get("scripts", {})
    for row in rows:
        disp = dispositions.get(row["file"])
        if disp is None:
            fail("script disposition", f"{row['file']} has no proposed disposition")
        row["proposed"] = disp

    payload = {
        "entrypoints": [
            {
                "name": e.name,
                "module": e.module,
                "file": rel(e.path),
                "invocations": e.invocations,
                "bulk_namespace_sites": scans[e.name].bulk,
                "dynamic_getattr_sites": scans[e.name].dynamic,
            }
            for e in entrypoints
        ],
        "flags": [flag.__dict__ | {"key": flag.key} for flag in flags],
        "web_templates": templates,
        "unknown_flags_in_commands": unknown_in_commands,
        "web_template_parse": template_parse,
        "shared_dests": {d: [f.entrypoint for f in g] for d, g in shared.items()},
        "concepts": concept_rows,
        "checks": checks,
        "scripts": rows,
        "scripts_facts": sfacts,
        "followups": follow,
        "duplicate_functions": dupes,
        "score_model": score,
        "config_field_reads": field_reads,
        "failures": [f.__dict__ for f in FAILURES],
    }
    OUT_JSON.write_text(
        _portable(json.dumps(payload, indent=1, default=str)) + "\n", encoding="utf-8"
    )
    OUT_MD.write_text(_portable(render(payload, flags, entrypoints, shared)), encoding="utf-8")
    print(
        f"{len(entrypoints)} entrypoints, {len(flags)} flags, {len(rows)} scripts files, "
        f"{len(FAILURES)} failures"
    )
    for failure in FAILURES:
        print(f"FAILURE {failure.what}: {failure.detail}")
    return 1 if FAILURES else 0


def _portable(text: str) -> str:
    """Replace machine-specific paths so the output is identical across runs and machines."""
    text = re.sub(re.escape(tempfile.gettempdir()) + r"/inventory_[\w-]+", "<tmp>", text)
    text = re.sub(r"\b(train)_\d{8}_\d{6}\b", r"\1_<timestamp>", text)
    return text.replace(str(ROOT), "<repo>")


def _cell(value: Any) -> str:
    """Render a table cell without breaking the Markdown table."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render(
    payload: dict[str, Any],
    flags: list[Flag],
    entrypoints: list[Entrypoint],
    shared: dict[str, list[Flag]],
) -> str:
    """Render the Markdown inventory."""
    out: list[str] = []
    w = out.append
    w("<!-- markdownlint-disable -->")
    w("<!-- Generated file: wide evidence tables and verbatim code cells; lint the generator. -->")
    w("")
    w("# CLI and scripts inventory (generated)")
    w("")
    w("Generated by `.venv/bin/python .agents/m60/inventory.py`; do not edit by hand. The full")
    w("data, including every read site, is in `inventory.json` beside this file. Hand judgments")
    w("come from `annotations.yaml` and are shown only with their re-checked evidence.")
    w("")
    w("## Summary")
    w("")
    counts: dict[str, int] = defaultdict(int)
    for flag in flags:
        counts[flag.auto_class] += 1
    w(f"- Entrypoints: {len(entrypoints)}; flags (parser actions, help excluded): {len(flags)}.")
    for name, count in sorted(counts.items()):
        w(f"- Mechanical class `{name}`: {count}.")
    judged = [f for f in flags if f.annotation]
    holding = sum(1 for f in judged if f.annotation and f.annotation["holds"])
    w(f"- Flags with a hand judgment: {len(judged)} ({holding} with evidence holding).")
    w(f"- Failures: {len(payload['failures'])}.")
    for failure in payload["failures"]:
        w(f"  - {failure['what']}: {failure['detail']}")
    w("")
    w("Mechanical classes: `active (read)` means the entrypoint reads the value at least once")
    w("outside a bulk `vars(args)` dump; it does not prove the value changes output. The")
    w("per-flag read sites and their sinks are the evidence for each class; hand judgments")
    w("(inert through a chain, tuning-only, duplicate, policy drift) are listed separately.")
    w("")

    w("## Entrypoints")
    w("")
    w("| entrypoint | file | invoked as | flags | bulk `vars(args)` sites | dynamic getattr |")
    w("| --- | --- | --- | --- | --- | --- |")
    for ep in payload["entrypoints"]:
        n = sum(1 for f in flags if f.entrypoint == ep["name"])
        bulk = ", ".join(f"{b['function']}:{b['line']}" for b in ep["bulk_namespace_sites"])
        dyn = ", ".join(f"{b['function']}:{b['line']}" for b in ep["dynamic_getattr_sites"])
        w(
            f"| {ep['name']} | `{ep['file']}` | `{' / '.join(ep['invocations'])}` | {n} | "
            f"{bulk or '-'} | {dyn or '-'} |"
        )
    w("")

    w("## Hand judgments (annotations.yaml, evidence re-checked)")
    w("")
    for flag in judged:
        note = flag.annotation or {}
        status = "holds" if note.get("holds") else "OVERTURNED"
        w(f"### {flag.entrypoint} `{flag.key}`: {note.get('class')} ({status})")
        w("")
        w(_cell(note.get("note", "")))
        w("")
        for line in note.get("evidence_lines", []):
            w(f"- {line}")
        w("")

    w("## Concepts spelled more than one way")
    w("")
    w("| concept | spelling | entrypoints |")
    w("| --- | --- | --- |")
    for concept in payload["concepts"]:
        for s in concept["spellings"]:
            w(f"| {concept['name']} | `{s['option']}` | {', '.join(s['entrypoints'])} |")
    w("")

    w("## Same dest in several entrypoints")
    w("")
    w("A dest shared by entrypoints, with each one's option strings and default. Differing")
    w("defaults are marked.")
    w("")
    w("| dest | entrypoint | options | default | differs |")
    w("| --- | --- | --- | --- | --- |")
    for dest, group in shared.items():
        defaults = {json.dumps(f.default, default=str) for f in group}
        for f in group:
            w(
                f"| `{dest}` | {f.entrypoint} | {_cell(' '.join(f.option_strings))} | "
                f"`{_cell(f.default)}` | {'yes' if len(defaults) > 1 else ''} |"
            )
    w("")

    w("## Checks of the recorded findings")
    w("")
    for check in payload["checks"] + payload["scripts_facts"]:
        verdict = "CONFIRMED" if check["confirmed"] else "OVERTURNED"
        w(f"- **{verdict}**: {check['claim']}")
        for line in check["evidence"]:
            w(f"  - {_cell(line)}")
    w("")

    w("## Follow-ups assigned to this step")
    w("")
    for item in payload["followups"]:
        verdict = ""
        if "confirmed" in item:
            verdict = " **CONFIRMED**" if item["confirmed"] else " **OVERTURNED**"
        w(f"- {item['item']}{verdict}")
        for line in item["evidence"]:
            w(f"  - {_cell(line)}")
    w("")

    w("## Config dataclass fields and where they are read")
    w("")
    w("A read counts when the value is `self` inside the class, a parameter annotated with the")
    w("class, or a local assigned from its constructor or `dataclasses.replace`. A field read")
    w("only by `to_dict` reaches metadata and the checkpoint fingerprint, nothing else. A read is")
    w("`tuning` when it sits in the Optuna search or under an `if <x>.enabled` / `tune` branch.")
    for cls, fields in payload["config_field_reads"].items():
        w("")
        w(f"### {cls}")
        w("")
        w("| field | reading functions (package) | reads outside tuning (package) | note |")
        w("| --- | --- | --- | --- |")
        for name, sites in fields.items():
            pkg = [x for x in sites if x["site"].startswith("nfl_predictor/")]
            funcs = sorted({x["function"] + "()" for x in pkg})
            outside = sorted({x["site"] for x in pkg if not x["tuning"]})
            note = ""
            if funcs == ["to_dict()"]:
                note = "**to_dict only**"
            elif pkg and not outside:
                note = "tuning only"
            elif not pkg:
                note = "**no package read**"
            w(f"| `{name}` | {_cell(', '.join(funcs))} | {len(outside)} | {note} |")
    w("")

    w("## ScoreModel and the `score` model kind")
    w("")
    score = payload["score_model"]
    w("Model-kind flags:")
    w("")
    for line in score["model_kind_flags"]:
        w(f"- {line}")
    w("")
    w(f"Model kinds recorded in `models/*/metadata.json`: `{score['run_metadata_model_kinds']}`")
    w("")
    w(
        f"Saved `models/**/model*.joblib` artifacts by class (checkpoints excluded): "
        f"`{score['saved_model_types']}`"
    )
    w("")
    w("Occurrences:")
    w("")
    for line in score["hits"]:
        w(f"- `{_cell(line)}`")
    w("")

    w("## Flags in commands that the entrypoint does not define")
    w("")
    w("Commands found in docs, CI or launchers that pass an option the named entrypoint does not")
    w("have (stale docs, or a tokenizer miss to check by hand).")
    w("")
    for line in payload["unknown_flags_in_commands"] or ["none"]:
        w(f"- `{line}`")
    w("")

    w("## Web job templates parsed by their target's real parser")
    w("")
    w("Every template's command, for every choice value and every model kind an active run can")
    w("carry, run through the target entrypoint's `parse_args` (weekly_run: config keys checked")
    w("against its dests). Rejections are listed; they are behavior defects, not move work.")
    w("")
    bad = [t for t in payload["web_template_parse"] if not t["ok"]]
    w(f"- Commands parsed: {len(payload['web_template_parse'])}; rejected: {len(bad)}.")
    for t in bad:
        w(f"  - `{t['template']}` with {t['param']}=`{t['value']}`: {_cell(t['error'])}")
    w("")

    w("## Functions defined in more than one place")
    w("")
    w("Top-level functions of `scripts/` also defined in another script or package module")
    w("(docstrings ignored when comparing bodies).")
    w("")
    w("| function | sites | bodies identical |")
    w("| --- | --- | --- |")
    for d in payload["duplicate_functions"]:
        w(f"| `{d['name']}` | {', '.join(d['sites'])} | {'yes' if d['identical'] else 'no'} |")
    w("")

    w("## Web job templates")
    w("")
    w("| template | launches | flags passed | config keys written |")
    w("| --- | --- | --- | --- |")
    for t in payload["web_templates"]:
        w(
            f"| {t['id']} | `{t['target']}` | {_cell(' '.join(t['flags']))} | "
            f"{_cell(' '.join(t['config_keys']))} |"
        )
    w("")

    w("## scripts/ files")
    w("")
    w(
        "| file | lines | imports scripts | test importers | code importers | web | CI/gate | "
        "docs (files) | launchers | proposed |"
    )
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["scripts"]:
        prop = row.get("proposed") or {}
        w(
            f"| `{row['file']}` | {row['lines']} | {_cell(', '.join(row['imports_scripts']))} | "
            f"{len(row['imported_by_tests'])} | "
            f"{_cell(', '.join(row['imported_by_code']))} | "
            f"{_cell(', '.join(row['web_templates']))} | {len(row['ci_gate'])} | "
            f"{len(row['docs'])} | {len(row['launchers'])} | {_cell(prop.get('disposition'))} |"
        )
    w("")
    for row in payload["scripts"]:
        w(f"### `{row['file']}`")
        w("")
        w(f"- Lines: {row['lines']}; `__main__` guard: {row['has_main_guard']}.")
        w(f"- Imports from scripts: {row['imports_scripts'] or 'none'}.")
        w(f"- Imported by tests: {row['imported_by_tests'] or 'none'}.")
        w(f"- Imported by code: {row['imported_by_code'] or 'none'}.")
        w(f"- Names importers use: {row['names_used_by_importers'] or 'none'}.")
        w(f"- Web templates: {row['web_templates'] or 'none'}.")
        w(f"- CI and gate: {row['ci_gate'] or 'none'}.")
        w(f"- Docs citing it: {row['docs'] or 'none'}.")
        w(f"- Python files naming its path: {row['python_path_strings'] or 'none'}.")
        w(
            f"- Package modules no other non-test code imports: "
            f"{row['exclusive_package_modules'] or 'none'}."
        )
        w(
            f"- Launchers under `models/` ({len(row['launchers'])}): "
            f"{', '.join(row['launchers']) or 'none'}."
        )
        prop = row.get("proposed") or {}
        w(f"- Proposed: **{prop.get('disposition', 'MISSING')}**. {_cell(prop.get('reason', ''))}")
        w("")

    w("## Every flag")
    w("")
    w("Consumers: `yaml` = set in `config/weekly_run.yaml`; `web` = job templates; `tests`,")
    w("`ci`, `docs`, `launch` = counts of attributed command or literal sites; `argv` = option")
    w("literals in other package or script files. Sinks list the first three read sites.")
    for ep in entrypoints:
        w("")
        w(f"### {ep.name}")
        w("")
        w(
            "| flag | dest | default | class | tags | yaml | web | tests | ci | docs | launch | "
            "argv | sinks |"
        )
        w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in (f for f in flags if f.entrypoint == ep.name):
            opts = " ".join(f.option_strings) or f.dest
            if f.subcommand:
                opts = f"{f.subcommand}: {opts}"
            cls = f.auto_class
            if f.annotation:
                cls += f"; judged {f.annotation['class']}"
            sinks = "; ".join(f"{r['line']} {r['sink']}" for r in f.reads[:3])
            if len(f.reads) > 3:
                sinks += f"; +{len(f.reads) - 3}"
            c = f.consumers
            yaml_cell = _cell(f.weekly_yaml) if f.weekly_yaml is not None else ""
            w(
                f"| `{_cell(opts)}` | {f.dest} | `{_cell(f.default)}` | {cls} | "
                f"{', '.join(f.tags)} | {yaml_cell} "
                f"| {_cell(', '.join(f.web_templates))} | {len(c.get('tests', []))} | "
                f"{len(c.get('ci_gate', []))} | {len(c.get('docs', []))} | "
                f"{len(c.get('launchers', []))} | {len(c.get('argv_builders', []))} | "
                f"{_cell(sinks)} |"
            )
    w("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
