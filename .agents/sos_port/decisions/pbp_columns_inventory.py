"""Inventory the play-by-play columns ``nfl-sos-ratings`` reads, from its code at a fixed commit.

Run from the root of an ``nfl-predictor`` checkout, naming the ``nfl-sos-ratings`` repository
(``../nfl-sos-ratings`` from the main checkout):

    .venv/bin/python .agents/sos_port/decisions/pbp_columns_inventory.py \
        --sos-repo ../nfl-sos-ratings

It writes nothing; the report goes to stdout (``pbp_columns_inventory_output.txt`` beside this
file holds the runs ``pbp_columns.md`` quotes). The exit status is 1 when a column the port needs
is missing from ``constants.PBP_COLUMNS``, when a primary module is missing at the commit, or
when a site that ``NOT_PLAY_BY_PLAY`` cites no longer holds; 2 on a usage or download error.

Method (static: nothing from that repository is imported or executed):

1. Each Python file is read at the commit with ``git show <commit>:<path>``: the modules the
   step plan names (``PRIMARY_MODULES``) and, as a cross-check, every other module of the
   package and the analysis helpers under ``.agents/findings_2026_10_08/``.
2. ``ast`` collects every string literal with its line, its enclosing function and its role
   (``pl.col`` argument, argument of a column helper such as ``value_expr(columns, name)``,
   ``.alias`` output name, element of a named collection, membership test, other). F-strings
   are listed too, since one could build a column name at run time.
3. The literals are intersected with the play-by-play column names nflverse publishes: the
   union of the columns of the seasons named with ``--schema-season`` (downloaded into memory
   with nflreadpy; nothing is written) or of the Parquet files named with ``--schema-file``.
4. A matched name counts as a play-by-play read unless ``NOT_PLAY_BY_PLAY`` names it, with the
   frame the code reads it from and a site that shows it; each cited site is checked.
5. The columns the port needs are those reads plus ``REQUESTED`` (columns the step plan asks
   for by name). Each must be in ``constants.PBP_COLUMNS``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants

DEFAULT_COMMIT = "9081bce"
PACKAGE = "nfl_sos_ratings"
FINDINGS = ".agents/findings_2026_10_08"
# The modules the step plan names; the play-by-play parts of data_loader are among its literals.
PRIMARY_MODULES = (
    "pbp_expressions",
    "team_stats",
    "team_stats_expanded",
    "qb_stats",
    "wp_bins",
    "wp_filter",
    "team_rating",
    "qb_rating",
    "player_team_repair",
    "data_loader",
)
# Seasons whose column names define the play-by-play universe by default.
DEFAULT_SCHEMA_SEASONS = (1999, 2025)
# Functions whose argument names a column of the frame they are applied to.
COLUMN_HELPERS = frozenset(
    {
        "_blank_unless_charted",
        "_count_and_epa",
        "_flag",
        "_flag_expr",
        "_if_charted",
        "_is_charted",
        "_non_null",
        "value_expr",
    }
)
FRAME_METHODS = frozenset(
    {"drop_nulls", "get_column", "group_by", "join", "rename", "select", "sort", "unique"}
)

# Matched names the code reads from some other frame: name -> (file, function, frame), where
# ``file`` is a primary module name or a path relative to the repository root.
NOT_PLAY_BY_PLAY: dict[str, tuple[str, str, str]] = {
    "away_score": ("team_stats", "_extract_points_per_team_game", "the schedule"),
    "home_score": ("team_stats", "_extract_points_per_team_game", "the schedule"),
    "receiving_yards": (
        "team_stats",
        "<module>",
        "a key of _RECEIVING_ALIAS_SOURCES: an output alias of the team-game frame",
    ),
    "result": (
        "team_stats_expanded",
        "_aggregate_drive_stats",
        "the per-drive frame, where it aliases fixed_drive_result",
    ),
    "total": ("wp_filter", "QbWpFilter.ratings", "a local frame of fitted weights"),
    "name": (
        f"{FINDINGS}/qb_play_types.py",
        "off_row_plays",
        "an output alias (an ESPN JSON key and a palette field elsewhere)",
    ),
    "rusher": (
        f"{FINDINGS}/aborted_snaps.py",
        "classify",
        "an output alias naming the rusher's role",
    ),
    "id": (
        f"{PACKAGE}/metrics/schema.py",
        "<module>",
        "a value of the metric registry's Shape (shape='id'), not a column",
    ),
}
# Columns the step plan asks for by name, with its reason.
REQUESTED: dict[str, str] = {
    "play_id": "orders plays within a game and keys a play",
    "rusher_player_id": "the rusher, for designed quarterback runs and the botched-snap rule",
    "receiver_player_id": "the receiver",
    "fumbled_1_player_id": "the player the scorer charged with the fumble",
    "fumbled_1_team": "the fumbling team",
    "fumbled_2_player_id": "the second fumbler",
    "fumbled_2_team": "the second fumbler's team",
    "aborted_play": "the botched-snap flag",
    "desc": "the play text, parsed for the penalized team in the repaired games",
}


@dataclass(frozen=True)
class Site:
    """One string literal in one file."""

    file: str
    line: int
    function: str
    role: str
    primary: bool


@dataclass
class Scan:
    """Everything the scan found."""

    files: list[str]
    primary_files: set[str]
    sites: dict[str, list[Site]] = field(default_factory=lambda: defaultdict(list))
    fstrings: list[Site] = field(default_factory=list)


def _git(repo: Path, *args: str) -> str:
    """Run a read-only git command in ``repo`` and return its output."""
    command = ["git", "-C", str(repo), "--no-optional-locks", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _git_files(repo: Path, commit: str) -> list[str]:
    """Return the Python files of the package and the findings helpers at ``commit``."""
    listing = _git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
    return sorted(
        path
        for path in listing
        if path.endswith(".py") and path.startswith((f"{PACKAGE}/", FINDINGS))
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Map every node to its parent."""
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Return the dotted name of the functions and classes enclosing ``node``."""
    names: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def _call_name(call: ast.Call) -> str:
    """Return the called function's or method's bare name."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _assigned_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Return the name a collection literal is assigned to, or an empty string."""
    current = parents.get(node)
    while current is not None and not isinstance(current, ast.stmt):
        current = parents.get(current)
    if isinstance(current, ast.Assign) and isinstance(current.targets[0], ast.Name):
        return current.targets[0].id
    if isinstance(current, ast.AnnAssign) and isinstance(current.target, ast.Name):
        return current.target.id
    return ""


def _call_role(call: ast.Call) -> str:
    """Classify a string literal passed to ``call``."""
    name = _call_name(call)
    if name == "col":
        return "pl.col"
    if name in COLUMN_HELPERS:
        return f"helper {name}"
    if name == "alias":
        return "alias"
    if name in FRAME_METHODS:
        return f"frame.{name}"
    return f"argument of {name}"


def _role(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> str:
    """Classify how a string literal is used, from its parent node."""
    parent = parents.get(node)
    if isinstance(parent, ast.Call):
        return _call_role(parent)
    if isinstance(parent, ast.keyword):
        call = parents.get(parent)
        called = _call_name(call) if isinstance(call, ast.Call) else ""
        return f"keyword {parent.arg} of {called}"
    if isinstance(parent, ast.Compare):
        membership = any(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
        return "membership" if membership else "comparison"
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        container = parents.get(parent)
        if isinstance(container, ast.Call) and _call_name(container) in FRAME_METHODS:
            return f"frame.{_call_name(container)} list"
        owner = _assigned_name(parent, parents)
        return f"element of {owner}" if owner else "collection element"
    return "other"


def _scan_file(scan: Scan, text: str, file: str, universe: set[str]) -> None:
    """Add the file's literals that name a play-by-play column, and its f-strings, to ``scan``."""
    tree = ast.parse(text, filename=file)
    parents = _parents(tree)
    primary = file in scan.primary_files
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in universe:
                function = _function(node, parents)
                role = _role(node, parents)
                scan.sites[node.value].append(Site(file, node.lineno, function, role, primary))
        elif isinstance(node, ast.JoinedStr):
            function = _function(node, parents)
            scan.fstrings.append(Site(file, node.lineno, function, "f-string", primary))


def _universe(arguments: argparse.Namespace) -> tuple[set[str], str]:
    """Return the play-by-play column names and a description of where they came from."""
    names: set[str] = set()
    if arguments.schema_file:
        for path in arguments.schema_file:
            names |= set(pl.read_parquet_schema(path))
        return names, f"{len(arguments.schema_file)} Parquet files"
    for season in arguments.schema_season:
        names |= set(nfl.load_pbp(seasons=[season]).columns)
    seasons = ", ".join(str(season) for season in arguments.schema_season)
    return names, f"seasons {seasons} from nflreadpy"


def _module_of(file: str) -> str:
    """Return the primary module name for a primary file, else the path."""
    stem = Path(file).stem
    if file == f"{PACKAGE}/{stem}.py" and stem in PRIMARY_MODULES:
        return stem
    return file


def _cited_site_problems(scan: Scan) -> list[str]:
    """Return a problem for every ``NOT_PLAY_BY_PLAY`` entry whose cited site no longer holds."""
    problems: list[str] = []
    for name, (module, function, _frame) in NOT_PLAY_BY_PLAY.items():
        cited = [
            site
            for site in scan.sites.get(name, [])
            if _module_of(site.file) == module and site.function == function
        ]
        if not cited:
            problems.append(f"{name}: no literal in {module} {function} any more")
    return problems


def _write_sites(out: TextIO, sites: list[Site]) -> None:
    """Write one indented line per site."""
    out.writelines(
        f"    {site.file}:{site.line} {site.function} [{site.role}]\n"
        for site in sorted(sites, key=lambda s: (s.file, s.line))
    )


def _report(out: TextIO, scan: Scan, commit: str, universe: tuple[set[str], str]) -> int:
    """Write the report and return the exit status."""
    names, source = universe
    problems = _cited_site_problems(scan)
    reads = {name: found for name, found in scan.sites.items() if name not in NOT_PLAY_BY_PLAY}
    primary_reads = {name for name, found in reads.items() if any(s.primary for s in found)}
    needed = sorted(set(reads) | set(REQUESTED))
    cached = set(constants.PBP_COLUMNS)
    missing = [name for name in needed if name not in cached]
    unpublished = sorted(name for name in REQUESTED if name not in names)
    if unpublished:
        problems.append(f"requested columns nflverse does not publish: {unpublished}")

    others = len(scan.files) - len(scan.primary_files)
    out.write(f"nfl-sos-ratings at {commit}: {len(scan.files)} Python files read ")
    out.write(f"({len(scan.primary_files)} primary modules, {others} others)\n")
    out.write(f"nflverse play-by-play columns: {len(names)} ({source})\n")
    out.write(f"Matched names: {len(scan.sites)}\n")
    out.write(f"Not play-by-play reads: {len(NOT_PLAY_BY_PLAY)}\n")
    out.write(
        f"Play-by-play reads: {len(reads)} ({len(primary_reads)} in the primary modules, "
        f"{len(reads) - len(primary_reads)} only in the others)\n"
    )
    found = sum(1 for name in REQUESTED if name in reads)
    out.write(f"Requested by name: {len(REQUESTED)} ({found} also found by the scan)\n")
    out.write(
        f"Columns the port needs: {len(needed)}; in constants.PBP_COLUMNS: "
        f"{len(needed) - len(missing)}; missing: {len(missing)}\n"
    )
    out.write(f"constants.PBP_COLUMNS: {len(constants.PBP_COLUMNS)} columns\n")

    out.write("\n## Missing from constants.PBP_COLUMNS\n\n")
    out.write(", ".join(missing) + "\n" if missing else "none\n")

    out.write("\n## Play-by-play reads (file:line function [role])\n\n")
    for name in sorted(reads):
        state = "cached" if name in cached else "MISSING"
        where = "primary" if name in primary_reads else "others only"
        out.write(f"{name}: {state}, {where}\n")
        _write_sites(out, reads[name])

    out.write("\n## Requested by name\n\n")
    for name, reason in REQUESTED.items():
        state = "found by the scan" if name in reads else "not read there"
        out.write(f"{name}: {reason} ({state})\n")

    out.write("\n## Not play-by-play reads (the frame, the cited site, every site)\n\n")
    for name, (module, function, frame) in NOT_PLAY_BY_PLAY.items():
        out.write(f"{name}: {frame}; cited {module} {function}\n")
        _write_sites(out, scan.sites.get(name, []))

    out.write("\n## F-strings in the primary modules (none may build a play-by-play column)\n\n")
    primary_fstrings = [site for site in scan.fstrings if site.primary]
    out.writelines(
        f"{site.file}:{site.line} {site.function}\n"
        for site in sorted(primary_fstrings, key=lambda s: (s.file, s.line))
    )
    if not primary_fstrings:
        out.write("none\n")

    out.write("\n## constants.PBP_COLUMNS entries nfl-sos-ratings does not read\n\n")
    unread = [name for name in constants.PBP_COLUMNS if name not in reads]
    out.write(", ".join(unread) + "\n" if unread else "none\n")

    if problems:
        out.write("\n## Problems\n\n" + "\n".join(problems) + "\n")
    return 1 if missing or problems else 0


def main() -> int:
    """Run the scan, print the inventory and return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sos-repo", type=Path, required=True, help="nfl-sos-ratings checkout.")
    parser.add_argument("--commit", default=DEFAULT_COMMIT, help="Commit to read.")
    parser.add_argument(
        "--schema-season",
        type=int,
        action="append",
        help="Season whose play-by-play columns join the universe (default 1999 and 2025).",
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        action="append",
        help="Raw play-by-play Parquet file whose columns join the universe (no download).",
    )
    arguments = parser.parse_args()
    if not arguments.schema_season:
        arguments.schema_season = list(DEFAULT_SCHEMA_SEASONS)
    try:
        universe = _universe(arguments)
        files = _git_files(arguments.sos_repo, arguments.commit)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    primary_files = {f"{PACKAGE}/{module}.py" for module in PRIMARY_MODULES}
    absent = sorted(primary_files - set(files))
    if absent:
        sys.stderr.write(f"error: primary modules not found at {arguments.commit}: {absent}\n")
        return 1
    scan = Scan(files=files, primary_files=primary_files)
    for file in files:
        text = _git(arguments.sos_repo, "show", f"{arguments.commit}:{file}")
        _scan_file(scan, text, file, universe[0])
    return _report(sys.stdout, scan, arguments.commit, universe)


if __name__ == "__main__":
    sys.exit(main())
