"""Generate the inventory of `nfl-sos-ratings` (at its frozen commit) against this repository.

Run from the repository root (``PYTHONDONTWRITEBYTECODE=1`` keeps the run from leaving
``__pycache__`` beside the scripts):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/inventory.py [--data-dir DIR]

``--data-dir`` is this repository's data directory to read (default: the main checkout's
``data/``). It writes ``INVENTORY.md`` and ``inventory.json`` beside this file. Everything listed
is found mechanically; only the classifications come from ``mapping.yaml``, and each is applied
only where it resolves cleanly:

- data sources: nflreadpy load calls, network fetches, web addresses and named data files, from
  the abstract syntax tree of each repository's Python source (``inventory_scan.py``);
- output columns: the schema of every ``{season}_{kind}.parquet`` it built for 1999-2025 (and the
  two rank-range-by-week files it builds only for the season in progress), each column resolved
  through its own metric registry, imported from the frozen commit, and checked against the
  tables this repository's ETL writes to ``data/ratings/`` (``port_catalog``);
- the metric registry: every entry, read from the registry object itself;
- ratings: its rating files' columns against this repository's strength snapshot and quarterback
  columns (a subset of the column classification);
- web: its API routes and pages, and this app's, from route decorators and the route tables.

The run exits non-zero when a column, source or web item matches no rule or more than one, when a
rule matches nothing, when a counterpart named for this repository does not exist, when an item
left missing has a counterpart here (the guards), when a search that must find something finds
nothing, or when an input is missing or not at its commit.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml
from inventory_report import render
from inventory_scan import (
    Load,
    Route,
    components,
    here_api_routes,
    here_python_files,
    page_routes,
    scan_loads,
    sos_api_routes,
    sos_python_files,
    tracked,
)
from port_audit import (
    HERE,
    ROOT,
    SEASONS,
    SOS_DATA,
    Audit,
    SosCode,
    check_sos_checkout,
    parse_args,
    portable,
)
from report_text import dump_json, season_ranges
from sos_sources import classify_game_log_columns

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.pbp import PBP_TEAM_BOX_SCORE_COLUMNS, PBP_TEAM_GAME_COLUMNS
from nfl_predictor.utils.polars.qb_stats import QB_GAME_SCHEMA
from nfl_predictor.utils.polars.ratings_build import IN_PROGRESS_KINDS, PUBLISHED_KINDS
from nfl_predictor.utils.polars.season_stats_schema import SEASON_STATS_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MAPPING = HERE / "mapping.yaml"
OUT_MD = HERE / "INVENTORY.md"
OUT_JSON = HERE / "inventory.json"
CLASSES = ("matched", "different", "missing")
# The counterpart catalog of the tables this repository's ETL writes to
# ``data/ratings/{season}_{kind}.parquet`` (the ported ratings and season statistics): a reference
# ``ratings:<column>`` names a column of the table of the same kind as the column it describes.
PORT = "ratings"
# Marks a counterpart read on the opponent's row of its team-game table (see check_counterpart).
OPPONENT_ROW = "[opponent]"
# The catalogs whose rows carry a team and its opponent, the only ones ``OPPONENT_ROW`` may follow.
TEAM_GAME_CATALOGS = ("team_game", "box_score", "team_stats")
FILE_NAME = re.compile(r"^(\d{4})_(.+)\.parquet$")
SIDES = ("home_", "away_")
# A column name inside a counterpart formula: letters, digits and underscores, not only digits.
NAME = re.compile(r"[A-Za-z0-9_]*[A-Za-z_][A-Za-z0-9_]*")
# File kinds whose columns describe the opponents a team or quarterback faced.
PROFILE_KINDS = frozenset({"opponent_profiles", "qb_opponent_profiles"})
# This app's views, against its pages and entity components: the page components, those of the
# ratings pages, and the shared table they draw with.
HERE_COMPONENTS = (
    "web/src/pages/*.tsx",
    "web/src/components/ratings/*.tsx",
    "web/src/components/table/*.tsx",
)
SCAN_SCOPE = (
    "Its package `nfl_sos_ratings/` at the frozen commit, and this repository's tracked Python "
    "under `src/` and `scripts/`. Not covered: its `.agents/` analysis scripts (one reads ESPN's "
    "summary API), and data files named through f-strings or variables, such as this repository's "
    "`meta_data.csv` (`data_collection.py`, `constants.QB_META_DATA_NAME`)."
)


@dataclass
class Column:
    """One output column of one file kind, resolved through its registry."""

    kind: str
    column: str
    seasons: list[int]
    base: str
    prefix: str
    suffix: str
    registry_source: str
    entity: str
    code_source: str = ""
    rule: str = ""
    classification: str = ""
    here: list[str] = field(default_factory=list)
    phase: str = ""
    note: str = ""


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def output_columns(audit: Audit) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[int]]]:
    """Return each output kind's columns with the seasons that have them, and each kind's seasons.

    A kind with a file in any of the completed seasons is read over those seasons only; a kind it
    writes only for the season in progress (the rank ranges by week) is read where it exists.
    """
    files: dict[str, dict[int, Any]] = defaultdict(dict)
    for path in sorted(SOS_DATA.glob("*.parquet")):
        match = FILE_NAME.match(path.name)
        if match is None:
            audit.fail("output files", f"unexpected file name {path.name}")
            continue
        files[match.group(2)][int(match.group(1))] = path
    if not files:
        audit.fail("output files", f"no Parquet output under {portable(str(SOS_DATA))}")
    columns: dict[str, dict[str, list[int]]] = {}
    seasons_by_kind: dict[str, list[int]] = {}
    for kind, by_season in sorted(files.items()):
        completed = sorted(season for season in by_season if season in SEASONS)
        chosen = completed or sorted(by_season)
        seasons_by_kind[kind] = chosen
        if completed and len(completed) != len(SEASONS):
            missing = sorted(set(SEASONS) - set(completed))
            audit.fail("output files", f"{kind} has no file for {missing}")
        found: dict[str, list[int]] = defaultdict(list)
        for season in chosen:
            for name in pl.read_parquet_schema(audit.record(by_season[season])):
                found[name].append(season)
        columns[kind] = dict(found)
    return columns, seasons_by_kind


def load_registry(code: SosCode) -> Any:
    """Import its metric registry from an extracted copy of the frozen commit."""
    root = code.extract(
        [
            "nfl_sos_ratings/__init__.py",
            "nfl_sos_ratings/metrics",
            "nfl_sos_ratings/rating_ranges.py",
        ]
    )
    sys.path.insert(0, str(root))
    try:
        from nfl_sos_ratings.metrics import get_registry  # extracted at run time

        return get_registry()
    finally:
        sys.path.remove(str(root))


def affixes(registry: Any, column: str, base: str) -> tuple[str, str]:
    """Return the prefix and suffix its registry resolves around ``base`` in ``column``."""
    if column == base:
        return "", ""
    for rule in registry._prefix_rules:  # the frozen registry's private resolution order
        if column.startswith(rule.prefix):
            core = column[len(rule.prefix) :]
            if registry._resolve_core(core) is not None:
                return rule.prefix, core[len(base) :] if core.startswith(base) else ""
    return "", column[len(base) :]


def here_catalog(data_dir: Path, audit: Audit) -> dict[str, list[str]]:
    """Return this repository's column and route sets a classification may name.

    ``dataset`` is the per-team name of every published game-dataset column (the header of the
    data directory's ``all_data_ml.csv``, with the ``home_``/``away_`` side prefix removed), and
    ``team_stats`` the columns of its cached weekly team stats of the last completed season.
    """
    dataset = data_dir / "all_data_ml.csv"
    team_stats = data_dir / "cache" / "nflreadpy" / f"team_stats_{SEASONS[-1]}_reg.parquet"
    header = audit.record(dataset).open(encoding="utf-8").readline().strip().split(",")
    team_level = sorted(
        {name.removeprefix(side) for name in header for side in SIDES if name.startswith(side)}
        | set(header)
    )
    catalog = {
        "team_game": list(PBP_TEAM_GAME_COLUMNS),
        "box_score": list(PBP_TEAM_BOX_SCORE_COLUMNS),
        "team_stats": list(pl.read_parquet_schema(audit.record(team_stats))),
        "qb_game": list(QB_GAME_SCHEMA),
        "dataset": team_level,
        "strength": list(constants.STRENGTH_SNAPSHOT_FILE_COLUMNS),
        "constants": sorted(
            {
                name
                for attr in dir(constants)
                if attr.isupper()
                and isinstance(values := getattr(constants, attr), list | tuple | set)
                and values
                and all(isinstance(value, str) for value in values)
                for name in values
            }
        ),
    }
    for name, values in catalog.items():
        if not values:
            audit.fail("catalog", f"this repository's {name} columns came back empty")
    return catalog


def port_catalog(data_dir: Path, audit: Audit) -> dict[str, list[str]]:
    """Return the columns of every table the ETL writes to ``data/ratings/``, by file kind.

    The season-statistics kinds come from ``SEASON_STATS_SCHEMAS``, the schema the ETL conforms
    each of their tables to. The ratings kinds have no such schema, so each is read from the data
    directory's file of the last completed season (``ratings_build.PUBLISHED_KINDS``) or of the
    season after it, the one in progress (``IN_PROGRESS_KINDS``); a missing file fails the run.
    """
    tables = {kind: list(schema) for kind, schema in SEASON_STATS_SCHEMAS.items()}
    folder = data_dir / constants.RATINGS_DIRNAME
    files = [(kind, SEASONS[-1]) for kind in PUBLISHED_KINDS]
    files += [(kind, SEASONS[-1] + 1) for kind in IN_PROGRESS_KINDS]
    for kind, season in files:
        path = folder / f"{season}_{kind}.parquet"
        if not path.is_file():
            audit.fail("port catalog", f"{portable(str(path))} does not exist")
            continue
        tables[kind] = list(pl.read_parquet_schema(audit.record(path)))
    return tables


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------


@dataclass
class Rule:
    """One classification rule from ``mapping.yaml``."""

    id: str
    spec: dict[str, Any]
    hits: int = 0

    def applies(self, kind: str) -> bool:
        """Return whether the rule covers file kind ``kind``."""
        kinds = self.spec.get("kinds", "*")
        return kinds == "*" or kind in kinds

    def matches(self, item: Column) -> bool:
        """Return whether every matcher the rule gives holds for ``item``."""
        spec = self.spec
        if not self.applies(item.kind) or spec.get("rest"):
            return False
        checks = [
            ("columns", lambda v: item.column in v),
            ("match", lambda v: re.fullmatch(v, item.column) is not None),
            ("base", lambda v: re.fullmatch(v, item.base) is not None),
            ("prefix", lambda v: item.prefix == v),
            ("suffix", lambda v: item.suffix == v),
            ("source", lambda v: re.fullmatch(v, item.code_source or item.registry_source)),
        ]
        given = [(key, check) for key, check in checks if key in spec]
        return bool(given) and all(check(spec[key]) for key, check in given)


def counterparts(rule: Rule, item: Column) -> list[str]:
    """Return the counterparts a rule names for ``item`` (templated with its names)."""
    here_map = rule.spec.get("here_map", {})
    raw = here_map.get(item.column, rule.spec.get("here", []))
    names = [raw] if isinstance(raw, str) else list(raw)
    core = item.column.removeprefix(item.prefix)
    return [n.format(column=item.column, base=item.base, core=core) for n in names]


def check_counterpart(
    ref: str, catalog: dict[str, list[str]], port: dict[str, list[str]], kind: str
) -> bool:
    """Return whether ``source:name`` names something that exists in this repository.

    A ``ratings:`` reference names columns of the ported table of ``kind``, the file kind of the
    column it describes. ``OPPONENT_ROW`` after a catalog (``box_score[opponent]:rush_attempts``)
    names the column on the opponent's row of the same team-game table, so it may follow only the
    ``TEAM_GAME_CATALOGS``.
    """
    source, _, formula = ref.partition(":")
    if source.endswith(OPPONENT_ROW):
        source = source.removesuffix(OPPONENT_ROW)
        if source not in TEAM_GAME_CATALOGS:
            return False
    names = NAME.findall(formula)
    known = port.get(kind) if source == PORT else catalog.get(source)
    return known is not None and bool(names) and all(part in known for part in names)


def port_reference(item: Column, port: dict[str, list[str]]) -> str | None:
    """Return the ported table's counterpart of ``item``: its own name in its own kind, if any."""
    return f"{PORT}:{item.column}" if item.column in port.get(item.kind, []) else None


def guard_port(items: list[Column], port: dict[str, list[str]], audit: Audit) -> int:
    """Fail every column left ``missing`` that the ported tables write under its kind and name.

    Unlike the same-name guard, this one exempts nothing: a column of the same name in the same
    kind is the same column. Returns the number of columns checked.
    """
    checked = 0
    for item in items:
        if item.classification != "missing":
            continue
        checked += 1
        if port_reference(item, port):
            audit.fail(
                "port guard",
                f"{item.kind}.{item.column} is missing, but {PORT}/{{season}}_{item.kind} has it",
            )
    return checked


def guard_missing(items: list[Column], catalog: dict[str, list[str]], audit: Audit) -> int:
    """Fail every column left ``missing`` that has a same-name counterpart in a catalog here.

    A column's own name is checked, and for a season total or per-game value (registry suffix
    ``_total`` or ``_per_game``) its base metric's too. Opponents-faced profile columns (a registry
    prefix, or the profile file kinds) are exempt: a same-name column here describes the team
    itself, not its opponents. Returns the number of columns checked.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for source, names in catalog.items():
        for name in names:
            owners[name].append(source)
    checked = 0
    for item in items:
        if item.classification != "missing" or item.prefix or item.kind in PROFILE_KINDS:
            continue
        checked += 1
        candidates = {item.column}
        if item.suffix in {"_total", "_per_game"}:
            candidates.add(item.base)
        found = sorted(f"{source}:{name}" for name in candidates for source in owners.get(name, []))
        if found:
            audit.fail(
                "missing guard", f"{item.kind}.{item.column} is missing, but here has {found}"
            )
    return checked


def _guard_names(item: Column) -> set[str]:
    """Return the names a column is known by: itself, and a total's or per-game value's base."""
    names = {item.column}
    if item.suffix in {"_total", "_per_game"}:
        names.add(item.base)
    return names


def guard_cross_kind(items: list[Column], exceptions: dict[str, str], audit: Audit) -> int:
    """Fail a column left ``missing`` that another file kind gives a counterpart.

    A column, or the base metric of a total or per-game column, that is ``matched`` or
    ``different`` in one kind cannot be ``missing`` in another, unless ``exceptions`` (keyed
    ``kind.column``) says why; an exception nothing needs fails too. Opponents-faced profile
    columns are exempt, as in ``guard_missing``. Returns the number of columns checked.
    """
    classified: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if (
            item.classification in {"matched", "different"}
            and not item.prefix
            and item.kind not in PROFILE_KINDS
        ):
            for name in _guard_names(item):
                classified[name].add(item.kind)
    checked, used = 0, set()
    for item in items:
        if item.classification != "missing" or item.prefix or item.kind in PROFILE_KINDS:
            continue
        checked += 1
        others = sorted(
            {kind for name in _guard_names(item) for kind in classified[name] if kind != item.kind}
        )
        key = f"{item.kind}.{item.column}"
        if not others:
            continue
        if key in exceptions:
            used.add(key)
            continue
        audit.fail("cross-kind guard", f"{key} is missing, but has a counterpart in {others}")
    for key in sorted(set(exceptions) - used):
        audit.fail("cross-kind guard", f"exception {key} is not needed")
    return checked


def resolve_mirrors(items: list[Column], mirrored: list[tuple[Column, Rule]], audit: Audit) -> None:
    """Copy each mirrored column's counterparts from the column its rule mirrors.

    The mirrored column must exist in the rule's ``mirror_kind``, be ``matched`` or ``different``
    there, and not be a mirror itself; otherwise the run fails. The mirrored column's ported
    table counterpart is not copied: the column takes its own (``add_port_counterparts``).
    """
    by_key = {(item.kind, item.column): item for item in items}
    mirrors = {(item.kind, item.column) for item, _ in mirrored}
    for item, rule in mirrored:
        name = item.base if rule.spec.get("mirror_base") else item.column
        key = (rule.spec["mirror_kind"], name)
        source = by_key.get(key)
        copied = [] if source is None else [r for r in source.here if not r.startswith(f"{PORT}:")]
        if source is None or source.classification not in {"matched", "different"}:
            problem = "which has no counterpart"
        elif key in mirrors:
            problem = "which is itself a mirror"
        elif not copied:
            problem = "which has no counterpart but the ported table"
        else:
            item.here = copied
            continue
        audit.fail(
            "column rules",
            f"{item.kind}.{item.column}: rule {rule.id} mirrors {key[0]}.{name}, {problem}",
        )


def add_port_counterparts(items: list[Column], port: dict[str, list[str]]) -> None:
    """Name each matched or different column's ported table counterpart first, once."""
    for item in items:
        ref = port_reference(item, port)
        if ref and item.classification in {"matched", "different"}:
            item.here = [ref, *(other for other in item.here if other != ref)]


def classify(
    items: list[Column],
    rules: list[Rule],
    catalog: dict[str, list[str]],
    port: dict[str, list[str]],
    audit: Audit,
) -> None:
    """Apply the rules: one explicit rule per column, else its kind's one rest rule.

    A rule with ``mirror_kind`` takes its counterparts from the same column's classification in
    that kind (with ``mirror_base``, from its base metric's column there), so a season value names
    what its game-log column names; mirrors resolve after every other rule. Every matched or
    different column the ported tables write (``port``) then names that table's column first.
    """
    mirrored: list[tuple[Column, Rule]] = []
    for item in items:
        explicit = [rule for rule in rules if rule.matches(item)]
        if len(explicit) > 1:
            names = ", ".join(rule.id for rule in explicit)
            audit.fail("column rules", f"{item.kind}.{item.column} matches {names}")
            continue
        chosen = explicit or [r for r in rules if r.spec.get("rest") and r.applies(item.kind)]
        if len(chosen) != 1:
            what = "no rule" if not chosen else "several rest rules"
            audit.fail("column rules", f"{item.kind}.{item.column} matches {what}")
            continue
        rule = chosen[0]
        rule.hits += 1
        item.rule = rule.id
        item.classification = rule.spec["class"]
        item.phase = str(rule.spec.get("phase", ""))
        item.note = str(rule.spec.get("note", ""))
        if "mirror_kind" in rule.spec:
            mirrored.append((item, rule))
            continue
        item.here = counterparts(rule, item)
        for ref in item.here:
            if not check_counterpart(ref, catalog, port, item.kind):
                audit.fail("counterpart", f"{item.kind}.{item.column}: {ref} does not exist here")
    resolve_mirrors(items, mirrored, audit)
    add_port_counterparts(items, port)
    for rule in rules:
        if not rule.hits:
            audit.fail("column rules", f"rule {rule.id} matches nothing")
        elif "expect" in rule.spec and rule.hits != rule.spec["expect"]:
            audit.fail(
                "column rules",
                f"rule {rule.id} takes {rule.hits} columns, "
                f"not the {rule.spec['expect']} it expects",
            )


def validate_rules(specs: list[dict[str, Any]], audit: Audit, section: str) -> list[Rule]:
    """Check each rule's class, counterpart and phase fields; return the rules."""
    rules = []
    seen: set[str] = set()
    for spec in specs:
        rule_id = spec.get("id", "?")
        if rule_id in seen:
            audit.fail(section, f"rule id {rule_id} is used twice")
        seen.add(rule_id)
        cls = spec.get("class")
        if cls not in CLASSES:
            audit.fail(section, f"{rule_id}: class {cls!r} is not one of {CLASSES}")
        has_here = "here" in spec or "here_map" in spec or "mirror_kind" in spec
        if cls in {"matched", "different"} and not has_here:
            audit.fail(section, f"{rule_id}: a {cls} rule names no counterpart here")
        if cls == "missing" and not spec.get("phase"):
            audit.fail(section, f"{rule_id}: a missing rule names no phase")
        if cls == "missing" and has_here:
            audit.fail(section, f"{rule_id}: a missing rule names a counterpart")
        if spec.get("rest") and not isinstance(spec.get("expect"), int):
            audit.fail(section, f"{rule_id}: a rest rule must say how many columns it takes")
        rules.append(Rule(rule_id, spec))
    return rules


def classify_simple(
    entries: list[tuple[str, Any]],
    specs: list[dict[str, Any]],
    known: set[str],
    audit: Audit,
    section: str,
) -> list[dict[str, Any]]:
    """Classify keyed entries (sources, web items) with ``match`` rules over their keys.

    ``known`` holds this repository's items as a counterpart names them; ``{key}`` in a rule's
    counterpart is the entry's own key.
    """
    rules = validate_rules(specs, audit, section)
    rows = []
    for key, entry in entries:
        hits = [rule for rule in rules if re.fullmatch(rule.spec["match"], key)]
        if len(hits) != 1:
            names = ", ".join(rule.id for rule in hits) or "no rule"
            audit.fail(section, f"{key} matches {names}")
            continue
        rule = hits[0]
        rule.hits += 1
        here = rule.spec.get("here", [])
        here = [ref.replace("{key}", key) for ref in ([here] if isinstance(here, str) else here)]
        for ref in here:
            if ref not in known:
                audit.fail(section, f"{key}: {ref} does not exist here")
        rows.append(
            {
                "key": key,
                "entry": entry,
                "rule": rule.id,
                "class": rule.spec["class"],
                "here": here,
                "phase": rule.spec.get("phase", ""),
                "note": rule.spec.get("note", ""),
            }
        )
    for rule in rules:
        if not rule.hits:
            audit.fail(section, f"rule {rule.id} matches nothing")
    return rows


def guard_simple(
    rows: list[dict[str, Any]], known: set[str], audit: Audit, section: str, ref_prefix: str = ""
) -> None:
    """Fail an entry left ``missing`` that this repository has under its own key.

    ``known`` holds this repository's items as a counterpart names them: ``ref_prefix`` and the
    key. The same-name guard does the same for output columns.
    """
    for row in rows:
        ref = f"{ref_prefix}{row['key']}"
        if row["class"] == "missing" and ref in known:
            audit.fail(section, f"{row['key']} is missing, but here has {ref}")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def _load_rows(loads: list[Load]) -> dict[str, list[dict[str, Any]]]:
    """Group load sites by source."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for load in loads:
        grouped[load.source].append(
            {"path": load.path, "line": load.line, "function": load.function}
        )
    return dict(sorted(grouped.items()))


def _route_key(route: Route) -> str:
    """Return a route's classification key: ``api GET /path``, ``page /path``, ``component X``."""
    if route.kind == "api":
        return f"api {route.method} {route.path}"
    return f"{route.kind} {route.path}"


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the inventory files, removing the extracted copy of its code whatever happens."""
    args = parse_args((__doc__ or "").splitlines()[0], argv)
    code = SosCode()
    try:
        return _run(code, args.data_dir)
    finally:
        code.cleanup()


def _run(code: SosCode, data_dir: Path) -> int:
    """Generate the inventory files from ``code``, its frozen commit, and ``data_dir``."""
    audit = Audit()
    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    checkout = check_sos_checkout(audit)

    # Data sources.
    sos_loads = scan_loads("sos", sos_python_files(code))
    here_loads = scan_loads("here", here_python_files())
    for repo, loads in (("nfl-sos-ratings", sos_loads), ("this repository", here_loads)):
        if not any(load.source.startswith("nflreadpy.") for load in loads):
            audit.fail("sources", f"found no nflreadpy load in {repo}")
    sos_sources = _load_rows(sos_loads)
    here_sources = _load_rows(here_loads)
    here_loads_known = {f"load:{source}" for source in here_sources}
    source_rows = classify_simple(
        list(sos_sources.items()), mapping["sources"], here_loads_known, audit, "source rules"
    )
    guard_simple(source_rows, here_loads_known, audit, "source rules", "load:")

    # Output columns and the registry.
    columns, seasons_by_kind = output_columns(audit)
    registry = load_registry(code)
    game_log_names = list(columns.get("team_game_logs", {}))
    if not game_log_names:
        audit.fail("output files", "no team_game_logs columns found")
    sources, source_facts = classify_game_log_columns(code, game_log_names, audit)
    items: list[Column] = []
    for kind, found in columns.items():
        for name, seasons in found.items():
            resolved = registry.resolve_column(name)
            if resolved is None:
                audit.fail("registry", f"{kind}.{name} does not resolve in its registry")
                continue
            prefix, suffix = affixes(registry, name, resolved.base.name)
            items.append(
                Column(
                    kind=kind,
                    column=name,
                    seasons=seasons,
                    base=resolved.base.name,
                    prefix=prefix,
                    suffix=suffix,
                    registry_source=resolved.base.source,
                    entity=resolved.base.entity,
                    code_source=sources[name].source
                    if kind == "team_game_logs" and name in sources
                    else "",
                )
            )
    catalog = here_catalog(data_dir, audit)
    port = port_catalog(data_dir, audit)
    rules = validate_rules(mapping["columns"], audit, "column rules")
    classify(items, rules, catalog, port, audit)
    guarded = guard_missing(items, catalog, audit)
    port_checked = guard_port(items, port, audit)
    exceptions = mapping.get("cross_kind_exceptions") or {}
    cross_checked = guard_cross_kind(items, exceptions, audit)

    # Web pages and API endpoints.
    sos_web = [
        *sos_api_routes(code),
        *page_routes("sos", "web/src/router.tsx", code.show("web/src/router.tsx")),
        *components("sos", code.glob(r"web/src/(pages|components/entity)/[^/]+\.tsx")),
    ]
    here_web = [
        *here_api_routes(),
        *page_routes("here", "web/src/router.tsx", (ROOT / "web/src/router.tsx").read_text()),
        *components("here", [path for spec in HERE_COMPONENTS for path in tracked(spec)]),
    ]
    for repo, routes in (("nfl-sos-ratings", sos_web), ("this app", here_web)):
        for kind in ("api", "page", "component"):
            if not any(route.kind == kind for route in routes):
                audit.fail("web", f"found no {kind} in {repo}")
    here_web_known = {_route_key(route) for route in here_web}
    web_rows = classify_simple(
        [(_route_key(route), route) for route in sos_web],
        mapping["web"],
        here_web_known,
        audit,
        "web rules",
    )
    guard_simple(web_rows, here_web_known, audit, "web rules")
    ported = {
        kind: sum(f"{PORT}:{item.column}" in item.here for item in items if item.kind == kind)
        for kind in sorted({item.kind for item in items})
    }

    metrics = [
        {
            "name": metric.name,
            "entity": metric.entity,
            "category": metric.category,
            "subcategory": metric.subcategory,
            "shape": metric.shape,
            "polarity": metric.polarity,
            "source": metric.source,
            "since": metric.since,
            "denominator": metric.denominator,
            "duplicate_of": metric.duplicate_of,
            "contextual": metric.contextual,
            "percent": metric.percent,
            "output_columns": sum(1 for item in items if item.base == metric.name),
        }
        for metric in registry.metrics.values()
    ]
    payload = {
        "generated_by": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/inventory.py "
            f"--data-dir {portable(str(data_dir))}"
        ),
        "data_dir": portable(str(data_dir)),
        "sos_checkout": checkout,
        "sources": {
            "sos": sos_sources,
            "here": here_sources,
            "classified": [{k: v for k, v in row.items() if k != "entry"} for row in source_rows],
        },
        "scan_scope": SCAN_SCOPE,
        "missing_guard": {
            "columns_checked": guarded,
            "exempt": [*sorted(PROFILE_KINDS), "columns with a registry prefix"],
        },
        "cross_kind_guard": {"columns_checked": cross_checked, "exceptions": exceptions},
        "port_guard": {"columns_checked": port_checked},
        "ported_columns": ported,
        "kinds": dict(sorted(seasons_by_kind.items())),
        "columns": [item.__dict__ for item in items],
        "rules": [{"id": rule.id, "hits": rule.hits, **rule.spec} for rule in rules],
        "game_log_source_facts": source_facts,
        "registry": {
            "metrics": metrics,
            "prefix_rules": [rule.prefix for rule in registry._prefix_rules],
            "suffix_rules": [rule.suffix for rule in registry._suffix_rules],
        },
        "here_catalog": catalog,
        "port_catalog": port,
        "web": {
            "sos": [route.__dict__ for route in sos_web],
            "here": [route.__dict__ for route in here_web],
            "classified": [{k: v for k, v in row.items() if k != "entry"} for row in web_rows],
        },
        "inputs": audit.files,
        "failures": [{"what": f.what, "detail": f.detail} for f in audit.failures],
    }
    OUT_MD.write_text(portable(render(payload)), encoding="utf-8")
    write_json(payload)
    log.info("%d output columns, %d failures", len(items), len(audit.failures))
    return 1 if audit.failures else 0


COLUMN_FIELDS = ["column", "base", "prefix", "suffix", "rule", "classification", "here"]


def json_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload as ``inventory.json`` stores it, compactly enough for the size limit.

    The column table is a list of rows of ``COLUMN_FIELDS`` per file kind. A column's seasons are
    its kind's (``kinds``) unless ``partial_seasons`` lists it; its phase and note are its rule's
    (``rules``; a rule with ``mirror_kind`` copies its counterparts from the same column, or its
    base metric's, in that kind), its source label and entity its base metric's (``registry``),
    and a game-log column's code-derived source is in ``game_log_sources``. The registry's metrics
    are rows of ``metric_fields``. Catalogs rebuilt from code or recorded inputs keep only their
    sizes: the constants catalog (every module-level list, tuple or set of strings in
    ``nfl_predictor.constants``) and the ported tables' columns.
    """
    columns: dict[str, list[list[Any]]] = defaultdict(list)
    partial = {}
    for column in payload["columns"]:
        columns[column["kind"]].append([column[field] for field in COLUMN_FIELDS])
        if column["seasons"] != payload["kinds"][column["kind"]]:
            partial[f"{column['kind']}.{column['column']}"] = season_ranges(column["seasons"])
    metrics = payload["registry"]["metrics"]
    fields = list(metrics[0]) if metrics else []
    catalog = payload["here_catalog"]
    return {
        **payload,
        "here_catalog": {**catalog, "constants": f"{len(catalog['constants'])} names"},
        "port_catalog": {
            kind: f"{len(names)} columns" for kind, names in payload["port_catalog"].items()
        },
        "registry": {
            **payload["registry"],
            "metric_fields": fields,
            "metrics": [[metric[field] for field in fields] for metric in metrics],
        },
        "game_log_sources": {
            column["column"]: column["code_source"]
            for column in payload["columns"]
            if column["kind"] == "team_game_logs"
        },
        "column_fields": COLUMN_FIELDS,
        "columns": dict(columns),
        "partial_seasons": partial,
    }


def write_json(payload: dict[str, Any]) -> None:
    """Write ``inventory.json`` (``json_body``)."""
    OUT_JSON.write_text(portable(dump_json(json_body(payload))) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
