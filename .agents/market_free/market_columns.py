"""Generate the inventory of every column derived from betting lines, from the code and a build.

Run from the repository root:

    .venv/bin/python .agents/market_free/market_columns.py \
        --data-path data/completed_games_ml.csv --nfeloqb ../nfeloqb

It writes ``MARKET_COLUMNS.md`` and ``market_columns.json`` beside this file and exits non-zero
when a check that must hold fails or a search that must find something comes back empty.
Nothing in the output is typed by hand:

- the published schema comes from ``finalize.build_final_column_order`` and the build's header;
- a column is named for the market when a ``_``-separated token of its name is a betting word
  (``BETTING_TOKENS``); the training-time transforms are ``feature_spec.MARKET_DERIVED_COLUMNS``;
- the ETL call sites are every reference to a line column name in the ETL modules' syntax
  trees, with the ``.alias(...)`` targets each referencing function writes, so a published
  column written from the lines would show up as an alias outside the five line columns;
- the empirical check correlates every numeric feature-range column of the build with the home
  spread, so a near copy of the line under another name would show up;
- the exclusion check builds the model's feature spec on the build with market features
  excluded, for both transform settings, and lists any inventory column that still enters;
- the indirect inputs are columns of a source this repo does not compute (the Elo family from
  ``data/qb_elos.csv``, produced by ``../nfeloqb``), with the evidence lines read from that
  repository (read only), never inferred.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_predictor import constants
from nfl_predictor.ml import feature_spec, walk_forward
from nfl_predictor.utils.polars.finalize import build_final_column_order

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

OUT_MD = HERE / "MARKET_COLUMNS.md"
MARKDOWN_WIDTH = 100
OUT_JSON = HERE / "market_columns.json"

# A betting word as a whole ``_``-separated token of a column name.
BETTING_TOKENS = frozenset(
    {"spread", "spreads", "moneyline", "line", "odds", "market", "implied", "vig", "over", "under"}
)
# Line column names as the ETL meets them: the published five, the nflverse schedule's own
# names before the rename, and nfelo's opening and latest lines.
LINE_NAMES = frozenset(
    {
        *constants.LINES_COLUMNS,
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "over_odds",
        "under_odds",
        "home_spread_open",
        "home_spread_last",
        "total_line_open",
        "total_line_last",
        "home_ml_open",
        "away_ml_open",
        "home_ml_last",
        "away_ml_last",
    }
)
# The ETL: everything that writes the datasets or their line columns.
ETL_SOURCES = (
    "src/nfl_predictor/data_collection.py",
    "src/nfl_predictor/lines_refresh.py",
    "src/nfl_predictor/week_builder.py",
    "src/nfl_predictor/utils",
)
# A non-line column whose absolute correlation with the home spread reaches this is reported as
# a possible copy of the line under another name.
NEAR_COPY_CORRELATION = 0.95
TOP_CORRELATIONS = 15
# Column families from the Elo file (``data/qb_elos.csv``): per side, diffs and trends.
ELO_STATS = (*constants.ELO_COLUMNS, "elo_4wk_trend", "qb_elo_4wk_trend", "qb_value_4wk_trend")
# Lines of ``../nfeloqb`` that show its Elo reads the betting market: the regression weight and
# the win-total ratings file, and the file's header.
NFELOQB_ELO = Path("nfeloqb/Resources/elo.py")
NFELOQB_EVIDENCE = re.compile(r"reg_vegas|wt_ratings|wt_rating_elo")
NFELOQB_WIN_TOTALS = Path("nfeloqb/Manual Data/wt_ratings.csv")
# The feature group that drops the Elo family, and how each of its columns carries the market.
ELO_GROUP = "elo"
MARKET_INFORMED_SOURCE = (
    "market-informed through the preseason Vegas win-total regression in ../nfeloqb "
    "Resources/elo.py"
)
# The two market-free command lines a backtest takes.
LINES_ONLY_COMMAND = (
    "nfl-predictor backtest --exclude-market --no-market-anchor --no-market-transform"
)
NO_MARKET_COMMAND = f"{LINES_ONLY_COMMAND} --disable-feature-groups {ELO_GROUP}"


class InventoryError(RuntimeError):
    """A check that must hold failed, or a search that must find something found nothing."""


def _market_named(columns: list[str]) -> list[str]:
    """Return the columns with a betting word as a whole name token."""
    return [column for column in columns if BETTING_TOKENS & set(column.split("_"))]


@dataclass(frozen=True)
class CallSite:
    """A reference to a line column name in an ETL module."""

    path: str
    line: int
    function: str
    name: str


def _etl_files() -> list[Path]:
    files: list[Path] = []
    for entry in ETL_SOURCES:
        path = ROOT / entry
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Map every line inside a function to the innermost function's qualified name."""
    owner: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                for line in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    owner[line] = name
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return owner


def _reads_lines(node: ast.AST, tainted: set[str]) -> bool:
    """Return whether an expression reads a line column, directly or through a tainted name."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value in LINE_NAMES:
            return True
        if isinstance(child, ast.Attribute) and child.attr == "LINES_COLUMNS":
            return True
        if isinstance(child, ast.Name) and child.id in tainted:
            return True
    return False


# DataFrame methods: a name assigned from one holds a whole frame, not a value computed from the
# lines, so it does not pass the lines on to the columns later read from it.
FRAME_METHODS = frozenset(
    {
        "agg",
        "cast",
        "concat",
        "drop",
        "drop_nulls",
        "filter",
        "group_by",
        "head",
        "join",
        "join_asof",
        "rename",
        "select",
        "sort",
        "tail",
        "unique",
        "with_columns",
        "with_row_index",
    }
)


def _is_frame_operation(value: ast.AST) -> bool:
    """Return whether an assigned value is a DataFrame method call (a frame, not a value)."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in FRAME_METHODS
    )


def _tainted_names(function: ast.AST) -> set[str]:
    """Return the local names a function assigns from an expression that reads the lines."""
    tainted: set[str] = set()
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
    ]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                child.id
                for target in targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            value = node.value
            if value is None or _is_frame_operation(value):
                continue
            if names - tainted and _reads_lines(value, tainted):
                tainted |= names
                changed = True
    return tainted


def _call_sites() -> tuple[list[CallSite], dict[str, list[str]]]:
    """Return every line-name reference in the ETL and the columns written from the lines.

    The second value maps each function to the ``.alias(...)`` targets whose expression reads a
    line column, directly or through a local name assigned from one.
    """
    sites: list[CallSite] = []
    written: dict[str, set[str]] = {}
    for path in _etl_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        relative = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Constant) and node.value in LINE_NAMES:
                name = str(node.value)
            elif isinstance(node, ast.Attribute) and node.attr == "LINES_COLUMNS":
                name = "LINES_COLUMNS"
            if name is not None:
                lineno = getattr(node, "lineno", 0)
                sites.append(CallSite(relative, lineno, owner.get(lineno, "<module>"), name))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            tainted = _tainted_names(function)
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "alias"
                    and node.args
                    and _reads_lines(node.func.value, tainted)
                ):
                    target = node.args[0]
                    # A name or expression target is listed as its source text, in angle
                    # brackets, so it stays visible even though it cannot be matched by name.
                    label = (
                        target.value
                        if isinstance(target, ast.Constant) and isinstance(target.value, str)
                        else f"<{ast.unparse(target)}>"
                    )
                    key = f"{relative}:{owner.get(node.lineno, function.name)}"
                    written.setdefault(key, set()).add(label)
    if not sites:
        msg = "no line column name found in the ETL modules; the search is broken"
        raise InventoryError(msg)
    if not written:
        msg = "no column written from the lines in the ETL modules; the search is broken"
        raise InventoryError(msg)
    sites.sort(key=lambda site: (site.path, site.line, site.name))
    return sites, {key: sorted(value) for key, value in sorted(written.items())}


def _published_aliases_outside_the_lines(
    aliases: dict[str, list[str]], schema: list[str]
) -> list[str]:
    """Return columns written from the lines that are published non-line columns.

    Published names are checked bare and with the side prefixes and diff suffix the final
    schema adds.
    """
    published = set(schema)
    flagged: list[str] = []
    for function, names in aliases.items():
        for name in names:
            candidates = {name, f"away_{name}", f"home_{name}", f"{name}_diff"}
            hits = sorted((candidates & published) - set(constants.LINES_COLUMNS))
            flagged.extend(f"{function}: {name} -> {hit}" for hit in hits)
    return flagged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlations(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    """Return each numeric feature-range column's absolute correlation with the home spread."""
    spread = pd.to_numeric(frame["home_spread"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        if column in constants.LINES_COLUMNS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & spread.notna()
        if valid.sum() < 2 or values[valid].nunique() < 2:
            continue
        correlation = float(np.corrcoef(values[valid], spread[valid])[0, 1])
        rows.append(
            {"column": column, "abs_corr": round(abs(correlation), 4), "games": int(valid.sum())}
        )
    rows.sort(key=lambda row: (-row["abs_corr"], row["column"]))
    return rows


# The selections the exclusion check builds a feature spec for: a label (the backtest options
# that give it), whether market features and the transform are on, and the groups disabled.
SELECTIONS: tuple[tuple[str, bool, bool, tuple[str, ...]], ...] = (
    ("--exclude-market --no-market-transform", False, False, ()),
    ("--exclude-market", False, True, ()),
    (
        "--exclude-market --no-market-transform --disable-feature-groups elo",
        False,
        False,
        (ELO_GROUP,),
    ),
    ("(default: market features, transformed)", True, True, ()),
)


def _exclusion(
    frame: pd.DataFrame, inventory: set[str], market_informed: set[str]
) -> dict[str, Any]:
    """Return which inventory and market-informed columns enter the spec under each selection."""
    result: dict[str, Any] = {}
    for label, include, transform, groups in SELECTIONS:
        dropped = walk_forward.resolve_feature_group_columns(list(frame.columns), groups)
        spec = feature_spec.build_feature_spec(
            frame.drop(columns=dropped),
            feature_spec.FeatureSelection(
                include_market=include, max_cardinality_ratio=0.5, market_transform=transform
            ),
        )
        features = set(spec.feature_columns)
        result[label] = {
            "include_market": include,
            "market_transform": transform,
            "disabled_feature_groups": list(groups),
            "feature_columns": len(spec.feature_columns),
            "inventory_columns_in_features": sorted(features & inventory),
            "market_informed_columns_in_features": sorted(features & market_informed),
        }
    return result


def _elo_columns(columns: list[str]) -> list[str]:
    stats = set(ELO_STATS)
    found = []
    for column in columns:
        base = column
        for prefix in ("away_", "home_"):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        base = base.removesuffix("_diff")
        if base in stats:
            found.append(column)
    return found


def _nfeloqb_evidence(nfeloqb: Path) -> dict[str, Any]:
    """Return the lines of ../nfeloqb that tie its Elo to the betting market (read only)."""
    elo_path = nfeloqb / NFELOQB_ELO
    evidence: dict[str, Any] = {"repository": str(nfeloqb), "elo_source": str(elo_path)}
    if not elo_path.is_file():
        evidence["status"] = "not found; the indirect-input check was not run"
        return evidence
    lines = elo_path.read_text(encoding="utf-8").splitlines()
    evidence["lines"] = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(lines, start=1)
        if NFELOQB_EVIDENCE.search(line)
    ]
    if not evidence["lines"]:
        msg = f"{elo_path} exists but names no Vegas regression; the evidence search is stale"
        raise InventoryError(msg)
    totals = nfeloqb / NFELOQB_WIN_TOTALS
    if totals.is_file():
        win_totals = pd.read_csv(totals)
        evidence["win_totals_file"] = str(totals)
        evidence["win_totals_columns"] = [
            c for c in win_totals.columns if not c.startswith("Unnamed")
        ]
        evidence["win_totals_seasons"] = [
            int(win_totals["season"].min()),
            int(win_totals["season"].max()),
        ]
        evidence["win_totals_rows"] = len(win_totals)
    evidence["status"] = "read"
    return evidence


def build_inventory(data_path: Path, nfeloqb: Path) -> dict[str, Any]:
    """Return the whole inventory as a JSON-ready dict."""
    schema = build_final_column_order()
    frame = pd.read_csv(data_path, low_memory=False)
    header = list(frame.columns)
    start = header.index(feature_spec.DEFAULT_FEATURE_START_COLUMN)
    end = header.index(feature_spec.DEFAULT_FEATURE_END_COLUMN)
    feature_range = header[start : end + 1]

    lines_in_build = [column for column in header if column in constants.LINES_COLUMNS]
    named_in_build = _market_named(header)
    named_in_schema = _market_named(schema)
    derived = list(feature_spec.MARKET_DERIVED_COLUMNS)
    inventory = sorted({*lines_in_build, *named_in_build, *derived})

    sites, aliases = _call_sites()
    flagged_aliases = _published_aliases_outside_the_lines(aliases, schema)
    completed = frame[frame["home_score"].notna()] if "home_score" in frame.columns else frame
    correlations = _correlations(completed, feature_range)
    near_copies = [row for row in correlations if row["abs_corr"] >= NEAR_COPY_CORRELATION]
    elo_columns = _elo_columns(header)
    elo_group = walk_forward.resolve_feature_group_columns(header, [ELO_GROUP])
    exclusion = _exclusion(completed, set(inventory), set(elo_columns))
    no_market_information = exclusion[SELECTIONS[2][0]]

    checks = {
        "build_lines_are_the_five_line_columns": sorted(lines_in_build)
        == sorted(constants.LINES_COLUMNS),
        "build_market_named_columns_are_the_lines": sorted(named_in_build)
        == sorted(constants.LINES_COLUMNS),
        "schema_market_named_columns_are_the_lines": sorted(named_in_schema)
        == sorted(constants.LINES_COLUMNS),
        "no_published_non_line_column_written_from_the_lines": not flagged_aliases,
        "no_non_line_column_near_a_copy_of_the_spread": not near_copies,
        "market_excluded_specs_carry_no_inventory_column": all(
            not entry["inventory_columns_in_features"]
            for entry in exclusion.values()
            if not entry["include_market"]
        ),
        "elo_group_drops_exactly_the_elo_family": sorted(elo_group) == sorted(elo_columns),
        "no_market_information_spec_carries_no_market_column": not (
            no_market_information["inventory_columns_in_features"]
            or no_market_information["market_informed_columns_in_features"]
        ),
    }
    return {
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "build_columns": len(header),
        "feature_range": {
            "start": feature_spec.DEFAULT_FEATURE_START_COLUMN,
            "end": feature_spec.DEFAULT_FEATURE_END_COLUMN,
            "columns": len(feature_range),
        },
        "completed_games": len(completed),
        "line_columns_in_build": lines_in_build,
        "market_named_columns_in_build": named_in_build,
        "market_named_columns_in_schema": named_in_schema,
        "training_time_transforms": derived,
        "inventory": inventory,
        "etl_call_sites": [site.__dict__ for site in sites],
        "etl_columns_written_from_lines": aliases,
        "published_non_line_columns_written_from_lines": flagged_aliases,
        "top_spread_correlations": correlations[:TOP_CORRELATIONS],
        "near_copies_of_the_spread": near_copies,
        "exclusion": exclusion,
        "indirect_market_inputs": {
            "elo_family_columns_in_build": elo_columns,
            "elo_group_columns": elo_group,
            "market_informed_columns": [
                {"column": column, "source": MARKET_INFORMED_SOURCE} for column in elo_columns
            ],
            "nfeloqb": _nfeloqb_evidence(nfeloqb),
        },
        "checks": checks,
    }


def _other_market_named(result: dict[str, Any]) -> int:
    """Return how many build columns are named for the market but are not line columns."""
    named = set(result["market_named_columns_in_build"])
    return len(named - set(result["line_columns_in_build"]))


def _wrapped(text: str) -> str:
    """Fill a prose line or list item to 100 columns, list items with a hanging indent."""
    if text.startswith("|") or len(text) <= MARKDOWN_WIDTH:
        return text
    stripped = text.lstrip(" ")
    indent = " " * (len(text) - len(stripped))
    hanging = indent + ("  " if stripped.startswith("- ") else "")
    return textwrap.fill(
        text,
        width=MARKDOWN_WIDTH,
        subsequent_indent=hanging,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _markdown(result: dict[str, Any]) -> str:
    out: list[str] = []

    def add(text: str) -> None:
        out.append(_wrapped(text))

    add("# Columns derived from betting lines")
    add("")
    add(
        "Generated by `.agents/market_free/market_columns.py`; do not edit by hand. Rerun it to "
        "refresh every number here."
    )
    add("")
    add(f"- Build: `{result['data_path']}` (sha256 `{result['data_sha256']}`)")
    add(
        f"- Build columns: {result['build_columns']}; feature range "
        f"`{result['feature_range']['start']}`..`{result['feature_range']['end']}`: "
        f"{result['feature_range']['columns']} columns; completed games: "
        f"{result['completed_games']}"
    )
    add(f"- Inventory (columns derived from betting lines): {len(result['inventory'])}")
    add(f"  - line columns in the build: {len(result['line_columns_in_build'])}")
    add(f"  - other build columns named for the market: {_other_market_named(result)}")
    add(f"  - training-time transforms of the lines: {len(result['training_time_transforms'])}")
    add(f"- ETL references to a line column name: {len(result['etl_call_sites'])}")
    add(
        "- Market-informed columns (the Elo family, not computed from game lines): "
        f"{len(result['indirect_market_inputs']['elo_family_columns_in_build'])}"
    )
    add("")
    add("## Checks")
    add("")
    add("| Check | Holds |")
    add("| --- | --- |")
    for name, holds in result["checks"].items():
        add(f"| {name} | {'yes' if holds else 'NO'} |")
    add("")
    add("## Inventory")
    add("")
    add("| Column | Where it comes from |")
    add("| --- | --- |")
    for column in result["inventory"]:
        source = (
            "computed at training time by `feature_spec.add_market_transforms`"
            if column in result["training_time_transforms"]
            else "a published line column (`constants.LINES_COLUMNS`)"
            if column in result["line_columns_in_build"]
            else "a published column named for the market"
        )
        add(f"| `{column}` | {source} |")
    add("")
    add("## Market-free command lines")
    add("")
    add(f"- No betting-line column (the Elo columns stay): `{LINES_ONLY_COMMAND}`")
    add(f"- No market information at all: `{NO_MARKET_COMMAND}`")
    add("")
    add("## Market selection and the feature spec on this build")
    add("")
    add("| Backtest options | Features | Line-derived among them | Market-informed among them |")
    add("| --- | --- | --- | --- |")
    for label, entry in result["exclusion"].items():
        lines = ", ".join(f"`{c}`" for c in entry["inventory_columns_in_features"]) or "none"
        informed = entry["market_informed_columns_in_features"]
        informed_text = f"{len(informed)} Elo columns" if informed else "none"
        add(f"| `{label}` | {entry['feature_columns']} | {lines} | {informed_text} |")
    add("")
    add("## Largest correlations of a non-line feature with the home spread")
    add("")
    add(
        f"Completed games; a column at or above {NEAR_COPY_CORRELATION} would be a possible copy "
        "of the line under another name."
    )
    add("")
    add("| Column | abs(corr) | Games |")
    add("| --- | --- | --- |")
    for row in result["top_spread_correlations"]:
        add(f"| `{row['column']}` | {row['abs_corr']} | {row['games']} |")
    add("")
    add("## Columns the ETL writes from the lines")
    add("")
    add(
        "Every `.alias()` target whose expression reads a line column, directly or through a "
        "local name assigned from one."
    )
    add("")
    add("| Function | Columns written |")
    add("| --- | --- |")
    for function, names in result["etl_columns_written_from_lines"].items():
        add(f"| `{function}` | {', '.join(f'`{n}`' for n in names)} |")
    add("")
    add(
        "Limits: only `.alias()` calls are traced, through local names assigned from a line "
        "expression; a column written under a variable name (the lines refresh's overlay of the "
        "five line columns, for one) shows as its source text or not at all. The correlation "
        "check above covers a copy the tracing misses."
    )
    add("")
    flagged = result["published_non_line_columns_written_from_lines"]
    add(
        "Published non-line columns among them: "
        + (", ".join(f"`{f}`" for f in flagged) if flagged else "none")
        + "."
    )
    add("")
    add("## Indirect market inputs")
    add("")
    indirect = result["indirect_market_inputs"]
    evidence = indirect["nfeloqb"]
    add(
        "The Elo family comes from `data/qb_elos.csv`, which `../nfeloqb` produces; this repo "
        "computes nothing in it from game lines. Its Elo code, read only:"
    )
    add("")
    add(f"- Source: `{evidence['elo_source']}` ({evidence['status']})")
    for line in evidence.get("lines", []):
        add(f"  - `{line}`")
    if "win_totals_file" in evidence:
        add(
            f"- `{evidence['win_totals_file']}`: seasons {evidence['win_totals_seasons'][0]}-"
            f"{evidence['win_totals_seasons'][1]}, {evidence['win_totals_rows']} rows, columns "
            + ", ".join(f"`{c}`" for c in evidence["win_totals_columns"])
        )
    add("")
    add(
        f"The `{ELO_GROUP}` feature group (`--disable-feature-groups {ELO_GROUP}`) drops "
        f"{len(indirect['elo_group_columns'])} columns of the build; the Elo family found from "
        f"the Elo stat names is {len(indirect['elo_family_columns_in_build'])} columns, and the "
        "check above says whether they are the same."
    )
    add("")
    add("| Column | How it carries the market |")
    add("| --- | --- |")
    for entry in indirect["market_informed_columns"]:
        add(f"| `{entry['column']}` | {entry['source']} |")
    add("")
    add("## ETL references to a line column name")
    add("")
    add("| File | Line | Function | Name |")
    add("| --- | --- | --- | --- |")
    for site in result["etl_call_sites"]:
        add(f"| `{site['path']}` | {site['line']} | `{site['function']}` | `{site['name']}` |")
    add("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "completed_games_ml.csv")
    parser.add_argument("--nfeloqb", type=Path, default=ROOT.parent / "nfeloqb")
    args = parser.parse_args()
    if not args.data_path.is_file():
        print(f"missing build: {args.data_path}", file=sys.stderr)
        return 2
    result = build_inventory(args.data_path.resolve(), args.nfeloqb.resolve())
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(_markdown(result), encoding="utf-8")
    failed = [name for name, holds in result["checks"].items() if not holds]
    print(
        f"inventory {len(result['inventory'])} columns; "
        f"lines {len(result['line_columns_in_build'])}, "
        f"transforms {len(result['training_time_transforms'])}, "
        f"other market-named {_other_market_named(result)}; "
        f"ETL references {len(result['etl_call_sites'])}; "
        f"Elo family {len(result['indirect_market_inputs']['elo_family_columns_in_build'])}; "
        f"failed checks: {failed or 'none'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
