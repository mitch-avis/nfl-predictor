"""Where each `nfl-sos-ratings` team game-log column comes from, read from its code.

Its loader builds every team-game row from play-by-play (``team_stats``), then overwrites some
columns with nflverse's official weekly team stats (``data_loader``), and joins scores from the
schedule and defensive stats from weekly player stats. This module reads those code paths at the
frozen commit with ``ast`` and returns one source per game-log column, so neither audit script
types a column's origin by hand.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from port_audit import Audit, SosCode

LOADER = "nfl_sos_ratings/data_loader.py"
TEAM_STATS = "nfl_sos_ratings/team_stats.py"
EXPANDED = "nfl_sos_ratings/team_stats_expanded.py"
KEY_COLUMNS = ("game_id", "week", "team", "opponent_team")
# A rate entry is a (numerator, denominator, rate) triple.
TRIPLE_LENGTH = 3

PLAY_BY_PLAY = "play-by-play"
OFFICIAL = "official weekly team stats"
SCHEDULE = "schedule"
PLAYER_STATS = "weekly player stats"
IDENTITY = "identity"


@dataclass(frozen=True)
class ColumnSource:
    """One game-log column's origin and the code that produces it."""

    source: str
    via: str


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Return the module-level function ``name``."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    msg = f"function {name} not found"
    raise LookupError(msg)


def _literal(tree: ast.Module, name: str) -> object:
    """Return the literal value assigned to the module-level name ``name``."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    msg = f"assignment {name} not found"
    raise LookupError(msg)


def _strings(value: object, name: str) -> list[str]:
    """Return ``value`` as a list of strings, or raise when it is not one."""
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    msg = f"{name} is not a sequence of strings"
    raise TypeError(msg)


def _string_triples(value: object, name: str) -> list[tuple[str, str, str]]:
    """Return ``value`` as a list of string triples, or raise when it is not one."""
    if isinstance(value, list | tuple):
        triples = [tuple(_strings(item, name)) for item in value]
        if all(len(triple) == TRIPLE_LENGTH for triple in triples):
            return [(triple[0], triple[1], triple[2]) for triple in triples]
    msg = f"{name} is not a sequence of string triples"
    raise TypeError(msg)


def _string_map(value: object, name: str) -> dict[str, str]:
    """Return ``value`` as a string-to-string mapping, or raise when it is not one."""
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        return {str(key): str(item) for key, item in value.items()}
    msg = f"{name} is not a string mapping"
    raise TypeError(msg)


def _alias_calls(node: ast.AST) -> list[tuple[str, ast.Call]]:
    """Return every ``.alias("name")`` call under ``node`` with its name."""
    return [
        (str(call.args[0].value), call)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "alias"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ]


def _col_names(node: ast.AST) -> set[str]:
    """Return the names of every ``pl.col("name")`` under ``node``."""
    return {
        call.args[0].value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "col"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }


def _coalesces_official(node: ast.AST) -> bool:
    """Return whether ``node`` holds ``pl.coalesce([pl.col("official_..."), ...])``."""
    for call in ast.walk(node):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "coalesce"
            and call.args
            and isinstance(call.args[0], ast.List)
            and call.args[0].elts
            and any(name.startswith("official_") for name in _col_names(call.args[0].elts[0]))
        ):
            return True
    return False


def _calls_in_order(function: ast.FunctionDef, names: list[str]) -> bool:
    """Return whether ``function`` calls each of ``names`` in that order."""
    positions = []
    for name in names:
        lines = [
            call.lineno
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == name
        ]
        if not lines:
            return False
        positions.append(min(lines))
    return positions == sorted(positions)


def _string_constants(tree: ast.AST) -> set[str]:
    """Return every string constant under ``tree``."""
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def classify_game_log_columns(
    code: SosCode, columns: list[str], audit: Audit
) -> tuple[dict[str, ColumnSource], dict[str, object]]:
    """Return each game-log column's source, and the facts the classification rests on."""
    loader = ast.parse(code.show(LOADER))
    stats = ast.parse(code.show(TEAM_STATS))
    expanded = ast.parse(code.show(EXPANDED))

    override = _function(loader, "_override_team_game_stats_with_official_weekly")
    official = {name for name, call in _alias_calls(override) if _coalesces_official(call)}
    points_fn = _function(stats, "_extract_points_per_team_game")
    schedule = {name for name, _ in _alias_calls(points_fn)} - set(KEY_COLUMNS)
    team_game_fn = _function(stats, "compute_team_game_stats_from_pbp")
    schedule |= {
        name
        for name, call in _alias_calls(team_game_fn)
        if _col_names(call) and _col_names(call) <= schedule
    }
    player_stats = set(
        _strings(_literal(stats, "_DEFENSE_ONLY_PLAYER_STATS"), "_DEFENSE_ONLY_PLAYER_STATS")
    )
    per_snap = {
        rate: (numerator, denominator)
        for numerator, denominator, rate in _string_triples(
            _literal(stats, "PER_SNAP_RATES"), "PER_SNAP_RATES"
        )
    }
    mirrors = _string_map(_literal(stats, "_RECEIVING_ALIAS_SOURCES"), "_RECEIVING_ALIAS_SOURCES")
    constant = {
        name
        for name, call in _alias_calls(team_game_fn)
        if isinstance(call.func, ast.Attribute)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "lit"
            for inner in ast.walk(call.func.value)
        )
    }
    pbp_names = _string_constants(expanded) | {
        name
        for function in ("compute_team_game_stats_from_pbp", "compute_team_snap_counts_from_pbp")
        for name, _ in _alias_calls(_function(stats, function))
    }

    weekly = _function(loader, "load_weekly_team_stats")
    official_after_pbp = _calls_in_order(
        weekly,
        ["compute_team_game_stats_from_pbp", "_override_team_game_stats_with_official_weekly"],
    )
    if not official_after_pbp:
        audit.fail(
            "game-log sources", "load_weekly_team_stats no longer overrides after the PBP build"
        )
    two_team_only = "two_team_games_only" in ast.unparse(
        _function(loader, "load_official_weekly_team_stats")
    )
    if not two_team_only:
        audit.fail("game-log sources", "official team stats no longer drop one-team games")
    for label, found in (
        ("official overrides", official),
        ("schedule columns", schedule),
        ("player-stat columns", player_stats),
        ("per-snap rates", per_snap),
        ("receiving mirrors", mirrors),
    ):
        if not found:
            audit.fail("game-log sources", f"found no {label} in its code")

    sources: dict[str, ColumnSource] = {}
    official_note = f"{OFFICIAL} (play-by-play where nflverse has one team's row only)"
    for column in columns:
        if column in KEY_COLUMNS or column in constant:
            sources[column] = ColumnSource(IDENTITY, "key or constant")
        elif column in schedule:
            sources[column] = ColumnSource(SCHEDULE, "team_stats._extract_points_per_team_game")
        elif column in player_stats:
            sources[column] = ColumnSource(PLAYER_STATS, "team_stats._DEFENSE_ONLY_PLAYER_STATS")
        elif column in per_snap:
            numerator, denominator = per_snap[column]
            numerator_source = (
                official_note
                if numerator in official
                else SCHEDULE
                if numerator in schedule
                else PLAYER_STATS
                if numerator in player_stats
                else PLAY_BY_PLAY
            )
            sources[column] = ColumnSource(
                f"{numerator_source} over {PLAY_BY_PLAY} {denominator}",
                "team_stats.PER_SNAP_RATES, rebuilt after the official override",
            )
        elif column in official:
            sources[column] = ColumnSource(
                official_note, "data_loader._override_team_game_stats_with_official_weekly"
            )
        elif column in mirrors:
            sources[column] = ColumnSource(
                PLAY_BY_PLAY,
                f"copy of {mirrors[column]} made before the official override "
                "(team_stats._add_receiving_display_mirrors)",
            )
        elif column in pbp_names:
            sources[column] = ColumnSource(PLAY_BY_PLAY, "team_stats / team_stats_expanded")
        else:
            audit.fail("game-log sources", f"{column}: no code path in its loader produces it")
    facts = {
        "official_overrides": sorted(official),
        "schedule_columns": sorted(schedule),
        "player_stat_columns": sorted(player_stats),
        "per_snap_rates": {rate: list(parts) for rate, parts in sorted(per_snap.items())},
        "receiving_mirrors": dict(sorted(mirrors.items())),
        "override_runs_after_play_by_play": official_after_pbp,
        "one_team_games_keep_play_by_play": two_team_only,
        "code_blobs": {path: code.blob(path) for path in (LOADER, TEAM_STATS, EXPANDED)},
    }
    return sources, facts
