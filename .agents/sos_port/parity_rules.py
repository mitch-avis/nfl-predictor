"""The parity check's rules: pair expressions, evidence checks and the cause adjustments.

``parity.py`` drives the check; this module holds the pieces ``parity.yaml`` names, so each named
cause has exactly one implementation here and the script can fail when the two lists drift apart.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl
from port_audit import ROOT

# ``_play_conditions`` holds the carry and dropback conditions the aggregates count with.
from nfl_predictor.utils.polars.pbp import _play_conditions

if TYPE_CHECKING:
    from collections.abc import Callable

    from port_audit import SosCode

Kind = Literal["count", "count_from_rate", "sum", "rate"]
KEYS = ["week", "team_abbr", "opponent_abbr"]
_NAME = re.compile(r"[A-Za-z0-9_]*[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Expression:
    """Arithmetic over named columns, with division that is null over a zero denominator."""

    text: str
    expr: pl.Expr
    columns: frozenset[str]


def expression(text: str) -> Expression:
    """Parse ``text`` (column names, numbers, ``+ - * /``) into a Polars expression.

    Column names may start with a digit (``2pt_conversions``), so every name is swapped for a
    placeholder before Python's parser reads the arithmetic.
    """
    names: list[str] = []

    def placeholder(match: re.Match[str]) -> str:
        if match.group(0).replace(".", "").isdigit():
            return match.group(0)
        names.append(match.group(0))
        return f"_c{len(names) - 1}"

    tree = ast.parse(_NAME.sub(placeholder, text), mode="eval")

    def build(node: ast.AST) -> pl.Expr:
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Name) and node.id.startswith("_c"):
            return pl.col(names[int(node.id[2:])]).cast(pl.Float64)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return pl.lit(float(node.value))
        if isinstance(node, ast.BinOp):
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return pl.when(right != 0).then(left / right).otherwise(None)
        msg = f"unsupported expression {text!r}"
        raise ValueError(msg)

    return Expression(text, build(tree), frozenset(names))


@dataclass(frozen=True)
class Pair:
    """One judged pair: this repository's value and the other repository's value."""

    id: str
    ours: Expression
    theirs: Expression
    kind: Kind
    causes: tuple[str, ...]
    null_as_zero: bool


def check_evidence(
    items: list[dict[str, str]], code: SosCode
) -> tuple[bool, list[dict[str, object]]]:
    """Verify each ``{repo, path, pattern, expect}`` item line by line; return (all held, rows)."""
    ok = True
    rows: list[dict[str, object]] = []
    for item in items:
        expect = item.get("expect", "present")
        if item["repo"] == "here":
            path = ROOT / item["path"]
            text = path.read_text(encoding="utf-8") if path.exists() else None
        else:
            text = code.show(item["path"]) if item["path"] in code.paths else None
        lines = [] if text is None else text.splitlines()
        hits = [n for n, line in enumerate(lines, start=1) if re.search(item["pattern"], line)]
        held = text is not None and (bool(hits) if expect == "present" else not hits)
        ok = ok and held
        rows.append(
            {
                "repo": item["repo"],
                "path": item["path"],
                "pattern": item["pattern"],
                "expect": expect,
                "lines": hits[:8],
                "held": held,
                "missing_file": text is None,
            }
        )
    return ok, rows


# ---------------------------------------------------------------------------------------------
# Cause adjustments
# ---------------------------------------------------------------------------------------------
#
# Each adjustment changes one rule in a season's play-by-play, as the ETL aggregates it, before
# this repository's own aggregates run again on it.


def _flag(name: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing."""
    return pl.col(name).cast(pl.Float64).fill_null(0) > 0


def _two_point_tries(pbp: pl.DataFrame) -> pl.DataFrame:
    """Stop treating two-point tries apart, so they count as dropbacks and carries."""
    return pbp.with_columns(pl.lit(0.0).alias("two_point_attempt"))


def _kneels_and_spikes(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a kneel-down as a designed run and a spike as a dropback, so their EPA counts.

    A kneel-down gets the run and rush-attempt flags (this repository's carries start from
    rush attempts) and a spike the dropback flag; both lose their own flag.
    """
    kneel = _flag("qb_kneel")
    return pbp.with_columns(
        pl.when(kneel).then(1.0).otherwise(pl.col("rush")).alias("rush"),
        pl.when(kneel).then(1.0).otherwise(pl.col("rush_attempt")).alias("rush_attempt"),
        pl.when(_flag("qb_spike")).then(1.0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"),
        pl.lit(0.0).alias("qb_kneel"),
        pl.lit(0.0).alias("qb_spike"),
    )


def _wiped_out_run_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a run a penalty wiped out (``rush`` 1, ``rush_attempt`` 0) as a rush attempt."""
    wiped = _flag("rush") & ~_flag("rush_attempt")
    return pbp.with_columns(
        pl.when(wiped).then(1.0).otherwise(pl.col("rush_attempt")).alias("rush_attempt")
    )


def _unflagged_rush_attempt_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Take out of carries the ones no scrimmage snap there: no run flag, and not a spike.

    The carries are this repository's (``_play_conditions``) as the earlier adjustments of a
    combination leave them, so a two-point try nflverse files as a rush attempt without the run
    flag leaves them too once ``two_point_tries`` has made it a carry. A spike filed as a rush
    attempt stays: it is a scrimmage snap there. The rush attempt is cleared.
    """
    unflagged = _play_conditions(pbp.columns)["rush"] & ~_flag("rush") & ~_flag("qb_spike")
    return pbp.with_columns(
        pl.when(unflagged).then(0.0).otherwise(pl.col("rush_attempt")).alias("rush_attempt")
    )


def _null_downs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Read a missing down as 0, so the play counts as an early down."""
    return pbp.with_columns(pl.col("down").fill_null(0.0))


def wiped_out_plays(pbp: pl.DataFrame) -> dict[str, float]:
    """Count the plays nflverse records as ``no_play`` by the flag they keep.

    ``plays`` is every such play with a team on each side; ``without_penalty_flag`` counts those
    whose ``penalty`` flag is not set. ``rush``, ``pass`` and ``qb_dropback`` count those with that
    flag still set, and ``rush_epa`` and ``pass_epa`` sum their EPA.
    """
    sides = pl.all_horizontal(
        pl.col(c).is_not_null() & (pl.col(c) != "") for c in ("posteam", "defteam")
    )
    wiped = pbp.filter(sides & (pl.col("play_type") == "no_play").fill_null(value=False))
    runs, passes = wiped.filter(_flag("rush")), wiped.filter(_flag("pass"))
    return {
        "plays": wiped.height,
        "without_penalty_flag": wiped.filter(~_flag("penalty")).height,
        "rush": runs.height,
        "pass": passes.height,
        "qb_dropback": wiped.filter(_flag("qb_dropback")).height,
        "rush_epa": float(runs["epa"].fill_null(0.0).sum()),
        "pass_epa": float(passes["epa"].fill_null(0.0).sum()),
    }


def _penalty_first_downs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count ``first_down`` on every play, first downs by penalty included."""
    return pbp.with_columns(
        pl.col("first_down").cast(pl.Float64).fill_null(0.0).alias("first_down_pass"),
        pl.lit(0.0).alias("first_down_rush"),
    )


def _fumbles_on_scrimmage_snaps(pbp: pl.DataFrame) -> pl.DataFrame:
    """Treat every play that is not a scrimmage snap by its definition as special teams."""
    run_play_scramble = _flag("qb_scramble") & (pl.col("play_type") == "run").fill_null(value=False)
    scrimmage = (
        _flag("qb_dropback")
        | run_play_scramble
        | _flag("rush")
        | _flag("qb_kneel")
        | _flag("qb_spike")
    )
    return pbp.with_columns(pl.when(scrimmage).then(0.0).otherwise(1.0).alias("special"))


@dataclass(frozen=True)
class Cause:
    """A named cause and how it adjusts one side of the comparison."""

    id: str
    side: Literal["ours", "theirs", "keys", "coverage", "identity"]
    ours: Callable[[pl.DataFrame], pl.DataFrame] | None = None


# Adjustments apply in this order when combined (``adjust``): the unflagged-carry rule reads the
# carries after the two-point, kneel-down and wiped-out-run rules have added plays to them.
CAUSES: dict[str, Cause] = {
    cause.id: cause
    for cause in (
        Cause("two_point_tries", "ours", ours=_two_point_tries),
        Cause("kneels_and_spikes", "ours", ours=_kneels_and_spikes),
        Cause("wiped_out_run_epa", "ours", ours=_wiped_out_run_epa),
        Cause("unflagged_rush_attempt_epa", "ours", ours=_unflagged_rush_attempt_epa),
        Cause("null_downs", "ours", ours=_null_downs),
        Cause("penalty_first_downs", "ours", ours=_penalty_first_downs),
        Cause("fumbles_on_scrimmage_snaps", "ours", ours=_fumbles_on_scrimmage_snaps),
        Cause("rams_written_la", "keys"),
        Cause("games_without_play_by_play", "coverage"),
        Cause("release_difference", "identity"),
    )
}


def adjust(pbp: pl.DataFrame, causes: frozenset[str]) -> pl.DataFrame:
    """Apply the play-by-play adjustments of ``causes`` to ``pbp``, in ``CAUSES`` order."""
    for cause in CAUSES.values():
        if cause.id in causes and cause.ours is not None:
            pbp = cause.ours(pbp)
    return pbp
