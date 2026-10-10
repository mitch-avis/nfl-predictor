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

from nfl_predictor.utils.polars.pbp import SITUATIONAL_PLAY_TYPES

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


def _flag(name: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing."""
    return pl.col(name).cast(pl.Float64).fill_null(0) > 0


def scramble_flag(columns: list[str], *, enabled: bool) -> tuple[pl.Expr, str]:
    """Return the per-play scramble flag and which implementation it is.

    ``qb_scramble`` when the play-by-play carries it; otherwise nflverse's scramble coding (a rush
    attempt coded as a pass play: ``pass`` 1, ``rush`` 0, ``rush_attempt`` 1), a stand-in. With
    ``enabled`` false no play is a scramble, which is how an attribution is tested for depending
    on the stand-in.
    """
    if not enabled:
        return pl.lit(value=False), "off"
    if "qb_scramble" in columns:
        return _flag("qb_scramble"), "flag"
    return _flag("pass") & ~_flag("rush") & _flag("rush_attempt"), "stand-in"


def _run_play() -> pl.Expr:
    """Return whether the play's type is ``run``."""
    return (pl.col("play_type") == "run").fill_null(value=False)


def _scramble_dropbacks(pbp: pl.DataFrame, scramble: pl.Expr) -> pl.DataFrame:
    """Count a run play that is a scramble as a dropback (its ``dropback_expr``)."""
    return pbp.with_columns(
        pl.when(scramble & _run_play())
        .then(1.0)
        .otherwise(pl.col("qb_dropback"))
        .alias("qb_dropback")
    )


def _two_point_tries(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Stop treating two-point tries apart, so they count as dropbacks and carries."""
    return pbp.with_columns(pl.lit(0.0).alias("two_point_attempt"))


def _kneels_and_spikes(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Count a kneel-down as a carry and a spike as a dropback, so their EPA enters the sums."""
    return pbp.with_columns(
        pl.when(_flag("qb_kneel")).then(1.0).otherwise(pl.col("rush")).alias("rush"),
        pl.when(_flag("qb_spike")).then(1.0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"),
        pl.lit(0.0).alias("qb_kneel"),
        pl.lit(0.0).alias("qb_spike"),
    )


def _wiped_out_runs(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Drop the rush flag from runs a penalty wiped out (``rush`` 1, ``rush_attempt`` 0)."""
    return pbp.with_columns(
        pl.when(_flag("rush") & ~_flag("rush_attempt"))
        .then(0.0)
        .otherwise(pl.col("rush"))
        .alias("rush")
    )


def _unflagged_rush_attempts(pbp: pl.DataFrame, scramble: pl.Expr) -> pl.DataFrame:
    """Flag as runs the rush attempts nflverse does not flag (not scrambles, not kneel-downs)."""
    unflagged = _flag("rush_attempt") & ~_flag("rush") & ~scramble & ~_flag("qb_kneel")
    return pbp.with_columns(pl.when(unflagged).then(1.0).otherwise(pl.col("rush")).alias("rush"))


def _null_downs(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Read a missing down as 0, so the play counts as an early down."""
    return pbp.with_columns(pl.col("down").fill_null(0.0))


def _every_play_type(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Give every play outside the four counted types the type ``run``."""
    counted = pl.col("play_type").is_in(list(SITUATIONAL_PLAY_TYPES)).fill_null(value=False)
    return pbp.with_columns(
        pl.when(counted).then(pl.col("play_type")).otherwise(pl.lit("run")).alias("play_type")
    )


SITUATIONAL_FLAGS = (
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "two_point_attempt",
)


def uncounted_situational_plays(pbp: pl.DataFrame) -> dict[str, int]:
    """Count, by play type, the flagged plays this repository's play-type filter leaves out.

    A play counts when it has a team on each side and any third-down, fourth-down or two-point
    flag, and its play type is not one of ``SITUATIONAL_PLAY_TYPES``; a missing type is ``none``.
    """
    sides = pl.all_horizontal(
        pl.col(c).is_not_null() & (pl.col(c) != "") for c in ("posteam", "defteam")
    )
    flagged = pl.any_horizontal(_flag(flag) for flag in SITUATIONAL_FLAGS)
    counted = pl.col("play_type").is_in(list(SITUATIONAL_PLAY_TYPES)).fill_null(value=False)
    left_out = pbp.filter(sides & flagged & ~counted).select(pl.col("play_type").fill_null("none"))
    return {str(k): int(v) for k, v in left_out.group_by("play_type").len().iter_rows()}


def _both_conversion_flags(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Count a down flagged both converted and failed once, as converted."""
    return pbp.with_columns(
        pl.when(_flag(f"{down}_down_converted") & _flag(f"{down}_down_failed"))
        .then(0.0)
        .otherwise(pl.col(f"{down}_down_failed"))
        .alias(f"{down}_down_failed")
        for down in ("third", "fourth")
    )


def _penalty_first_downs(pbp: pl.DataFrame, _scramble: pl.Expr) -> pl.DataFrame:
    """Count ``first_down`` on every play, first downs by penalty included."""
    return pbp.with_columns(
        pl.col("first_down").cast(pl.Float64).fill_null(0.0).alias("first_down_pass"),
        pl.lit(0.0).alias("first_down_rush"),
    )


def _fumbles_on_scrimmage_snaps(pbp: pl.DataFrame, scramble: pl.Expr) -> pl.DataFrame:
    """Treat every play that is not a scrimmage snap by its definition as special teams."""
    scrimmage = (
        _flag("qb_dropback")
        | (scramble & _run_play())
        | _flag("rush")
        | _flag("qb_kneel")
        | _flag("qb_spike")
    )
    return pbp.with_columns(pl.when(scrimmage).then(0.0).otherwise(1.0).alias("special"))


def one_team_games(pbp: pl.DataFrame) -> dict[str, str]:
    """Return each game whose cached ``penalty_team`` and ``td_team`` values name one team.

    The value is that team. Its ``player_team_repair.one_team_games`` reads sixteen player team
    columns; this repository caches two of them, so this reads those two.
    """
    present = [column for column in ("penalty_team", "td_team") if column in pbp.columns]
    values = (
        pbp.select("game_id", *present)
        .unpivot(index="game_id", value_name="team")
        .filter(pl.col("team").is_not_null() & (pl.col("team") != ""))
        .group_by("game_id")
        .agg(pl.col("team").n_unique().alias("teams"), pl.col("team").first().alias("named"))
        .filter(pl.col("teams") == 1)
        .sort("game_id")
    )
    return dict(zip(values["game_id"].to_list(), values["named"].to_list(), strict=True))


def unrepair_penalties(theirs: pl.DataFrame, broken: dict[str, str]) -> pl.DataFrame:
    """Undo its player-team repair: credit a broken game's penalties to the team nflverse named."""
    if not broken:
        return theirs
    named = pl.DataFrame(
        {"game_id": list(broken), "_named": list(broken.values())},
        schema={"game_id": pl.String, "_named": pl.String},
    )
    totals = theirs.group_by("game_id").agg(
        pl.col("penalties").sum().alias("_penalties"),
        pl.col("penalty_yards").sum().alias("_penalty_yards"),
    )
    adjusted = theirs.join(named, on="game_id", how="left").join(totals, on="game_id", how="left")
    exprs = [
        pl.when(pl.col("_named").is_null())
        .then(pl.col(column))
        .when(pl.col("team_abbr") == pl.col("_named"))
        .then(pl.col(f"_{column}"))
        .otherwise(0)
        .cast(theirs.schema[column])
        .alias(column)
        for column in ("penalties", "penalty_yards")
    ]
    return adjusted.with_columns(exprs).drop("_named", "_penalties", "_penalty_yards")


@dataclass(frozen=True)
class Cause:
    """A named cause and how it adjusts one side of the comparison."""

    id: str
    side: Literal["ours", "theirs", "keys", "coverage", "none"]
    uses_scrambles: bool = False
    ours: Callable[[pl.DataFrame, pl.Expr], pl.DataFrame] | None = None


CAUSES: dict[str, Cause] = {
    cause.id: cause
    for cause in (
        Cause("scramble_dropbacks", "ours", uses_scrambles=True, ours=_scramble_dropbacks),
        Cause("two_point_tries", "ours", ours=_two_point_tries),
        Cause("kneels_and_spikes", "ours", ours=_kneels_and_spikes),
        Cause("wiped_out_runs", "ours", ours=_wiped_out_runs),
        Cause(
            "unflagged_rush_attempts", "ours", uses_scrambles=True, ours=_unflagged_rush_attempts
        ),
        Cause("null_downs", "ours", ours=_null_downs),
        Cause("every_play_type", "ours", ours=_every_play_type),
        Cause("both_conversion_flags", "ours", ours=_both_conversion_flags),
        Cause("penalty_first_downs", "ours", ours=_penalty_first_downs),
        Cause(
            "fumbles_on_scrimmage_snaps",
            "ours",
            uses_scrambles=True,
            ours=_fumbles_on_scrimmage_snaps,
        ),
        Cause("jacksonville_player_teams", "theirs"),
        Cause("rams_written_la", "keys"),
        Cause("games_without_play_by_play", "coverage"),
        Cause("release_difference", "none"),
    )
}
