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
from nfl_predictor.utils.polars.pbp import (
    SITUATIONAL_PLAY_TYPES,
    _play_conditions,
    regular_season_plays,
)

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
# Each adjustment changes one rule in a season's play-by-play (this repository's cached columns
# plus the snapshot's ``qb_scramble`` and ``play_id``) before this repository's own aggregates
# run again on it.


def _flag(name: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing."""
    return pl.col(name).cast(pl.Float64).fill_null(0) > 0


def _run_play() -> pl.Expr:
    """Return whether the play's type is ``run``."""
    return (pl.col("play_type") == "run").fill_null(value=False)


def _scramble_dropbacks(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a run play flagged ``qb_scramble`` as a dropback (its ``dropback_expr``)."""
    return pbp.with_columns(
        pl.when(_flag("qb_scramble") & _run_play())
        .then(1.0)
        .otherwise(pl.col("qb_dropback"))
        .alias("qb_dropback")
    )


def _scrambles_coded_as_runs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Clear the rush flag of a play flagged both ``qb_scramble`` and ``rush``."""
    coded = _flag("qb_scramble") & _flag("rush")
    return pbp.with_columns(pl.when(coded).then(0.0).otherwise(pl.col("rush")).alias("rush"))


def _dropback_and_carry(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a play flagged both a dropback and a run once, as a dropback."""
    both = _flag("qb_dropback") & _flag("rush")
    return pbp.with_columns(pl.when(both).then(0.0).otherwise(pl.col("rush")).alias("rush"))


def _spike_rush_attempts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a spike nflverse also records as a rush attempt as a carry, not a spike."""
    both = _flag("qb_spike") & _flag("rush_attempt")
    return pbp.with_columns(
        pl.when(both).then(0.0).otherwise(pl.col("qb_spike")).alias("qb_spike"),
        pl.when(both).then(1.0).otherwise(pl.col("rush")).alias("rush"),
    )


def _two_point_tries(pbp: pl.DataFrame) -> pl.DataFrame:
    """Stop treating two-point tries apart, so they count as dropbacks and carries."""
    return pbp.with_columns(pl.lit(0.0).alias("two_point_attempt"))


def _kneels_and_spikes(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a kneel-down as a carry and a spike as a dropback, so their EPA enters the sums."""
    return pbp.with_columns(
        pl.when(_flag("qb_kneel")).then(1.0).otherwise(pl.col("rush")).alias("rush"),
        pl.when(_flag("qb_spike")).then(1.0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"),
        pl.lit(0.0).alias("qb_kneel"),
        pl.lit(0.0).alias("qb_spike"),
    )


def _wiped_out_runs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Drop the rush flag from runs a penalty wiped out (``rush`` 1, ``rush_attempt`` 0)."""
    wiped = _flag("rush") & ~_flag("rush_attempt")
    return pbp.with_columns(pl.when(wiped).then(0.0).otherwise(pl.col("rush")).alias("rush"))


def _unflagged_rush_attempts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Flag as runs the rush attempts nflverse does not flag (not scrambles, not kneel-downs)."""
    unflagged = _flag("rush_attempt") & ~_flag("rush") & ~_flag("qb_scramble") & ~_flag("qb_kneel")
    return pbp.with_columns(pl.when(unflagged).then(1.0).otherwise(pl.col("rush")).alias("rush"))


def _null_downs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Read a missing down as 0, so the play counts as an early down."""
    return pbp.with_columns(pl.col("down").fill_null(0.0))


def _every_play_type(pbp: pl.DataFrame) -> pl.DataFrame:
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


# What the play text of a ``no_play`` row without the penalty flag says, tried in this order.
NO_PLAY_TEXTS = {
    "offsetting_penalties": "(?i)offsetting",
    "declined_penalties": "(?i)declined",
    "plays_under_review": "(?i)under review",
    "timeouts": "(?i)timeout",
}


def unflagged_no_plays(pbp: pl.DataFrame) -> dict[str, int]:
    """Count the ``no_play`` rows without the penalty flag by what their play text (``desc``) says.

    The rows are those ``wiped_out_plays`` counts as ``without_penalty_flag``; each falls under the
    first ``NO_PLAY_TEXTS`` pattern its text matches, else ``other``.
    """
    sides = pl.all_horizontal(
        pl.col(c).is_not_null() & (pl.col(c) != "") for c in ("posteam", "defteam")
    )
    no_play = (pl.col("play_type") == "no_play").fill_null(value=False)
    rows = pbp.filter(sides & no_play & ~_flag("penalty"))
    text = pl.col("desc").fill_null("")
    kind: pl.Expr = pl.lit("other")
    for name, pattern in reversed(NO_PLAY_TEXTS.items()):
        kind = pl.when(text.str.contains(pattern)).then(pl.lit(name)).otherwise(kind)
    found = dict(rows.select(kind.alias("kind")).group_by("kind").len().iter_rows())
    return {name: int(found.get(name, 0)) for name in [*NO_PLAY_TEXTS, "other"]}


def carries_without_rush_attempt(pbp: pl.DataFrame) -> dict[str, int]:
    """Count this repository's carries that lack the ``rush_attempt`` flag, by play type.

    These are the plays a carry rule built from ``rush_attempt`` takes out of carries; they keep
    the ``rush`` flag, so they stay scrimmage snaps. A missing play type is ``none``.
    """
    plays = regular_season_plays(pbp)
    today = _play_conditions(plays.columns)["rush"]
    rows = plays.filter(today & ~_flag("rush_attempt"))
    found = rows.select(pl.col("play_type").fill_null("none")).group_by("play_type").len()
    return {str(kind): int(count) for kind, count in sorted(found.iter_rows())}


def carries_from_rush_attempts(pbp: pl.DataFrame) -> dict[str, int]:
    """Count how carry rules built from ``rush_attempt`` differ from this repository's carries.

    Over the plays this repository's aggregates read, ``today`` counts its carries (the ``rush``
    condition of ``_play_conditions``). ``with_flag`` is its designed-run rule,
    ``rush_attempt & ~two_point_attempt & ~qb_scramble & ~qb_kneel``; ``without_flag`` is the same
    rule without ``~qb_scramble``, the closest a cache without the scramble flag can build. Each
    counts the plays it adds to and removes from today's carries; the ``without_flag_added_*``
    counts split that rule's added plays by the scramble flag and by today's dropback condition.
    """
    plays = regular_season_plays(pbp)
    conditions = _play_conditions(plays.columns)
    today, dropback = conditions["rush"], conditions["dropback"]
    without_flag = _flag("rush_attempt") & ~_flag("two_point_attempt") & ~_flag("qb_kneel")
    with_flag = without_flag & ~_flag("qb_scramble")
    added = plays.filter(without_flag & ~today)
    return {
        "today": plays.filter(today).height,
        "with_flag_added": plays.filter(with_flag & ~today).height,
        "with_flag_removed": plays.filter(today & ~with_flag).height,
        "without_flag_added": added.height,
        "without_flag_removed": plays.filter(today & ~without_flag).height,
        "without_flag_added_scrambles": added.filter(_flag("qb_scramble")).height,
        "without_flag_added_dropbacks": added.filter(dropback).height,
        "without_flag_added_scramble_dropbacks": added.filter(
            _flag("qb_scramble") & dropback
        ).height,
    }


def _both_conversion_flags(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count a down flagged both converted and failed once, as converted."""
    return pbp.with_columns(
        pl.when(_flag(f"{down}_down_converted") & _flag(f"{down}_down_failed"))
        .then(0.0)
        .otherwise(pl.col(f"{down}_down_failed"))
        .alias(f"{down}_down_failed")
        for down in ("third", "fourth")
    )


def _penalty_first_downs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count ``first_down`` on every play, first downs by penalty included."""
    return pbp.with_columns(
        pl.col("first_down").cast(pl.Float64).fill_null(0.0).alias("first_down_pass"),
        pl.lit(0.0).alias("first_down_rush"),
    )


def _fumbles_on_scrimmage_snaps(pbp: pl.DataFrame) -> pl.DataFrame:
    """Treat every play that is not a scrimmage snap by its definition as special teams."""
    scrimmage = (
        _flag("qb_dropback")
        | (_flag("qb_scramble") & _run_play())
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
    side: Literal["ours", "theirs", "keys", "coverage", "identity"]
    ours: Callable[[pl.DataFrame], pl.DataFrame] | None = None


# Adjustments apply in this order when combined: the dropback-and-carry rule reads the dropback flag
# as cached, before the scramble rule sets it on run plays flagged qb_scramble.
CAUSES: dict[str, Cause] = {
    cause.id: cause
    for cause in (
        Cause("dropback_and_carry", "ours", ours=_dropback_and_carry),
        Cause("scramble_dropbacks", "ours", ours=_scramble_dropbacks),
        Cause("scrambles_coded_as_runs", "ours", ours=_scrambles_coded_as_runs),
        Cause("spike_rush_attempts", "ours", ours=_spike_rush_attempts),
        Cause("two_point_tries", "ours", ours=_two_point_tries),
        Cause("kneels_and_spikes", "ours", ours=_kneels_and_spikes),
        Cause("wiped_out_runs", "ours", ours=_wiped_out_runs),
        Cause("unflagged_rush_attempts", "ours", ours=_unflagged_rush_attempts),
        Cause("null_downs", "ours", ours=_null_downs),
        Cause("every_play_type", "ours", ours=_every_play_type),
        Cause("both_conversion_flags", "ours", ours=_both_conversion_flags),
        Cause("penalty_first_downs", "ours", ours=_penalty_first_downs),
        Cause("fumbles_on_scrimmage_snaps", "ours", ours=_fumbles_on_scrimmage_snaps),
        Cause("jacksonville_player_teams", "theirs"),
        Cause("rams_written_la", "keys"),
        Cause("games_without_play_by_play", "coverage"),
        Cause("release_difference", "identity"),
    )
}
