"""Check this repository's play-by-play team-game aggregates against `nfl-sos-ratings` game logs.

Run from the repository root:

    .venv/bin/python .agents/sos_port/parity.py

For every regular season 1999-2025 it aggregates this repository's cached play-by-play (the main
checkout's ``data/cache/nflreadpy/pbp_{season}_reg.parquet``) with
``aggregate_pbp_team_game_stats`` and ``aggregate_pbp_team_box_score_stats`` and compares every
team-game with ``{season}_team_game_logs.parquet`` of `nfl-sos-ratings` at its frozen commit, on
the pairs ``parity.yaml`` judges to share a definition. Which game-log columns come from
play-by-play is read from that repository's code (``sos_sources.py``); only those are judged.

Counts must be equal, sums and rates within the tolerance. A mismatching team-game is attributed
to a named cause only when that cause's adjustment (``parity_rules.CAUSES``), alone or with the
fewest other causes the pair allows, makes it match exactly. It writes ``PARITY.md`` and
``parity.json`` beside this file and exits non-zero when any team-game stays unexplained, when an
attribution holds only with the stand-in for the ``qb_scramble`` flag, when a team-game exists on
one side only, when an evidence pattern or a source check fails, or when an input is missing or
not at its expected commit.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import yaml
from parity_report import render
from parity_rules import (
    CAUSES,
    KEYS,
    Pair,
    check_evidence,
    expression,
    one_team_games,
    scramble_flag,
    unrepair_penalties,
)
from port_audit import (
    DATA,
    HERE,
    SEASONS,
    Audit,
    SosCode,
    check_sos_checkout,
    portable,
    sos_output,
    team_code_map,
)
from report_text import dump_json
from sos_sources import PLAY_BY_PLAY, classify_game_log_columns

from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.pbp import (
    aggregate_pbp_team_box_score_stats,
    aggregate_pbp_team_game_stats,
)

RULES = HERE / "parity.yaml"
OUT_MD = HERE / "PARITY.md"
OUT_JSON = HERE / "parity.json"
CACHE = DATA / "cache" / "nflreadpy"
TEAM_COLUMNS = ("posteam", "defteam", "home_team", "away_team", "td_team", "penalty_team")
ADJUSTABLE = ("ours", "theirs")
KINDS = ("count", "count_from_rate", "sum", "rate")
_COUNT_KEYS = ("compared", "mismatched", "explained", "provisional", "unexplained")
# Marks a cause combination that matches only with the stand-in's scrambles.
STAND_IN_MARK = " (stand-in)"


def load_pairs(rules: dict[str, Any], audit: Audit) -> list[Pair]:
    """Build the judged pairs and check that every cause they name exists."""
    pairs = []
    for entry in rules["pairs"]:
        causes = tuple(entry["causes"])
        if entry["kind"] not in KINDS:
            audit.fail("pair", f"{entry['id']}: unknown kind {entry['kind']}")
        for cause in causes:
            if cause not in CAUSES or cause not in rules["causes"]:
                audit.fail("pair", f"{entry['id']}: unknown cause {cause}")
            elif CAUSES[cause].side not in ADJUSTABLE:
                audit.fail("pair", f"{entry['id']}: cause {cause} adjusts no value")
        pairs.append(
            Pair(
                id=entry["id"],
                ours=expression(str(entry["ours"])),
                theirs=expression(str(entry["theirs"])),
                kind=entry["kind"],
                causes=causes,
                null_as_zero=bool(entry.get("null_as_zero", False)),
            )
        )
    implemented, described = set(CAUSES), set(rules["causes"])
    for cause in sorted(implemented ^ described):
        side = "described in parity.yaml only" if cause in described else "implemented only"
        audit.fail("cause", f"{cause}: {side}")
    return pairs


def ours_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return this repository's two play-by-play aggregates joined per team-game."""
    team_game = aggregate_pbp_team_game_stats(pbp)
    box_score = aggregate_pbp_team_box_score_stats(pbp).drop("season_type")
    return team_game.join(box_score, on=["season", *KEYS], how="full", coalesce=True).drop("season")


def compare(pair: Pair, ours: pl.DataFrame, theirs: pl.DataFrame, tolerance: float) -> pl.DataFrame:
    """Return one row per team-game on both sides with both values and whether they match."""
    their_value = pair.theirs.expr
    if pair.null_as_zero:
        their_value = their_value.fill_null(0.0)
    joined = theirs.select(*KEYS, "game_id", their_value.alias("theirs")).join(
        ours.select(*KEYS, pair.ours.expr.alias("ours")), on=KEYS, how="inner"
    )
    both_null = pl.col("ours").is_null() & pl.col("theirs").is_null()
    if pair.kind == "count":
        equal = pl.col("ours") == pl.col("theirs")
    else:
        equal = (pl.col("ours") - pl.col("theirs")).abs() <= tolerance
    return joined.with_columns(
        (both_null | equal.fill_null(value=False)).alias("match"),
        (pl.col("ours") - pl.col("theirs")).alias("diff"),
    )


@dataclass
class SeasonFrames:
    """One season's inputs and its adjusted frames, each built once."""

    season: int
    pbp: pl.DataFrame
    theirs_base: pl.DataFrame
    broken_games: dict[str, str]
    flag_kind: str
    _ours: dict[tuple[frozenset[str], bool], pl.DataFrame] = field(default_factory=dict)
    _theirs: dict[frozenset[str], pl.DataFrame] = field(default_factory=dict)

    def ours(self, causes: frozenset[str], *, scrambles: bool) -> pl.DataFrame:
        """Return this repository's aggregates after the ``causes`` play-by-play adjustments."""
        key = (causes, scrambles)
        if key not in self._ours:
            flag, _ = scramble_flag(self.pbp.columns, enabled=scrambles)
            pbp = self.pbp
            for cause in CAUSES.values():
                if cause.id in causes and cause.ours is not None:
                    pbp = cause.ours(pbp, flag)
            self._ours[key] = ours_frame(pbp)
        return self._ours[key]

    def theirs(self, causes: frozenset[str]) -> pl.DataFrame:
        """Return its game logs after the ``causes`` adjustments on its side."""
        if causes not in self._theirs:
            frame = self.theirs_base
            if "jacksonville_player_teams" in causes:
                frame = unrepair_penalties(frame, self.broken_games)
            self._theirs[causes] = frame
        return self._theirs[causes]


def _split(combo: tuple[str, ...]) -> tuple[frozenset[str], frozenset[str]]:
    """Split a cause combination into its play-by-play and its game-log adjustments."""
    ours = frozenset(cause for cause in combo if CAUSES[cause].side == "ours")
    return ours, frozenset(combo) - ours


def attribute(
    pair: Pair, frames: SeasonFrames, base: pl.DataFrame, tolerance: float
) -> list[dict[str, Any]]:
    """Attribute every mismatching team-game of ``base`` to its fewest matching causes."""
    mismatched = base.filter(~pl.col("match"))
    records: dict[tuple[int, str, str], dict[str, Any]] = {
        (row["week"], row["team_abbr"], row["opponent_abbr"]): {
            "season": frames.season,
            "game_id": row["game_id"],
            "week": row["week"],
            "team": row["team_abbr"],
            "opponent": row["opponent_abbr"],
            "pair": pair.id,
            "ours": row["ours"],
            "theirs": row["theirs"],
            "diff": row["diff"],
            "causes": [],
            "status": "unexplained",
        }
        for row in mismatched.iter_rows(named=True)
    }
    remaining = set(records)
    for size in range(1, len(pair.causes) + 1):
        if not remaining:
            break
        found: dict[tuple[int, str, str], list[tuple[str, str]]] = defaultdict(list)
        for combo in itertools.combinations(pair.causes, size):
            ours_causes, their_causes = _split(combo)
            adjusted = compare(
                pair,
                frames.ours(ours_causes, scrambles=True),
                frames.theirs(their_causes),
                tolerance,
            ).filter(pl.col("match"))
            keys = {(r[0], r[1], r[2]) for r in adjusted.select(KEYS).iter_rows()} & remaining
            if not keys:
                continue
            provisional: set[tuple[int, str, str]] = set()
            if frames.flag_kind == "stand-in" and any(CAUSES[c].uses_scrambles for c in combo):
                without = compare(
                    pair,
                    frames.ours(ours_causes, scrambles=False),
                    frames.theirs(their_causes),
                    tolerance,
                ).filter(pl.col("match"))
                held = {(r[0], r[1], r[2]) for r in without.select(KEYS).iter_rows()}
                provisional = keys - held
            label = " + ".join(combo)
            for key in keys:
                found[key].append((label, "provisional" if key in provisional else "confirmed"))
        for key, matches in found.items():
            records[key]["causes"] = sorted(
                label if status == "confirmed" else f"{label}{STAND_IN_MARK}"
                for label, status in matches
            )
            confirmed = any(status == "confirmed" for _, status in matches)
            records[key]["status"] = "explained" if confirmed else "provisional"
        remaining -= set(found)
    return list(records.values())


def theirs_frame(season: int, audit: Audit, code_map: dict[str, str]) -> pl.DataFrame:
    """Read one season's game logs, keyed with this repository's team codes."""
    frame = pl.read_parquet(audit.record(sos_output(season, "team_game_logs")))
    mapping = team_code_map(frame["team"].to_list() + frame["opponent_team"].to_list())
    code_map.update({code: ours for code, ours in mapping.items() if code != ours})
    return frame.with_columns(
        pl.col("team").replace_strict(mapping).alias("team_abbr"),
        pl.col("opponent_team").replace_strict(mapping).alias("opponent_abbr"),
        pl.col("week").cast(pl.Int64),
    )


def flagless_plays(pbp: pl.DataFrame) -> dict[tuple[str, int, str, str], int]:
    """Count, per team-game side, the plays only the ``qb_scramble`` flag can classify.

    These are rush attempts coded as pass plays without a dropback flag (``pass`` 1, ``rush`` 0,
    ``rush_attempt`` 1, ``qb_dropback`` 0): before 2006 most scrambles, and in any season the
    aborted snaps nflverse codes the same way. The stand-in calls every one of them a scramble.
    """
    coded = pbp.filter(
        (pl.col("pass").fill_null(0) > 0)
        & (pl.col("rush").fill_null(0) == 0)
        & (pl.col("rush_attempt").fill_null(0) > 0)
        & (pl.col("qb_dropback").fill_null(0) == 0)
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
    )
    counts: dict[tuple[str, int, str, str], int] = {}
    for (week, offense, defense), group in coded.group_by(["week", "posteam", "defteam"]):
        counts[("offense", int(week), str(offense), str(defense))] = group.height
        counts[("defense", int(week), str(defense), str(offense))] = group.height
    return counts


def check_keys(
    season: int, ours: pl.DataFrame, theirs: pl.DataFrame, pbp: pl.DataFrame, audit: Audit
) -> dict[str, Any]:
    """Compare the team-game keys of both sides; re-key one-sided rows with LA read as LAR."""
    our_keys = {tuple(row) for row in ours.select(KEYS).iter_rows()}
    their_keys = {tuple(row) for row in theirs.select(KEYS).iter_rows()}

    def rams(key: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple("LAR" if value == "LA" else value for value in key)

    ours_only, theirs_only = our_keys - their_keys, their_keys - our_keys
    rekeyed = {rams(key) for key in ours_only} & {rams(key) for key in theirs_only}
    unpaired = sorted(str(key) for key in (ours_only | theirs_only) if rams(key) not in rekeyed)
    for key in unpaired:
        audit.fail("team-game keys", f"{season} {key} exists on one side only")
    la_values = sum(
        pbp.filter(pl.col(column) == "LA").height
        for column in TEAM_COLUMNS
        if column in pbp.columns
    )
    return {
        "both": len(our_keys & their_keys),
        "ours_only": len(ours_only),
        "theirs_only": len(theirs_only),
        "rekeyed_rams": sorted(str(key) for key in rekeyed),
        "unpaired": unpaired,
        "cached_la_values": la_values,
    }


def check_coverage(
    season: int,
    ours: pl.DataFrame,
    theirs: pl.DataFrame,
    pbp: pl.DataFrame,
    expected_missing: set[str],
    audit: Audit,
) -> dict[str, Any]:
    """Check every played regular-season game of the cached schedule against both sides."""
    schedule = pl.read_parquet(audit.record(CACHE / f"schedule_{season}.parquet")).filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    codes = team_code_map(schedule["home_abbr"].to_list() + schedule["away_abbr"].to_list())
    our_keys = {tuple(row) for row in ours.select(KEYS).iter_rows()}
    their_games = set(theirs["game_id"].to_list())
    played_games = set(pbp["game_id"].drop_nulls().to_list())
    without_rows, unexpected = [], []
    for row in schedule.iter_rows(named=True):
        home, away = codes[row["home_abbr"]], codes[row["away_abbr"]]
        ours_has = {(row["week"], home, away), (row["week"], away, home)} <= our_keys
        theirs_has = row["game_id"] in their_games
        if ours_has and theirs_has:
            continue
        explained = (
            row["game_id"] in expected_missing
            and not ours_has
            and not theirs_has
            and row["game_id"] not in played_games
        )
        (without_rows if explained else unexpected).append(row["game_id"])
    for game in unexpected:
        audit.fail("coverage", f"{game} has no team-game rows on one or both sides")
    for game in sorted(expected_missing):
        if game.startswith(f"{season}_") and game not in without_rows:
            audit.fail("coverage", f"{game} is listed without play-by-play but has rows or plays")
    return {
        "played_games": schedule.height,
        "games_without_play_by_play": sorted(without_rows),
        "unexpected": sorted(unexpected),
    }


def main() -> int:
    """Run the parity check and write the reports."""
    audit = Audit()
    code = SosCode()
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    tolerance = float(rules["tolerance"])
    expected_missing = set(rules["games_without_play_by_play"])
    checkout = check_sos_checkout(audit)
    pairs = load_pairs(rules, audit)
    evidence = {}
    for cause_id, cause in rules["causes"].items():
        held, rows = check_evidence(cause.get("evidence", []), code)
        evidence[cause_id] = rows
        if not held:
            audit.fail("evidence", f"{cause_id}: an evidence pattern no longer holds")

    first_logs = pl.read_parquet_schema(sos_output(SEASONS[0], "team_game_logs"))
    log_columns = list(first_logs)
    sources, source_facts = classify_game_log_columns(code, log_columns, audit)

    code_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    per_pair: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seasons_report: dict[int, dict[str, Any]] = {}
    flag_kinds: set[str] = set()
    broken_all: dict[str, str] = {}
    for season in SEASONS:
        log.info("Season %d", season)
        pbp = pl.read_parquet(audit.record(CACHE / f"pbp_{season}_reg.parquet"))
        theirs = theirs_frame(season, audit, code_map)
        if list(theirs.columns[: len(log_columns)]) != log_columns:
            audit.fail("game-log columns", f"{season} columns differ from {SEASONS[0]}'s")
        _, flag_kind = scramble_flag(pbp.columns, enabled=True)
        flag_kinds.add(flag_kind)
        broken = one_team_games(pbp)
        broken_all.update(broken)
        frames = SeasonFrames(season, pbp, theirs, broken, flag_kind)
        base_ours = frames.ours(frozenset(), scrambles=True)
        if season == SEASONS[0]:
            before = len(audit.failures)
            for pair in pairs:
                for column in sorted(pair.ours.columns - set(base_ours.columns)):
                    audit.fail("pair", f"{pair.id}: {column} is not in this repo's aggregates")
                for column in sorted(pair.theirs.columns):
                    source = sources.get(column)
                    if column not in log_columns:
                        audit.fail("pair", f"{pair.id}: {column} is not in its game logs")
                    elif source is None or source.source != PLAY_BY_PLAY:
                        found = source.source if source else "unknown"
                        audit.fail("pair", f"{pair.id}: {column} comes from {found}")
            if len(audit.failures) > before:
                break
        keys = check_keys(season, base_ours, theirs, pbp, audit)
        coverage = check_coverage(season, base_ours, theirs, pbp, expected_missing, audit)
        season_pairs = {}
        flagless = flagless_plays(pbp)
        for pair in pairs:
            base = compare(pair, base_ours, theirs, tolerance)
            found = attribute(pair, frames, base, tolerance)
            for record in found:
                role = "defense" if pair.id == "defensive_snaps" else "offense"
                record["flagless_plays"] = flagless.get(
                    (role, record["week"], record["team"], record["opponent"]), 0
                )
            records.extend(found)
            counts = per_pair[pair.id]
            counts["compared"] += base.height
            counts["mismatched"] += len(found)
            for record in found:
                counts[record["status"]] += 1
            season_pairs[pair.id] = {"compared": base.height, "mismatched": len(found)}
        seasons_report[season] = {
            "keys": keys,
            "coverage": coverage,
            "one_team_games": broken,
            "scramble_flag": flag_kind,
            "pairs": season_pairs,
        }

    order = {pair.id: index for index, pair in enumerate(pairs)}
    records.sort(key=lambda r: (order[r["pair"]], r["season"], r["week"], r["team"]))
    for record in records:
        if record["status"] == "unexplained":
            audit.fail("unexplained", _describe(record))
    provisional = Counter((r["pair"], r["season"]) for r in records if r["status"] == "provisional")
    for (pair_id, season), count in sorted(provisional.items()):
        audit.fail(
            "provisional",
            f"{pair_id} {season}: {count} team-games match only with the stand-in's scrambles",
        )

    payload = {
        "generated_by": ".venv/bin/python .agents/sos_port/parity.py",
        "tolerance": tolerance,
        "sos_checkout": checkout,
        "team_code_map": dict(sorted(code_map.items())),
        "scramble_flag": sorted(flag_kinds),
        "one_team_games": dict(sorted(broken_all.items())),
        "game_log_sources": {
            column: {"source": sources[column].source, "via": sources[column].via}
            for column in log_columns
            if column in sources
        },
        "game_log_source_facts": source_facts,
        "pairs": [
            {
                "id": pair.id,
                "ours": pair.ours.text,
                "theirs": pair.theirs.text,
                "kind": pair.kind,
                "causes": list(pair.causes),
                **{k: per_pair[pair.id].get(k, 0) for k in _COUNT_KEYS},
            }
            for pair in pairs
        ],
        "causes": summarize_causes(rules, records, seasons_report),
        "evidence": evidence,
        "seasons": {str(season): report for season, report in seasons_report.items()},
        "inputs": audit.files,
        "failures": [{"what": f.what, "detail": f.detail} for f in audit.failures],
    }
    write_json(payload, records)
    OUT_MD.write_text(portable(render(payload, rules, records)), encoding="utf-8")
    code.cleanup()
    log.info("%d mismatching pair values, %d failures", len(records), len(audit.failures))
    return 1 if audit.failures else 0


def _describe(record: dict[str, Any]) -> str:
    """Return a one-line description of a mismatching team-game."""
    return (
        f"{record['pair']} {record['game_id']} {record['team']}: here {record['ours']}, "
        f"there {record['theirs']}"
    )


def _labels(record: dict[str, Any], cause_id: str, *, confirmed: bool | None = None) -> list[str]:
    """Return the record's matching cause combinations that include ``cause_id``.

    ``confirmed`` true keeps those that hold without the stand-in, false those that need it.
    """
    return [
        label
        for label in record["causes"]
        if cause_id in label.removesuffix(STAND_IN_MARK).split(" + ")
        and (confirmed is None or label.endswith(STAND_IN_MARK) != confirmed)
    ]


def summarize_causes(
    rules: dict[str, Any], records: list[dict[str, Any]], seasons: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Return, per cause, the team-games and games it explains and the size of the difference."""
    summary: dict[str, Any] = {}
    for cause_id, cause in rules["causes"].items():
        mine = [r for r in records if _labels(r, cause_id)]
        columns: dict[str, Any] = {}
        for pair_id in sorted({r["pair"] for r in mine}):
            rows = [r for r in mine if r["pair"] == pair_id]
            diffs = [float(r["diff"]) for r in rows if r["diff"] is not None]
            columns[pair_id] = {
                "team_games": len(rows),
                "explained": sum(bool(_labels(r, cause_id, confirmed=True)) for r in rows),
                "provisional": sum(not _labels(r, cause_id, confirmed=True) for r in rows),
                "shared_with_another_cause": sum(len(r["causes"]) > 1 for r in rows),
                "seasons": sorted({r["season"] for r in rows}),
                "signed_diff_sum": round(sum(diffs), 6),
                "abs_diff_sum": round(sum(abs(d) for d in diffs), 6),
                "max_abs_diff": round(max((abs(d) for d in diffs), default=0.0), 6),
            }
        entry: dict[str, Any] = {
            "name": cause["name"],
            "team_games": len({(r["season"], r["week"], r["team"]) for r in mine}),
            "games": len({(r["season"], r["game_id"]) for r in mine}),
            "seasons": sorted({r["season"] for r in mine}),
            "columns": columns,
        }
        if cause_id == "games_without_play_by_play":
            games = [
                g for s in seasons.values() for g in s["coverage"]["games_without_play_by_play"]
            ]
            entry.update(games=len(games), team_games=2 * len(games), listed=sorted(games))
            entry["seasons"] = sorted(int(g[:4]) for g in games)
        if cause_id == "rams_written_la":
            rekeyed = [k for s in seasons.values() for k in s["keys"]["rekeyed_rams"]]
            entry.update(
                team_games=len(rekeyed),
                cached_la_values=sum(s["keys"]["cached_la_values"] for s in seasons.values()),
            )
        summary[cause_id] = entry
    return summary


UNEXPLAINED_COLUMNS = [
    "season",
    "game_id",
    "week",
    "team",
    "opponent",
    "pair",
    "ours",
    "theirs",
    "diff",
    "causes",
    "flagless_plays",
]
ATTRIBUTION_COLUMNS = [
    "pair",
    "season",
    "status",
    "causes",
    "team_games",
    "signed_diff_sum",
    "abs_diff_sum",
    "max_abs_diff",
]


def attribution_rows(records: list[dict[str, Any]]) -> list[list[object]]:
    """Group the mismatching team-games by pair, season, status and matching causes."""
    groups: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for record in records:
        key = (record["pair"], record["season"], record["status"], " | ".join(record["causes"]))
        groups[key].append(float(record["diff"]) if record["diff"] is not None else 0.0)
    return [
        [
            *key,
            len(diffs),
            round(sum(diffs), 6),
            round(sum(abs(d) for d in diffs), 6),
            round(max(abs(d) for d in diffs), 6),
        ]
        for key, diffs in sorted(groups.items())
    ]


def write_json(payload: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write ``parity.json``: the payload, then one line per table row.

    ``attributions`` groups every mismatching team-game by pair, season, status and the cause
    combinations that make it match; ``unexplained`` lists each unexplained team-game.
    """
    unexplained = [
        [_rounded(record[column]) for column in UNEXPLAINED_COLUMNS]
        for record in sorted(records, key=lambda r: (r["pair"], r["season"], r["week"], r["team"]))
        if record["status"] == "unexplained"
    ]
    body = {
        **payload,
        "attributions": {"columns": ATTRIBUTION_COLUMNS, "rows": attribution_rows(records)},
        "unexplained": {"columns": UNEXPLAINED_COLUMNS, "rows": unexplained},
    }
    OUT_JSON.write_text(portable(dump_json(body)) + "\n", encoding="utf-8")


def _rounded(value: object) -> object:
    """Round floats so the report is stable across platforms."""
    return round(value, 9) if isinstance(value, float) else value


if __name__ == "__main__":
    sys.exit(main())
