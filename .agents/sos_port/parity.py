"""Check this repository's play-by-play team-game aggregates against `nfl-sos-ratings` game logs.

Run from the repository root (``PYTHONDONTWRITEBYTECODE=1`` keeps the run from leaving
``__pycache__`` beside the scripts):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/parity.py [--data-dir DIR]

For every regular season 1999-2025 it reads this repository's cached play-by-play
(``cache/nflreadpy/pbp_{season}_reg.parquet`` under ``--data-dir``, by default the main
checkout's ``data/``) through its own loader, repairs it as the ETL does, aggregates it with
``aggregate_pbp_team_game_stats`` and ``aggregate_pbp_team_box_score_stats``, and compares every
team-game with ``{season}_team_game_logs.parquet`` of `nfl-sos-ratings` at its frozen commit, on
the pairs ``parity.yaml`` judges to share a definition. Which game-log columns come from
play-by-play is read from that repository's code (``sos_sources.py``); only those are judged.
Every nflverse download is blocked, so nothing is written to the data directory.

The play-by-play `nfl-sos-ratings` read is the snapshot of nflreadpy's on-disk cache it used
(``port_audit.NFLREADPY_SNAPSHOT``). Each season's file is found by its content and checked
against the snapshot's manifest, and the run fails unless that file, prepared as this repository
prepares its cache, equals the cache row by row on every cached column (``season_pbp``).

Counts must be equal, sums and rates within the tolerance. A mismatching team-game is attributed
to a named cause only when that cause's adjustment (``parity_rules.CAUSES``), alone or with the
fewest other causes the pair allows, makes it match exactly. It writes ``PARITY.md``,
``parity.json`` and ``parity_attributions.parquet`` (every mismatching team-game with its causes)
beside this file and exits non-zero when any team-game stays unexplained, when a team-game exists
on one side only, when an evidence pattern or a source check fails, or when an input is missing,
not at its expected commit, or not identical to this repository's cache.
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml
from parity_report import render
from parity_rules import (
    CAUSES,
    KEYS,
    Pair,
    adjust,
    check_evidence,
    expression,
    wiped_out_plays,
)
from port_audit import (
    HERE,
    NFLREADPY_SNAPSHOT,
    NFLREADPY_SNAPSHOT_MANIFEST,
    SEASONS,
    Audit,
    SosCode,
    check_sos_checkout,
    parse_args,
    portable,
    sos_output,
    team_code_map,
)
from report_text import dump_json
from snapshot_cache import block_downloads
from sos_sources import PLAY_BY_PLAY, classify_game_log_columns

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import loaders, nflverse_sources, player_team_repair
from nfl_predictor.utils.polars.loaders import _pbp_cache_path, _prepare_pbp, _read_cached_frame
from nfl_predictor.utils.polars.pbp import (
    aggregate_pbp_team_box_score_stats,
    aggregate_pbp_team_game_stats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

RULES = HERE / "parity.yaml"
OUT_MD = HERE / "PARITY.md"
OUT_JSON = HERE / "parity.json"
OUT_ATTRIBUTIONS = HERE / "parity_attributions.parquet"
# Every audited season is history to the loaders, so they read the cache and never refresh it.
CURRENT_SEASON = SEASONS[-1] + 1
TEAM_COLUMNS = tuple(constants.PBP_TEAM_COLUMNS)
ADJUSTABLE = ("ours", "theirs")
KINDS = ("count", "count_from_rate", "sum", "rate")
ROWS = ("own", "opponent")
COUNT_KEYS = ("compared", "mismatched", "explained", "unexplained")
PBP_MARKERS = {"play_id", "game_id", "qb_scramble", "posteam", "epa", "season"}


def load_pairs(rules: dict[str, Any], audit: Audit) -> list[Pair]:
    """Build the judged pairs and check that every cause they name exists."""
    pairs = []
    for entry in rules["pairs"]:
        causes = tuple(entry["causes"])
        if entry["kind"] not in KINDS:
            audit.fail("pair", f"{entry['id']}: unknown kind {entry['kind']}")
        row = entry.get("row", "own")
        if row not in ROWS:
            audit.fail("pair", f"{entry['id']}: unknown row {row}")
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
                row=row,
            )
        )
    implemented, described = set(CAUSES), set(rules["causes"])
    for cause in sorted(implemented ^ described):
        side = "described in parity.yaml only" if cause in described else "implemented only"
        audit.fail("cause", f"{cause}: {side}")
    return pairs


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def snapshot_files(audit: Audit) -> dict[int, Any]:
    """Find each season's play-by-play file in the snapshot by its content.

    A file is play-by-play when its schema has ``PBP_MARKERS``; its season is the one value of its
    ``season`` column. Every completed season must have exactly one such file, and every file used
    must match the snapshot's sha256 manifest.
    """
    manifest: dict[str, str] = {}
    if NFLREADPY_SNAPSHOT_MANIFEST.exists():
        for line in audit.record(NFLREADPY_SNAPSHOT_MANIFEST).read_text().splitlines():
            digest, _, name = line.strip().partition("  ")
            manifest[name] = digest
    else:
        audit.fail("snapshot", f"no manifest at {portable(str(NFLREADPY_SNAPSHOT_MANIFEST))}")
    found: dict[int, list[Any]] = defaultdict(list)
    for path in sorted(NFLREADPY_SNAPSHOT.glob("*.parquet")):
        if not PBP_MARKERS <= set(pl.read_parquet_schema(path)):
            continue
        seasons = pl.read_parquet(path, columns=["season"])["season"].unique().to_list()
        if len(seasons) == 1:
            found[int(seasons[0])].append(path)
        else:
            audit.fail("snapshot", f"{path.name} holds seasons {sorted(seasons)}")
    chosen: dict[int, Any] = {}
    for season in SEASONS:
        paths = found.get(season, [])
        if len(paths) != 1:
            audit.fail("snapshot", f"{season}: {len(paths)} play-by-play files, not one")
            continue
        path = paths[0]
        audit.record(path)
        if manifest.get(path.name) != audit.files[portable(str(path))]:
            audit.fail("snapshot", f"{path.name} does not match the snapshot's manifest")
        chosen[season] = path
    return chosen


def season_pbp(
    season: int, snapshot: Path, cache_dir: Path, audit: Audit
) -> tuple[pl.DataFrame, dict[str, Any]] | None:
    """Return the season's play-by-play as the ETL aggregates it, checked against the snapshot.

    The season is read through this repository's loader from ``cache_dir`` (a cache hit only:
    every audited season is historical, and a cache file that predates the columns the code
    reads fails the run instead of downloading), then repaired as the ETL repairs it
    (``nflverse_sources.repair_one_team_pbp_games``, reading the season rosters from the same
    cache). Before the repair the snapshot's regular-season rows, prepared with this repository's
    own ``_prepare_pbp``, must equal the cached rows on every cached column, ``play_id``,
    ``qb_scramble`` and ``desc`` included, or the run fails. Returns ``None`` when the season
    cannot be read.
    """
    path = _pbp_cache_path(cache_dir, season, regular_season_only=True)
    if not path.is_file():
        audit.fail("cache", f"{season}: {portable(str(path))} does not exist")
        return None
    if _read_cached_frame(audit.record(path), tuple(constants.PBP_COLUMNS)) is None:
        audit.fail("cache", f"{season}: {portable(str(path))} predates the columns the code reads")
        return None
    cached = loaders.load_pbp([season], cache_dir=cache_dir, current_season=CURRENT_SEASON)
    schema = pl.read_parquet_schema(snapshot)
    raw = pl.read_parquet(snapshot, columns=[c for c in constants.PBP_COLUMNS if c in schema])
    prepared = _prepare_pbp(raw, regular_season_only=True)
    regular = raw.filter(pl.col("season_type") == "REG") if "season_type" in raw.columns else raw
    identical = prepared.equals(cached) and regular.height == cached.height
    if not identical:
        audit.fail("snapshot", f"{season}: the snapshot is not row by row equal to the cache")
    repaired_games = repair_games(cached)
    for roster_season in sorted({int(game[:4]) for game in repaired_games}):
        roster = nflverse_sources.ROSTERS.cache_path(
            cache_dir, roster_season, regular_season_only=False
        )
        if not roster.is_file():
            audit.fail("cache", f"{season}: the repair's {portable(str(roster))} does not exist")
            return None
        audit.record(roster)
    repaired = nflverse_sources.repair_one_team_pbp_games(
        cached, cache_dir=cache_dir, current_season=CURRENT_SEASON
    )
    report: dict[str, Any] = {
        "file": snapshot.name,
        "plays": cached.height,
        "identical_to_cache": identical,
        "cached_columns_compared": len(cached.columns),
        "repaired_games": repaired_games,
    }
    return repaired, report


def repair_games(pbp: pl.DataFrame) -> list[str]:
    """Return the games the ETL's repair rebuilds: their player team columns name one team."""
    return player_team_repair.one_team_games(pbp, list(constants.PBP_PLAYER_TEAM_ID_COLUMNS))


def ours_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return this repository's two play-by-play aggregates joined per team-game."""
    team_game = aggregate_pbp_team_game_stats(pbp)
    box_score = aggregate_pbp_team_box_score_stats(pbp).drop("season_type")
    return team_game.join(box_score, on=["season", *KEYS], how="full", coalesce=True).drop("season")


def canonical(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` with its team codes in this repository's canonical spelling."""
    codes = team_code_map(frame["team_abbr"].to_list() + frame["opponent_abbr"].to_list())
    return frame.with_columns(
        pl.col("team_abbr").replace_strict(codes),
        pl.col("opponent_abbr").replace_strict(codes),
    )


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


# ---------------------------------------------------------------------------------------------
# Comparison and attribution
# ---------------------------------------------------------------------------------------------


def compare(pair: Pair, ours: pl.DataFrame, theirs: pl.DataFrame, tolerance: float) -> pl.DataFrame:
    """Return one row per team-game on both sides with both values and whether they match.

    Both frames are keyed in canonical team codes, so a team-game one source writes ``LA`` lines up.
    An opponent-row pair reads its value from the opponent's row of the same game: that frame's team
    and opponent codes swap before the join, so the row keyed by the team holds the opponent's value.
    """
    their_value = pair.theirs.expr
    if pair.null_as_zero:
        their_value = their_value.fill_null(0.0)
    if pair.row == "opponent":
        theirs = theirs.rename({"team_abbr": "opponent_abbr", "opponent_abbr": "team_abbr"})
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
    _ours: dict[frozenset[str], pl.DataFrame] = field(default_factory=dict)

    def ours(self, causes: frozenset[str]) -> pl.DataFrame:
        """Return this repository's aggregates, canonically keyed, after ``causes``' adjustments."""
        if causes not in self._ours:
            self._ours[causes] = canonical(ours_frame(adjust(self.pbp, causes)))
        return self._ours[causes]

    def theirs(self, causes: frozenset[str]) -> pl.DataFrame:
        """Return its game logs after the ``causes`` adjustments on its side.

        No cause adjusts its side today (the one that undid its Jacksonville repair explains no
        team-game since this repository repairs those games too), so this is its game logs.
        """
        del causes
        return self.theirs_base


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
        found: dict[tuple[int, str, str], list[str]] = defaultdict(list)
        for combo in itertools.combinations(pair.causes, size):
            ours_causes, their_causes = _split(combo)
            adjusted = compare(
                pair, frames.ours(ours_causes), frames.theirs(their_causes), tolerance
            ).filter(pl.col("match"))
            keys = {(r[0], r[1], r[2]) for r in adjusted.select(KEYS).iter_rows()} & remaining
            for key in keys:
                found[key].append(" + ".join(combo))
        for key, labels in found.items():
            records[key]["causes"] = sorted(labels)
            records[key]["status"] = "explained"
        remaining -= set(found)
    return list(records.values())


def cause_effects(
    pairs: list[Pair], rules: dict[str, Any], frames: SeasonFrames
) -> dict[str, dict[str, list[float]]]:
    """Return each cause's own effect this season, alone, on its pairs and effect columns.

    The effect on a value is the adjusted value minus the unadjusted one, per team-game, on the
    side the cause adjusts: this repository's for a play-by-play adjustment, the other's for an
    undone repair. Returns cause -> value name -> [team-games changed, total, absolute total].
    """
    effects: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for cause_id, cause in CAUSES.items():
        if cause.side not in ADJUSTABLE:
            continue
        values = {
            f"{pair.id} ({'here' if cause.side == 'ours' else 'there'})": (
                pair.ours.expr if cause.side == "ours" else pair.theirs.expr
            )
            for pair in pairs
            if cause_id in pair.causes
        }
        values |= {name: pl.col(name) for name in rules["causes"][cause_id].get("effects", [])}
        if cause.side == "ours":
            base, adjusted = frames.ours(frozenset()), frames.ours(frozenset({cause_id}))
        else:
            base, adjusted = frames.theirs(frozenset()), frames.theirs(frozenset({cause_id}))
        for name, value in values.items():
            joined = base.select(*KEYS, value.alias("base")).join(
                adjusted.select(*KEYS, value.alias("adjusted")), on=KEYS, how="inner"
            )
            delta = (pl.col("adjusted").fill_null(0.0) - pl.col("base").fill_null(0.0)).alias("d")
            change = joined.select(delta)["d"]
            effects[cause_id][name] = [
                float((change.abs() > 0).sum()),
                float(change.sum()),
                float(change.abs().sum()),
            ]
    return effects


# ---------------------------------------------------------------------------------------------
# Keys and coverage
# ---------------------------------------------------------------------------------------------


def check_keys(
    season: int, ours_raw: pl.DataFrame, theirs: pl.DataFrame, pbp: pl.DataFrame, audit: Audit
) -> dict[str, Any]:
    """Compare raw team-game keys; a one-sided key that lines up in canonical codes is re-keyed.

    The comparison itself joins on canonical keys (``canonical``), so a re-keyed team-game is
    compared like any other.
    """
    our_keys = {tuple(row) for row in ours_raw.select(KEYS).iter_rows()}
    their_keys = {tuple(row) for row in theirs.select(KEYS).iter_rows()}
    codes = team_code_map([k[i] for k in our_keys | their_keys for i in (1, 2)])

    def canonical_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        return (key[0], codes[key[1]], codes[key[2]])

    ours_only, theirs_only = our_keys - their_keys, their_keys - our_keys
    rekeyed = {canonical_key(k) for k in ours_only} & {canonical_key(k) for k in theirs_only}
    unpaired = sorted(
        str(key) for key in (ours_only | theirs_only) if canonical_key(key) not in rekeyed
    )
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


def check_opponent_rows(season: int, ours: pl.DataFrame, theirs: pl.DataFrame, audit: Audit) -> int:
    """Fail every team-game of ``ours`` whose opponent's row ``theirs`` lacks; return how many.

    An opponent-row pair reads each team-game's value from the opponent's row (``compare``), whose
    inner join would drop such a team-game without a word. Both frames are canonically keyed.
    """
    their_keys = {tuple(row) for row in theirs.select(KEYS).iter_rows()}
    missing = sorted(
        str(key)
        for key in (tuple(row) for row in ours.select(KEYS).iter_rows())
        if (key[0], key[2], key[1]) not in their_keys
    )
    for key in missing:
        audit.fail("team-game keys", f"{season} {key} has no opponent row on their side")
    return len(missing)


def check_coverage(
    cached_schedule: pl.DataFrame,
    ours: pl.DataFrame,
    theirs: pl.DataFrame,
    pbp: pl.DataFrame,
    expected_missing: set[str],
    audit: Audit,
) -> dict[str, Any]:
    """Check every played regular-season game of the season's cached schedule against both sides.

    An empty schedule names no season to check, so it fails the audit.
    """
    if cached_schedule.is_empty():
        audit.fail("coverage", "the cached schedule has no games")
        return {"played_games": 0, "games_without_play_by_play": [], "unexpected": []}
    season = int(cached_schedule["season"][0])
    schedule = cached_schedule.filter(
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


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity check and write the reports, with every nflverse download blocked."""
    args = parse_args((__doc__ or "").splitlines()[0], argv)
    block_downloads()
    code = SosCode()
    try:
        return _run(code, args.data_dir)
    finally:
        code.cleanup()


def _run(code: SosCode, data_dir: Path) -> int:
    """Run the parity check on ``data_dir``'s play-by-play cache and write the reports."""
    audit = Audit()
    cache_dir = data_dir / "cache" / "nflreadpy"
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

    log_columns = list(pl.read_parquet_schema(sos_output(SEASONS[0], "team_game_logs")))
    sources, source_facts = classify_game_log_columns(code, log_columns, audit)
    snapshots = snapshot_files(audit)

    code_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    per_pair: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seasons_report: dict[int, dict[str, Any]] = {}
    effects: dict[str, dict[str, list[Any]]] = defaultdict(dict)
    for season in SEASONS:
        if season not in snapshots:
            continue
        log.info("Season %d", season)
        read = season_pbp(season, snapshots[season], cache_dir, audit)
        if read is None:
            continue
        pbp, snapshot_report = read
        theirs = theirs_frame(season, audit, code_map)
        if list(theirs.columns[: len(log_columns)]) != log_columns:
            audit.fail("game-log columns", f"{season} columns differ from {SEASONS[0]}'s")
        frames = SeasonFrames(season, pbp, theirs)
        base_ours = frames.ours(frozenset())
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
        keys = check_keys(season, ours_frame(pbp), theirs, pbp, audit)
        schedule = pl.read_parquet(audit.record(cache_dir / f"schedule_{season}.parquet"))
        coverage = check_coverage(schedule, base_ours, theirs, pbp, expected_missing, audit)
        if any(pair.row == "opponent" for pair in pairs):
            check_opponent_rows(season, base_ours, theirs, audit)
        season_pairs = {}
        for pair in pairs:
            base = compare(pair, base_ours, theirs, tolerance)
            found = attribute(pair, frames, base, tolerance)
            records.extend(found)
            counts = per_pair[pair.id]
            counts["compared"] += base.height
            counts["mismatched"] += len(found)
            for record in found:
                counts[record["status"]] += 1
            season_pairs[pair.id] = {"compared": base.height, "mismatched": len(found)}
        for cause_id, values in cause_effects(pairs, rules, frames).items():
            for name, (changed, total, absolute) in values.items():
                running = effects[cause_id].setdefault(name, [0.0, 0.0, 0.0, []])
                running[0] += changed
                running[1] += total
                running[2] += absolute
                if changed:
                    running[3].append(season)
        seasons_report[season] = {
            "snapshot": snapshot_report,
            "keys": keys,
            "coverage": coverage,
            "repaired_games": snapshot_report.pop("repaired_games"),
            "wiped_out_plays": wiped_out_plays(pbp),
            "pairs": season_pairs,
        }

    order = {pair.id: index for index, pair in enumerate(pairs)}
    records.sort(key=lambda r: (order[r["pair"]], r["season"], r["week"], r["team"]))
    for record in records:
        if record["status"] == "unexplained":
            audit.fail("unexplained", _describe(record))

    payload = {
        "generated_by": (
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/parity.py "
            f"--data-dir {portable(str(data_dir))}"
        ),
        "data_dir": portable(str(data_dir)),
        "tolerance": tolerance,
        "sos_checkout": checkout,
        "team_code_map": dict(sorted(code_map.items())),
        "snapshot": {
            "directory": str(NFLREADPY_SNAPSHOT),
            "manifest": str(NFLREADPY_SNAPSHOT_MANIFEST),
        },
        "repaired_games": sorted(
            game for report in seasons_report.values() for game in report["repaired_games"]
        ),
        "game_log_sources": {
            column: {"source": sources[column].source, "via": sources[column].via}
            for column in log_columns
            if column in sources
        },
        "game_log_source_facts": source_facts,
        "pairs": pair_entries(pairs, per_pair),
        "causes": summarize_causes(rules, records, seasons_report, effects),
        "evidence": evidence,
        "seasons": {str(season): report for season, report in seasons_report.items()},
        "inputs": audit.files,
        "failures": [{"what": f.what, "detail": f.detail} for f in audit.failures],
    }
    write_outputs(payload, records)
    OUT_MD.write_text(portable(render(payload, rules, records)), encoding="utf-8")
    log.info("%d mismatching pair values, %d failures", len(records), len(audit.failures))
    return 1 if audit.failures else 0


def pair_entries(pairs: list[Pair], per_pair: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Return the report's entry for each judged pair: its definition, its row and its counts."""
    return [
        {
            "id": pair.id,
            "ours": pair.ours.text,
            "theirs": pair.theirs.text,
            "row": pair.row,
            "kind": pair.kind,
            "causes": list(pair.causes),
            **{k: per_pair.get(pair.id, {}).get(k, 0) for k in COUNT_KEYS},
        }
        for pair in pairs
    ]


def _describe(record: dict[str, Any]) -> str:
    """Return a one-line description of a mismatching team-game."""
    return (
        f"{record['pair']} {record['game_id']} {record['team']}: here {record['ours']}, "
        f"there {record['theirs']}"
    )


# Per-season counts summed into the ``wiped_out_run_epa`` cause: the plays a penalty wiped out.
WIPED_OUT_DIAGNOSTICS = ("wiped_out_plays",)


def _season_sums(seasons: dict[int, dict[str, Any]], name: str) -> dict[str, float]:
    """Sum the per-season counts under ``name`` over every season, key by key (missing keys 0)."""
    keys = dict.fromkeys(key for season in seasons.values() for key in season[name])
    return {key: round(sum(s[name].get(key, 0) for s in seasons.values()), 6) for key in keys}


def summarize_causes(
    rules: dict[str, Any],
    records: list[dict[str, Any]],
    seasons: dict[int, dict[str, Any]],
    effects: dict[str, dict[str, list[Any]]],
) -> dict[str, Any]:
    """Return, per cause, what it explains (attribution) and what it changes on its own (effect)."""
    summary: dict[str, Any] = {}
    for cause_id, cause in rules["causes"].items():
        mine = [r for r in records if any(cause_id in c.split(" + ") for c in r["causes"])]
        columns: dict[str, Any] = {}
        for pair_id in sorted({r["pair"] for r in mine}):
            rows = [r for r in mine if r["pair"] == pair_id]
            columns[pair_id] = {
                "team_games": len(rows),
                "alone": sum(cause_id in r["causes"] for r in rows),
                "shared_with_another_cause": sum(len(r["causes"]) > 1 for r in rows),
                "seasons": sorted({r["season"] for r in rows}),
            }
        entry: dict[str, Any] = {
            "name": cause["name"],
            "team_games": len({(r["season"], r["week"], r["team"]) for r in mine}),
            "games": len({(r["season"], r["game_id"]) for r in mine}),
            "seasons": sorted({r["season"] for r in mine}),
            "columns": columns,
            "effect": {
                name: {
                    "team_games_changed": int(values[0]),
                    "adjusted_minus_unadjusted": round(values[1], 6),
                    "absolute": round(values[2], 6),
                    "seasons": sorted(set(values[3])),
                }
                for name, values in sorted(effects.get(cause_id, {}).items())
            },
        }
        if cause_id == "games_without_play_by_play":
            games = [
                g for s in seasons.values() for g in s["coverage"]["games_without_play_by_play"]
            ]
            entry.update(games=len(games), team_games=2 * len(games), listed=sorted(games))
            entry["seasons"] = sorted({int(g[:4]) for g in games})
        if cause_id == "wiped_out_run_epa":
            entry.update((name, _season_sums(seasons, name)) for name in WIPED_OUT_DIAGNOSTICS)
        if cause_id == "rams_written_la":
            rekeyed = [k for s in seasons.values() for k in s["keys"]["rekeyed_rams"]]
            entry.update(
                team_games=len(rekeyed),
                cached_la_values=sum(s["keys"]["cached_la_values"] for s in seasons.values()),
            )
        if cause_id == "release_difference":
            snapshots = [s["snapshot"] for s in seasons.values()]
            entry.update(
                plays_compared=sum(s["plays"] for s in snapshots),
                seasons_identical=sum(s["identical_to_cache"] for s in snapshots),
                seasons_compared=len(snapshots),
                cached_columns_compared=max(
                    (s["cached_columns_compared"] for s in snapshots), default=0
                ),
            )
        summary[cause_id] = entry
    return summary


ATTRIBUTION_COLUMNS = [
    "season",
    "game_id",
    "week",
    "team",
    "opponent",
    "pair",
    "ours",
    "theirs",
    "diff",
    "status",
    "causes",
]


def write_outputs(payload: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write ``parity.json`` and every mismatching team-game to ``parity_attributions.parquet``.

    A team-game's ``causes`` are the cause combinations that make it match with the fewest
    causes, separated by `` | ``; ``parity.json`` lists the unexplained ones too.
    """
    frame = pl.DataFrame(
        [[record[column] for column in ATTRIBUTION_COLUMNS] for record in records],
        schema={
            "season": pl.Int64,
            "game_id": pl.String,
            "week": pl.Int64,
            "team": pl.String,
            "opponent": pl.String,
            "pair": pl.String,
            "ours": pl.Float64,
            "theirs": pl.Float64,
            "diff": pl.Float64,
            "status": pl.String,
            "causes": pl.List(pl.String),
        },
        orient="row",
    ).with_columns(pl.col("causes").list.join(" | "))
    frame.write_parquet(OUT_ATTRIBUTIONS, statistics=False)
    unexplained = [
        [_rounded(record[column]) for column in ATTRIBUTION_COLUMNS[:-2]]
        for record in records
        if record["status"] == "unexplained"
    ]
    body = {
        **payload,
        "attributions": {
            "file": OUT_ATTRIBUTIONS.name,
            "rows": frame.height,
            "columns": ATTRIBUTION_COLUMNS,
        },
        "unexplained": {"columns": ATTRIBUTION_COLUMNS[:-2], "rows": unexplained},
    }
    OUT_JSON.write_text(portable(dump_json(body)) + "\n", encoding="utf-8")


def _rounded(value: object) -> object:
    """Round floats so the report is stable across platforms."""
    return round(value, 9) if isinstance(value, float) else value


if __name__ == "__main__":
    sys.exit(main())
