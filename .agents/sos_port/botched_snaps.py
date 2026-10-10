r"""What the botched-snap charge rule changes in the ported quarterback rating: a measurement.

The plan is ``protocols/botched_snaps.md``, written before measuring. The rule (the
``nfl-sos-ratings`` maintainer's, approved 2026-10-09): a botched snap counts against a
quarterback only when the official scorer charged the fumble to him (``fumbled_1_player_id``).
The approved change removes from the rating the aborted snaps the scorer charged to a center or
other lineman that went on as a pass or sack, which the ported rating counts today as dropbacks of
the passer. Nothing is adopted here: the script refits the rating with and without those plays and
reports how far it moves (full-season ratings and ranks, the weekly history, and the pre-week
ratings the model reads as ``ratings_qb``).

Run from the repository root, without ``-I`` (it imports ``rating_tests_common`` from its own
directory):

    VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/botched_snaps.py \
        --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy

It reads the play-by-play and nflverse player sources through this repository's loaders with
downloads blocked, and writes only ``BOTCHED_SNAPS.md`` and ``botched_snaps.json`` beside itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rating_tests_common as common
from threadpoolctl import threadpool_limits

from nfl_predictor.ratings.inputs import team_game_rows
from nfl_predictor.ratings.pbp_inputs import (
    passer_games_from_pbp,
    quarterback_rows,
    team_games_from_pbp,
)
from nfl_predictor.ratings.pre_week import RatingsHistory
from nfl_predictor.ratings.qb_rating import (
    QB_DROPBACKS_COLUMN,
    QB_ID_COLUMN,
    QB_RATING_COLUMN,
    fit_qb_ratings,
    fit_qb_ratings_by_week,
    qb_season_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

HERE = Path(__file__).resolve().parent
OUT_MD = "BOTCHED_SNAPS.md"
OUT_JSON = "botched_snaps.json"
FIRST_SEASON = 1999
APPROVED = (2006, 2025)
LAST_SEASON = 2026
LINE_POSITIONS = ("C", "G", "T", "OL", "OT", "OG")
GAME_KEYS = ["game_id", "team", "qb_id"]
MOVED = 1e-9
PRE_WEEK_MOVED = 1e-12
# The source's count for the approved change (its aborted_snaps.py 2006 2025).
SOURCE_COUNT = (60, -21.3)
_REMOVAL_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "team": pl.String(),
        "qb_id": pl.String(),
        "plays": pl.Int64(),
        "qb_epa": pl.Float64(),
        "epa": pl.Float64(),
    }
)


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def classify_aborted(
    plays: pl.DataFrame, qb_rows: pl.DataFrame, positions: pl.DataFrame
) -> pl.DataFrame:
    """Return a season's aborted plays with the charged player's and the rusher's roles.

    Aborted plays are flagged ``aborted_play``, not two-point tries, not wiped out by a penalty,
    with the offense known. A role is ``qb`` (a quarterback row of that team and game, or listed at
    QB in the players file), ``line`` (for the charged player only: listed at C, G, T, OL, OT or
    OG), ``other``, or ``none`` (no player), the source helper's classes.
    """
    aborted = plays.filter(
        _flag("aborted_play")
        & ~_flag("two_point_attempt")
        & (pl.col("play_type") != "no_play").fill_null(value=True)
        & pl.col("posteam").is_not_null()
    )
    rows = qb_rows.select(GAME_KEYS).unique().with_columns(pl.lit(value=True).alias("_row"))
    places = positions.unique(subset=["player_id"], keep="first", maintain_order=True)

    def role(column: str, alias: str, *, line: bool) -> pl.Series:
        joined = (
            aborted.select("game_id", "posteam", column)
            .join(
                rows,
                left_on=["game_id", "posteam", column],
                right_on=GAME_KEYS,
                how="left",
                maintain_order="left",
            )
            .join(places, left_on=column, right_on="player_id", how="left", maintain_order="left")
        )
        is_qb = pl.col("_row").fill_null(value=False) | (pl.col("position") == "QB").fill_null(
            value=False
        )
        on_line = (
            pl.col("position").is_in(LINE_POSITIONS).fill_null(value=False)
            if line
            else pl.lit(value=False)
        )
        return joined.select(
            pl.when(pl.col(column).is_null())
            .then(pl.lit("none"))
            .when(is_qb)
            .then(pl.lit("qb"))
            .when(on_line)
            .then(pl.lit("line"))
            .otherwise(pl.lit("other"))
            .alias(alias)
        ).to_series()

    return aborted.with_columns(
        role("fumbled_1_player_id", "charged", line=True),
        role("rusher_player_id", "rusher", line=False),
    )


def lineman_dropbacks(classified: pl.DataFrame, qb_rows: pl.DataFrame) -> pl.DataFrame:
    """Return the approved change's plays per passer-game: a lineman's snap that became a dropback.

    The scorer charged a lineman, no one ran with the ball, and the play is one of the rating's
    dropbacks: ``qb_dropback`` set and the passer a quarterback row of that team and game with a
    dropback. Columns ``game_id``, ``team``, ``qb_id``, ``plays``, ``qb_epa`` and ``epa``.
    """
    fit_rows = qb_rows.filter(pl.col("qb_dropbacks") > 0).select(GAME_KEYS)
    chosen = classified.filter(
        (pl.col("charged") == "line")
        & (pl.col("rusher") == "none")
        & _flag("qb_dropback")
        & pl.col("passer_player_id").is_not_null()
    )
    if chosen.is_empty():
        return pl.DataFrame(schema=_REMOVAL_SCHEMA)
    return (
        chosen.join(
            fit_rows,
            left_on=["game_id", "posteam", "passer_player_id"],
            right_on=GAME_KEYS,
            how="semi",
        )
        .group_by("game_id", "posteam", "passer_player_id", maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias("plays"),
            pl.col("qb_epa").fill_null(0.0).sum().alias("qb_epa"),
            pl.col("epa").fill_null(0.0).sum().alias("epa"),
        )
        .rename({"posteam": "team", "passer_player_id": "qb_id"})
        .select(list(_REMOVAL_SCHEMA))
        .cast(_REMOVAL_SCHEMA)
    )


def remove_plays(qb_rows: pl.DataFrame, removals: pl.DataFrame) -> pl.DataFrame:
    """Return the quarterback rows with each removed play's dropback and ``qb_epa`` taken out."""
    joined = qb_rows.join(
        removals.select(*GAME_KEYS, "plays", "qb_epa"),
        on=GAME_KEYS,
        how="left",
        maintain_order="left",
    )
    return joined.with_columns(
        (pl.col("qb_dropbacks") - pl.col("plays").fill_null(0))
        .cast(qb_rows.schema["qb_dropbacks"])
        .alias("qb_dropbacks"),
        (pl.col("qb_dropback_epa") - pl.col("qb_epa").fill_null(0.0))
        .cast(qb_rows.schema["qb_dropback_epa"])
        .alias("qb_dropback_epa"),
    ).select(qb_rows.columns)


def _ranks(ratings: pl.DataFrame, qualified: Sequence[str]) -> dict[str, int]:
    """Return each qualified passer's rank on the rating, best first."""
    ranked = ratings.filter(pl.col(QB_ID_COLUMN).is_in(list(qualified))).with_columns(
        pl.col(QB_RATING_COLUMN).rank("min", descending=True).cast(pl.Int64).alias("rank")
    )
    return dict(ranked.select(QB_ID_COLUMN, "rank").iter_rows())


def rating_movement(
    before: pl.DataFrame, after: pl.DataFrame, *, qualified: Sequence[str]
) -> dict[str, Any]:
    """Return how far a season's full-season rating moves when ``before`` becomes ``after``.

    Each side is fit as published: ``fit_qb_ratings`` with its own cross-validated penalty.
    """
    fit_before, fit_after = fit_qb_ratings(before), fit_qb_ratings(after)
    joined = fit_before.ratings.join(fit_after.ratings, on=QB_ID_COLUMN, suffix="_after")
    change = (pl.col(QB_RATING_COLUMN) - pl.col(f"{QB_RATING_COLUMN}_after")).abs()
    ranks_before = _ranks(fit_before.ratings, qualified)
    ranks_after = _ranks(fit_after.ratings, qualified)
    rank_moves = [abs(ranks_before[passer] - ranks_after[passer]) for passer in ranks_before]
    return {
        "passers": joined.height,
        "moved": joined.filter(change > MOVED).height,
        "largest": common.as_float(joined.select(change.max()).item()),
        "penalty_before": fit_before.ridge_lambda,
        "penalty_after": fit_after.ridge_lambda,
        "qualified": len(ranks_before),
        "rank_changes": sum(1 for move in rank_moves if move),
        "largest_rank_change": max(rank_moves, default=0),
    }


def history_movement(before: pl.DataFrame, after: pl.DataFrame) -> float:
    """Return the largest change in the published weekly history, each side with its own fit."""
    history_before = fit_qb_ratings_by_week(before, fit_qb_ratings(before))
    history_after = fit_qb_ratings_by_week(after, fit_qb_ratings(after))
    joined = history_before.join(history_after, on=["week", QB_ID_COLUMN], suffix="_after")
    return common.as_float(
        joined.select(
            (pl.col(QB_RATING_COLUMN) - pl.col(f"{QB_RATING_COLUMN}_after")).abs().max()
        ).item()
    )


# --- the measurement -------------------------------------------------------------------------


@dataclass
class Seasons:
    """Every season's team rows, passer-games, removals and the plays behind them."""

    team_rows: dict[int, pl.DataFrame]
    passers: dict[int, pl.DataFrame]
    removals: dict[int, pl.DataFrame]
    plays: dict[int, pl.DataFrame]
    others: dict[int, dict[str, float]]


def _other_charges(classified: pl.DataFrame) -> dict[str, float]:
    """Return the aborted snaps that went on as a pass or sack charged to another or to no one."""
    passes = classified.filter(
        (pl.col("rusher") == "none")
        & _flag("qb_dropback")
        & pl.col("passer_player_id").is_not_null()
    )
    result: dict[str, float] = {}
    for role in ("other", "none"):
        chosen = passes.filter(pl.col("charged") == role)
        result[f"{role}_plays"] = float(chosen.height)
        result[f"{role}_epa"] = common.as_float(chosen.get_column("epa").fill_null(0.0).sum())
    return result


def load_seasons(cache_dir: Path, ledger: common.InputLedger) -> Seasons:
    """Load every season through this repository's loaders and find the plays to remove."""
    common.block_downloads()
    players = common.load_players(cache_dir, ledger)
    positions = players.select(pl.col("gsis_id").alias("player_id"), "position").drop_nulls(
        "player_id"
    )
    seasons = Seasons(team_rows={}, passers={}, removals={}, plays={}, others={})
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        sources, schedule = common.load_season_sources(cache_dir, season, players, ledger)
        team = team_games_from_pbp(sources.plays, schedule)
        seasons.team_rows[season] = team_game_rows(team)
        passers = passer_games_from_pbp(sources, team.drop("season"))
        seasons.passers[season] = passers
        qb_rows = quarterback_rows(passers)
        classified = classify_aborted(sources.plays, qb_rows, positions)
        removals = lineman_dropbacks(classified, qb_rows)
        seasons.removals[season] = removals
        seasons.others[season] = _other_charges(classified)
        chosen = classified.join(
            removals.select("game_id", pl.col("team").alias("posteam"), pl.col("qb_id")),
            left_on=["game_id", "posteam", "passer_player_id"],
            right_on=["game_id", "posteam", "qb_id"],
            how="semi",
        ).filter((pl.col("charged") == "line") & (pl.col("rusher") == "none"))
        seasons.plays[season] = chosen.select(
            pl.lit(season).alias("season"),
            "game_id",
            "play_id",
            "posteam",
            "passer_player_id",
            "fumbled_1_player_id",
            "epa",
            "qb_epa",
            pl.col("desc").str.slice(0, 110).alias("desc"),
        )
    return seasons


def approved_removals(seasons: Seasons, season: int) -> pl.DataFrame:
    """Return the plays the approved change removes in ``season``: none outside 2006-2025."""
    if APPROVED[0] <= season <= APPROVED[1]:
        return seasons.removals[season]
    return pl.DataFrame(schema=_REMOVAL_SCHEMA)


def _history(seasons: Seasons, *, removed: bool) -> RatingsHistory:
    """Return the pre-week ratings' history with or without the approved change's plays."""

    def adjust(rows: pl.DataFrame, season: int) -> pl.DataFrame:
        return remove_plays(rows, approved_removals(seasons, season)) if removed else rows

    def whole(season: int) -> pl.DataFrame:
        return adjust(quarterback_rows(seasons.passers[season]), season)

    def before(season: int, week: int) -> pl.DataFrame:
        return adjust(quarterback_rows(seasons.passers[season], before_week=week), season)

    return RatingsHistory(
        lambda season: seasons.team_rows[season],
        whole,
        first_season=FIRST_SEASON,
        load_qb_games_before=before,
    )


def pre_week_movement(
    before: RatingsHistory, after: RatingsHistory, season: int, weeks: Sequence[int]
) -> dict[str, float]:
    """Return how many pre-week passer rows a season's weeks move, and the largest change."""
    rows = moved = dropbacks_moved = 0
    largest = 0.0
    for week in weeks:
        old = before.pre_week_quarterbacks(season, week)
        new = after.pre_week_quarterbacks(season, week)
        joined = old.join(new, on=QB_ID_COLUMN, how="full", suffix="_after", coalesce=True)
        change = (pl.col(QB_RATING_COLUMN) - pl.col(f"{QB_RATING_COLUMN}_after")).abs()
        missing = pl.col(QB_RATING_COLUMN).is_null() | pl.col(f"{QB_RATING_COLUMN}_after").is_null()
        rows += joined.height
        moved += joined.filter(missing | (change > PRE_WEEK_MOVED)).height
        dropbacks_moved += joined.filter(
            missing | (pl.col(QB_DROPBACKS_COLUMN) != pl.col(f"{QB_DROPBACKS_COLUMN}_after"))
        ).height
        largest = max(largest, common.as_float(joined.select(change.max()).item()))
    return {
        "rows": float(rows),
        "moved": float(moved),
        "dropbacks_moved": float(dropbacks_moved),
        "largest": largest,
    }


def _weeks(team_rows: pl.DataFrame) -> list[int]:
    """Return the weeks whose pre-week ratings read games: 2 through the week after the last."""
    last = int(common.as_float(team_rows.get_column("week").max()))
    return list(range(2, last + 2))


def measure(seasons: Seasons) -> list[dict[str, Any]]:
    """Return each season's plays the rule names and how far the approved change moves the rating.

    Every season counts its plays; the movement is measured with the approved change, which
    removes the 2006-2025 plays only: the full-season rating and its history move with the
    season's own plays (each uses its own penalty), the pre-week ratings also with the previous
    season's (their penalty), so 2026's pre-week ratings move too.
    """
    before, after = _history(seasons, removed=False), _history(seasons, removed=True)
    table: list[dict[str, Any]] = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        removals = seasons.removals[season]
        record: dict[str, Any] = {
            "season": season,
            "plays": int(removals.get_column("plays").sum()),
            "epa": common.as_float(removals.get_column("epa").sum()),
            "qb_epa": common.as_float(removals.get_column("qb_epa").sum()),
            "passers": removals.get_column("qb_id").n_unique(),
            **{f"other_{key}": value for key, value in seasons.others[season].items()},
        }
        applied = approved_removals(seasons, season)
        if applied.height:
            rows = quarterback_rows(seasons.passers[season])
            adjusted = remove_plays(rows, applied)
            summary = qb_season_summary(rows, seasons.team_rows[season])
            qualified = summary.filter(pl.col("qb_is_eligible")).get_column(QB_ID_COLUMN)
            movement = rating_movement(rows, adjusted, qualified=qualified.to_list())
            record |= {f"season_{key}": value for key, value in movement.items()}
            record["history_largest"] = history_movement(rows, adjusted)
        if APPROVED[0] <= season <= APPROVED[1] + 1:
            movement = pre_week_movement(before, after, season, _weeks(seasons.team_rows[season]))
            record |= {f"pre_week_{key}": value for key, value in movement.items()}
        table.append(record)
    return table


# --- the report ------------------------------------------------------------------------------


def _total(table: Sequence[Mapping[str, Any]], key: str, seasons: tuple[int, int]) -> float:
    """Return the sum of ``key`` over the seasons in the inclusive range."""
    return sum(
        float(row.get(key, 0.0) or 0.0)
        for row in table
        if seasons[0] <= row["season"] <= seasons[1]
    )


def _largest(table: Sequence[Mapping[str, Any]], key: str) -> float:
    """Return the largest ``key`` over the seasons that have it."""
    return max((float(row[key]) for row in table if key in row), default=0.0)


def headline(table: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the headline bullets."""
    approved = APPROVED
    plays = int(_total(table, "plays", approved))
    epa = _total(table, "epa", approved)
    qb_epa = _total(table, "qb_epa", approved)
    moved = int(_total(table, "season_moved", approved))
    rank_changes = int(_total(table, "season_rank_changes", approved))
    pre_rows = int(_total(table, "pre_week_rows", (approved[0], approved[1] + 1)))
    pre_moved = int(_total(table, "pre_week_moved", (approved[0], approved[1] + 1)))
    pre_dropbacks = int(_total(table, "pre_week_dropbacks_moved", (approved[0], approved[1] + 1)))
    return [
        (
            f"- The approved change removes {plays} plays of 2006-2025 ({epa:+.1f} EPA by `epa`, "
            f"{qb_epa:+.1f} by `qb_epa`, the rating's EPA), against the source's count of "
            f"{SOURCE_COUNT[0]} at {SOURCE_COUNT[1]:+.1f}."
        ),
        (
            "- Full-season ratings (each season's own penalty, as published): "
            f"{moved} passer-seasons move by more than {MOVED:g}, by at most "
            f"{_largest(table, 'season_largest'):.4f} EPA per dropback; {rank_changes} ranks "
            "among qualified passers change, by at most "
            f"{int(_largest(table, 'season_largest_rank_change'))}."
        ),
        (
            "- The published weekly history moves by at most "
            f"{_largest(table, 'history_largest'):.4f}."
        ),
        (
            "- Pre-week ratings (the model's `ratings_qb` and `ratings_qb_dropbacks`), weeks of "
            f"2006-2026: {pre_moved:,} of {pre_rows:,} passer-weeks move beyond "
            f"{PRE_WEEK_MOVED:g}, by at most {_largest(table, 'pre_week_largest'):.4f}; "
            f"{pre_dropbacks:,} passer-weeks change their dropback count. A season's removals "
            "also move the next season's pre-week ratings through its penalty."
        ),
    ]


def _season_table(table: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the per-season table."""
    columns = (
        ("season", "season", "{}"),
        ("plays", "plays", "{}"),
        ("epa", "epa", "{:+.2f}"),
        ("qb_epa", "qb_epa", "{:+.2f}"),
        ("passers moved", "season_moved", "{:.0f}"),
        ("largest change", "season_largest", "{:.4f}"),
        ("penalty before", "season_penalty_before", "{:.3f}"),
        ("penalty after", "season_penalty_after", "{:.3f}"),
        ("rank changes", "season_rank_changes", "{:.0f}"),
        ("history largest", "history_largest", "{:.4f}"),
        ("pre-week moved", "pre_week_moved", "{:.0f}"),
        ("pre-week rows", "pre_week_rows", "{:.0f}"),
        ("pre-week largest", "pre_week_largest", "{:.4f}"),
    )
    lines = [
        "| " + " | ".join(title for title, _, _ in columns) + " |",
        "|" + " --- |" * len(columns),
    ]
    for row in table:
        cells = [
            form.format(row[key]) if key in row and row[key] is not None else "-"
            for _, key, form in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _plays_table(plays: pl.DataFrame) -> list[str]:
    """Return a table of every play the approved change removes."""
    lines = [
        "| season | game | play | passer | charged | epa | qb_epa | text |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in plays.iter_rows(named=True):
        text = str(row["desc"]).replace("|", "/").strip()
        lines.append(
            f"| {row['season']} | {row['game_id']} | {int(row['play_id'])} | "
            f"{row['passer_player_id']} | {row['fumbled_1_player_id']} | {row['epa']:+.3f} | "
            f"{row['qb_epa']:+.3f} | {text} |"
        )
    return lines


def build_report(
    table: Sequence[Mapping[str, Any]], plays: pl.DataFrame, provenance: Mapping[str, Any]
) -> list[str]:
    """Return the report's lines, wrapped for the Markdown linter."""
    outside = [row for row in table if not APPROVED[0] <= row["season"] <= APPROVED[1]]
    other_plays = int(_total(table, "other_other_plays", APPROVED))
    none_plays = int(_total(table, "other_none_plays", APPROVED))
    lines = [
        "# The botched-snap charge rule in the quarterback rating: measurement",
        "",
        (
            "Planned in `protocols/botched_snaps.md` before measuring. The rule: a botched snap "
            "counts against a quarterback only when the official scorer charged the fumble to "
            "him. The approved change removes the aborted snaps of 2006-2025 the scorer charged "
            "to a lineman that went on as a pass or sack; the ported rating counts them today as "
            "the passer's dropbacks. Nothing is adopted here."
        ),
        "",
        "## Headline",
        "",
        *headline(table),
        "",
        "## By season",
        "",
        (
            "Movement is measured with the approved change (2006-2025 plays only); 1999-2005 and "
            "2026 are counted but not removed. A dash marks a part the season's change does not "
            "reach (no play removed that season, or no pre-week week measured)."
        ),
        "",
        *_season_table(table),
        "",
        "## The plays the approved change removes",
        "",
        *_plays_table(plays.filter(pl.col("season").is_between(*APPROVED))),
        "",
        "## Outside the approved change (descriptive)",
        "",
        "- The same plays outside 2006-2025, which the rule as written also reaches: "
        + "; ".join(f"{row['season']} {row['plays']}" for row in outside if row["plays"])
        + ".",
        (
            "- Aborted snaps of 2006-2025 that went on as a pass or sack, charged to another "
            f"player ({other_plays}) or to no one ({none_plays}): the rule's wording would remove "
            "these too; the approved change names only the lineman's."
        ),
        (
            "- Attempts stay as they are (they come from the official statistics and decide only "
            "the rankings qualifier)."
        ),
        "",
        "## Provenance",
        "",
        f"- nfl-predictor {provenance['version']}; code SHA-256: "
        + ", ".join(f"`{name}` {digest[:12]}" for name, digest in provenance["code"].items())
        + ".",
        (
            f"- Inputs: {provenance['inputs']} files, fingerprint `{provenance['fingerprint']}` "
            f"(each file's SHA-256 is in `{OUT_JSON}`), read only, downloads blocked."
        ),
        f"- Command: `{provenance['command']}`; {provenance['seconds']:.0f} seconds.",
        "",
    ]
    return common.wrap_markdown(lines)


def run(cache_dir: Path, out_dir: Path) -> int:
    """Measure the rule and write the report; return the exit code."""
    start = time.perf_counter()
    ledger = common.InputLedger()
    with threadpool_limits(limits=1, user_api="blas"):
        seasons = load_seasons(cache_dir, ledger)
        table = measure(seasons)
    plays = pl.concat(list(seasons.plays.values())).sort("season", "game_id", "play_id")
    code = [HERE / "botched_snaps.py", HERE / "rating_tests_common.py"]
    provenance = {
        "version": metadata.version("nfl-predictor"),
        "code": {path.name: common.sha256_file(path) for path in code},
        "inputs": len(ledger.files),
        "fingerprint": ledger.fingerprint(),
        "command": 'VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python '
        f".agents/sos_port/botched_snaps.py --cache-dir {cache_dir}",
        "seconds": time.perf_counter() - start,
    }
    lines = build_report(table, plays, provenance)
    (out_dir / OUT_MD).write_text("\n".join(lines).rstrip("\n") + "\n")
    payload = {
        "seasons": table,
        "plays": plays.to_dicts(),
        "provenance": {**provenance, "input_files": ledger.files},
    }
    (out_dir / OUT_JSON).write_text(json.dumps(payload, indent=1, default=_json_default) + "\n")
    return 0


def _json_default(value: object) -> object:
    """Return a JSON-friendly form of a value ``json`` cannot write."""
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True, help="nflreadpy cache, read only.")
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Where the report goes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement."""
    args = _parse_args(argv)
    return run(args.cache_dir, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
