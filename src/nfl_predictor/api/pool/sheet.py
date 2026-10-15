"""The confidence pool sheet: a week's picks with the points to assign, scored once games end.

The sheet reads the confidence ranks the predictions already carry and never re-ranks: the weekly
run and the ``predict`` command assign each week's unique ranks ``1..N`` from the unrounded win
probability, and the pool is scored by the repository's rules (``README.md``, "Confidence pool
rules"): a correct pick earns its rank in points, and a tie scores as incorrect for both sides.
Picks are single-shot before the week's first game, so the sheet names that kickoff and flags
predictions written after it. When a week's predictions were rewritten after the lock (a lines
refresh chains a new prediction) and the weekly run's picks file was written before it, the sheet
and its points come from that picks file, with the newer ranks only as a note.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from nfl_predictor import constants
from nfl_predictor.api.pool.kickoff import first_kickoff, kickoff_eastern
from nfl_predictor.api.readers.market import is_number

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    import polars as pl

Result = Literal["win", "loss", "tie"]
SCHEDULE_COLUMNS = ("game_datetime", "date", "gametime", "away_score", "home_score")
"""Columns a pre-lock sheet takes from the week's predictions: kickoffs and final scores."""


@dataclass(frozen=True, kw_only=True)
class SheetRow:
    """One game on the sheet: the pick, its points and, once played, how it scored.

    ``newer_points`` and ``newer_pick`` are set on a sheet built from the picks entered before
    the lock, for a game whose points or pick differ in the predictions written after it.
    """

    game_id: str
    points: int | None
    pick: str | None
    opponent: str | None
    pick_is_home: bool | None
    away_abbr: str
    home_abbr: str
    kickoff: str | None
    pick_win_prob: float | None
    expected_points: float | None
    away_score: int | None
    home_score: int | None
    result: Result | None
    points_won: int | None
    newer_points: int | None = None
    newer_pick: str | None = None


@dataclass(frozen=True, kw_only=True)
class ConfidenceSheet:
    """A week's confidence pool sheet and its totals.

    ``picks_file`` names the picks file the sheet was built from when the week's predictions were
    rewritten after the lock; ``newer_predictions_at`` is when, and ``changed_after_lock`` counts
    the games whose pick or points the newer predictions changed.
    """

    rows: list[SheetRow]
    games: int
    max_points: int
    expected_points: float | None
    games_final: int
    points_won: int | None
    points_possible: int | None
    lock_at: str | None
    generated_after_lock: bool | None
    rank_problem: str | None
    picks_file: str | None = None
    newer_predictions_at: str | None = None
    changed_after_lock: int = 0


def _int(value: object) -> int | None:
    """Return ``value`` as an int when it is a finite number."""
    return int(value) if is_number(value) else None


def _float(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite number."""
    return float(value) if is_number(value) else None


def _pick(row: Mapping[str, object]) -> str | None:
    """Return the published pick, else the home side when its chance is at least even."""
    pick = row.get("predicted_winner")
    if isinstance(pick, str) and pick:
        return pick
    p_home = _float(row.get("home_win_prob"))
    if p_home is None:
        return None
    return str(row["home_abbr"] if p_home >= constants.EVEN_ODDS_PROBABILITY else row["away_abbr"])


def _pick_win_prob(row: Mapping[str, object], *, pick_is_home: bool) -> float | None:
    """Return the published win probability of the picked side."""
    if pick_is_home:
        return _float(row.get("home_win_prob"))
    away = _float(row.get("away_win_prob"))
    if away is not None:
        return away
    home = _float(row.get("home_win_prob"))
    return None if home is None else 1.0 - home


def _final_score(
    row: Mapping[str, object], finals: Mapping[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Return ``(away, home)`` final score from the row itself, else from ``finals``."""
    away, home = _int(row.get("away_score")), _int(row.get("home_score"))
    if away is not None and home is not None:
        return away, home
    return finals.get(str(row.get("game_id")))


def _result(score: tuple[int, int] | None, *, pick_is_home: bool | None) -> Result | None:
    """Return how the pick did: a tie is incorrect for both sides."""
    if score is None or pick_is_home is None:
        return None
    away, home = score
    if away == home:
        return "tie"
    return "win" if (home > away) == pick_is_home else "loss"


def _sheet_row(row: Mapping[str, object], finals: Mapping[str, tuple[int, int]]) -> SheetRow:
    """Build one sheet row from one predictions row."""
    away, home = str(row.get("away_abbr")), str(row.get("home_abbr"))
    pick = _pick(row)
    pick_is_home = None if pick is None else pick == home
    points = _int(row.get("confidence_rank"))
    win_prob = None if pick_is_home is None else _pick_win_prob(row, pick_is_home=pick_is_home)
    score = _final_score(row, finals)
    result = _result(score, pick_is_home=pick_is_home)
    kickoff = kickoff_eastern(row)
    return SheetRow(
        game_id=str(row.get("game_id")),
        points=points,
        pick=pick,
        opponent=None if pick is None else (away if pick_is_home else home),
        pick_is_home=pick_is_home,
        away_abbr=away,
        home_abbr=home,
        kickoff=kickoff.isoformat() if kickoff else None,
        pick_win_prob=win_prob,
        expected_points=points * win_prob if points is not None and win_prob is not None else None,
        away_score=score[0] if score else None,
        home_score=score[1] if score else None,
        result=result,
        points_won=None if result is None or points is None else (points if result == "win" else 0),
    )


def rank_problem(ranks: list[int | None], *, has_ranks: bool) -> str | None:
    """Return why ``ranks`` are not the unique ``1..N`` the pool needs, else ``None``."""
    if not has_ranks:
        return "The predictions file has no confidence ranks."
    games = len(ranks)
    unranked = sum(rank is None for rank in ranks)
    if unranked:
        return f"{unranked} of {games} games have no confidence rank."
    counts = Counter(rank for rank in ranks if rank is not None)
    repeated = sorted(rank for rank, count in counts.items() if count > 1)
    missing = sorted(set(range(1, games + 1)) - set(counts))
    if not repeated and not missing:
        return None
    parts = [f"{rank} repeats" for rank in repeated] + [f"{rank} is missing" for rank in missing]
    return f"Confidence ranks should run 1 to {games} once each: {', '.join(parts)}."


def _order(row: SheetRow) -> tuple[int, str, str]:
    """Sort key: most points first, then kickoff, then game id."""
    return (-(row.points or 0), row.kickoff or "", row.game_id)


def build_sheet(
    frame: pl.DataFrame,
    finals: Mapping[str, tuple[int, int]],
    generated_at: datetime | None,
) -> ConfidenceSheet:
    """Build the sheet for one week's predictions.

    Args:
        frame: The week's predictions, one row per game.
        finals: ``game_id -> (away_score, home_score)`` for games already played; scores in the
            predictions file itself take precedence.
        generated_at: When the predictions were written (aware), to compare with the lock.

    Returns:
        The rows, most points first, and the week's totals.

    """
    records = frame.to_dicts()
    rows = sorted((_sheet_row(row, finals) for row in records), key=_order)
    games = len(rows)
    problem = rank_problem([row.points for row in rows], has_ranks="confidence_rank" in frame)
    final = [row for row in rows if row.result is not None]
    points_won = sum(row.points_won or 0 for row in final) if final else None
    open_points = sum(row.points or 0 for row in rows if row.result is None)
    expected = [row.expected_points for row in rows]
    lock = first_kickoff(records)
    return ConfidenceSheet(
        rows=rows,
        games=games,
        max_points=games * (games + 1) // 2,
        expected_points=None
        if not rows or any(value is None for value in expected)
        else sum(value for value in expected if value is not None),
        games_final=len(final),
        points_won=points_won,
        points_possible=None if problem else (points_won or 0) + open_points,
        lock_at=lock.isoformat() if lock else None,
        generated_after_lock=None if lock is None or generated_at is None else generated_at > lock,
        rank_problem=problem,
    )


@dataclass(frozen=True)
class SubmittedPicks:
    """A weekly run's picks file for one week, written before that week's lock."""

    frame: pl.DataFrame
    file_name: str
    written_at: datetime


def build_pre_lock_sheet(
    predictions: pl.DataFrame,
    submitted: SubmittedPicks,
    finals: Mapping[str, tuple[int, int]],
    predictions_written_at: datetime,
) -> ConfidenceSheet:
    """Build the sheet from the picks entered before the lock, noting what changed after it.

    Args:
        predictions: The week's predictions, written after the lock; they supply the kickoffs
            and any final scores, and their ranks become the notes.
        submitted: The weekly run's picks file, written before the lock.
        finals: ``game_id -> (away_score, home_score)`` for games already played.
        predictions_written_at: When the predictions were rewritten (aware).

    Returns:
        The sheet of the pre-lock picks, scored like any sheet.

    """
    schedule = predictions.select(
        "game_id", *(column for column in SCHEDULE_COLUMNS if column in predictions.columns)
    )
    picks = submitted.frame
    entered = picks.drop([column for column in SCHEDULE_COLUMNS if column in picks.columns]).join(
        schedule, on="game_id", how="left"
    )
    sheet = build_sheet(entered, finals, submitted.written_at)
    newer = {row.game_id: row for row in build_sheet(predictions, finals, None).rows}
    rows: list[SheetRow] = []
    changed = 0
    for row in sheet.rows:
        latest = newer.get(row.game_id)
        if latest is None or (latest.points, latest.pick) == (row.points, row.pick):
            rows.append(row)
            continue
        changed += 1
        rows.append(replace(row, newer_points=latest.points, newer_pick=latest.pick))
    return replace(
        sheet,
        rows=rows,
        picks_file=submitted.file_name,
        newer_predictions_at=predictions_written_at.isoformat(),
        changed_after_lock=changed,
    )
