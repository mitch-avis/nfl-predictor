"""Payloads for the pool helpers: the confidence sheet, the tiebreaker card and survivor plans."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_USED_TEAMS = 32
"""A survivor entry can have used at most every team in the league."""


class SheetRowOut(BaseModel):
    """One game on the confidence sheet.

    ``points`` is the confidence rank to assign; ``result`` and ``points_won`` are set once the
    game is final, and a tie is a ``tie`` worth no points for either pick. On a sheet built from
    the picks entered before the lock, ``newer_points`` and ``newer_pick`` note a game whose
    points or pick the predictions written after the lock changed.
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
    result: Literal["win", "loss", "tie"] | None
    points_won: int | None
    newer_points: int | None = None
    newer_pick: str | None = None


class ConfidenceSheetOut(BaseModel):
    """A week's confidence pool sheet.

    ``lock_at`` is the week's first kickoff (picks are single-shot before it), and
    ``generated_after_lock`` flags predictions written after it. ``rank_problem`` explains ranks
    that are not the unique ``1..N`` the pool needs. When the week's predictions were rewritten
    after the lock and the run's picks file was written before it, the sheet comes from that file:
    ``picks_file`` names it, ``generated_at`` is when it was written, ``newer_predictions_at`` is
    when the predictions were, and ``changed_after_lock`` counts the games they changed.
    """

    run_id: str | None
    source: str
    season: int | None
    week: int | None
    generated_at: str | None
    rows: list[SheetRowOut]
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


class TiebreakGameOut(BaseModel):
    """One game's whole-point tiebreak entry and the p10 to p90 bands behind it."""

    game_id: str
    away_abbr: str
    home_abbr: str
    kickoff: str | None
    slot: str | None
    total_points: int | None
    home_points: int | None
    away_points: int | None
    total_raw: float | None
    home_raw: float | None
    away_raw: float | None
    total_p10: float | None
    total_p90: float | None
    home_p10: float | None
    home_p90: float | None
    away_p10: float | None
    away_p90: float | None


class ScoreEstimateOut(BaseModel):
    """The week's highest or lowest team score, and the team projected to have it."""

    points: int | None
    median: float | None
    p10: float | None
    p90: float | None
    projected_team: str | None
    projected_points: float | None


class TiebreakersOut(BaseModel):
    """The tiebreaker card for a week."""

    run_id: str | None
    source: str
    season: int | None
    week: int | None
    generated_at: str | None
    games: list[TiebreakGameOut]
    default_game_ids: list[str]
    highest: ScoreEstimateOut
    lowest: ScoreEstimateOut


class SurvivorRequest(BaseModel):
    """What the survivor planner needs: the week to pick now and the teams already used.

    ``horizon_weeks`` only sets how many weeks from the start week the plans show as near-term
    ("planned") picks; every week with predictions is solved exactly either way.
    """

    season: int
    start_week: int = Field(ge=1)
    used_teams: list[str] = Field(default_factory=list, max_length=MAX_USED_TEAMS)
    horizon_weeks: int = Field(default=4, ge=1)


class SurvivorPickOut(BaseModel):
    """One week's pick in a plan; ``later`` marks a lookahead pick rather than a planned one."""

    week: int
    team: str
    opponent: str
    is_home: bool
    win_prob: float
    game_id: str
    kickoff: str | None
    later: bool
    cumulative_survival: float


class SurvivorPlanOut(BaseModel):
    """A ranked plan: the best found for its first-week pick."""

    rank: int
    picks: list[SurvivorPickOut]
    survival: float
    later_survival: float | None
    total_survival: float


class FutureValueOut(BaseModel):
    """The share of the survival chance in the weeks after this one lost by using ``team`` now."""

    team: str
    value: float


class SurvivorOut(BaseModel):
    """Survivor plans from ``start_week`` and the weeks behind them."""

    season: int
    start_week: int
    planned_weeks: list[int]
    later_weeks: list[int]
    missing_weeks: list[int]
    used_teams: list[str]
    teams: list[str]
    plans: list[SurvivorPlanOut]
    future_values: list[FutureValueOut]
