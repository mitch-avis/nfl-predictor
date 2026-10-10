"""The survivor planner: one team a week, each team once, and a loss or a tie ends the entry.

The objective is the chance of surviving every week with predictions, read from the predicted win
probabilities and treating the games as independent. The planned weeks (the next few) are laid out
by a beam search over states of (teams used, log survival), one search per possible pick in the
first week, so the plans that come back answer "what happens if I pick this team now". The later
weeks with predictions are the lookahead: each finished plan is completed by the best assignment
of the teams it leaves over the later weeks, solved exactly, and while the beam searches, using a
team costs its future value: how much the best later assignment loses without it. That cost is
what saves a strong team for a later week. Weeks without predictions are not planned for.

Nothing here changes a prediction: the planner only combines the published win probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from nfl_predictor.api.pool.kickoff import kickoff_eastern
from nfl_predictor.api.readers.market import is_number

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import polars as pl

MIN_PROBABILITY = 1e-9
"""Floor on a win probability before its logarithm, so a certain loss stays finite."""
UNAVAILABLE_COST = 1e6
"""Assignment cost of a team that does not play that week (far above any ``-log p``)."""
LOST_CAUSE_VALUE = 50.0
"""Future value (in log survival) of a team without which the later weeks cannot be covered."""


@dataclass(frozen=True, kw_only=True)
class Option:
    """One team that can be picked in one week, and its chance to win."""

    week: int
    team: str
    opponent: str
    is_home: bool
    win_prob: float
    game_id: str
    kickoff: str | None = None


@dataclass(frozen=True, kw_only=True)
class Pick:
    """A plan's pick for one week, with the chance of surviving through that week."""

    option: Option
    later: bool
    cumulative_survival: float


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A plan: its picks in week order and its chances of surviving.

    ``survival`` covers the planned weeks, ``later_survival`` the later weeks with predictions
    (``None`` when there are none), and ``total_survival`` both.
    """

    picks: tuple[Pick, ...]
    survival: float
    later_survival: float | None
    total_survival: float


@dataclass(frozen=True, kw_only=True)
class SurvivorResult:
    """Ranked plans, one per first-week pick, and each available team's future value.

    A team's future value is the share of the later weeks' survival chance lost by using it now.
    """

    plans: list[Plan]
    future_values: dict[str, float]


@dataclass(frozen=True)
class _State:
    """A partial plan in the beam."""

    teams: frozenset[str]
    picks: tuple[Option, ...]
    log_survival: float
    penalty: float

    @property
    def score(self) -> float:
        """Return the log survival so far minus the future value of the teams used."""
        return self.log_survival - self.penalty


def _log(probability: float) -> float:
    """Return the logarithm of a win probability, floored so a sure loss stays finite."""
    return math.log(max(probability, MIN_PROBABILITY))


def _probability(value: object) -> float | None:
    """Return ``value`` when it is a probability, else ``None``."""
    return float(value) if is_number(value) and 0.0 <= float(value) <= 1.0 else None


def week_options(
    frame: pl.DataFrame, week: int, finals: Mapping[str, tuple[int, int]]
) -> list[Option]:
    """Return both sides of every game in ``frame`` that is not already final.

    Args:
        frame: One week's predictions.
        week: The week the predictions are for.
        finals: ``game_id -> (away_score, home_score)`` of played games.

    """
    options: list[Option] = []
    for row in frame.to_dicts():
        game_id = str(row.get("game_id"))
        played = is_number(row.get("home_score")) and is_number(row.get("away_score"))
        p_home = _probability(row.get("home_win_prob"))
        if played or game_id in finals or p_home is None:
            continue
        p_away = _probability(row.get("away_win_prob"))
        home, away = str(row.get("home_abbr")), str(row.get("away_abbr"))
        kickoff = kickoff_eastern(row)
        stamp = kickoff.isoformat() if kickoff else None
        options.append(
            Option(
                week=week,
                team=home,
                opponent=away,
                is_home=True,
                win_prob=p_home,
                game_id=game_id,
                kickoff=stamp,
            )
        )
        options.append(
            Option(
                week=week,
                team=away,
                opponent=home,
                is_home=False,
                win_prob=1.0 - p_home if p_away is None else p_away,
                game_id=game_id,
                kickoff=stamp,
            )
        )
    return options


class _Lookahead:
    """The best assignment of available teams to the later weeks, solved exactly and cached."""

    def __init__(self, weeks: Sequence[Sequence[Option]]) -> None:
        """Index the later weeks' options by week and team."""
        self._weeks = [{option.team: option for option in week} for week in weeks]
        self._cache: dict[frozenset[str], tuple[float, tuple[Option, ...]] | None] = {}

    @property
    def active(self) -> bool:
        """Return whether there is any later week to cover."""
        return bool(self._weeks)

    def best(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
        """Return the best later log survival with ``available`` teams and its picks.

        ``None`` means the later weeks cannot all be covered with distinct available teams.
        """
        if available not in self._cache:
            self._cache[available] = self._solve(available)
        return self._cache[available]

    def _solve(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
        """Solve the assignment of later weeks to distinct teams that maximizes survival."""
        if not self._weeks:
            return 0.0, ()
        teams = sorted(available)
        if len(teams) < len(self._weeks):
            return None
        cost = np.full((len(self._weeks), len(teams)), UNAVAILABLE_COST)
        for row, options in enumerate(self._weeks):
            for column, team in enumerate(teams):
                option = options.get(team)
                if option is not None:
                    cost[row, column] = -_log(option.win_prob)
        rows, columns = linear_sum_assignment(cost)
        assigned = cost[rows, columns]
        if (assigned >= UNAVAILABLE_COST).any():
            return None
        picks = tuple(
            self._weeks[row][teams[column]] for row, column in zip(rows, columns, strict=True)
        )
        return -float(assigned.sum()), picks

    def future_values(self, available: frozenset[str]) -> dict[str, float]:
        """Return each team's future value in log survival: what the later weeks lose without it.

        Only the teams in the best assignment can lose anything; every other team's value is 0.
        """
        values = dict.fromkeys(available, 0.0)
        best = self.best(available)
        if best is None:
            return values
        log_best, picks = best
        for option in picks:
            without = self.best(available - {option.team})
            values[option.team] = LOST_CAUSE_VALUE if without is None else log_best - without[0]
        return values


def _beam_key(state: _State) -> tuple[float, tuple[str, ...]]:
    """Order states best first, ties by their teams in pick order."""
    return (-state.score, tuple(option.team for option in state.picks))


def _search(
    first: Option,
    planned: Sequence[Sequence[Option]],
    values: Mapping[str, float],
    used: frozenset[str],
    beam_width: int,
) -> list[_State]:
    """Run the beam over the planned weeks after ``first``; return the final beam."""
    beam = [
        _State(frozenset({first.team}), (first,), _log(first.win_prob), values.get(first.team, 0.0))
    ]
    for options in planned[1:]:
        candidates: dict[frozenset[str], _State] = {}
        for state in beam:
            for option in options:
                if option.team in used or option.team in state.teams:
                    continue
                child = _State(
                    state.teams | {option.team},
                    (*state.picks, option),
                    state.log_survival + _log(option.win_prob),
                    state.penalty + values.get(option.team, 0.0),
                )
                kept = candidates.get(child.teams)
                if kept is None or _beam_key(child) < _beam_key(kept):
                    candidates[child.teams] = child
        beam = sorted(candidates.values(), key=_beam_key)[:beam_width]
    return beam


def _plan(
    state: _State, later: tuple[float, tuple[Option, ...]] | None, *, has_later: bool
) -> Plan:
    """Turn a finished state and its later assignment into a plan with cumulative survival."""
    later_picks = later[1] if later is not None else ()
    ordered = sorted(
        [(option, False) for option in state.picks] + [(option, True) for option in later_picks],
        key=lambda item: item[0].week,
    )
    picks: list[Pick] = []
    cumulative = 1.0
    for option, is_later in ordered:
        cumulative *= option.win_prob
        picks.append(Pick(option=option, later=is_later, cumulative_survival=cumulative))
    survival = math.exp(state.log_survival)
    later_survival = (math.exp(later[0]) if later is not None else 0.0) if has_later else None
    return Plan(
        picks=tuple(picks),
        survival=survival,
        later_survival=later_survival,
        total_survival=survival * (1.0 if later_survival is None else later_survival),
    )


def plan_survivor(
    planned: Sequence[Sequence[Option]],
    later: Sequence[Sequence[Option]],
    used: Collection[str],
    *,
    beam_width: int,
    max_plans: int = 10,
) -> SurvivorResult:
    """Rank survivor plans, one per possible pick in the first planned week.

    Args:
        planned: Each planned week's options, in week order; the first is the week to pick now.
        later: Each later week's options, covered by the exact lookahead.
        used: Teams already used this season.
        beam_width: Partial plans kept per week in each search.
        max_plans: Plans to return, best first.

    Returns:
        The ranked plans and the future value of every available team.

    Raises:
        ValueError: When ``beam_width`` is below one.

    """
    if beam_width < 1:
        msg = f"beam_width must be at least 1, got {beam_width}"
        raise ValueError(msg)
    used_teams = frozenset(used)
    available = (
        frozenset(option.team for week in (*planned, *later) for option in week) - used_teams
    )
    lookahead = _Lookahead(later)
    values = lookahead.future_values(available)
    plans: list[tuple[tuple[float, float, str], Plan]] = []
    for first in planned[0] if planned else ():
        if first.team in used_teams:
            continue
        finished: list[tuple[tuple[float, float, str], Plan]] = []
        for state in _search(first, planned, values, used_teams, beam_width):
            later_best = lookahead.best(available - state.teams)
            later_log = -math.inf if later_best is None else later_best[0]
            key = (-(state.log_survival + later_log), -state.log_survival, first.team)
            finished.append((key, _plan(state, later_best, has_later=lookahead.active)))
        if finished:
            plans.append(min(finished, key=lambda item: item[0]))
    plans.sort(key=lambda item: item[0])
    return SurvivorResult(
        plans=[plan for _key, plan in plans[:max_plans]],
        future_values={team: 1.0 - math.exp(-value) for team, value in sorted(values.items())},
    )
