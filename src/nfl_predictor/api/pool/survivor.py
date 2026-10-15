"""The survivor planner: one team a week, each team once, and a loss or a tie ends the entry.

The objective is the chance of surviving every week with predictions, read from the predicted win
probabilities and treating the games as independent. Maximizing that product with each team used
at most once is an assignment of weeks to teams at cost ``-log p``, which
``scipy.optimize.linear_sum_assignment`` solves exactly. The planner solves it once per possible
pick this week: the first week is fixed to that pick and the weeks after it are assigned exactly,
so the plans that come back answer "what happens if I pick this team now". A team's future value
is what the best assignment of the weeks after this one loses without it; it is why a strong team
is saved for a week where it is worth more. The planned weeks (the next few) and the later weeks
differ only in how they are shown. Weeks without predictions are not planned for.

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
"""Future value (in log survival) of a team without which the weeks after this one go uncovered."""


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

    A team's future value is the share of the survival chance in the weeks after this one lost by
    using it now.
    """

    plans: list[Plan]
    future_values: dict[str, float]


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


class _Assignment:
    """The best assignment of available teams to a run of weeks, solved exactly and cached."""

    def __init__(self, weeks: Sequence[Sequence[Option]]) -> None:
        """Index the weeks' options by week and team."""
        self._weeks = [{option.team: option for option in week} for week in weeks]
        self._cache: dict[frozenset[str], tuple[float, tuple[Option, ...]] | None] = {}

    def best(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
        """Return the best log survival over the weeks with ``available`` teams, and its picks.

        ``None`` means the weeks cannot all be covered with distinct available teams.
        """
        if available not in self._cache:
            self._cache[available] = self._solve(available)
        return self._cache[available]

    def _solve(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
        """Solve the assignment of the weeks to distinct teams that maximizes survival."""
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
        """Return each team's future value in log survival: what the weeks lose without it.

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


def _plan(
    first: Option,
    rest: tuple[float, tuple[Option, ...]] | None,
    planned_weeks: frozenset[int],
    *,
    has_later: bool,
) -> Plan:
    """Build the plan of a first pick and the exact assignment of the weeks after it.

    A plan whose weeks cannot all be covered keeps only its first pick and has no chance.
    """
    if rest is None:
        alone = len(planned_weeks) == 1
        return Plan(
            picks=(Pick(option=first, later=False, cumulative_survival=first.win_prob),),
            survival=first.win_prob if alone else 0.0,
            later_survival=0.0 if has_later else None,
            total_survival=0.0,
        )
    picks: list[Pick] = []
    cumulative = 1.0
    for option in sorted((first, *rest[1]), key=lambda pick: pick.week):
        cumulative *= option.win_prob
        later = option.week not in planned_weeks
        picks.append(Pick(option=option, later=later, cumulative_survival=cumulative))
    survival = math.prod(pick.option.win_prob for pick in picks if not pick.later)
    later_survival = (
        math.prod(pick.option.win_prob for pick in picks if pick.later) if has_later else None
    )
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
    max_plans: int = 10,
) -> SurvivorResult:
    """Rank survivor plans, one per possible pick in the first planned week, each exact.

    Args:
        planned: The planned weeks' options, in week order; the first is the week to pick now.
        later: The later weeks' options, in week order. Planned and later weeks are solved the
            same way; the split only says which picks the plan shows as near-term.
        used: Teams already used this season.
        max_plans: Plans to return, best first.

    Returns:
        The ranked plans and the future value of every available team.

    """
    if not planned:
        return SurvivorResult(plans=[], future_values={})
    used_teams = frozenset(used)
    weeks = [*planned, *later]
    available = frozenset(option.team for week in weeks for option in week) - used_teams
    planned_weeks = frozenset(option.week for week in planned for option in week)
    rest = _Assignment(weeks[1:])
    ranked: list[tuple[tuple[float, float, str], Plan]] = []
    for first in planned[0]:
        if first.team in used_teams:
            continue
        best_rest = rest.best(available - {first.team})
        rest_log = -math.inf if best_rest is None else best_rest[0]
        key = (-(_log(first.win_prob) + rest_log), -first.win_prob, first.team)
        ranked.append((key, _plan(first, best_rest, planned_weeks, has_later=bool(later))))
    ranked.sort(key=lambda item: item[0])
    values = rest.future_values(available)
    return SurvivorResult(
        plans=[plan for _key, plan in ranked[:max_plans]],
        future_values={team: 1.0 - math.exp(-value) for team, value in sorted(values.items())},
    )
