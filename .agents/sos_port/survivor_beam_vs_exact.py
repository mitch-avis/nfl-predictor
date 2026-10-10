"""Compare the survivor planner's first design (a beam search) with the exact optimum.

The web app's first survivor planner laid out the next weeks with a beam search and filled the
weeks after them with an exact assignment. The independent review of 2026-10-10 found plans below
the exact optimum on simulated seasons, so the planner became exact. This script keeps the
evidence: it copies the beam search as it was written (commit ``e08ec30``,
``src/nfl_predictor/api/pool/survivor.py``) and compares it on 30 simulated seasons with the exact
optimum, one assignment of every week to distinct teams (``scipy.optimize.linear_sum_assignment``
on the cost ``-log p``), and with the planner the app ships now.

Simulation, as the review described it: 32 teams with persistent ratings drawn from
``Normal(0, 6)``, 14 weeks of random pairings, four teams on a bye in each of weeks 3 to 12, home
win probability ``Phi((r_home - r_away + 1.5) / 13.5)``, seeds ``default_rng(1000 + s)`` for
``s`` in ``0..29``. The beam plans 10 weeks (the exact lookahead covers weeks 11 to 14).

Run from the repository root:

    .venv/bin/python .agents/sos_port/survivor_beam_vs_exact.py

It prints, for beam widths 64 and 16, how many seasons got a best plan below the exact optimum,
the largest shortfall, and how many recommended a different first pick; then the seasons each
width got wrong, and the shipped planner's largest gap to the optimum (zero when it is exact).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.optimize import linear_sum_assignment

from nfl_predictor.api.pool.survivor import Option, plan_survivor

SEASONS = 30
TEAMS = 32
WEEKS = 14
BYE_WEEKS = range(3, 13)
BYES_PER_WEEK = 4
PLANNED_WEEKS = 10
HOME_EDGE = 1.5
SIGMA = 13.5
RATING_SD = 6.0
MIN_PROBABILITY = 1e-9
UNAVAILABLE_COST = 1e6
LOST_CAUSE_VALUE = 50.0
NORMAL = NormalDist()


def season(seed: int) -> list[list[Option]]:
    """Return one simulated season's options, week by week."""
    rng = np.random.default_rng(1000 + seed)
    names = [f"T{index:02d}" for index in range(TEAMS)]
    ratings = rng.normal(0.0, RATING_SD, TEAMS)
    weeks: list[list[Option]] = []
    for week in range(1, WEEKS + 1):
        order = rng.permutation(TEAMS)
        playing = order[BYES_PER_WEEK:] if week in BYE_WEEKS else order
        options: list[Option] = []
        for index in range(0, len(playing), 2):
            home, away = int(playing[index]), int(playing[index + 1])
            p_home = NORMAL.cdf((ratings[home] - ratings[away] + HOME_EDGE) / SIGMA)
            game_id = f"{week:02d}_{names[away]}_{names[home]}"
            options.append(
                Option(week=week, team=names[home], opponent=names[away], is_home=True,
                       win_prob=p_home, game_id=game_id)
            )
            options.append(
                Option(week=week, team=names[away], opponent=names[home], is_home=False,
                       win_prob=1.0 - p_home, game_id=game_id)
            )
        weeks.append(options)
    return weeks


def log_p(probability: float) -> float:
    """Return the logarithm of a win probability, floored."""
    return math.log(max(probability, MIN_PROBABILITY))


def exact_optimum(weeks: list[list[Option]]) -> tuple[float, str]:
    """Return the best chance to survive every week and its first pick, by one assignment."""
    teams = sorted({option.team for week in weeks for option in week})
    by_week = [{option.team: option for option in week} for week in weeks]
    cost = np.full((len(weeks), len(teams)), UNAVAILABLE_COST)
    for row, options in enumerate(by_week):
        for column, team in enumerate(teams):
            if team in options:
                cost[row, column] = -log_p(options[team].win_prob)
    rows, columns = linear_sum_assignment(cost)
    first = teams[columns[list(rows).index(0)]]
    return math.exp(-float(cost[rows, columns].sum())), first


# ---- The beam search as written in commit e08ec30 (frozen copy) -------------------------------


@dataclass(frozen=True)
class _State:
    teams: frozenset[str]
    picks: tuple[Option, ...]
    log_survival: float
    penalty: float

    @property
    def score(self) -> float:
        return self.log_survival - self.penalty


class _Lookahead:
    def __init__(self, weeks: list[list[Option]]) -> None:
        self._weeks = [{option.team: option for option in week} for week in weeks]
        self._cache: dict[frozenset[str], tuple[float, tuple[Option, ...]] | None] = {}

    def best(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
        if available not in self._cache:
            self._cache[available] = self._solve(available)
        return self._cache[available]

    def _solve(self, available: frozenset[str]) -> tuple[float, tuple[Option, ...]] | None:
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
                    cost[row, column] = -log_p(option.win_prob)
        rows, columns = linear_sum_assignment(cost)
        assigned = cost[rows, columns]
        if (assigned >= UNAVAILABLE_COST).any():
            return None
        picks = tuple(self._weeks[r][teams[c]] for r, c in zip(rows, columns, strict=True))
        return -float(assigned.sum()), picks

    def future_values(self, available: frozenset[str]) -> dict[str, float]:
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
    return (-state.score, tuple(option.team for option in state.picks))


def _search(
    first: Option,
    planned: list[list[Option]],
    values: dict[str, float],
    used: frozenset[str],
    beam_width: int,
) -> list[_State]:
    beam = [
        _State(frozenset({first.team}), (first,), log_p(first.win_prob), values.get(first.team, 0.0))
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
                    state.log_survival + log_p(option.win_prob),
                    state.penalty + values.get(option.team, 0.0),
                )
                kept = candidates.get(child.teams)
                if kept is None or _beam_key(child) < _beam_key(kept):
                    candidates[child.teams] = child
        beam = sorted(candidates.values(), key=_beam_key)[:beam_width]
    return beam


def beam_best(weeks: list[list[Option]], beam_width: int) -> tuple[float, str]:
    """Return the beam design's best plan: its chance to survive every week and its first pick."""
    planned, later = weeks[:PLANNED_WEEKS], weeks[PLANNED_WEEKS:]
    available = frozenset(option.team for week in weeks for option in week)
    lookahead = _Lookahead(later)
    values = lookahead.future_values(available)
    best: tuple[float, str] = (0.0, "")
    for first in planned[0]:
        for state in _search(first, planned, values, frozenset(), beam_width):
            later_best = lookahead.best(available - state.teams)
            if later_best is None:
                continue
            total = math.exp(state.log_survival + later_best[0])
            if total > best[0]:
                best = (total, first.team)
    return best


# ---- Comparison ---------------------------------------------------------------------------------


def main() -> None:
    """Print the comparison."""
    seasons = [season(s) for s in range(SEASONS)]
    optima = [exact_optimum(weeks) for weeks in seasons]
    for width in (64, 16):
        below: list[tuple[int, float]] = []
        wrong_first: list[int] = []
        for s, weeks in enumerate(seasons):
            optimum, optimum_first = optima[s]
            total, first = beam_best(weeks, width)
            if total < optimum * (1 - 1e-9):
                below.append((s, 1 - total / optimum))
            if first != optimum_first:
                wrong_first.append(s)
        worst = max((gap for _s, gap in below), default=0.0)
        print(
            f"width {width}: {len(below)} of {SEASONS} seasons below the exact optimum "
            f"(largest shortfall {worst:.1%}); first pick differs in {len(wrong_first)} of {SEASONS}"
        )
        print(f"  seasons below: {[s for s, _gap in below]}")
        print(f"  seasons with a different first pick: {wrong_first}")
    shipped_gap = 0.0
    seconds: list[float] = []
    for s, weeks in enumerate(seasons):
        start = time.perf_counter()
        result = plan_survivor(weeks[:PLANNED_WEEKS], weeks[PLANNED_WEEKS:], set())
        seconds.append(time.perf_counter() - start)
        shipped_gap = max(shipped_gap, abs(1 - result.plans[0].total_survival / optima[s][0]))
    print(f"shipped planner: largest relative gap to the exact optimum {shipped_gap:.2e}")
    print(
        f"shipped planner: {sum(seconds) / len(seconds) * 1000:.0f} ms per season on average, "
        f"{max(seconds) * 1000:.0f} ms at most (machine-dependent)"
    )


if __name__ == "__main__":
    main()
