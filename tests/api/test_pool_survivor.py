"""Tests for the survivor planner: exact plans, one per pick this week, over predicted weeks."""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest
from scipy.optimize import linear_sum_assignment

from nfl_predictor.api.pool.survivor import Option, plan_survivor, week_options


def _game(week: int, home: str, away: str, p_home: float) -> list[Option]:
    """Return both sides of one game as options."""
    game_id = f"2026_{week:02d}_{away}_{home}"
    return [
        Option(week=week, team=home, opponent=away, is_home=True, win_prob=p_home, game_id=game_id),
        Option(
            week=week, team=away, opponent=home, is_home=False, win_prob=1 - p_home, game_id=game_id
        ),
    ]


def _week(week: int, *games: tuple[str, str, float]) -> list[Option]:
    """Return every option of a week given ``(home, away, p_home)`` games."""
    return [option for home, away, p in games for option in _game(week, home, away, p)]


def _brute_force(weeks: list[list[Option]], used: set[str], first: str | None = None) -> float:
    """Return the best chance to survive every week with distinct teams, by enumeration."""
    best = 0.0
    for combo in itertools.product(*weeks):
        teams = [option.team for option in combo]
        if len(set(teams)) < len(teams) or used & set(teams):
            continue
        if first is not None and teams[0] != first:
            continue
        best = max(best, math.prod(option.win_prob for option in combo))
    return best


def _simulated_season(seed: int) -> list[list[Option]]:
    """Return a 32-team, 14-week season with persistent ratings, as the review simulated it.

    Ratings are drawn from ``Normal(0, 6)``, four teams rest in each of weeks 3 to 12 and the home
    win probability is ``Phi((r_home - r_away + 1.5) / 13.5)``.
    """
    rng = np.random.default_rng(1000 + seed)
    names = [f"T{index:02d}" for index in range(32)]
    ratings = rng.normal(0.0, 6.0, 32)
    weeks: list[list[Option]] = []
    for week in range(1, 15):
        order = rng.permutation(32)
        playing = [int(team) for team in (order[4:] if 3 <= week <= 12 else order)]
        games: list[tuple[str, str, float]] = []
        for home, away in itertools.batched(playing, 2, strict=True):
            edge = float(ratings[home] - ratings[away]) + 1.5
            games.append((names[home], names[away], NormalDist().cdf(edge / 13.5)))
        weeks.append(_week(week, *games))
    return weeks


def _exact_optimum(weeks: list[list[Option]]) -> tuple[float, str]:
    """Return the best chance to survive every week with distinct teams, and its first pick.

    One assignment of every week to a team at cost ``-log p``, solved directly, as the oracle.
    """
    teams = sorted({option.team for week in weeks for option in week})
    cost = np.full((len(weeks), len(teams)), 1e6)
    for row, week in enumerate(weeks):
        for option in week:
            cost[row, teams.index(option.team)] = -math.log(option.win_prob)
    rows, columns = linear_sum_assignment(cost)
    first = teams[columns[list(rows).index(0)]]
    return math.exp(-float(cost[rows, columns].sum())), first


def _random_weeks(seed: int, weeks: int, teams: int) -> list[list[Option]]:
    """Return ``weeks`` weeks in which ``teams`` teams pair off into games at random chances."""
    rng = np.random.default_rng(seed)
    names = [f"T{index}" for index in range(teams)]
    schedule: list[list[Option]] = []
    for week in range(1, weeks + 1):
        order = rng.permutation(teams)
        games = [
            (names[order[i]], names[order[i + 1]], float(rng.uniform(0.15, 0.9)))
            for i in range(0, teams, 2)
        ]
        schedule.append(_week(week, *games))
    return schedule


def test_single_week_ranks_one_plan_per_pick() -> None:
    """With one week, every playable team gets a plan, ranked by its chance to win."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8), ("B", "Y", 0.7), ("C", "Z", 0.6))]

    # Act
    result = plan_survivor(planned, [], set())

    # Assert
    assert [plan.picks[0].option.team for plan in result.plans] == ["A", "B", "C", "Z", "Y", "X"]
    assert result.plans[0].survival == pytest.approx(0.8)
    assert result.plans[0].later_survival is None
    assert result.plans[0].total_survival == pytest.approx(0.8)


def test_used_teams_are_never_picked() -> None:
    """Teams already used this season are left out of every plan."""
    # Arrange
    planned = [
        _week(1, ("A", "X", 0.8), ("B", "Y", 0.7)),
        _week(2, ("A", "Y", 0.9), ("B", "X", 0.6)),
    ]

    # Act
    result = plan_survivor(planned, [], {"A"})

    # Assert
    teams = {pick.option.team for plan in result.plans for pick in plan.picks}
    assert "A" not in teams


def test_a_team_is_picked_at_most_once_per_plan() -> None:
    """A team that is the best pick in two weeks is used in only one of them."""
    # Arrange
    planned = [
        _week(1, ("A", "X", 0.9), ("B", "Y", 0.6)),
        _week(2, ("A", "Y", 0.9), ("B", "X", 0.7)),
    ]

    # Act
    result = plan_survivor(planned, [], set())

    # Assert
    for plan in result.plans:
        teams = [pick.option.team for pick in plan.picks]
        assert len(teams) == len(set(teams))
    assert [pick.option.team for pick in result.plans[0].picks] == ["A", "B"]
    assert result.plans[0].survival == pytest.approx(0.9 * 0.7)


def test_lookahead_saves_a_team_for_a_later_week() -> None:
    """A team worth more later is saved: its later week lifts the whole plan."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.85), ("B", "Y", 0.80))]
    later = [_week(2, ("A", "Z", 0.95), ("B", "W", 0.55), ("C", "V", 0.60))]

    # Act
    result = plan_survivor(planned, later, set())

    # Assert
    best = result.plans[0]
    assert [(pick.option.week, pick.option.team, pick.later) for pick in best.picks] == [
        (1, "B", False),
        (2, "A", True),
    ]
    assert best.survival == pytest.approx(0.80)
    assert best.later_survival == pytest.approx(0.95)
    assert best.total_survival == pytest.approx(0.76)
    assert result.future_values["A"] == pytest.approx(1 - 0.60 / 0.95)
    assert result.future_values["B"] == 0.0


def test_cumulative_survival_multiplies_the_picks_in_week_order() -> None:
    """Each pick carries the chance of surviving every week up to and including it."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8)), _week(2, ("B", "Y", 0.5))]
    later = [_week(3, ("C", "Z", 0.9))]

    # Act
    result = plan_survivor(planned, later, set())

    # Assert
    cumulative = [pick.cumulative_survival for pick in result.plans[0].picks]
    assert cumulative == pytest.approx([0.8, 0.4, 0.36])


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_plans_match_an_exhaustive_search(seed: int) -> None:
    """Each first pick's plan is the best that pick allows, over planned and later weeks."""
    # Arrange
    weeks = _random_weeks(seed, weeks=5, teams=6)
    planned, later = weeks[:3], weeks[3:]

    # Act
    result = plan_survivor(planned, later, set(), max_plans=12)

    # Assert
    assert result.plans[0].total_survival == pytest.approx(_brute_force(weeks, set()))
    for plan in result.plans:
        first = plan.picks[0].option.team
        assert plan.total_survival == pytest.approx(_brute_force(weeks, set(), first))


def test_plans_respect_byes() -> None:
    """A team without a game in a week is never planned for that week."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.9)), _week(2, ("B", "Y", 0.8))]

    # Act
    result = plan_survivor(planned, [], set())

    # Assert
    for plan in result.plans:
        for pick in plan.picks:
            assert pick.option.team in {o.team for o in planned[pick.option.week - 1]}


def test_max_plans_caps_the_list() -> None:
    """Only the requested number of plans comes back, best first."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8), ("B", "Y", 0.7), ("C", "Z", 0.6))]

    # Act
    result = plan_survivor(planned, [], set(), max_plans=2)

    # Assert
    assert [plan.picks[0].option.team for plan in result.plans] == ["A", "B"]


def test_later_weeks_that_cannot_be_covered_leave_no_later_chance() -> None:
    """When the teams left cannot cover a later week, the later chance is zero for every plan."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8))]
    later = [_week(2, ("B", "Y", 0.6))]

    # Act
    result = plan_survivor(planned, later, {"B", "Y"})

    # Assert
    assert [plan.later_survival for plan in result.plans] == [0.0, 0.0]
    assert result.future_values == {"A": 0.0, "X": 0.0}


def test_a_team_needed_to_cover_a_later_week_is_worth_all_of_it() -> None:
    """A team without which a later week cannot be covered carries a future value of one."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8))]
    later = [_week(2, ("A", "Y", 0.6))]

    # Act
    result = plan_survivor(planned, later, {"Y"})

    # Assert
    assert result.future_values["A"] == pytest.approx(1.0)
    assert [plan.picks[0].option.team for plan in result.plans] == ["X", "A"]


def test_a_first_pick_that_strands_a_planned_week_has_no_chance() -> None:
    """A first pick that leaves no open team for a later planned week ranks last, at zero."""
    # Arrange
    planned = [_week(1, ("A", "X", 0.8)), [*_game(2, "A", "Y", 0.7)[:1]]]

    # Act
    result = plan_survivor(planned, [], set())

    # Assert
    assert [plan.picks[0].option.team for plan in result.plans] == ["X", "A"]
    stranded = result.plans[1]
    assert (stranded.survival, stranded.total_survival) == (0.0, 0.0)
    assert [pick.option.team for pick in stranded.picks] == ["A"]


@pytest.mark.parametrize("seed", [0, 6, 8, 13, 25])
def test_plans_reach_the_exact_optimum_on_a_full_season(seed: int) -> None:
    """On a simulated 32-team, 14-week season the best plan is the exact optimum.

    Seasons 6, 8 and 13 are ones the first design (a beam of 64 over 10 planned weeks) planned
    below the optimum; in seasons 0 and 25 a beam of 16 recommended the wrong first pick
    (``.agents/sos_port/survivor_beam_vs_exact.py``).
    """
    # Arrange
    weeks = _simulated_season(seed)
    optimum, optimum_first = _exact_optimum(weeks)

    # Act
    result = plan_survivor(weeks[:10], weeks[10:], set())

    # Assert
    best = result.plans[0]
    assert best.total_survival == pytest.approx(optimum, rel=1e-9)
    assert best.picks[0].option.team == optimum_first


def test_every_first_pick_gets_its_own_best_plan_on_a_full_season() -> None:
    """Each plan covers every week with distinct teams and is the best its first pick allows."""
    # Arrange
    weeks = _simulated_season(6)

    # Act
    result = plan_survivor(weeks[:4], weeks[4:], set(), max_plans=32)

    # Assert
    assert len(result.plans) == 32
    for plan in result.plans:
        teams = [pick.option.team for pick in plan.picks]
        assert [pick.option.week for pick in plan.picks] == list(range(1, 15))
        assert len(set(teams)) == 14
        first = plan.picks[0].option
        rest = [[o for o in week if o.team != first.team] for week in weeks[1:]]
        rest_optimum, _first = _exact_optimum(rest)
        assert plan.total_survival == pytest.approx(first.win_prob * rest_optimum, rel=1e-9)


def test_no_planned_week_gives_no_plan() -> None:
    """Without a week to plan there is nothing to rank."""
    # Act
    result = plan_survivor([], [], set())

    # Assert
    assert result.plans == []
    assert result.future_values == {}


def test_week_options_read_both_sides_of_each_game() -> None:
    """Each game offers its home side at the home chance and its away side at the away chance."""
    # Arrange
    frame = pl.DataFrame(
        {
            "game_id": ["2026_05_TB_DAL", "2026_05_HOU_TEN"],
            "away_abbr": ["TB", "HOU"],
            "home_abbr": ["DAL", "TEN"],
            "game_datetime": ["2026-10-08T20:15:00", "2026-10-11T13:00:00"],
            "home_win_prob": [0.7928, 0.291],
            "away_win_prob": [0.2072, None],
        }
    )

    # Act
    options = week_options(frame, 5, {})

    # Assert
    assert [(o.team, o.opponent, o.is_home, o.win_prob) for o in options] == [
        ("DAL", "TB", True, 0.7928),
        ("TB", "DAL", False, 0.2072),
        ("TEN", "HOU", True, 0.291),
        ("HOU", "TEN", False, pytest.approx(0.709)),
    ]
    assert options[0].kickoff == "2026-10-08T20:15:00-04:00"
    assert options[0].game_id == "2026_05_TB_DAL"


def test_week_options_skip_games_already_played() -> None:
    """A game with a final score can no longer be picked."""
    # Arrange
    frame = pl.DataFrame(
        {
            "game_id": ["2026_05_TB_DAL", "2026_05_HOU_TEN"],
            "away_abbr": ["TB", "HOU"],
            "home_abbr": ["DAL", "TEN"],
            "home_win_prob": [0.7928, 0.291],
        }
    )

    # Act
    options = week_options(frame, 5, {"2026_05_TB_DAL": (10, 31)})

    # Assert
    assert {o.team for o in options} == {"TEN", "HOU"}
