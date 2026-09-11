"""Tests for the pre-week schedule-adjusted team strength snapshot."""

from __future__ import annotations

import math

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import strength_snapshot


def _team_game(
    season: int,
    week: int,
    team: str,
    opponent: str,
    *,
    is_home: bool,
    pass_epa: float = 0.0,
    rush_epa: float = 0.0,
    snaps: float = 60.0,
    points_scored: float = 20.0,
    points_allowed: float = 17.0,
    st_for: float = 0.0,
    st_against: float = 0.0,
    st_plays: float = 10.0,
) -> dict[str, object]:
    """Build one synthetic team-game row in the shape the snapshot builder consumes."""
    return {
        "season": season,
        "week": week,
        "team_abbr": team,
        "opponent_abbr": opponent,
        "is_home": is_home,
        "offensive_snaps": snaps,
        "pass_epa_sum": pass_epa * snaps,
        "rush_epa_sum": rush_epa * snaps,
        "points_scored": points_scored,
        "points_allowed": points_allowed,
        "st_epa_for": st_for,
        "st_epa_against": st_against,
        "st_plays": st_plays,
    }


def _round_robin(
    season: int,
    weeks: tuple[int, ...],
    teams: tuple[str, ...],
    *,
    offense: dict[str, float] | None = None,
    epa_scale: float = 1.0,
) -> pl.DataFrame:
    """Build a balanced double round-robin spread evenly across the given weeks.

    Every ordered ``(home, away)`` pair is played exactly once, so each team hosts
    and visits every other team and the solve is not confounded by an unbalanced
    home/away split.
    """
    offense = offense or {}
    pairs = [(home, away) for home in teams for away in teams if home != away]
    rows: list[dict[str, object]] = []
    for index, (home, away) in enumerate(pairs):
        week = weeks[index % len(weeks)]
        for team, opponent, home_flag in ((home, away, True), (away, home, False)):
            strength = offense.get(team, 0.0) * epa_scale
            rows.append(
                _team_game(
                    season,
                    week,
                    team,
                    opponent,
                    is_home=home_flag,
                    pass_epa=strength,
                    rush_epa=strength / 2.0,
                    points_scored=20.0 + 10.0 * strength,
                    points_allowed=20.0 - 10.0 * strength,
                    st_for=strength,
                )
            )
    return pl.DataFrame(rows)


_TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
_OFFENSE = {"AAA": 0.30, "BBB": 0.18, "CCC": 0.06, "DDD": -0.06, "EEE": -0.18, "FFF": -0.30}


def test_snapshot_emits_every_documented_column_for_every_team() -> None:
    """The snapshot names one row per team with the full published column set."""
    games = _round_robin(2024, (1, 2, 3, 4), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(games, season=2024, week=5)

    assert snapshot.columns == ["team_abbr", *constants.STRENGTH_SNAPSHOT_STATS]
    assert sorted(snapshot["team_abbr"].to_list()) == sorted(_TEAMS)


def test_snapshot_orders_teams_by_their_known_offensive_strength() -> None:
    """A team given a stronger offense ranks above a weaker one on the adjusted value."""
    games = _round_robin(2024, (1, 2, 3, 4, 5, 6), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=7, blend_prior=False
    )
    ranked = snapshot.sort("adj_off_pass_epa_snap", descending=True)["team_abbr"].to_list()

    assert ranked == sorted(_TEAMS, key=lambda team: -_OFFENSE[team])


def test_adjusted_components_are_centered_across_the_league() -> None:
    """Offense and defense coefficients each average to zero across the league."""
    games = _round_robin(2024, (1, 2, 3, 4), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=5, blend_prior=False
    )

    for column in (
        "adj_off_pass_epa_snap",
        "adj_off_rush_epa_snap",
        "adj_def_pass_epa_snap",
        "adj_def_rush_epa_snap",
        "adj_srs",
        "st_rating",
    ):
        assert snapshot[column].mean() == pytest.approx(0.0, abs=1e-6), column


def test_snapshot_uses_only_games_before_the_target_week() -> None:
    """Rewriting the target week and every later week cannot move a pre-week value."""
    baseline = strength_snapshot.build_strength_snapshot(
        _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE), season=2024, week=4
    )
    with_future = strength_snapshot.build_strength_snapshot(
        pl.concat(
            [
                _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE),
                _round_robin(2024, (4, 5, 6), _TEAMS, offense=_OFFENSE, epa_scale=50.0),
            ]
        ),
        season=2024,
        week=4,
    )

    assert_snapshots_equal(baseline, with_future)


def test_snapshot_ignores_other_seasons() -> None:
    """Games from a different season never enter the solve for this one."""
    current = _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE)
    decoy = _round_robin(2023, (1, 2, 3), _TEAMS, offense=_OFFENSE, epa_scale=50.0)

    baseline = strength_snapshot.build_strength_snapshot(current, season=2024, week=4)
    with_decoy = strength_snapshot.build_strength_snapshot(
        pl.concat([current, decoy]), season=2024, week=4
    )

    assert_snapshots_equal(baseline, with_decoy)


def test_playoff_week_uses_the_full_regular_season_and_ignores_playoff_games() -> None:
    """Playoff rows switch from "before week N" to the whole regular season.

    The playoff games are placed *before* the target week on purpose: a naive
    "everything before week N" cutoff would swallow them, so this pins the branch
    rather than relying on the target week already being the earliest playoff week.
    """
    regular_weeks = (1, 2, 3)
    regular_season_weeks = constants.get_regular_season_weeks(2024)
    earlier_playoff_weeks = (regular_season_weeks + 1, regular_season_weeks + 2)
    target_week = regular_season_weeks + 3
    regular = _round_robin(2024, regular_weeks, _TEAMS, offense=_OFFENSE)
    playoff = _round_robin(2024, earlier_playoff_weeks, _TEAMS, offense=_OFFENSE, epa_scale=50.0)

    full_season = strength_snapshot.build_strength_snapshot(
        regular, season=2024, week=regular_season_weeks + 1
    )
    later_round = strength_snapshot.build_strength_snapshot(
        pl.concat([regular, playoff]), season=2024, week=target_week
    )

    assert_snapshots_equal(full_season, later_round)


def test_week_one_without_a_prior_snapshot_returns_nulls_not_zeros() -> None:
    """With no in-season games and no prior season, the snapshot is null, not a fake zero."""
    games = _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(games, season=2024, week=1)

    assert snapshot.height == len(_TEAMS)
    for column in constants.STRENGTH_SNAPSHOT_STATS:
        if column == "strength_games_played":
            # Zero games played is a fact, not a missing value.
            assert snapshot[column].to_list() == [0.0] * snapshot.height
            continue
        assert snapshot[column].null_count() == snapshot.height, column


def test_week_one_falls_back_to_the_regressed_prior_snapshot() -> None:
    """Week 1 carries the previous season's final snapshot regressed toward zero."""
    prior = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            **{column: [0.30, -0.30] for column in constants.STRENGTH_SNAPSHOT_STATS},
        }
    )
    games = _round_robin(2024, (1, 2), ("AAA", "BBB"), offense={"AAA": 0.2})

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=1, prior_snapshot=prior
    )

    expected = 0.30 * (1.0 - constants.WEEK1_REGRESSION_FACTOR)
    row = snapshot.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    assert row["adj_off_pass_epa_snap"] == pytest.approx(expected)


def test_prior_blend_weight_follows_the_documented_games_formula() -> None:
    """The in-season weight is games / (games + K), so the prior fades as games accrue."""
    prior = pl.DataFrame(
        {
            "team_abbr": list(_TEAMS),
            **{column: [1.0] * len(_TEAMS) for column in constants.STRENGTH_SNAPSHOT_STATS},
        }
    )
    games = _round_robin(2024, (1, 2), _TEAMS, offense=_OFFENSE)

    blended = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=3, prior_snapshot=prior
    )
    in_season_only = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=3, prior_snapshot=prior, blend_prior=False
    )

    played = blended.filter(pl.col("team_abbr") == "AAA")["strength_games_played"].item()
    weight = played / (played + constants.PRIOR_BLEND_GAMES)
    regressed_prior = 1.0 * (1.0 - constants.WEEK1_REGRESSION_FACTOR)
    solved = in_season_only.filter(pl.col("team_abbr") == "AAA")["adj_off_pass_epa_snap"].item()
    expected = weight * solved + (1.0 - weight) * regressed_prior

    got = blended.filter(pl.col("team_abbr") == "AAA")["adj_off_pass_epa_snap"].item()
    assert got == pytest.approx(expected)


def test_disabling_the_prior_blend_leaves_the_in_season_solve_untouched() -> None:
    """The ablation switch removes the prior entirely rather than down-weighting it."""
    prior = pl.DataFrame(
        {
            "team_abbr": list(_TEAMS),
            **{column: [5.0] * len(_TEAMS) for column in constants.STRENGTH_SNAPSHOT_STATS},
        }
    )
    games = _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE)

    without_prior = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=4, prior_snapshot=prior, blend_prior=False
    )
    no_prior_supplied = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=4, blend_prior=False
    )

    assert_snapshots_equal(without_prior, no_prior_supplied)


def test_composite_uses_the_documented_weights_over_standardized_components() -> None:
    """The composite is the weighted sum of within-snapshot z-scored components."""
    games = _round_robin(2024, (1, 2, 3, 4), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=5, blend_prior=False
    )

    expected = [0.0] * snapshot.height
    for column, weight in strength_snapshot.COMPOSITE_WEIGHTS.items():
        values = snapshot[column].to_list()
        mean = sum(values) / len(values)
        deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        for index, value in enumerate(values):
            expected[index] += weight * ((value - mean) / deviation if deviation else 0.0)

    for index, value in enumerate(snapshot["adj_strength_composite"].to_list()):
        assert value == pytest.approx(expected[index], abs=1e-9)


def test_composite_weights_are_documented_and_sum_to_one() -> None:
    """The default display weights are named per component and form a convex blend."""
    assert set(strength_snapshot.COMPOSITE_WEIGHTS) == {
        "adj_off_pass_epa_snap",
        "adj_off_rush_epa_snap",
        "adj_def_pass_epa_snap",
        "adj_def_rush_epa_snap",
        "st_rating",
    }
    assert sum(strength_snapshot.COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)
    assert all(weight > 0.0 for weight in strength_snapshot.COMPOSITE_WEIGHTS.values())


def test_a_better_defense_earns_a_higher_defense_rating() -> None:
    """A team that allows less EPA per snap scores higher, so the sign is not inverted."""
    rows: list[dict[str, object]] = []
    for week in (1, 2, 3):
        # STOUT smothers every opponent; SIEVE lets the same opponents move the ball.
        rows.append(_team_game(2024, week, "OFFA", "STOUT", is_home=False, pass_epa=-0.20))
        rows.append(_team_game(2024, week, "STOUT", "OFFA", is_home=True, pass_epa=0.0))
        rows.append(_team_game(2024, week, "OFFB", "SIEVE", is_home=False, pass_epa=0.20))
        rows.append(_team_game(2024, week, "SIEVE", "OFFB", is_home=True, pass_epa=0.0))
        rows.append(_team_game(2024, week, "OFFA", "SIEVE", is_home=True, pass_epa=0.20))
        rows.append(_team_game(2024, week, "SIEVE", "OFFA", is_home=False, pass_epa=0.0))
        rows.append(_team_game(2024, week, "OFFB", "STOUT", is_home=True, pass_epa=-0.20))
        rows.append(_team_game(2024, week, "STOUT", "OFFB", is_home=False, pass_epa=0.0))

    snapshot = strength_snapshot.build_strength_snapshot(
        pl.DataFrame(rows), season=2024, week=4, blend_prior=False
    )
    ratings = dict(zip(snapshot["team_abbr"], snapshot["adj_def_pass_epa_snap"], strict=True))

    assert ratings["STOUT"] > ratings["SIEVE"]


def test_home_field_term_is_shared_by_every_team() -> None:
    """The solve fits one home-field value, so every row reports the same number."""
    games = _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE)

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=4, blend_prior=False
    )

    assert snapshot["adj_hfa"].n_unique() == 1


def test_empty_input_returns_a_typed_empty_snapshot() -> None:
    """No team-game rows at all yields the empty frame, not an exception."""
    snapshot = strength_snapshot.build_strength_snapshot(
        strength_snapshot.empty_snapshot(), season=2024, week=4
    )

    assert snapshot.height == 0
    assert snapshot.columns == ["team_abbr", *constants.STRENGTH_SNAPSHOT_STATS]


def test_special_teams_rating_is_null_when_the_source_is_missing() -> None:
    """A season without special-teams play data emits nulls rather than a zero rating."""
    games = _round_robin(2024, (1, 2, 3), _TEAMS, offense=_OFFENSE).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("st_epa_for"),
        pl.lit(None, dtype=pl.Float64).alias("st_epa_against"),
        pl.lit(None, dtype=pl.Float64).alias("st_plays"),
    )

    snapshot = strength_snapshot.build_strength_snapshot(
        games, season=2024, week=4, blend_prior=False
    )

    assert snapshot["st_rating"].null_count() == snapshot.height
    # The rest of the snapshot still solves.
    assert snapshot["adj_off_pass_epa_snap"].null_count() == 0


def assert_snapshots_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Assert two snapshots agree on every team and every published value."""
    assert left.columns == right.columns
    left_sorted = left.sort("team_abbr")
    right_sorted = right.sort("team_abbr")
    assert left_sorted["team_abbr"].to_list() == right_sorted["team_abbr"].to_list()
    for column in constants.STRENGTH_SNAPSHOT_STATS:
        for lhs, rhs in zip(left_sorted[column], right_sorted[column], strict=True):
            if lhs is None or rhs is None:
                assert lhs is rhs, column
            else:
                assert lhs == pytest.approx(rhs, abs=1e-9), column
