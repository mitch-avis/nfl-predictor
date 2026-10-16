"""Unit tests for power rankings utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nfl_predictor.reporting.power_rankings import (
    COMPOSITE_PUBLISHED_COLUMNS,
    LeagueStandings,
    build_power_rankings_and_standings,
    composite_points_scale,
    compute_projected_standings,
    fit_bradley_terry_ratings,
    outcome_to_home_prob,
    ratings_to_power_0_to_10,
    scale_ratings_1_to_10,
    win_prob_to_power_0_10,
    win_prob_to_power_1_10,
)
from nfl_predictor.utils.polars import strength_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable


def test_outcome_to_home_prob_encodes_results() -> None:
    """Encodes win/loss/tie into probability targets."""
    # Arrange
    home_score = pd.Series([21, 10, 14])
    away_score = pd.Series([17, 13, 14])

    # Act
    p = outcome_to_home_prob(home_score, away_score, eps=0.1)

    # Assert
    assert p.tolist() == [0.9, 0.1, 0.5]


def test_fit_bradley_terry_ratings_orders_strength() -> None:
    """A consistently winning team should rank above a losing team."""
    # Arrange
    games = pd.DataFrame(
        {
            "home_abbr": ["A", "A", "B", "C"],
            "away_abbr": ["B", "C", "C", "A"],
            "p_home": [0.8, 0.75, 0.6, 0.2],
        }
    )

    # Act
    ratings, _home_adv = fit_bradley_terry_ratings(games, ridge_alpha=1.0)

    # Assert
    assert ratings["A"] > ratings["C"]


def test_scale_ratings_1_to_10_bounds() -> None:
    """Scaled ratings stay in [1, 10] with a stable mapping."""
    # Arrange
    raw = pd.Series({"A": -2.0, "B": 0.0, "C": 2.0})

    # Act
    scaled = scale_ratings_1_to_10(raw)

    # Assert
    assert scaled.min() >= 1.0
    assert scaled.max() <= 10.0
    # Average team (~0) maps to midscale.
    assert scaled["B"] == 5.5


def test_ratings_to_power_0_to_10_bounds() -> None:
    """0-10 scale stays in [0, 10] with a stable mapping."""
    # Arrange
    raw = pd.Series({"A": -10.0, "B": 0.0, "C": 10.0})

    # Act
    scaled = ratings_to_power_0_to_10(raw)

    # Assert
    assert scaled.min() >= 0.0
    assert scaled.max() <= 10.0
    assert scaled["B"] == 5.0


def test_build_power_rankings_and_standings_smoke() -> None:
    """End-to-end smoke test on a tiny synthetic league."""
    # Arrange
    current_records = pd.DataFrame(
        {
            "team_abbr": ["A", "B", "C", "D"],
            "wins": [2, 1, 0, 1],
            "losses": [0, 1, 2, 1],
            "ties": [0, 0, 0, 0],
            "games_played": [2, 2, 2, 2],
        }
    )

    games_for_ratings = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 3],
            "away_abbr": ["B", "C", "D", "B"],
            "home_abbr": ["A", "D", "A", "C"],
            "p_home": [0.8, 0.55, 0.7, 0.6],
        }
    )

    future_games = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [4, 4],
            "away_abbr": ["C", "D"],
            "home_abbr": ["A", "B"],
            "home_win_prob": [0.75, 0.6],
        }
    )

    # Act
    result = build_power_rankings_and_standings(
        LeagueStandings(2025, 3, current_records, future_games), games_for_ratings=games_for_ratings
    )

    # Assert
    assert not result.power_rankings.empty
    assert set(result.power_rankings.columns).issuperset(
        {"team_abbr", "power_rating_1_10", "power_rating_0_10", "rank"}
    )
    assert not result.projected_standings.empty
    assert "projected_wins" in result.projected_standings.columns
    assert not result.projected_division_standings.empty


def test_compute_projected_standings_no_future_games_ok() -> None:
    """Handles postseason weeks (no remaining REG games) without error."""
    # Arrange
    current_records = pd.DataFrame(
        {
            "team_abbr": ["A", "B"],
            "wins": [10, 7],
            "losses": [7, 10],
            "ties": [0, 0],
            "games_played": [17, 17],
        }
    )

    # Intentionally missing away/home columns; should be treated as empty schedule.
    future_games = pd.DataFrame(columns=["home_win_prob"])

    # Act
    out = compute_projected_standings(
        current_records=current_records,
        future_games=future_games,
        season=2025,
    )

    # Assert
    assert set(out["team_abbr"]) == {"A", "B"}
    assert (out["games_remaining"] == 0).all()


def test_projected_standings_list_every_team_before_any_game_is_played() -> None:
    """Before week 1 there are no records yet, but every scheduled team is projected."""
    # Arrange
    no_records = pd.DataFrame(columns=["team_abbr", "wins", "losses", "ties", "games_played"])
    future_games = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "away_abbr": ["AAA", "CCC"],
            "home_abbr": ["BBB", "DDD"],
            "home_win_prob": [0.75, 0.4],
        }
    )

    # Act
    out = compute_projected_standings(
        current_records=no_records, future_games=future_games, season=2026
    ).set_index("team_abbr")

    # Assert
    assert sorted(out.index) == ["AAA", "BBB", "CCC", "DDD"]
    assert (out["wins"] == 0).all()
    assert (out["games_remaining"] == 1).all()
    assert out.loc["BBB", "projected_wins"] == pytest.approx(0.75)
    assert out.loc["CCC", "projected_wins"] == pytest.approx(0.6)


def test_outcome_to_home_prob_binary_mode_cannot_tell_blowouts_from_squeakers() -> None:
    """Binary targets carry only who won."""
    # Arrange
    home = pd.Series([31.0, 20.0, 17.0])
    away = pd.Series([3.0, 19.0, 24.0])

    # Act
    binary = outcome_to_home_prob(home, away)

    # Assert
    # The binary mapping cannot tell a 28-point win from a 1-point win.
    assert binary[0] == binary[1]


def test_outcome_to_home_prob_margin_mode_separates_blowouts_from_squeakers() -> None:
    """Margin-based targets carry how decisively a game was won, not just who won."""
    # Arrange
    home = pd.Series([31.0, 20.0, 17.0])
    away = pd.Series([3.0, 19.0, 24.0])

    # Act
    margin = outcome_to_home_prob(home, away, target="margin")

    # Assert
    # The margin mapping can, and still agrees on who won.
    assert margin[0] > margin[1] > 0.5
    assert margin[2] < 0.5


def test_outcome_to_home_prob_margin_mode_keeps_ties_at_even() -> None:
    """A tie is an even result under either target."""
    # Arrange
    home = pd.Series([20.0])
    away = pd.Series([20.0])

    # Act
    margin = outcome_to_home_prob(home, away, target="margin")

    # Assert
    assert margin[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("weights", "stronger", "weaker"),
    [([1.0, 0.05], "AAA", "BBB"), ([0.05, 1.0], "BBB", "AAA")],
    ids=["old_heavy", "recent_heavy"],
)
def test_fit_bradley_terry_ratings_honors_sample_weights(
    weights: list[float], stronger: str, weaker: str
) -> None:
    """Down-weighting a season pulls the fit toward the weighted-up games."""
    # Arrange
    # AAA beats BBB in the old season; BBB beats AAA in the recent one.
    games = pd.DataFrame(
        {
            "home_abbr": ["AAA", "BBB"],
            "away_abbr": ["BBB", "AAA"],
            "p_home": [0.97, 0.97],
        }
    )

    # Act
    ratings, _ = fit_bradley_terry_ratings(
        games, sample_weights=pd.Series(weights), include_home_advantage=False
    )

    # Assert
    assert ratings[stronger] > ratings[weaker]


def test_fit_bradley_terry_ratings_unweighted_matches_uniform_weights() -> None:
    """Passing uniform weights is the same as passing none."""
    # Arrange
    games = pd.DataFrame(
        {
            "home_abbr": ["AAA", "BBB", "CCC"],
            "away_abbr": ["BBB", "CCC", "AAA"],
            "p_home": [0.9, 0.8, 0.4],
        }
    )

    plain, plain_hfa = fit_bradley_terry_ratings(games)

    # Act
    uniform, uniform_hfa = fit_bradley_terry_ratings(games, sample_weights=pd.Series([1.0] * 3))

    # Assert
    pd.testing.assert_series_equal(plain, uniform)
    assert plain_hfa == pytest.approx(uniform_hfa)


def _composite_snapshot(
    composites: dict[str, float], *, points_per_unit: float = 7.0
) -> pd.DataFrame:
    """Build one week's strength snapshot whose SRS is an exact multiple of the composite."""
    teams = sorted(composites)
    frame = pd.DataFrame({"team_abbr": teams})
    frame["adj_strength_composite"] = [composites[team] for team in teams]
    for column in strength_snapshot.COMPOSITE_WEIGHTS:
        frame[column] = frame["adj_strength_composite"] / 10.0
    frame["adj_srs"] = frame["adj_strength_composite"] * points_per_unit
    frame["strength_games_played"] = 8.0
    return frame


def _empty_records(teams: list[str]) -> pd.DataFrame:
    """Return a zero record for every team."""
    return pd.DataFrame(
        {
            "team_abbr": teams,
            "wins": [0] * len(teams),
            "losses": [0] * len(teams),
            "ties": [0] * len(teams),
            "games_played": [0] * len(teams),
        }
    )


def _composite_rankings(
    snapshot: pd.DataFrame, records: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rank one snapshot with no future games."""
    if records is None:
        records = _empty_records(snapshot["team_abbr"].tolist())
    return build_power_rankings_and_standings(
        LeagueStandings(2024, 8, records, pd.DataFrame()), strength_snapshot=snapshot
    ).power_rankings


def test_composite_ranking_puts_the_strongest_team_first() -> None:
    """Teams are ordered by the adjusted composite, strongest first."""
    # Act
    rankings = _composite_rankings(
        _composite_snapshot({"AAA": -0.4, "BBB": 1.3, "CCC": 0.2, "DDD": -1.1})
    )

    # Assert
    assert rankings["team_abbr"].tolist() == ["BBB", "CCC", "AAA", "DDD"]
    assert rankings["rank"].tolist() == [1, 2, 3, 4]


def test_composite_ranking_publishes_its_components_next_to_the_rank() -> None:
    """A reader can see what drove each rank; the Bradley-Terry columns do not appear."""
    # Act
    rankings = _composite_rankings(_composite_snapshot({"AAA": 0.5, "BBB": -0.5}))

    # Assert
    assert set(COMPOSITE_PUBLISHED_COLUMNS) <= set(rankings.columns)
    assert {"adj_strength_composite", "points_vs_average", "rank"} <= set(rankings.columns)
    assert set(strength_snapshot.COMPOSITE_WEIGHTS) <= set(COMPOSITE_PUBLISHED_COLUMNS)
    assert "home_advantage_logit" not in rankings.columns


def test_composite_points_scale_is_the_slope_of_srs_on_the_composite() -> None:
    """One composite unit is worth as many points as the league's SRS says it is."""
    # Arrange
    snapshot = _composite_snapshot({"AAA": -1.0, "BBB": 0.0, "CCC": 2.0}, points_per_unit=6.5)

    # Act
    scale = composite_points_scale(snapshot)

    # Assert
    assert scale == pytest.approx(6.5)


def test_points_scale_is_unavailable_without_a_positive_srs_relation() -> None:
    """A missing SRS relation yields no scale rather than a misleading one."""
    # Arrange
    no_srs = _composite_snapshot({"AAA": -1.0, "BBB": 0.0, "CCC": 1.0})
    no_srs["adj_srs"] = np.nan

    # Act
    scale = composite_points_scale(no_srs)

    # Assert
    assert np.isnan(scale)


def test_points_scale_is_unavailable_with_an_inverted_srs_relation() -> None:
    """An inverted SRS relation yields no scale rather than a misleading one."""
    # Arrange
    inverted = _composite_snapshot({"AAA": -1.0, "BBB": 0.0, "CCC": 1.0}, points_per_unit=-3.0)

    # Act
    scale = composite_points_scale(inverted)

    # Assert
    assert np.isnan(scale)


def test_points_vs_average_expresses_the_composite_in_points() -> None:
    """The published points column is the composite's distance from average, in points."""
    # Arrange
    composites = {"AAA": -1.0, "BBB": 0.5, "CCC": 2.0}

    # Act
    rankings = _composite_rankings(_composite_snapshot(composites)).set_index("team_abbr")

    # Assert
    mean = float(np.mean(list(composites.values())))
    assert rankings.loc["CCC", "points_vs_average"] == pytest.approx(7.0 * (2.0 - mean))


def test_composite_power_ratings_are_monotone_and_bounded() -> None:
    """Stronger teams never score lower, and even extreme composites stay on the scale."""
    # Act
    rankings = _composite_rankings(
        _composite_snapshot({"AAA": -40.0, "BBB": -1.0, "CCC": 0.0, "DDD": 1.0, "EEE": 40.0})
    )

    # Assert
    one_to_ten = rankings["power_rating_1_10"].to_numpy(dtype=float)
    zero_to_ten = rankings["power_rating_0_10"].to_numpy(dtype=float)
    assert np.all(np.diff(one_to_ten) <= 0.0)
    assert np.all(np.diff(zero_to_ten) <= 0.0)
    assert one_to_ten.min() >= 1.0
    assert one_to_ten.max() <= 10.0
    assert zero_to_ten.min() >= 0.0
    assert zero_to_ten.max() <= 10.0


def test_an_average_team_sits_mid_scale() -> None:
    """A team exactly at the league-average composite maps to the middle of both scales."""
    # Act
    rankings = _composite_rankings(
        _composite_snapshot({"AAA": -1.0, "BBB": 0.0, "CCC": 1.0})
    ).set_index("team_abbr")

    # Assert
    assert rankings.loc["BBB", "power_rating_1_10"] == pytest.approx(5.5)
    assert rankings.loc["BBB", "power_rating_0_10"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("to_scale", "low", "high"),
    [(win_prob_to_power_1_10, 1.0, 10.0), (win_prob_to_power_0_10, 0.0, 10.0)],
    ids=["one_to_ten", "zero_to_ten"],
)
def test_win_probability_scales_are_monotone_and_bounded(
    to_scale: Callable[[np.ndarray], np.ndarray], low: float, high: float
) -> None:
    """The shared probability-to-scale maps span exactly 1-10 and 0-10."""
    # Arrange
    grid = np.linspace(0.0, 1.0, 11)

    # Act
    scaled = to_scale(grid)

    # Assert
    assert np.all(np.diff(scaled) > 0.0)
    assert scaled[0] == pytest.approx(low)
    assert scaled[-1] == pytest.approx(high)


def test_composite_ranking_survives_a_missing_points_scale() -> None:
    """Without a usable points scale the rank still stands; only the 1-10 values go null."""
    # Arrange
    snapshot = _composite_snapshot({"AAA": -1.0, "BBB": 2.0, "CCC": 0.5})
    snapshot["adj_srs"] = np.nan

    # Act
    rankings = _composite_rankings(snapshot)

    # Assert
    assert rankings["team_abbr"].tolist() == ["BBB", "CCC", "AAA"]
    assert rankings["power_rating_1_10"].isna().all()
    assert rankings["power_rating_0_10"].isna().all()


def test_a_team_missing_from_the_snapshot_is_kept_and_ranked_last() -> None:
    """A team with a record but no snapshot row is reported, not silently dropped."""
    # Arrange
    snapshot = _composite_snapshot({"AAA": -1.0, "BBB": 1.0})

    # Act
    rankings = _composite_rankings(snapshot, _empty_records(["AAA", "BBB", "CCC"]))

    # Assert
    assert rankings["team_abbr"].tolist() == ["BBB", "AAA", "CCC"]
    assert rankings.loc[rankings["team_abbr"] == "CCC", "adj_strength_composite"].isna().all()


def test_exactly_one_ratings_source_is_required() -> None:
    """Rankings need a Bradley-Terry game table or a snapshot; neither is an error."""
    # Arrange
    records = _empty_records(["AAA", "BBB"])

    # Act & Assert
    with pytest.raises(ValueError, match="exactly one"):
        build_power_rankings_and_standings(LeagueStandings(2024, 1, records, pd.DataFrame()))


def test_both_ratings_sources_at_once_are_an_error() -> None:
    """Rankings come either from a Bradley-Terry game table or from a snapshot, not both."""
    # Arrange
    records = _empty_records(["AAA", "BBB"])
    snapshot = _composite_snapshot({"AAA": -1.0, "BBB": 1.0})
    games = pd.DataFrame({"home_abbr": ["AAA"], "away_abbr": ["BBB"], "p_home": [0.6]})

    # Act & Assert
    with pytest.raises(ValueError, match="exactly one"):
        build_power_rankings_and_standings(
            LeagueStandings(2024, 1, records, pd.DataFrame()),
            games_for_ratings=games,
            strength_snapshot=snapshot,
        )


def _rotating_schedule(teams: tuple[str, ...], weeks: int) -> list[tuple[int, str, str]]:
    """Return ``(week, away, home)`` for a rotating round-robin with no byes."""
    order = list(teams)
    games: list[tuple[int, str, str]] = []
    for week in range(1, weeks + 1):
        for index in range(len(order) // 2):
            away, home = order[index], order[-1 - index]
            if week % 2 == 0:
                away, home = home, away
            games.append((week, away, home))
        order = [order[0], order[-1], *order[1:-1]]
    return games


def _season_team_games(season: int, weeks: int, strength: dict[str, float]) -> pl.DataFrame:
    """Build per-team-game rows whose per-snap output follows the given team strengths."""
    rows: list[dict[str, object]] = []
    for week, away, home in _rotating_schedule(tuple(sorted(strength)), weeks):
        for team, opponent, is_home in ((away, home, False), (home, away, True)):
            edge = strength[team] - strength[opponent]
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team_abbr": team,
                    "opponent_abbr": opponent,
                    "is_home": is_home,
                    "offensive_snaps": 60.0,
                    "pass_epa_sum": 60.0 * edge,
                    "rush_epa_sum": 30.0 * edge,
                    "points_scored": 20.0 + 20.0 * edge,
                    "points_allowed": 20.0 - 20.0 * edge,
                    "st_epa_for": edge,
                    "st_epa_against": -edge,
                    "st_plays": 10.0,
                }
            )
    return pl.DataFrame(rows)


def _rank_of_breakout(week: int) -> tuple[str, int]:
    """Return the leader and the breakout team's rank at a week of the breakout season."""
    last_year = {"AAA": 0.30, "BBB": 0.15, "CCC": 0.05, "DDD": -0.05, "EEE": -0.15, "FFF": -0.30}
    this_year = {"AAA": 0.15, "BBB": 0.10, "CCC": 0.05, "DDD": -0.05, "EEE": -0.20, "FFF": 0.45}
    prior = strength_snapshot.build_strength_snapshot(
        _season_team_games(2023, 15, last_year), season=2023, week=19
    )
    games = _season_team_games(2024, 15, this_year)
    snapshot = strength_snapshot.build_strength_snapshot(
        games,
        season=2024,
        week=week,
        prior=strength_snapshot.StrengthPrior(prior),
        teams=sorted(this_year),
    )
    rankings = _composite_rankings(pd.DataFrame(snapshot.to_dicts()))
    leader = str(rankings.iloc[0]["team_abbr"])
    return leader, int(rankings.loc[rankings["team_abbr"] == "FFF", "rank"].iloc[0])


def test_a_breakout_team_is_held_down_early_in_the_season() -> None:
    """Early on the regressed prior still holds last year's worst team down."""
    # Act
    early_leader, early_rank = _rank_of_breakout(2)

    # Assert
    assert early_leader != "FFF"
    assert early_rank > 1


def test_a_breakout_team_ranks_first_late_in_the_season() -> None:
    """Last year's worst team that dominates this year leads once the season has spoken.

    Early on the regressed prior still holds it down; by week 16 its in-season results
    carry most of the weight and it ranks first.
    """
    # Act
    late_leader, late_rank = _rank_of_breakout(16)

    # Assert
    assert late_leader == "FFF"
    assert late_rank == 1
