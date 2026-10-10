"""Tests for the preseason prior of the team fit: carryover slopes, the fade, and the means."""

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from nfl_predictor.ratings.ridge import UnitPrior, fit_unit_ridge
from nfl_predictor.ratings.team_prior import (
    MIN_CARRYOVER_PAIRS,
    PRIOR_HORIZON_GAMES,
    CarryoverSlopes,
    PriorHistory,
    SeasonPrior,
    carryover_slope,
    fade,
    games_played,
    prior_means,
    prior_without_team,
    read_team_prior_table,
    snapshot_fit,
    snapshot_prior,
    snapshot_ratings,
    team_prior_table,
)
from nfl_predictor.ratings.team_rating import TEAM_UNIT_COLUMNS, fit_team_ratings, scrimmage_rows

if TYPE_CHECKING:
    from collections.abc import Mapping

_TEAMS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
_PLAYS = 60
_ST_PLAYS = 12
_SPREAD = 0.05


def _round_robin_weeks(teams: tuple[str, ...]) -> list[list[tuple[str, str]]]:
    """Return a double round robin by week (circle method): each team plays once a week."""
    rotation = list(teams)
    weeks: list[list[tuple[str, str]]] = []
    for _ in range(len(teams) - 1):
        half = len(rotation) // 2
        weeks.append(list(zip(rotation[:half], reversed(rotation[half:]), strict=True)))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return weeks + [[(away, home) for home, away in week] for week in weeks]


def _season_logs(
    offense: Mapping[str, float],
    defense: Mapping[str, float],
    *,
    noise: float = 0.0,
    seed: int = 0,
    teams: tuple[str, ...] = _TEAMS,
) -> pl.DataFrame:
    """Return a season of team-game rows from known per-play effects, with optional noise."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for week_index, week in enumerate(_round_robin_weeks(teams)):
        for home, away in week:
            game_id = f"w{week_index + 1:02d}_{away}_{home}"
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                sign = 1.0 if is_home else -1.0
                epa = 0.02 + offense[team] - defense[opponent] + 0.01 * sign
                epa += noise * float(rng.standard_normal())
                rows.append(
                    {
                        "game_id": game_id,
                        "week": week_index + 1,
                        "team": team,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "offensive_snaps": _PLAYS,
                        "offensive_epa": epa * _PLAYS,
                        "st_plays": _ST_PLAYS,
                        "st_epa": 0.0,
                    }
                )
    return pl.DataFrame(rows)


def _carryover_league(
    seasons: int,
    carryover: float,
    *,
    noise: float,
    seed: int,
    teams: tuple[str, ...] = _TEAMS,
) -> tuple[dict[int, pl.DataFrame], dict[int, pl.DataFrame]]:
    """Return seasons whose true effects carry over, and each season's true per-game ratings.

    Each season's effects are ``carryover`` times the last season's plus fresh variation, so the
    true carryover slope is ``carryover``. Every effect has a spread of ``_SPREAD`` per play.
    """
    rng = np.random.default_rng(seed)
    fresh = math.sqrt(1.0 - carryover**2)
    offense = rng.normal(0.0, _SPREAD, len(teams))
    defense = rng.normal(0.0, _SPREAD, len(teams))
    league: dict[int, pl.DataFrame] = {}
    truth: dict[int, pl.DataFrame] = {}
    for season in range(2000, 2000 + seasons):
        centered_offense, centered_defense = offense - offense.mean(), defense - defense.mean()
        league[season] = _season_logs(
            dict(zip(teams, centered_offense.tolist(), strict=True)),
            dict(zip(teams, centered_defense.tolist(), strict=True)),
            noise=noise,
            seed=seed + season,
            teams=teams,
        )
        truth[season] = pl.DataFrame(
            {
                "team": teams,
                "offense_rating": centered_offense * _PLAYS,
                "defense_rating": centered_defense * _PLAYS,
            }
        )
        offense = carryover * offense + fresh * rng.normal(0.0, _SPREAD, len(teams))
        defense = carryover * defense + fresh * rng.normal(0.0, _SPREAD, len(teams))
    return league, truth


def _history(league: dict[int, pl.DataFrame]) -> PriorHistory:
    """Return a prior history over ``league``, whose first season is its earliest."""
    return PriorHistory(league.__getitem__, first_season=min(league))


def _season_prior(
    offense: Mapping[str, float], defense: Mapping[str, float], slope: float = 1.0
) -> SeasonPrior:
    """Return a prior built from the given previous-season effects and one slope for both sides."""
    return SeasonPrior(
        season=2001,
        offense=offense,
        defense=defense,
        slopes=CarryoverSlopes(offense=slope, defense=slope, pairs=MIN_CARRYOVER_PAIRS),
    )


def _rating(ratings: pl.DataFrame, team: str, column: str) -> float:
    """Return one team's value for one rating column."""
    return float(ratings.filter(pl.col("team") == team).get_column(column).item())


@pytest.mark.parametrize(
    ("games", "horizon", "expected"),
    [(0, 6.0, 1.0), (3, 6.0, 0.5), (6, 6.0, 0.0), (9, 6.0, 0.0), (5, math.inf, 1.0)],
)
def test_fade_shrinks_the_prior_linearly_to_zero_at_the_horizon(
    games: int, horizon: float, expected: float
) -> None:
    # Act
    weight = fade(games, horizon)

    # Assert
    assert weight == pytest.approx(expected)


def test_carryover_slope_is_the_least_squares_slope_through_the_origin() -> None:
    # Act
    slope = carryover_slope([1.0, 2.0, -1.0], [2.0, 4.2, -1.8])

    # Assert
    assert slope == pytest.approx(12.2 / 6.0)


def test_carryover_slope_without_previous_spread_raises() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="no spread"):
        carryover_slope([0.0, 0.0], [1.0, -1.0])


def _teams(count: int) -> tuple[str, ...]:
    """Return ``count`` team codes for a larger synthetic league."""
    return tuple(f"T{index:02d}" for index in range(count))


def test_slopes_recover_the_true_carryover_where_published_on_published_runs_low() -> None:
    """The near-unpenalized slope estimates the carryover; regressing shrunk fits on shrunk fits
    shrinks it a second time.

    Over seeds 0-11 of this league the near-unpenalized slopes fell within 0.09 of 0.7 and the
    published-on-published slope at least 0.16 below them, so the tolerances hold with room.
    """
    # Arrange
    teams = _teams(24)
    league, _ = _carryover_league(30, 0.7, noise=0.2, seed=11, teams=teams)
    history = _history(league)
    seasons = sorted(league)[1:]
    published_pairs = [
        (history.published_scrimmage(season - 1), history.published_scrimmage(season))
        for season in seasons
    ]
    naive = carryover_slope(
        [before.offense[team] for before, _ in published_pairs for team in teams],
        [after.offense[team] for _, after in published_pairs for team in teams],
    )

    # Act
    slopes = history.slopes(max(league) + 1)

    # Assert
    assert slopes is not None
    assert slopes.pairs == len(seasons)
    assert slopes.offense == pytest.approx(0.7, abs=0.12)
    assert slopes.defense == pytest.approx(0.7, abs=0.12)
    assert naive < slopes.offense - 0.1


def test_prior_beats_the_zero_prior_early_in_a_season() -> None:
    """Over seeds 0-19 of this league the prior's squared error after two games was at most 0.86
    times the zero prior's."""
    # Arrange
    teams = _teams(16)
    league, truth = _carryover_league(16, 0.8, noise=0.2, seed=3, teams=teams)
    history = _history(league)
    season = max(league)
    early = league[season].filter(pl.col("week") <= 2)
    penalties = history.penalties(season)
    season_prior = history.season_prior(season)
    without = snapshot_ratings(early, teams, penalties, None, 6.0)

    # Act
    with_prior = snapshot_ratings(early, teams, penalties, season_prior, 6.0)

    # Assert
    assert season_prior is not None
    assert _squared_error(with_prior, truth[season]) < _squared_error(without, truth[season])


def _squared_error(ratings: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Return the summed squared offense and defense rating error against ``truth``."""
    joined = ratings.join(truth, on="team", suffix="_true")
    return float(
        joined.select(
            (pl.col("offense_rating") - pl.col("offense_rating_true")) ** 2
            + (pl.col("defense_rating") - pl.col("defense_rating_true")) ** 2
        )
        .sum()
        .item()
    )


def test_each_prior_pulls_its_own_side_the_right_way() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    one_game = _season_logs(flat, flat).filter(pl.col("week") == 1)
    season_prior = _season_prior({**flat, "AAA": 0.08}, {**flat, "BBB": 0.08})
    penalties = fit_team_ratings(one_game, scrimmage_lambda=200.0, special_teams_lambda=200.0)

    # Act
    ratings = snapshot_ratings(one_game, _TEAMS, penalties, season_prior, 6.0)

    # Assert
    assert _rating(ratings, "AAA", "offense_rating") > 0.0
    assert _rating(ratings, "BBB", "defense_rating") > 0.0


def test_overwhelming_penalty_rates_each_team_at_its_centered_prior() -> None:
    """A rating is a per-play prior mean times the snapshot's plays per game."""
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(pl.col("week") <= 3)
    season_prior = _season_prior({**flat, "AAA": 0.08}, flat, slope=0.5)
    penalties = fit_team_ratings(logs, scrimmage_lambda=1e12, special_teams_lambda=40.0)
    means = prior_means(season_prior, games_played(logs, _TEAMS), 6.0, _TEAMS)

    # Act
    ratings = snapshot_ratings(logs, _TEAMS, penalties, season_prior, 6.0)

    # Assert
    assert _rating(ratings, "AAA", "offense_rating") == pytest.approx(
        means.offense["AAA"] * _PLAYS, abs=1e-6
    )
    assert _rating(ratings, "BBB", "offense_rating") == pytest.approx(
        means.offense["BBB"] * _PLAYS, abs=1e-6
    )


def test_prior_means_fade_by_games_played_and_center_over_the_fitted_teams() -> None:
    # Arrange
    season_prior = _season_prior({"AAA": 0.1, "BBB": 0.1}, {"AAA": 0.0, "BBB": 0.2}, slope=0.5)

    # Act
    means = prior_means(season_prior, {"AAA": 0, "BBB": 6}, 6.0, ("AAA", "BBB"))

    # Assert
    assert means.offense == pytest.approx({"AAA": 0.025, "BBB": -0.025})
    assert means.defense == pytest.approx({"AAA": 0.0, "BBB": 0.0})


def test_prior_means_shift_an_unplayed_team_by_the_fitted_teams_centering() -> None:
    # Arrange
    season_prior = _season_prior(
        {"AAA": 0.1, "BBB": -0.1, "CCC": 0.3}, {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}
    )

    # Act
    means = prior_means(season_prior, {"AAA": 1, "BBB": 1, "CCC": 0}, 2.0, ("AAA", "BBB"))

    # Assert
    assert means.offense == pytest.approx({"AAA": 0.05, "BBB": -0.05, "CCC": 0.3})


def test_prior_means_without_a_previous_season_effect_raise() -> None:
    # Arrange
    season_prior = _season_prior({"AAA": 0.1}, {"AAA": 0.0})

    # Act & Assert
    with pytest.raises(ValueError, match="BBB"):
        prior_means(season_prior, {"AAA": 1, "BBB": 1}, 6.0, ("AAA", "BBB"))


def test_centered_prior_keeps_the_effects_averaging_zero_with_byes() -> None:
    # Arrange
    offense = dict(zip(_TEAMS, np.linspace(-0.06, 0.08, len(_TEAMS)), strict=True))
    defense = dict(zip(_TEAMS, np.linspace(0.05, -0.03, len(_TEAMS)), strict=True))
    logs = _season_logs(offense, defense, noise=0.2, seed=4).filter(
        (pl.col("week") <= 3) & ~((pl.col("week") == 2) & pl.col("team").is_in(["AAA", "HHH"]))
    )
    logs = logs.filter(pl.col("game_id").is_in(_complete_games(logs)))
    season_prior = _season_prior(offense, defense, slope=0.6)
    penalties = fit_team_ratings(logs, scrimmage_lambda=300.0, special_teams_lambda=40.0)

    # Act
    ratings = snapshot_ratings(logs, _TEAMS, penalties, season_prior, 6.0)

    # Assert
    assert ratings.get_column("offense_rating").mean() == pytest.approx(0.0, abs=1e-12)
    assert ratings.get_column("defense_rating").mean() == pytest.approx(0.0, abs=1e-12)


def _complete_games(logs: pl.DataFrame) -> list[str]:
    """Return the games that still have both teams' rows."""
    counts = logs.group_by("game_id").len()
    return counts.filter(pl.col("len") == 2).get_column("game_id").to_list()


def test_snapshot_ratings_rate_a_team_without_games_at_its_prior() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(
        (pl.col("week") == 1) & ~pl.col("team").is_in(["AAA", "HHH"])
    )
    logs = logs.filter(pl.col("game_id").is_in(_complete_games(logs)))
    season_prior = _season_prior({**flat, "AAA": 0.08}, {**flat, "AAA": -0.04}, slope=0.5)
    penalties = fit_team_ratings(logs, scrimmage_lambda=200.0, special_teams_lambda=40.0)
    fitted = sorted(logs.get_column("team").unique().to_list())
    means = prior_means(season_prior, games_played(logs, _TEAMS), 6.0, fitted)

    # Act
    ratings = snapshot_ratings(logs, _TEAMS, penalties, season_prior, 6.0)

    # Assert
    assert _rating(ratings, "AAA", "offense_rating") == pytest.approx(means.offense["AAA"] * _PLAYS)
    assert _rating(ratings, "AAA", "defense_rating") == pytest.approx(means.defense["AAA"] * _PLAYS)
    assert _rating(ratings, "AAA", "special_teams_rating") == 0.0


def test_snapshot_ratings_without_a_prior_match_the_published_fit() -> None:
    # Arrange
    offense = dict(zip(_TEAMS, np.linspace(-0.06, 0.08, len(_TEAMS)), strict=True))
    logs = _season_logs(offense, dict.fromkeys(_TEAMS, 0.0), noise=0.2).filter(pl.col("week") <= 4)
    penalties = fit_team_ratings(logs, scrimmage_lambda=150.0, special_teams_lambda=40.0)
    expected = fit_team_ratings(logs, scrimmage_lambda=150.0, special_teams_lambda=40.0)

    # Act
    ratings = snapshot_ratings(logs, _TEAMS, penalties, None, 6.0)

    # Assert
    assert ratings.sort("team").equals(expected.ratings.sort("team"))


def test_games_played_counts_each_teams_games_and_zero_for_the_rest() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(pl.col("week") <= 2)

    # Act
    games = games_played(logs, (*_TEAMS, "ZZZ"))

    # Assert
    assert games == {**dict.fromkeys(_TEAMS, 2), "ZZZ": 0}


def test_history_penalties_of_the_first_season_are_its_own_cross_validation() -> None:
    # Arrange
    league, _ = _carryover_league(3, 0.7, noise=0.3, seed=1)
    history = _history(league)

    # Act
    penalties = history.penalties(2000)

    # Assert
    assert penalties is history.cross_validated_fit(2000)


def test_history_penalties_come_from_the_previous_seasons_cross_validation() -> None:
    # Arrange
    league, _ = _carryover_league(3, 0.7, noise=0.3, seed=1)
    history = _history(league)

    # Act
    penalties = history.penalties(2001)

    # Assert
    assert penalties is history.cross_validated_fit(2000)


def test_history_has_no_prior_before_enough_season_pairs() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    history = _history(league)

    # Act
    too_early = history.season_prior(2003)

    # Assert
    assert too_early is None


def test_history_first_prior_pools_three_pairs_and_the_previous_published_fit() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    history = _history(league)

    # Act
    first_prior = history.season_prior(2004)

    # Assert
    assert first_prior is not None
    assert first_prior.slopes.pairs == MIN_CARRYOVER_PAIRS
    assert first_prior.offense == history.published_scrimmage(2003).offense


def test_history_reads_only_the_seasons_before_the_one_rated() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    earlier = {season: logs for season, logs in league.items() if season < 2004}
    expected = _history(league).season_prior(2004)
    restricted = PriorHistory(earlier.__getitem__, first_season=2000)

    # Act
    actual = restricted.season_prior(2004)

    # Assert
    assert actual == expected


def test_unit_prior_type_is_what_the_fit_takes() -> None:
    # Arrange
    season_prior = _season_prior({"AAA": 0.1, "BBB": -0.1}, {"AAA": 0.0, "BBB": 0.0})

    # Act
    means = prior_means(season_prior, {"AAA": 0, "BBB": 0}, 6.0, ("AAA", "BBB"))

    # Assert
    assert isinstance(means, UnitPrior)


def test_carryover_slope_of_unpaired_values_raises() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="paired values"):
        carryover_slope([1.0, 2.0], [1.0])


def test_prior_means_for_a_fitted_team_without_a_game_count_raise() -> None:
    # Arrange
    season_prior = _season_prior({"AAA": 0.1, "BBB": 0.0}, {"AAA": 0.0, "BBB": 0.0})

    # Act & Assert
    with pytest.raises(ValueError, match="without a game count: BBB"):
        prior_means(season_prior, {"AAA": 1}, 6.0, ("AAA", "BBB"))


def test_snapshot_ratings_with_a_prior_but_no_penalties_raise() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(pl.col("week") == 1)

    # Act & Assert
    with pytest.raises(ValueError, match="needs the season's penalties"):
        snapshot_ratings(logs, _TEAMS, None, _season_prior(flat, flat), 6.0)


def test_snapshot_ratings_with_a_one_sided_game_raise() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    one_sided = _season_logs(flat, flat).filter((pl.col("week") == 1) & (pl.col("team") != "AAA"))
    penalties = fit_team_ratings(one_sided, scrimmage_lambda=50.0, special_teams_lambda=50.0)

    # Act & Assert
    with pytest.raises(ValueError, match="both teams' scrimmage rows"):
        snapshot_ratings(one_sided, _TEAMS, penalties, _season_prior(flat, flat), 6.0)


def test_history_fits_each_season_once() -> None:
    # Arrange
    league, _ = _carryover_league(2, 0.7, noise=0.3, seed=1)
    history = _history(league)
    first = history.near_unpenalized_scrimmage(2000)

    # Act
    again = history.near_unpenalized_scrimmage(2000)

    # Assert
    assert again is first


def test_snapshot_fit_returns_the_means_it_fit_with() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(pl.col("week") <= 2)
    season_prior = _season_prior({**flat, "AAA": 0.08}, flat, slope=0.5)
    penalties = fit_team_ratings(logs, scrimmage_lambda=200.0, special_teams_lambda=40.0)
    expected = prior_means(season_prior, games_played(logs, _TEAMS), 6.0, _TEAMS)

    # Act
    ratings, means = snapshot_fit(logs, _TEAMS, penalties, season_prior, 6.0)

    # Assert
    assert means == expected
    assert ratings.equals(snapshot_ratings(logs, _TEAMS, penalties, season_prior, 6.0))


def test_snapshot_fit_without_a_prior_has_no_means() -> None:
    # Arrange
    flat = dict.fromkeys(_TEAMS, 0.0)
    logs = _season_logs(flat, flat).filter(pl.col("week") <= 2)
    penalties = fit_team_ratings(logs, scrimmage_lambda=200.0, special_teams_lambda=40.0)

    # Act
    _, means = snapshot_fit(logs, _TEAMS, penalties, None, 6.0)

    # Assert
    assert means is None


def _flat_season(weeks: int) -> pl.DataFrame:
    """Return the first ``weeks`` weeks of a season of average teams."""
    zero = dict.fromkeys(_TEAMS, 0.0)
    return _season_logs(zero, zero).filter(pl.col("week") <= weeks)


def _spread_prior() -> SeasonPrior:
    """Return a prior whose previous-season effects differ team by team."""
    return _season_prior(
        {team: 0.01 * index for index, team in enumerate(_TEAMS)},
        {team: -0.005 * index for index, team in enumerate(_TEAMS)},
        slope=0.7,
    )


def test_snapshot_prior_fades_and_centers_by_the_games_played_so_far() -> None:
    # Arrange
    games = _flat_season(3)
    expected = prior_means(
        _spread_prior(), games_played(games, _TEAMS), PRIOR_HORIZON_GAMES, sorted(_TEAMS)
    )

    # Act
    prior = snapshot_prior(_spread_prior(), games)

    # Assert
    assert prior == expected


def test_snapshot_prior_is_gone_once_every_team_reaches_the_horizon() -> None:
    # Arrange
    games = _flat_season(int(PRIOR_HORIZON_GAMES))

    # Act
    prior = snapshot_prior(_spread_prior(), games)

    # Assert
    assert prior is None


def test_snapshot_prior_without_a_season_prior_is_none() -> None:
    # Act
    prior = snapshot_prior(None, _flat_season(2))

    # Assert
    assert prior is None


def test_season_prior_without_a_team_refits_the_previous_season_without_its_games() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    history = _history(league)
    others = league[2003].filter((pl.col("team") != "AAA") & (pl.col("opponent_team") != "AAA"))
    expected = fit_unit_ridge(
        scrimmage_rows(others),
        TEAM_UNIT_COLUMNS,
        ridge_lambda=history.penalties(2003).scrimmage_lambda,
    )
    full = history.season_prior(2004)

    # Act
    prior = history.season_prior_without(2004, "AAA")

    # Assert
    assert prior is not None
    assert full is not None
    assert (prior.offense, prior.defense) == (expected.offense, expected.defense)
    assert "AAA" not in prior.offense
    assert prior.slopes == full.slopes


def test_season_prior_without_a_team_ignores_that_teams_previous_games() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    changed = dict(league)
    changed[2003] = league[2003].with_columns(
        pl.when(pl.col("team") == "AAA")
        .then(pl.col("offensive_epa") + 30.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    expected = _history(league).season_prior_without(2004, "AAA")
    history = PriorHistory(changed.__getitem__, first_season=2000)

    # Act
    prior = history.season_prior_without(2004, "AAA")

    # Assert
    assert prior is not None
    assert expected is not None
    assert (prior.offense, prior.defense) == (expected.offense, expected.defense)


def test_season_prior_without_a_team_before_enough_pairs_is_none() -> None:
    # Arrange
    league, _ = _carryover_league(4, 0.7, noise=0.3, seed=2)

    # Act
    prior = _history(league).season_prior_without(2003, "AAA")

    # Assert
    assert prior is None


def test_prior_without_team_fades_opponents_by_their_snapshot_games() -> None:
    """An opponent's fade counts its game against the left-out team, who leaves the centering."""
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    history = _history(league)
    games = league[2004].filter(pl.col("week") <= 3)
    others = games.filter((pl.col("team") != "AAA") & (pl.col("opponent_team") != "AAA"))
    expected = snapshot_prior(
        history.season_prior_without(2004, "AAA"),
        others,
        games=games_played(games, _TEAMS),
    )

    # Act
    prior = prior_without_team(history, 2004, games)("AAA")

    # Assert
    assert prior == expected
    assert prior is not None
    assert "AAA" not in prior.offense
    assert sum(prior.offense.values()) == pytest.approx(0.0, abs=1e-12)


def test_prior_without_team_is_none_when_no_other_game_remains() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    first_game = league[2004].filter(pl.col("game_id") == league[2004].get_column("game_id")[0])
    team = str(first_game.get_column("team")[0])

    # Act
    prior = prior_without_team(_history(league), 2004, first_game)(team)

    # Assert
    assert prior is None


def test_team_prior_table_lists_the_season_means_and_each_left_out_teams() -> None:
    # Arrange
    season_means = UnitPrior(offense={"AAA": 0.01, "BBB": -0.01}, defense={"AAA": 0.0, "BBB": 0.0})
    without = {
        "AAA": UnitPrior(offense={"BBB": 0.0}, defense={"BBB": 0.0}),
        "BBB": None,
    }

    # Act
    table = team_prior_table(season_means, without.__getitem__, ["AAA", "BBB"])

    # Assert
    assert table.rows() == [
        (None, "AAA", 0.01, 0.0),
        (None, "BBB", -0.01, 0.0),
        ("AAA", "BBB", 0.0, 0.0),
    ]


def test_read_team_prior_table_returns_the_means_it_was_built_from() -> None:
    # Arrange
    season_means = UnitPrior(offense={"AAA": 0.01, "BBB": -0.01}, defense={"AAA": 0.0, "BBB": 0.0})
    without_aaa = UnitPrior(offense={"BBB": 0.0}, defense={"BBB": 0.0})
    table = team_prior_table(season_means, {"AAA": without_aaa, "BBB": None}.__getitem__, ["AAA"])

    # Act
    means = read_team_prior_table(table)

    # Assert
    assert means == (season_means, {"AAA": without_aaa})


def test_team_prior_table_without_a_prior_is_empty() -> None:
    # Act
    table = team_prior_table(None, None, ["AAA"])

    # Assert
    assert table.is_empty()
    assert table.columns == ["excluded_team", "team", "offense_prior", "defense_prior"]


def test_read_team_prior_table_of_an_empty_table_has_no_means() -> None:
    # Arrange
    table = team_prior_table(None, None, ["AAA"])

    # Act
    means = read_team_prior_table(table)

    # Assert
    assert means == (None, {})


def test_season_prior_without_a_team_is_built_once() -> None:
    # Arrange
    league, _ = _carryover_league(5, 0.7, noise=0.3, seed=2)
    history = _history(league)
    first = history.season_prior_without(2004, "AAA")

    # Act
    second = history.season_prior_without(2004, "AAA")

    # Assert
    assert second is first


def test_history_first_season_defaults_to_the_first_play_by_play_season() -> None:
    """1999 has no earlier season, so it cross-validates its own penalties."""
    # Arrange
    league, _ = _carryover_league(2, 0.7, noise=0.3, seed=1)
    renumbered = {1999 + offset: league[season] for offset, season in enumerate(sorted(league))}
    history = PriorHistory(renumbered.__getitem__)

    # Act
    penalties = history.penalties(1999)

    # Assert
    assert penalties is history.cross_validated_fit(1999)


def test_history_reports_its_first_season() -> None:
    # Arrange
    league, _ = _carryover_league(2, 0.7, noise=0.3, seed=1)

    # Act
    history = _history(league)

    # Assert
    assert history.first_season == 2000
