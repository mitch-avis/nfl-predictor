"""Tests for the leakage-safe pre-week ratings that feed model features."""

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.ratings.pre_week import (
    QB_PRE_WEEK_COLUMNS,
    TEAM_PRE_WEEK_COLUMNS,
    RatingsHistory,
)
from nfl_predictor.ratings.qb_rating import (
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
)
from nfl_predictor.ratings.season import BootstrapSettings, build_team_season
from nfl_predictor.ratings.team_prior import (
    PRIOR_HORIZON_GAMES,
    PriorHistory,
    prior_means,
    prior_without_team,
    snapshot_prior,
)
from nfl_predictor.ratings.team_rating import (
    compute_team_schedule_strength,
    fit_team_ratings_with_previous_penalties,
)

if TYPE_CHECKING:
    from nfl_predictor.ratings.team_prior import SeasonPrior

_TEAMS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
_FIRST = 2000
_SEASONS = range(_FIRST, _FIRST + 6)
# 2004 is the first season with a prior: the carryover slopes need three earlier season pairs.
_PRIOR_SEASON = 2004
_WEEKS = 14


def _round_robin_weeks(teams: tuple[str, ...]) -> list[list[tuple[str, str]]]:
    """Return a double round robin by week (circle method): each team plays once a week."""
    rotation = list(teams)
    weeks: list[list[tuple[str, str]]] = []
    for _ in range(len(teams) - 1):
        half = len(rotation) // 2
        weeks.append(list(zip(rotation[:half], reversed(rotation[half:]), strict=True)))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return weeks + [[(away, home) for home, away in week] for week in weeks]


def _effect(team: str, season: int, salt: int) -> float:
    """Return a deterministic per-play effect that carries over from season to season."""
    index = _TEAMS.index(team)
    return 0.012 * ((index * 7 + salt) % 9 - 4) + 0.003 * ((season * (index + 3) + salt) % 5 - 2)


def _noise(team: str, week: int, season: int) -> float:
    """Return deterministic game-level noise in EPA per play."""
    index = _TEAMS.index(team)
    return 0.04 * (((index + 1) * (week + 3) * (season % 7 + 2)) % 11 - 5) / 5


def _team_games(season: int) -> pl.DataFrame:
    """Return one season of team-game rows, sorted by team, week, and game."""
    rows: list[dict[str, object]] = []
    for week_index, week in enumerate(_round_robin_weeks(_TEAMS)):
        for home, away in week:
            game_id = f"{season}_{week_index + 1:02d}_{away}_{home}"
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                plays = 58 + (week_index * 3 + _TEAMS.index(team)) % 9
                epa = (
                    0.01
                    + _effect(team, season, 1)
                    - _effect(opponent, season, 4)
                    + (0.01 if is_home else -0.01)
                    + _noise(team, week_index, season)
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "week": week_index + 1,
                        "team": team,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "offensive_snaps": plays,
                        "offensive_epa": epa * plays,
                        "st_plays": 12,
                        "st_epa": 0.3 * _noise(opponent, week_index, season),
                        "point_margin": round(epa * 100),
                    }
                )
    return pl.DataFrame(rows).sort("team", "week", "game_id")


def _qb_games(season: int) -> pl.DataFrame:
    """Return one passer per team-game (a second passer for AAA from week 8 on)."""
    team_games = _team_games(season)
    rows: list[dict[str, object]] = []
    for row in team_games.iter_rows(named=True):
        passer = "AAA-2" if row["team"] == "AAA" and int(row["week"]) >= 8 else row["team"]
        dropbacks = 30 + int(row["week"]) % 7
        rate = float(row["offensive_epa"]) / float(row["offensive_snaps"]) + 0.02
        rows.append(
            {
                "game_id": row["game_id"],
                "week": row["week"],
                "team": row["team"],
                "opponent_team": row["opponent_team"],
                "is_home": row["is_home"],
                "qb_id": f"qb-{passer}",
                "qb_name": f"{passer} Passer",
                "qb_dropbacks": dropbacks,
                "qb_dropback_epa": rate * dropbacks,
                "qb_attempts": dropbacks - 2,
            }
        )
    return pl.DataFrame(rows).sort("team", "week", "game_id", "qb_name")


def _history(
    team_games=_team_games,
    qb_games=_qb_games,
) -> RatingsHistory:
    """Return a ratings history over the synthetic league, whose first season is 2000."""
    return RatingsHistory(team_games, qb_games, first_season=_FIRST)


def _changed_from(season: int, week: int):
    """Return loaders that change every game of ``season`` from ``week`` on and every later one."""

    def later(frame: pl.DataFrame, frame_season: int) -> pl.Expr:
        return (pl.lit(frame_season) > season) | (
            (pl.lit(frame_season) == season) & (pl.col("week") >= week)
        )

    def team_games(frame_season: int) -> pl.DataFrame:
        frame = _team_games(frame_season)
        return frame.with_columns(
            pl.when(later(frame, frame_season))
            .then(pl.col("offensive_epa") * -3.0 + 9.0)
            .otherwise(pl.col("offensive_epa"))
            .alias("offensive_epa")
        )

    def qb_games(frame_season: int) -> pl.DataFrame:
        frame = _qb_games(frame_season)
        return frame.with_columns(
            pl.when(later(frame, frame_season))
            .then(pl.col("qb_dropback_epa") * -2.0 - 5.0)
            .otherwise(pl.col("qb_dropback_epa"))
            .alias("qb_dropback_epa")
        )

    return team_games, qb_games


def test_pre_week_team_ratings_publish_one_row_per_team_in_order() -> None:
    # Act
    ratings = _history().pre_week(2005, 5, _TEAMS)

    # Assert
    assert ratings.teams.columns == list(TEAM_PRE_WEEK_COLUMNS)
    assert ratings.teams.get_column("team").to_list() == list(_TEAMS)
    assert (ratings.season, ratings.week) == (2005, 5)


def test_pre_week_team_ratings_never_see_the_week_or_later_games() -> None:
    # Arrange
    expected = _history().pre_week(2005, 6, _TEAMS).teams
    changed = _history(*_changed_from(2005, 6))

    # Act
    teams = changed.pre_week(2005, 6, _TEAMS).teams

    # Assert
    assert_frame_equal(teams, expected)


def test_pre_week_qb_ratings_never_see_the_week_or_later_games() -> None:
    # Arrange
    expected = _history().pre_week(2005, 6, _TEAMS).quarterbacks
    changed = _history(*_changed_from(2005, 6))

    # Act
    quarterbacks = changed.pre_week(2005, 6, _TEAMS).quarterbacks

    # Assert
    assert_frame_equal(quarterbacks, expected)


def test_pre_week_changed_games_before_the_week_do_change_the_ratings() -> None:
    """The guard above is not vacuous: the games just before the week do move the ratings."""
    # Arrange
    expected = _history().pre_week(2005, 6, _TEAMS).teams
    changed = _history(*_changed_from(2005, 5))

    # Act
    teams = changed.pre_week(2005, 6, _TEAMS).teams

    # Assert
    assert not teams.equals(expected)


def test_pre_week_team_ratings_are_the_history_of_the_week_before() -> None:
    # Arrange
    season = build_team_season(
        _PRIOR_SEASON,
        _team_games(_PRIOR_SEASON),
        PriorHistory(_team_games, first_season=_FIRST),
        bootstrap=BootstrapSettings(resamples=1),
    )
    week_four = season.ratings_by_week.filter(pl.col("week") == 4).drop("week")

    # Act
    teams = _history().pre_week(_PRIOR_SEASON, 5, _TEAMS).teams

    # Assert
    assert_frame_equal(teams.select(week_four.columns), week_four)


def test_pre_week_sos_refits_without_each_team_and_its_previous_season() -> None:
    # Arrange
    priors = PriorHistory(_team_games, first_season=_FIRST)
    snapshot = _team_games(_PRIOR_SEASON).filter(pl.col("week") < 5)
    means = snapshot_prior(priors.season_prior(_PRIOR_SEASON), snapshot)
    fit = fit_team_ratings_with_previous_penalties(
        snapshot, priors.penalties(_PRIOR_SEASON), scrimmage_prior=means
    )
    expected = compute_team_schedule_strength(
        snapshot, fit, prior_without=prior_without_team(priors, _PRIOR_SEASON, snapshot)
    )

    # Act
    teams = _history().pre_week(_PRIOR_SEASON, 5, _TEAMS).teams

    # Assert
    assert means is not None
    assert_frame_equal(teams.select("team", "sos"), expected.sort("team"))


def test_pre_week_week_one_rates_every_team_at_its_preseason_prior() -> None:
    # Arrange
    priors = PriorHistory(_team_games, first_season=_FIRST)
    scale = priors.penalties(_PRIOR_SEASON).scrimmage_plays_per_game
    season_prior = cast("SeasonPrior", priors.season_prior(_PRIOR_SEASON))
    means = prior_means(season_prior, dict.fromkeys(_TEAMS, 0), PRIOR_HORIZON_GAMES, _TEAMS)
    expected_offense, expected_defense = means.offense["CCC"] * scale, means.defense["CCC"] * scale

    # Act
    teams = _history().pre_week(_PRIOR_SEASON, 1, _TEAMS).teams

    # Assert
    row = teams.filter(pl.col("team") == "CCC").row(0, named=True)
    assert row["offense_rating"] == pytest.approx(expected_offense)
    assert row["defense_rating"] == pytest.approx(expected_defense)
    assert (row["special_teams_rating"], row["games_played"], row["sos"]) == (0.0, 0, None)
    assert row["team_rating"] == pytest.approx(row["offense_rating"] + row["defense_rating"])


def test_pre_week_week_one_priors_average_zero_over_the_league() -> None:
    # Act
    teams = _history().pre_week(_PRIOR_SEASON, 1, _TEAMS).teams

    # Assert
    assert teams.get_column("offense_rating").sum() == pytest.approx(0.0, abs=1e-9)
    assert teams.get_column("defense_rating").sum() == pytest.approx(0.0, abs=1e-9)


def test_pre_week_week_one_without_a_prior_has_no_team_ratings() -> None:
    # Act
    teams = _history().pre_week(_PRIOR_SEASON - 1, 1, _TEAMS).teams

    # Assert
    assert teams.height == len(_TEAMS)
    assert teams.get_column("team_rating").null_count() == len(_TEAMS)
    assert teams.get_column("games_played").to_list() == [0] * len(_TEAMS)


def test_pre_week_first_season_has_no_ratings_without_an_earlier_penalty() -> None:
    # Act
    ratings = _history().pre_week(_FIRST, 9, _TEAMS)

    # Assert
    assert ratings.teams.get_column("team_rating").null_count() == len(_TEAMS)
    assert ratings.teams.get_column("games_played").to_list() == [8] * len(_TEAMS)
    assert ratings.quarterbacks.is_empty()


def test_pre_week_team_without_a_game_yet_rates_at_its_prior() -> None:
    # Arrange
    def bye_week_one(season: int) -> pl.DataFrame:
        games = _team_games(season)
        if season != _PRIOR_SEASON:
            return games
        opening = games.filter(pl.col("week") == 1)
        bye = opening.filter(pl.col("team").is_in(["AAA", "HHH"])).get_column("game_id")
        return games.filter(~pl.col("game_id").is_in(bye.to_list()))

    history = _history(team_games=bye_week_one)

    # Act
    teams = history.pre_week(_PRIOR_SEASON, 2, _TEAMS).teams

    # Assert
    rested = teams.filter(pl.col("team").is_in(["AAA", "HHH"]))
    assert rested.get_column("games_played").to_list() == [0, 0]
    assert rested.get_column("sos").null_count() == 2
    assert rested.get_column("offense_rating").null_count() == 0
    assert rested.get_column("special_teams_rating").to_list() == [0.0, 0.0]


def test_pre_week_adds_teams_from_the_games_to_the_universe() -> None:
    """A season without a prior rates a team without games as unknown."""
    # Act
    teams = _history().pre_week(_PRIOR_SEASON - 1, 5, ["AAA", "ZZZ"]).teams

    # Assert
    assert teams.get_column("team").to_list() == [*_TEAMS, "ZZZ"]
    assert teams.filter(pl.col("team") == "ZZZ").get_column("team_rating").to_list() == [None]


def test_pre_week_team_without_a_previous_season_raises_in_a_prior_season() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="no 2004 effect for ZZZ"):
        _history().pre_week(2005, 5, [*_TEAMS, "ZZZ"])


def test_pre_week_qb_ratings_take_the_previous_seasons_penalty() -> None:
    # Arrange
    snapshot = _qb_games(2005).filter(pl.col("week") < 9)
    penalty = fit_qb_ratings(_qb_games(2004)).ridge_lambda
    fit = fit_qb_ratings(snapshot, ridge_lambda=penalty)
    faced = compute_qb_faced_pass_defense(snapshot, fit)

    # Act
    quarterbacks = _history().pre_week(2005, 9, _TEAMS).quarterbacks

    # Assert
    assert quarterbacks.columns == list(QB_PRE_WEEK_COLUMNS)
    assert_frame_equal(
        quarterbacks.select("qb_id", "adj_qb_epa_per_dropback", "qb_faced_pass_defense"),
        fit.ratings.join(faced, on="qb_id").sort("qb_id"),
    )


def test_pre_week_qb_penalty_is_the_previous_seasons_cross_validated_one() -> None:
    # Arrange
    expected = fit_qb_ratings(_qb_games(2002)).ridge_lambda

    # Act
    penalty = _history().qb_penalty(2003)

    # Assert
    assert penalty == expected


def test_pre_week_first_season_has_no_qb_penalty() -> None:
    # Act
    penalty = _history().qb_penalty(_FIRST)

    # Assert
    assert penalty is None


def test_pre_week_qb_volume_counts_the_games_and_dropbacks_before_the_week() -> None:
    # Arrange
    snapshot = _qb_games(2005).filter((pl.col("week") < 9) & (pl.col("qb_id") == "qb-AAA"))

    # Act
    quarterbacks = _history().pre_week(2005, 9, _TEAMS).quarterbacks

    # Assert
    row = quarterbacks.filter(pl.col("qb_id") == "qb-AAA").row(0, named=True)
    assert (row["qb_games_played"], row["qb_dropbacks"]) == (
        snapshot.height,
        snapshot.get_column("qb_dropbacks").sum(),
    )


def test_pre_week_week_one_has_no_qb_ratings() -> None:
    # Act
    quarterbacks = _history().pre_week(2005, 1, _TEAMS).quarterbacks

    # Assert
    assert quarterbacks.is_empty()
    assert quarterbacks.columns == list(QB_PRE_WEEK_COLUMNS)


def test_pre_week_postseason_week_fits_the_whole_regular_season() -> None:
    # Arrange
    last_week = _history().pre_week(2005, _WEEKS + 1, _TEAMS)

    # Act
    postseason = _history().pre_week(2005, _WEEKS + 4, _TEAMS)

    # Assert
    assert_frame_equal(postseason.teams, last_week.teams)
    assert_frame_equal(postseason.quarterbacks, last_week.quarterbacks)


def test_pre_week_loaded_rows_must_have_the_input_columns() -> None:
    # Arrange
    history = _history(team_games=lambda season: _team_games(season).drop("st_epa"))

    # Act & Assert
    with pytest.raises(ValueError, match="st_epa"):
        history.pre_week(2003, 5, _TEAMS)


def test_pre_week_teams_are_the_team_part_of_the_pre_week_ratings() -> None:
    # Arrange
    expected = _history().pre_week(2005, 7, _TEAMS).teams

    # Act
    teams = _history().pre_week_teams(2005, 7, _TEAMS)

    # Assert
    assert_frame_equal(teams, expected)


def test_pre_week_quarterbacks_are_the_qb_part_of_the_pre_week_ratings() -> None:
    # Arrange
    expected = _history().pre_week(2005, 7, _TEAMS).quarterbacks

    # Act
    quarterbacks = _history().pre_week_quarterbacks(2005, 7)

    # Assert
    assert_frame_equal(quarterbacks, expected)
