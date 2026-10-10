"""Tests for the points-based team rating and its head-to-head-excluded schedule strength."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_predictor.ratings import team_rating
from nfl_predictor.ratings.ridge import UnitPrior, fit_unit_ridge
from nfl_predictor.ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    TEAM_UNIT_COLUMNS,
    TeamRatingResampler,
    bootstrap_team_ratings,
    compute_team_schedule_strength,
    fit_team_ratings,
    fit_team_ratings_by_week,
    fit_team_ratings_with_previous_penalties,
    scrimmage_rows,
    special_teams_rows,
    team_ratings_from_unit_fits,
)

# The first four teams have mean-zero effects, so a round-robin among them recovers them exactly.
_OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": -0.10, "EEE": 0.03, "FFF": -0.02}
_DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.00, "DDD": -0.04, "EEE": 0.02, "FFF": 0.06}
_ST_POSSESSION = {"AAA": 0.20, "BBB": -0.10, "CCC": 0.00, "DDD": -0.10, "EEE": 0.0, "FFF": 0.1}
_ST_COVERAGE = {"AAA": 0.05, "BBB": 0.05, "CCC": -0.05, "DDD": -0.05, "EEE": 0.0, "FFF": 0.0}
_ROUND_ROBIN = tuple(itertools.permutations(("AAA", "BBB", "CCC", "DDD"), 2))
# Six teams that each play only three opponents (home and away), so opponent sets differ.
_PARTIAL_PAIRS = (
    ("AAA", "BBB"),
    ("BBB", "CCC"),
    ("CCC", "DDD"),
    ("DDD", "EEE"),
    ("EEE", "FFF"),
    ("FFF", "AAA"),
    ("AAA", "CCC"),
    ("DDD", "FFF"),
    ("BBB", "EEE"),
)
_PARTIAL = tuple(pair for home, away in _PARTIAL_PAIRS for pair in ((home, away), (away, home)))
_SCRIMMAGE_PLAYS = 60
_ST_PLAYS = 12
_TINY_LAMBDA = 1e-6


def _game_logs(games: tuple[tuple[str, str], ...] = _ROUND_ROBIN) -> pl.DataFrame:
    """Return noiseless team-game rows for ``(home, away)`` games built from known effects."""
    rows: list[dict[str, object]] = []
    for game_number, (home, away) in enumerate(games):
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            sign = 1.0 if is_home else -1.0
            epa_per_play = 0.02 + _OFFENSE[team] - _DEFENSE[opponent] + 0.01 * sign
            st_per_play = _ST_POSSESSION[team] - _ST_COVERAGE[opponent]
            rows.append(
                {
                    "game_id": f"g{game_number:02d}",
                    "week": game_number + 1,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "offensive_snaps": _SCRIMMAGE_PLAYS,
                    "offensive_epa": epa_per_play * _SCRIMMAGE_PLAYS,
                    "st_plays": _ST_PLAYS,
                    "st_epa": st_per_play * _ST_PLAYS,
                }
            )
    return pl.DataFrame(rows)


def _rating(ratings: pl.DataFrame, team: str, column: str) -> float:
    """Return one team's value for one rating column."""
    return float(ratings.filter(pl.col("team") == team).get_column(column).item())


def test_fit_team_ratings_team_rating_is_sum_of_unit_ratings() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(game_logs).ratings

    # Assert
    summed = ratings.select(
        pl.col("offense_rating") + pl.col("defense_rating") + pl.col("special_teams_rating")
    ).to_series()
    assert ratings.get_column("team_rating").to_list() == pytest.approx(summed.to_list())


def test_fit_team_ratings_offense_rating_is_points_per_game() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    assert _rating(ratings, "AAA", "offense_rating") == pytest.approx(0.10 * 60, abs=1e-4)


def test_fit_team_ratings_better_defense_has_positive_defense_rating() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    assert _rating(ratings, "BBB", "defense_rating") == pytest.approx(0.08 * 60, abs=1e-4)


def test_fit_team_ratings_special_teams_rating_adds_both_special_teams_units() -> None:
    # Arrange
    game_logs = _game_logs()
    expected = (0.20 + 0.05) * 12

    # Act
    ratings = fit_team_ratings(
        game_logs, scrimmage_lambda=_TINY_LAMBDA, special_teams_lambda=_TINY_LAMBDA
    ).ratings

    # Assert
    assert _rating(ratings, "AAA", "special_teams_rating") == pytest.approx(expected, abs=1e-4)


def test_fit_team_ratings_missing_special_teams_columns_raises_value_error() -> None:
    # Arrange
    game_logs = _game_logs().drop("st_plays")

    # Act & Assert
    with pytest.raises(ValueError, match="st_plays"):
        fit_team_ratings(game_logs)


def test_compute_team_schedule_strength_ignores_the_subject_teams_own_games() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    blowouts = game_logs.with_columns(
        pl.when((pl.col("team") == "AAA") | (pl.col("opponent_team") == "AAA"))
        .then(pl.col("offensive_epa") * 3.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    baseline_fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)
    blowout_fit = fit_team_ratings(blowouts, scrimmage_lambda=10.0, special_teams_lambda=10.0)
    baseline = _rating(compute_team_schedule_strength(game_logs, baseline_fit), "AAA", "sos")

    # Act
    blowout_sos = compute_team_schedule_strength(blowouts, blowout_fit)

    # Assert
    assert _rating(blowout_sos, "AAA", "sos") == pytest.approx(baseline)


def test_compute_team_schedule_strength_other_teams_see_the_changed_games() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    blowouts = game_logs.with_columns(
        pl.when((pl.col("team") == "AAA") | (pl.col("opponent_team") == "AAA"))
        .then(pl.col("offensive_epa") * 3.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa")
    )
    baseline_fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)
    blowout_fit = fit_team_ratings(blowouts, scrimmage_lambda=10.0, special_teams_lambda=10.0)
    baseline = _rating(compute_team_schedule_strength(game_logs, baseline_fit), "BBB", "sos")

    # Act
    blowout_sos = compute_team_schedule_strength(blowouts, blowout_fit)

    # Assert
    assert _rating(blowout_sos, "BBB", "sos") != pytest.approx(baseline)


def test_fit_team_ratings_ignores_game_outcomes() -> None:
    # Arrange
    game_logs = _game_logs().with_columns(
        pl.lit(0).alias("point_margin"), pl.lit(0.0).alias("win_value")
    )
    flipped = game_logs.with_columns(
        pl.lit(30).alias("point_margin"), pl.lit(1.0).alias("win_value")
    )
    baseline = fit_team_ratings(game_logs).ratings

    # Act
    result = fit_team_ratings(flipped).ratings

    # Assert
    assert result.equals(baseline)


def test_compute_team_schedule_strength_skips_opponents_without_other_games() -> None:
    # Arrange
    games = (
        ("AAA", "BBB"),
        ("BBB", "AAA"),
        ("CCC", "DDD"),
        ("DDD", "CCC"),
        ("AAA", "CCC"),
        ("EEE", "FFF"),
        ("FFF", "EEE"),
    )
    game_logs = _game_logs(games)
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Act
    sos = compute_team_schedule_strength(game_logs, fit)

    # Assert
    by_team = dict(sos.iter_rows())
    assert by_team["EEE"] is None
    assert by_team["AAA"] is not None


def test_compute_team_schedule_strength_with_no_other_games_is_null() -> None:
    # Arrange
    game_logs = _game_logs((("AAA", "BBB"), ("BBB", "AAA")))
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Act
    sos = compute_team_schedule_strength(game_logs, fit)

    # Assert
    assert sos.get_column("sos").null_count() == 2


def test_fit_team_ratings_by_week_refits_each_week_with_the_season_penalties() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    season_fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    through_week_nine = fit_team_ratings(
        game_logs.filter(pl.col("week") <= 9), scrimmage_lambda=10.0, special_teams_lambda=20.0
    ).ratings

    # Act
    history = fit_team_ratings_by_week(game_logs, season_fit)

    # Assert
    week_nine = history.filter(pl.col("week") == 9).select(through_week_nine.columns)
    assert week_nine.sort("team").equals(through_week_nine.sort("team"))


def test_fit_team_ratings_by_week_last_week_is_the_season_rating() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    season_fit = fit_team_ratings(game_logs)

    # Act
    history = fit_team_ratings_by_week(game_logs, season_fit)

    # Assert
    last_week = history.filter(pl.col("week") == history.get_column("week").max())
    assert (
        last_week.select(season_fit.ratings.columns)
        .sort("team")
        .equals(season_fit.ratings.sort("team"))
    )


def test_fit_team_ratings_by_week_rates_only_teams_that_have_played() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)

    # Act
    history = fit_team_ratings_by_week(game_logs, fit_team_ratings(game_logs))

    # Assert
    week_one_teams = history.filter(pl.col("week") == 1).get_column("team").sort().to_list()
    assert week_one_teams == ["AAA", "BBB"]


def test_fit_team_ratings_by_week_counts_games_played_through_each_week() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)

    # Act
    history = fit_team_ratings_by_week(game_logs, fit_team_ratings(game_logs))

    # Assert
    # AAA plays BBB in weeks 1-2, FFF in weeks 11-12, and CCC in weeks 13-14.
    aaa = history.filter(pl.col("team") == "AAA").sort("week")
    assert dict(aaa.select("week", "games_played").iter_rows())[12] == 4


def test_fit_team_ratings_by_week_publishes_columns_in_order() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)

    # Act
    history = fit_team_ratings_by_week(game_logs, fit_team_ratings(game_logs))

    # Assert
    assert history.columns == ["week", "team", "games_played", *TEAM_RATING_COLUMNS]


def test_fit_team_ratings_by_week_without_a_week_column_raises_value_error() -> None:
    # Arrange
    game_logs = _game_logs()
    season_fit = fit_team_ratings(game_logs)

    # Act & Assert
    with pytest.raises(ValueError, match="week"):
        fit_team_ratings_by_week(game_logs.drop("week"), season_fit)


def test_scrimmage_rows_hold_epa_per_play_weighted_by_plays() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    rows = scrimmage_rows(game_logs)

    # Assert
    first = rows.row(0, named=True)
    assert (first["team"], first["opponent_team"]) == ("AAA", "BBB")
    assert (first["plays"], first["epa_per_play"]) == pytest.approx((60, 0.02 + 0.10 - 0.08 + 0.01))


def test_fit_team_ratings_with_previous_penalties_reuses_the_previous_seasons_penalties() -> None:
    # Arrange
    previous = fit_team_ratings(
        _game_logs(_PARTIAL), scrimmage_lambda=12.0, special_teams_lambda=34.0
    )

    # Act
    fit = fit_team_ratings_with_previous_penalties(_game_logs(), previous)

    # Assert
    assert (fit.scrimmage_lambda, fit.special_teams_lambda) == (12.0, 34.0)


def test_fit_team_ratings_with_previous_penalties_without_one_cross_validates() -> None:
    # Arrange
    game_logs = _game_logs()
    expected = fit_team_ratings(game_logs).ratings

    # Act
    fit = fit_team_ratings_with_previous_penalties(game_logs, None)

    # Assert
    assert fit.ratings.equals(expected)


def _duplicated(game_logs: pl.DataFrame, game_ids: list[str], counts: np.ndarray) -> pl.DataFrame:
    """Return ``game_logs`` with each game repeated ``count`` times under distinct game IDs."""
    return pl.concat(
        [
            game_logs.filter(pl.col("game_id") == game_id).with_columns(
                pl.lit(f"{game_id}#{copy}").alias("game_id")
            )
            for game_id, count in zip(game_ids, counts, strict=True)
            for copy in range(int(count))
        ]
    )


def test_team_rating_resampler_matches_a_fit_on_duplicated_games() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    resampler = TeamRatingResampler(game_logs, fit)
    counts = np.array([(index * 7) % 3 for index in range(len(resampler.game_ids))])
    expected = fit_team_ratings(
        _duplicated(game_logs, resampler.game_ids, counts),
        scrimmage_lambda=10.0,
        special_teams_lambda=20.0,
    ).ratings.sort("team")

    # Act
    ratings = resampler.ratings(counts).sort("team")

    # Assert
    assert ratings.get_column("team").to_list() == expected.get_column("team").to_list()
    assert ratings.get_column("team_rating").to_list() == pytest.approx(
        expected.get_column("team_rating").to_list()
    )


def test_bootstrap_team_ratings_is_reproducible_for_a_seed() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    first = bootstrap_team_ratings(game_logs, fit, resamples=5, seed=3)

    # Act
    second = bootstrap_team_ratings(game_logs, fit, resamples=5, seed=3)

    # Assert
    assert second.equals(first)
    assert second.columns == ["draw", "team", *TEAM_RATING_COLUMNS]
    assert second.get_column("draw").n_unique() == 5


def test_bootstrap_team_ratings_units_add_up_to_the_team_rating() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)

    # Act
    draws = bootstrap_team_ratings(game_logs, fit, resamples=4, seed=1)

    # Assert
    units = draws.select(
        pl.col("offense_rating") + pl.col("defense_rating") + pl.col("special_teams_rating")
    ).to_series()
    assert units.to_list() == pytest.approx(draws.get_column("team_rating").to_list())


def test_fit_team_ratings_zero_scrimmage_prior_matches_the_plain_fit() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    plain = fit_team_ratings(game_logs, scrimmage_lambda=40.0, special_teams_lambda=40.0)

    # Act
    fit = fit_team_ratings(
        game_logs,
        scrimmage_lambda=40.0,
        special_teams_lambda=40.0,
        scrimmage_prior=UnitPrior(offense={}, defense={}),
    )

    # Assert
    assert fit.ratings.equals(plain.ratings)


def test_fit_team_ratings_overwhelming_penalty_rates_scrimmage_at_the_prior() -> None:
    """At a huge penalty each scrimmage rating is its per-play prior mean times plays per game."""
    # Arrange
    prior = UnitPrior(offense={"AAA": 0.04, "BBB": -0.04}, defense={"CCC": 0.02, "DDD": -0.02})

    # Act
    fit = fit_team_ratings(
        _game_logs(), scrimmage_lambda=1e12, special_teams_lambda=40.0, scrimmage_prior=prior
    )

    # Assert
    assert _rating(fit.ratings, "AAA", "offense_rating") == pytest.approx(
        0.04 * _SCRIMMAGE_PLAYS, abs=1e-6
    )
    assert _rating(fit.ratings, "CCC", "defense_rating") == pytest.approx(
        0.02 * _SCRIMMAGE_PLAYS, abs=1e-6
    )


def test_fit_team_ratings_scrimmage_prior_needs_a_fixed_penalty() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="fixed penalty"):
        fit_team_ratings(_game_logs(), scrimmage_prior=UnitPrior(offense={}, defense={}))


# A prior that pulls the first four teams' scrimmage effects apart.
_PRIOR = UnitPrior(
    offense={"AAA": 0.04, "BBB": -0.04, "CCC": 0.01, "DDD": -0.01},
    defense={"AAA": 0.02, "BBB": -0.02, "CCC": 0.0, "DDD": 0.0},
)


def test_fit_team_ratings_with_previous_penalties_takes_the_scrimmage_prior() -> None:
    # Arrange
    previous = fit_team_ratings(_game_logs(), scrimmage_lambda=40.0, special_teams_lambda=20.0)
    expected = fit_team_ratings(
        _game_logs(), scrimmage_lambda=40.0, special_teams_lambda=20.0, scrimmage_prior=_PRIOR
    )

    # Act
    fit = fit_team_ratings_with_previous_penalties(_game_logs(), previous, scrimmage_prior=_PRIOR)

    # Assert
    assert fit.ratings.equals(expected.ratings)


def test_team_rating_resampler_with_a_prior_matches_a_prior_fit_on_duplicated_games() -> None:
    # Arrange
    game_logs = _game_logs()
    fit = fit_team_ratings(
        game_logs, scrimmage_lambda=40.0, special_teams_lambda=20.0, scrimmage_prior=_PRIOR
    )
    resampler = TeamRatingResampler(game_logs, fit, scrimmage_prior=_PRIOR)
    counts = np.array([(index * 5) % 3 for index in range(len(resampler.game_ids))])
    expected = fit_team_ratings(
        _duplicated(game_logs, resampler.game_ids, counts),
        scrimmage_lambda=40.0,
        special_teams_lambda=20.0,
        scrimmage_prior=_PRIOR,
    ).ratings.sort("team")

    # Act
    ratings = resampler.ratings(counts).sort("team")

    # Assert
    assert ratings.get_column("team_rating").to_list() == pytest.approx(
        expected.get_column("team_rating").to_list()
    )


def test_bootstrap_team_ratings_keep_the_snapshots_prior_in_every_draw() -> None:
    """At a huge penalty every draw rates each offense at its prior, however often it is drawn."""
    # Arrange
    fit = fit_team_ratings(
        _game_logs(), scrimmage_lambda=1e12, special_teams_lambda=20.0, scrimmage_prior=_PRIOR
    )

    # Act
    draws = bootstrap_team_ratings(_game_logs(), fit, resamples=3, seed=2, scrimmage_prior=_PRIOR)

    # Assert
    offense = draws.filter(pl.col("team") == "AAA").get_column("offense_rating").to_list()
    assert offense == pytest.approx([0.04 * _SCRIMMAGE_PLAYS] * 3, abs=1e-6)


def test_fit_team_ratings_by_week_fits_each_week_with_that_weeks_prior() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    season_fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    seen: list[int] = []

    def prior_for(games: pl.DataFrame) -> UnitPrior:
        seen.append(games.get_column("week").n_unique())
        return _PRIOR

    through_week_nine = game_logs.filter(pl.col("week") <= 9)
    expected = fit_team_ratings(
        through_week_nine,
        scrimmage_lambda=10.0,
        special_teams_lambda=20.0,
        scrimmage_prior=_PRIOR,
    ).ratings

    # Act
    history = fit_team_ratings_by_week(game_logs, season_fit, prior_for=prior_for)

    # Assert
    week_nine = history.filter(pl.col("week") == 9).select(expected.columns)
    assert week_nine.sort("team").equals(expected.sort("team"))
    assert seen == list(range(1, game_logs.get_column("week").n_unique() + 1))


def test_compute_team_schedule_strength_refits_with_the_prior_left_without_the_team() -> None:
    """At a huge penalty each opponent rates at its prior from the refit without the team."""
    # Arrange
    game_logs = _game_logs()
    fit = fit_team_ratings(game_logs, scrimmage_lambda=1e12, special_teams_lambda=1e12)
    asked: list[str] = []

    def prior_without(team: str) -> UnitPrior:
        asked.append(team)
        return UnitPrior(offense={"BBB": 0.04}, defense={"BBB": -0.01})

    # Act
    sos = compute_team_schedule_strength(game_logs, fit, prior_without=prior_without)

    # Assert
    aaa = float(sos.filter(pl.col("team") == "AAA").get_column("sos").item())
    # AAA's opponents: BBB (prior offense 0.04, defense -0.01), CCC and DDD (prior zero).
    assert aaa == pytest.approx((0.04 - 0.01) * _SCRIMMAGE_PLAYS / 3, abs=1e-4)
    assert sorted(asked) == ["AAA", "BBB", "CCC", "DDD"]


def test_special_teams_rows_rate_the_possession_unit_against_the_coverage_unit() -> None:
    # Arrange
    game_logs = _game_logs()

    # Act
    rows = special_teams_rows(game_logs)

    # Assert
    first = rows.row(0, named=True)
    assert (first["team"], first["opponent_team"]) == ("AAA", "BBB")
    assert (first["plays"], first["epa_per_play"]) == pytest.approx((12, 0.20 - 0.05))


def test_fit_team_ratings_leaves_rows_without_plays_out_of_the_fit_and_the_scale() -> None:
    # Arrange
    game_logs = _game_logs()
    padded = pl.concat(
        [
            game_logs,
            game_logs.head(1).with_columns(
                pl.lit("g99").alias("game_id"),
                pl.lit(0, dtype=pl.Int64).alias("offensive_snaps"),
                pl.lit(50.0).alias("offensive_epa"),
                pl.lit(0, dtype=pl.Int64).alias("st_plays"),
            ),
        ]
    )
    expected = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Act
    fit = fit_team_ratings(padded, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Assert
    assert fit.scrimmage_plays_per_game == expected.scrimmage_plays_per_game
    assert fit.ratings.equals(expected.ratings)


def test_fit_team_ratings_scale_is_the_league_average_plays_per_team_game() -> None:
    # Arrange
    game_logs = _game_logs().with_columns(
        pl.when(pl.col("team") == "AAA").then(80).otherwise(60).alias("offensive_snaps")
    )

    # Act
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)

    # Assert
    # AAA has 6 of the 24 team-game rows at 80 plays; the other 18 have 60.
    assert fit.scrimmage_plays_per_game == pytest.approx((6 * 80 + 18 * 60) / 24)
    assert fit.special_teams_plays_per_game == pytest.approx(_ST_PLAYS)


def test_team_ratings_from_unit_fits_use_the_season_fits_scales() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL)
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    scrimmage = fit_unit_ridge(scrimmage_rows(game_logs), TEAM_UNIT_COLUMNS, ridge_lambda=10.0)
    special = fit_unit_ridge(special_teams_rows(game_logs), TEAM_UNIT_COLUMNS, ridge_lambda=20.0)

    # Act
    ratings = team_ratings_from_unit_fits(scrimmage, special, fit)

    # Assert
    assert ratings.equals(fit.ratings)


def test_fit_team_ratings_by_week_scales_each_week_by_its_own_plays_per_game() -> None:
    # Arrange
    game_logs = _game_logs(_PARTIAL).with_columns(
        pl.when(pl.col("week") == 1).then(90).otherwise(60).alias("offensive_snaps")
    )
    season_fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=20.0)
    week_one = fit_team_ratings(
        game_logs.filter(pl.col("week") == 1), scrimmage_lambda=10.0, special_teams_lambda=20.0
    )

    # Act
    history = fit_team_ratings_by_week(game_logs, season_fit)

    # Assert
    first_week = history.filter(pl.col("week") == 1).select(week_one.ratings.columns)
    assert first_week.sort("team").equals(week_one.ratings.sort("team"))
    assert week_one.scrimmage_plays_per_game == pytest.approx(90.0)


def test_compute_team_schedule_strength_sums_opponents_in_the_ratings_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mean adds the opponents in the order of their ratings (by team), not of the games.

    With these ratings the two orders give different floating-point sums: BBB, CCC, DDD gives
    (1e16 - 1e16) + 1, while the game order DDD, BBB, CCC gives (1 + 1e16) - 1e16 = 0.
    """
    # Arrange
    game_logs = _game_logs((("AAA", "DDD"), ("AAA", "BBB"), ("AAA", "CCC")))
    fit = fit_team_ratings(game_logs, scrimmage_lambda=10.0, special_teams_lambda=10.0)
    opponents = pl.DataFrame({"team": ["BBB", "CCC", "DDD"], "team_rating": [1e16, -1e16, 1.0]})
    monkeypatch.setattr(team_rating, "_ratings_without", lambda *_args: opponents)
    expected = float(np.mean(np.array([1e16, -1e16, 1.0])))

    # Act
    sos = compute_team_schedule_strength(game_logs, fit)

    # Assert
    assert _rating(sos, "AAA", "sos") == expected
    assert expected != 0.0
