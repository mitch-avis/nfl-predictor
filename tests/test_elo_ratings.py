"""Tests for this repository's own Elo ratings, FiveThirtyEight's framework first."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_game_context, elo_ratings

FIXTURE = Path(__file__).parent / "fixtures" / "fivethirtyeight_elo" / "nfl_games_2017_2019.csv"


def _fivethirtyeight_games() -> pl.DataFrame:
    """Return the fixture as schedule rows, with FiveThirtyEight's published numbers beside them.

    ``team1`` is the home team. Weeks count seven-day windows from each season's first game, and
    the zero-padded row number is the game id, so sorting by week, date and id keeps the file's
    order.
    """
    raw = pl.read_csv(FIXTURE, infer_schema_length=None).with_row_index("row")
    dates = pl.col("date").str.to_date()
    first_date = dates.min().over("season")
    return raw.select(
        pl.col("row").cast(pl.String).str.zfill(4).alias("game_id"),
        "season",
        ((dates - first_date).dt.total_days() // 7 + 1).cast(pl.Int64).alias("week"),
        dates.alias("date"),
        pl.when(pl.col("playoff") == 1)
        .then(pl.lit("POST"))
        .otherwise(pl.lit("REG"))
        .alias("game_type"),
        pl.col("team1").alias("home_abbr"),
        pl.col("team2").alias("away_abbr"),
        pl.col("score1").alias("home_score"),
        pl.col("score2").alias("away_score"),
        "neutral",
        "elo1",
        "elo2",
        "elo_prob1",
    )


def _first_published_ratings(games: pl.DataFrame) -> dict[str, float]:
    """Return each team's published rating before its first game in the frame."""
    sides = pl.concat(
        [
            games.select("game_id", pl.col("home_abbr").alias("team"), pl.col("elo1").alias("elo")),
            games.select("game_id", pl.col("away_abbr").alias("team"), pl.col("elo2").alias("elo")),
        ]
    ).sort("game_id")
    first = sides.unique("team", keep="first", maintain_order=True)
    return dict(zip(first["team"].to_list(), first["elo"].to_list(), strict=True))


def _game(
    game_id: str,
    when: tuple[int, int],
    teams: tuple[str, str],
    score: tuple[int, int] | None = None,
    **context: object,
) -> dict[str, object]:
    """Return one regular-season schedule row from (season, week), (home, away) and the score.

    ``context`` overrides any column, for example ``neutral=1`` or ``game_type="SB"``.
    """
    season, week = when
    home, away = teams
    home_score, away_score = score if score is not None else (None, None)
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": "REG",
        "home_abbr": home,
        "away_abbr": away,
        "home_score": home_score,
        "away_score": away_score,
        "neutral": 0,
        "stadium_id": f"{home}00",
        **context,
    }


def _schedule(*rows: dict[str, object]) -> pl.DataFrame:
    """Return schedule rows with this repository's column types."""
    return pl.DataFrame(
        list(rows),
        schema={
            "game_id": pl.String,
            "season": pl.Int64,
            "week": pl.Int64,
            "game_type": pl.String,
            "home_abbr": pl.String,
            "away_abbr": pl.String,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "neutral": pl.Int64,
            "stadium_id": pl.String,
        },
    )


def _largest_gap(frame: pl.DataFrame, left: str, right: str) -> float:
    """Return the largest absolute difference between two float columns."""
    return float(np.max(np.abs(frame[left].to_numpy() - frame[right].to_numpy())))


def _logistic(diff: float) -> float:
    """Return FiveThirtyEight's win probability for an Elo difference."""
    return 1.0 / (math.pow(10.0, -diff / 400.0) + 1.0)


def test_framework_reproduces_fivethirtyeight_published_ratings_and_probabilities() -> None:
    """Seeded with 2017's first ratings, three seasons match 538's numbers to float precision."""
    # Arrange
    games = _fivethirtyeight_games()
    seeds = _first_published_ratings(games)

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games.drop("elo1", "elo2", "elo_prob1"),
        inputs=elo_ratings.EloInputs(initial_ratings=seeds),
    )

    # Assert
    compared = games.join(rated, on="game_id", how="inner")
    assert compared.height == games.height == 801
    assert _largest_gap(compared, "home_rating", "elo1") < 1e-9
    assert _largest_gap(compared, "away_rating", "elo2") < 1e-9
    assert _largest_gap(compared, "home_win_prob", "elo_prob1") < 1e-12


def test_framework_home_win_probability_adds_65_points_for_the_home_team() -> None:
    """Two teams at the mean: the home side's edge is 65 Elo points."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated["home_rating"].to_list() == [1505.0]
    assert rated["away_rating"].to_list() == [1505.0]
    assert rated["home_win_prob"].item() == pytest.approx(_logistic(65.0), abs=1e-15)


def test_framework_neutral_site_has_no_home_field() -> None:
    """At a neutral site two equal teams are a coin flip."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB"), neutral=1))

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated["home_win_prob"].item() == 0.5


def test_framework_win_moves_ratings_by_the_margin_of_victory_shift() -> None:
    """A 10-point home win moves K * ln(11) * 2.2 / (diff * 0.001 + 2.2) * (1 - p) points."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB"), (27, 17)),
        _game("g2", (2020, 2), ("AAA", "BBB")),
    )
    p = _logistic(65.0)
    shift = 20.0 * math.log(11.0) * (2.2 / (65.0 * 0.001 + 2.2)) * (1.0 - p)

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    second = rated.filter(pl.col("game_id") == "g2")
    assert second["home_rating"].item() == pytest.approx(1505.0 + shift, abs=1e-12)
    assert second["away_rating"].item() == pytest.approx(1505.0 - shift, abs=1e-12)


def test_framework_tie_uses_the_published_tie_multiplier() -> None:
    """A tie moves K * 2.2 * ln(2) * (0.5 - p): the multiplier is 1.525 for any Elo gap."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB"), (20, 20)),
        _game("g2", (2020, 2), ("AAA", "BBB")),
    )
    shift = 20.0 * (math.log(2.0) * 2.2) * (0.5 - _logistic(65.0))

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    second = rated.filter(pl.col("game_id") == "g2")
    assert second["home_rating"].item() == pytest.approx(1505.0 + shift, abs=1e-12)


def test_framework_unplayed_game_leaves_ratings_unchanged() -> None:
    """A game without a score is rated but moves nobody's rating."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB")),
        _game("g2", (2020, 2), ("AAA", "BBB")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated["home_rating"].to_list() == [1505.0, 1505.0]
    assert rated["home_rating_post"].to_list() == [None, None]


def test_framework_offseason_reverts_one_third_toward_the_mean() -> None:
    """A team's first game of a new season starts one third of the way back toward 1505."""
    # Arrange
    games = _schedule(
        _game("g1", (2019, 17), ("AAA", "BBB"), (40, 0)),
        _game("g2", (2020, 1), ("AAA", "BBB")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    first, second = rated.sort("game_id").iter_rows(named=True)
    end_of_season = first["home_rating_post"]
    assert second["home_rating"] == pytest.approx(1505.0 / 3.0 + end_of_season * 2.0 / 3.0)


def test_framework_team_first_seen_after_the_first_season_starts_as_an_expansion_team() -> None:
    """A team that joins after the first season starts at 1300, the modern expansion rating."""
    # Arrange
    games = _schedule(
        _game("g1", (2001, 1), ("AAA", "BBB")),
        _game("g2", (2002, 1), ("AAA", "NEW")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated.filter(pl.col("game_id") == "g2")["away_rating"].item() == 1300.0


def test_framework_initial_ratings_seed_teams_at_their_first_game() -> None:
    """A seeded team starts at its seed, and an unseeded first-season team at the mean."""
    # Arrange
    games = _schedule(_game("g1", (1999, 1), ("AAA", "BBB")))

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games, inputs=elo_ratings.EloInputs(initial_ratings={"AAA": 1600.0})
    )

    # Assert
    assert rated["home_rating"].item() == 1600.0
    assert rated["away_rating"].item() == 1505.0


def test_framework_games_are_rated_in_season_and_week_order() -> None:
    """Rows given out of order are rated chronologically and returned in that order."""
    # Arrange
    games = _schedule(
        _game("late", (2020, 2), ("AAA", "BBB")),
        _game("early", (2020, 1), ("AAA", "BBB"), (30, 0)),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated["game_id"].to_list() == ["early", "late"]
    assert rated["home_rating"][1] > 1505.0


def test_compute_elo_ratings_rejects_a_frame_without_the_score_columns() -> None:
    """A schedule missing a required column fails with the column named."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB"))).drop("home_score")

    # Act & Assert
    with pytest.raises(ValueError, match="home_score"):
        elo_ratings.compute_elo_ratings(games)


def _diffs(rated: pl.DataFrame) -> dict[str, float]:
    """Return each game's pre-game Elo difference, by game id."""
    return dict(zip(rated["game_id"].to_list(), rated["elo_diff"].to_list(), strict=True))


def test_settings_default_to_the_framework() -> None:
    """Calling without settings is the same as calling with every switch off."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB"), (27, 17)),
        _game("g2", (2020, 3), ("AAA", "BBB")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games, settings=elo_ratings.EloSettings())

    # Assert
    assert rated.equals(elo_ratings.compute_elo_ratings(games))


def test_rest_switch_credits_25_points_to_a_home_team_off_a_bye() -> None:
    """With rest on, a home team that sat out the previous week starts 25 points stronger."""
    # Arrange
    games = _schedule(
        _game("w1", (2020, 1), ("AAA", "BBB")),
        _game("w2", (2020, 2), ("BBB", "CCC")),
        _game("w3", (2020, 3), ("AAA", "BBB")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games, settings=elo_ratings.EloSettings(rest=True))

    # Assert
    assert _diffs(rated)["w3"] == 65.0 + 25.0


def test_travel_switch_uses_55_points_plus_the_visitors_trip() -> None:
    """With travel on, the home edge is 55 plus 4 points per 1,000 miles the visitor travels."""
    # Arrange
    games = _schedule(
        _game("ne_home", (2020, 1), ("NE", "LAC"), stadium_id="BOS00"),
        _game("lac_home", (2020, 2), ("LAC", "NE"), stadium_id="LAX01"),
    )
    miles = elo_game_context.great_circle_miles(
        constants.STADIUM_COORDINATES["LAX01"], constants.STADIUM_COORDINATES["BOS00"]
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(games, settings=elo_ratings.EloSettings(travel=True))

    # Assert
    assert _diffs(rated)["ne_home"] == pytest.approx(55.0 + 4.0 * miles / 1000.0, abs=1e-9)


def test_compute_elo_ratings_with_travel_needs_the_stadium_column() -> None:
    """Travel reads ``stadium_id``; a schedule without it fails with the column named."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB"))).drop("stadium_id")

    # Act & Assert
    with pytest.raises(ValueError, match="stadium_id"):
        elo_ratings.compute_elo_ratings(games, settings=elo_ratings.EloSettings(travel=True))


def test_playoff_boost_multiplies_a_playoff_games_difference_by_1_2() -> None:
    """With the boost on, a playoff game's whole difference, home field included, grows 20%."""
    # Arrange
    games = _schedule(
        _game("regular", (2020, 17), ("AAA", "BBB")),
        _game("playoff", (2020, 19), ("AAA", "BBB"), game_type="DIV"),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games, settings=elo_ratings.EloSettings(playoff_boost=True)
    )

    # Assert
    assert _diffs(rated) == {"regular": 65.0, "playoff": 65.0 * 1.2}


def test_playoff_boost_feeds_the_boosted_difference_to_the_margin_multiplier() -> None:
    """The update after a boosted playoff game damps the favorite's win with the boosted gap."""
    # Arrange
    games = _schedule(
        _game("playoff", (2020, 19), ("AAA", "BBB"), (31, 10), game_type="DIV"),
        _game("next", (2020, 20), ("AAA", "BBB"), game_type="CON"),
    )
    diff = 65.0 * 1.2
    multiplier = math.log(22.0) * (2.2 / (diff * 0.001 + 2.2))
    shift = 20.0 * multiplier * (1.0 - _logistic(diff))

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games, settings=elo_ratings.EloSettings(playoff_boost=True)
    )

    # Assert
    nxt = rated.filter(pl.col("game_id") == "next")
    assert nxt["home_rating"].item() == pytest.approx(1505.0 + shift, abs=1e-12)


def test_rolling_home_field_switch_uses_the_rolling_ten_season_base() -> None:
    """With the rolling home field on, a later season's base is 25 points per point of margin."""
    # Arrange
    games = _schedule(
        _game("first", (2000, 1), ("AAA", "BBB"), (22, 20)),
        _game("second", (2001, 1), ("CCC", "DDD")),
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games, settings=elo_ratings.EloSettings(rolling_home_field=True)
    )

    # Assert
    assert _diffs(rated) == {"first": 65.0, "second": 50.0}


def test_compute_elo_ratings_rejects_a_repeated_game_id() -> None:
    """Every game must appear once; a repeated id fails instead of rating the game twice."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB")),
        _game("g1", (2020, 2), ("AAA", "BBB")),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="more than once"):
        elo_ratings.compute_elo_ratings(games)


def test_compute_elo_ratings_of_an_empty_schedule_is_an_empty_typed_frame() -> None:
    """No games, no rows, but the full column set."""
    # Arrange
    games = _schedule()

    # Act
    rated = elo_ratings.compute_elo_ratings(games)

    # Assert
    assert rated.height == 0
    assert "home_win_prob" in rated.columns


def _qb_adjustments(*rows: tuple[str, float, float]) -> pl.DataFrame:
    """Return quarterback adjustments from (game id, home points, away points) tuples."""
    return pl.DataFrame(
        rows,
        schema={"game_id": pl.String, "home_qb_adj": pl.Float64, "away_qb_adj": pl.Float64},
        orient="row",
    )


def test_quarterback_switch_adds_each_sides_adjustment_to_its_rating() -> None:
    """With the quarterback adjustment on, each side's adjustment joins its pre-game rating."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))
    adjustments = _qb_adjustments(("g1", 30.0, -20.0))

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games,
        settings=elo_ratings.EloSettings(quarterback=True),
        inputs=elo_ratings.EloInputs(qb_adjustments=adjustments),
    )

    # Assert
    row = rated.row(0, named=True)
    assert row["elo_diff"] == 65.0 + 30.0 + 20.0
    assert (row["home_qb_adj"], row["away_qb_adj"]) == (30.0, -20.0)
    assert (row["home_rating"], row["away_rating"]) == (1505.0, 1505.0)


def test_quarterback_adjusted_result_moves_only_the_base_ratings() -> None:
    """The shift uses the adjusted difference but lands on the team ratings, not the adjustment."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB"), (24, 21)),
        _game("g2", (2020, 2), ("AAA", "BBB")),
    )
    adjustments = _qb_adjustments(("g1", 100.0, 0.0))
    diff = 165.0
    shift = 20.0 * math.log(4.0) * (2.2 / (diff * 0.001 + 2.2)) * (1.0 - _logistic(diff))

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games,
        settings=elo_ratings.EloSettings(quarterback=True),
        inputs=elo_ratings.EloInputs(qb_adjustments=adjustments),
    )

    # Assert
    second = rated.filter(pl.col("game_id") == "g2").row(0, named=True)
    assert second["home_rating"] == pytest.approx(1505.0 + shift, abs=1e-12)
    assert second["home_qb_adj"] == 0.0


def test_quarterback_switch_needs_the_adjustments() -> None:
    """Switching the adjustment on without the adjustments fails rather than rating without it."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))

    # Act & Assert
    with pytest.raises(ValueError, match="qb_adjustments"):
        elo_ratings.compute_elo_ratings(games, settings=elo_ratings.EloSettings(quarterback=True))


def test_quarterback_adjustments_without_the_switch_are_refused() -> None:
    """Adjustments passed with the switch off fail rather than being silently ignored."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))

    # Act & Assert
    with pytest.raises(ValueError, match="quarterback"):
        elo_ratings.compute_elo_ratings(
            games, inputs=elo_ratings.EloInputs(qb_adjustments=_qb_adjustments(("g1", 1.0, 1.0)))
        )


def _win_totals(*rows: tuple[int, str, float]) -> pl.DataFrame:
    """Return win-total ratings from (season, team, rating) tuples."""
    return pl.DataFrame(
        rows,
        schema={"season": pl.Int64, "team_abbr": pl.String, "win_total_elo": pl.Float64},
        orient="row",
    )


WIN_TOTALS = elo_ratings.EloSettings(win_totals=True)


def test_win_total_pull_blends_the_reverted_rating_toward_the_win_total_rating() -> None:
    """At a team's first game of a season its reverted rating moves toward its win-total rating."""
    # Arrange
    games = _schedule(
        _game("g1", (2000, 17), ("AAA", "BBB"), (30, 0)),
        _game("g2", (2001, 1), ("AAA", "BBB")),
    )
    totals = _win_totals((2001, "AAA", 1400.0))

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games,
        settings=WIN_TOTALS,
        inputs=elo_ratings.EloInputs(win_totals=totals, win_total_weights={2001: 0.5}),
    )

    # Assert
    first, second = rated.iter_rows(named=True)
    home_reverted = 1505.0 / 3.0 + first["home_rating_post"] * 2.0 / 3.0
    away_reverted = 1505.0 / 3.0 + first["away_rating_post"] * 2.0 / 3.0
    assert second["home_rating"] == pytest.approx(0.5 * home_reverted + 0.5 * 1400.0)
    assert second["away_rating"] == pytest.approx(away_reverted)


def test_win_total_switch_needs_the_win_totals() -> None:
    """Switching the pull on without the ratings fails rather than silently regressing."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))

    # Act & Assert
    with pytest.raises(ValueError, match="win_totals"):
        elo_ratings.compute_elo_ratings(games, settings=WIN_TOTALS)


def test_win_totals_are_refused_without_the_switch_because_they_are_market_information() -> None:
    """A market-free setting never reads the win totals, even when they are passed."""
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB")))

    # Act & Assert
    with pytest.raises(ValueError, match="market"):
        elo_ratings.compute_elo_ratings(
            games, inputs=elo_ratings.EloInputs(win_totals=_win_totals((2020, "AAA", 1600.0)))
        )


def _strong_favorite_seasons(*, last_score: tuple[int, int] = (30, 0)) -> pl.DataFrame:
    """Return 2000 even, then 2001 and 2002 seasons the win totals call for AAA, which AAA wins."""
    return _schedule(
        _game("a", (2000, 1), ("AAA", "BBB"), (20, 20)),
        *[_game(f"b{week}", (2001, week), ("AAA", "BBB"), (30, 0)) for week in range(1, 5)],
        _game("c1", (2002, 1), ("AAA", "BBB"), last_score),
        _game("d1", (2003, 1), ("AAA", "BBB")),
    )


STRONG_FAVORITE_TOTALS = _win_totals(
    *[
        (season, team, rating)
        for season in (2001, 2002, 2003)
        for team, rating in (("AAA", 1800.0), ("BBB", 1210.0))
    ]
)


def test_fitted_win_total_weight_is_zero_in_the_first_win_total_season() -> None:
    """With no earlier season to fit on, the first season with win totals gets no pull."""
    # Arrange
    games = _strong_favorite_seasons()

    # Act
    weights = elo_ratings.fit_win_total_weights(
        games, inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS), grid=(0.0, 1.0)
    )

    # Assert
    assert weights[2001] == 0.0


def test_fitted_win_total_weight_takes_the_lowest_brier_over_earlier_seasons() -> None:
    """When the win totals called 2001 right, 2002's weight is the full pull."""
    # Arrange
    games = _strong_favorite_seasons()

    # Act
    weights = elo_ratings.fit_win_total_weights(
        games, inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS), grid=(0.0, 1.0)
    )

    # Assert
    assert weights[2002] == 1.0


def test_fitted_win_total_weight_never_reads_its_own_seasons_results() -> None:
    """Reversing every result of 2002 leaves 2002's weight unchanged."""
    # Arrange
    games = _strong_favorite_seasons(last_score=(0, 30))

    # Act
    weights = elo_ratings.fit_win_total_weights(
        games, inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS), grid=(0.0, 1.0)
    )

    # Assert
    assert weights[2002] == 1.0


def test_compute_elo_ratings_fits_the_weights_when_none_are_given() -> None:
    """Without explicit weights the pull uses the weights fitted on earlier seasons."""
    # Arrange
    games = _strong_favorite_seasons()
    fitted = elo_ratings.fit_win_total_weights(
        games, inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS)
    )

    # Act
    rated = elo_ratings.compute_elo_ratings(
        games,
        settings=WIN_TOTALS,
        inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS),
    )

    # Assert
    explicit = elo_ratings.compute_elo_ratings(
        games,
        settings=WIN_TOTALS,
        inputs=elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS, win_total_weights=fitted),
    )
    assert rated.equals(explicit)


def test_fit_win_total_weights_needs_the_win_totals() -> None:
    """There is nothing to fit without the win totals."""
    # Arrange
    games = _strong_favorite_seasons()

    # Act & Assert
    with pytest.raises(ValueError, match="win_totals"):
        elo_ratings.fit_win_total_weights(games)


def test_market_free_settings_are_the_components_the_evaluation_kept() -> None:
    """Travel, the rolling home field and the quarterback adjustment; no market information."""
    # Act
    settings = elo_ratings.MARKET_FREE_SETTINGS

    # Assert
    assert settings == elo_ratings.EloSettings(
        travel=True, rolling_home_field=True, quarterback=True
    )


def test_win_total_settings_add_only_the_win_total_pull() -> None:
    """The market-informed preset is the market-free one plus the win-total pull."""
    # Act
    settings = elo_ratings.WIN_TOTAL_SETTINGS

    # Assert
    assert settings == elo_ratings.EloSettings(
        travel=True, rolling_home_field=True, quarterback=True, win_totals=True
    )


def test_quarterback_adjustments_with_a_repeated_game_id_are_refused() -> None:
    """Two adjustment rows for one game fail instead of misaligning every later game."""
    # Arrange
    games = _schedule(
        _game("g1", (2020, 1), ("AAA", "BBB")),
        _game("g2", (2020, 2), ("AAA", "BBB")),
    )
    adjustments = _qb_adjustments(("g1", 10.0, 0.0), ("g1", 20.0, 0.0), ("g2", 0.0, 0.0))

    # Act & Assert
    with pytest.raises(ValueError, match=r"qb_adjustments.*more than once"):
        elo_ratings.compute_elo_ratings(
            games,
            settings=elo_ratings.EloSettings(quarterback=True),
            inputs=elo_ratings.EloInputs(qb_adjustments=adjustments),
        )


def test_fitting_win_total_weights_refuses_a_repeated_quarterback_game_id() -> None:
    """The weight fit reads the same adjustments and refuses the same repeated id."""
    # Arrange
    games = _strong_favorite_seasons()
    adjustments = _qb_adjustments(("a", 1.0, 0.0), ("a", 2.0, 0.0))
    inputs = elo_ratings.EloInputs(win_totals=STRONG_FAVORITE_TOTALS, qb_adjustments=adjustments)

    # Act & Assert
    with pytest.raises(ValueError, match=r"qb_adjustments.*more than once"):
        elo_ratings.fit_win_total_weights(
            games, settings=elo_ratings.EloSettings(quarterback=True), inputs=inputs
        )


def _market_free_inputs() -> elo_ratings.EloInputs:
    """Return quarterback adjustments for the two-game schedule of the builder tests."""
    return elo_ratings.EloInputs(qb_adjustments=_qb_adjustments(("g1", 10.0, -5.0)))


def _two_games() -> pl.DataFrame:
    """Return two games at known stadiums, the later one listed first."""
    return _schedule(
        _game("g2", (2020, 2), ("LAC", "NE"), stadium_id="LAX01"),
        _game("g1", (2020, 1), ("NE", "LAC"), (24, 20), stadium_id="BOS00"),
    )


def test_build_own_elo_features_publishes_the_market_free_columns_in_schedule_order() -> None:
    """By default: one row per schedule row, in its order, with the market-free columns."""
    # Arrange
    games = _two_games()

    # Act
    features = elo_ratings.build_own_elo_features(games, inputs=_market_free_inputs())

    # Assert
    assert features.frame.columns == ["game_id", *constants.OWN_ELO_COLUMNS]
    assert features.frame["game_id"].to_list() == ["g2", "g1"]


def test_build_own_elo_features_rate_each_side_with_its_quarterback_adjustment() -> None:
    """A side's own Elo is its rating plus its adjustment; the diff is away minus home."""
    # Arrange
    games = _two_games()
    inputs = _market_free_inputs()
    rated = elo_ratings.compute_elo_ratings(
        games, settings=elo_ratings.MARKET_FREE_SETTINGS, inputs=inputs
    )

    # Act
    features = elo_ratings.build_own_elo_features(games, inputs=inputs)

    # Assert
    joined = rated.join(features.frame, on="game_id").filter(pl.col("game_id") == "g1")
    row = joined.row(0, named=True)
    assert row["home_own_elo"] == row["home_rating"] + 10.0
    assert row["away_own_elo"] == row["away_rating"] - 5.0
    assert row["own_elo_diff"] == row["away_own_elo"] - row["home_own_elo"]
    assert row["own_elo_home_win_prob"] == row["home_win_prob"]


def _win_total_build() -> elo_ratings.OwnEloFeatures:
    """Return the win-total variant built on the strong-favorite seasons."""
    inputs = elo_ratings.EloInputs(
        qb_adjustments=_qb_adjustments(), win_totals=STRONG_FAVORITE_TOTALS
    )
    return elo_ratings.build_own_elo_features(
        _strong_favorite_seasons(), variant="win_totals", inputs=inputs
    )


def test_market_free_variant_never_publishes_a_win_total_column() -> None:
    """The market-free build carries exactly its four columns, none of the win-total family."""
    # Act
    features = elo_ratings.build_own_elo_features(_two_games(), inputs=_market_free_inputs())

    # Assert
    assert set(features.frame.columns) & set(constants.OWN_WT_ELO_COLUMNS) == set()
    assert features.metadata["variant"] == "market_free"


def test_win_total_variant_never_publishes_a_market_free_column() -> None:
    """The win-total build carries exactly its four columns, none of the market-free family."""
    # Act
    features = _win_total_build()

    # Assert
    assert features.frame.columns == ["game_id", *constants.OWN_WT_ELO_COLUMNS]
    assert set(features.frame.columns) & set(constants.OWN_ELO_COLUMNS) == set()


def test_market_free_variant_refuses_the_win_totals() -> None:
    """Win totals passed to the market-free variant fail instead of leaking into it."""
    # Arrange
    inputs = elo_ratings.EloInputs(
        qb_adjustments=_qb_adjustments(), win_totals=STRONG_FAVORITE_TOTALS
    )

    # Act & Assert
    with pytest.raises(ValueError, match="market"):
        elo_ratings.build_own_elo_features(_strong_favorite_seasons(), inputs=inputs)


def test_win_total_variant_records_its_preset_and_fitted_weights() -> None:
    """The metadata names the variant, its switches, its columns and the weights it used."""
    # Act
    features = _win_total_build()

    # Assert
    metadata = features.metadata
    assert metadata["variant"] == "win_totals"
    assert metadata["settings"] == dataclasses.asdict(elo_ratings.WIN_TOTAL_SETTINGS)
    assert metadata["columns"] == constants.OWN_WT_ELO_COLUMNS
    assert metadata["first_season"] == 2000
    weights = metadata["win_total_weights"]
    assert isinstance(weights, dict)
    assert set(weights) == {"2001", "2002", "2003"}


def test_market_free_variant_records_no_win_total_weights() -> None:
    """The market-free metadata says no weights were used."""
    # Act
    features = elo_ratings.build_own_elo_features(_two_games(), inputs=_market_free_inputs())

    # Assert
    assert features.metadata["win_total_weights"] is None
    assert features.metadata["settings"] == dataclasses.asdict(elo_ratings.MARKET_FREE_SETTINGS)


def test_variants_publish_the_constants_column_families() -> None:
    """Each variant's names are the constants' lists, so the feature groups resolve to them."""
    # Act
    families = {name: variant.columns for name, variant in elo_ratings.OWN_ELO_VARIANTS.items()}

    # Assert
    assert families == {
        "market_free": constants.OWN_ELO_COLUMNS,
        "win_totals": constants.OWN_WT_ELO_COLUMNS,
    }


def test_build_own_elo_features_rejects_an_unknown_variant() -> None:
    """Only the two presets exist; any other name fails."""
    # Arrange
    options: dict[str, Any] = {"variant": "custom"}

    # Act & Assert
    with pytest.raises(ValueError, match="variant"):
        elo_ratings.build_own_elo_features(_two_games(), **options)


def test_win_total_variant_uses_and_records_weights_it_is_given() -> None:
    """Explicit weights are used as given, not refitted, and the metadata records them."""
    # Arrange
    inputs = elo_ratings.EloInputs(
        qb_adjustments=_qb_adjustments(),
        win_totals=STRONG_FAVORITE_TOTALS,
        win_total_weights={2001: 0.25, 2002: 0.5, 2003: 0.75},
    )

    # Act
    features = elo_ratings.build_own_elo_features(
        _strong_favorite_seasons(), variant="win_totals", inputs=inputs
    )

    # Assert
    assert features.metadata["win_total_weights"] == {"2001": 0.25, "2002": 0.5, "2003": 0.75}
