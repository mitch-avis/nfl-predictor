"""Tests for this repository's own Elo ratings, FiveThirtyEight's framework first."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl_predictor.utils.polars import elo_ratings

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
        games.drop("elo1", "elo2", "elo_prob1"), initial_ratings=seeds
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
    rated = elo_ratings.compute_elo_ratings(games, initial_ratings={"AAA": 1600.0})

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
