"""Tests for the own Elo's game-level adjustments: byes, travel and a rolling home field."""

from __future__ import annotations

import logging

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_game_context as context

SCHEMA = {
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
}


def _game(
    game_id: str,
    when: tuple[int, int],
    teams: tuple[str, str],
    score: tuple[int, int] | None = None,
    **columns: object,
) -> dict[str, object]:
    """Return one regular-season home game at the home team's venue ``<HOME>00`` by default."""
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
        **columns,
    }


def _schedule(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=SCHEMA)


def _by_game(frame: pl.DataFrame, column: str) -> dict[str, object]:
    return dict(zip(frame["game_id"].to_list(), frame[column].to_list(), strict=True))


def test_great_circle_miles_one_degree_along_the_equator_is_about_69_miles() -> None:
    # Act
    miles = context.great_circle_miles((0.0, 0.0), (0.0, 1.0))

    # Assert
    assert miles == pytest.approx(69.09, abs=0.01)


def test_bye_weeks_flag_a_team_whose_previous_game_was_two_weeks_earlier() -> None:
    # Arrange
    games = _schedule(
        _game("w1", (2020, 1), ("AAA", "BBB")),
        _game("w2", (2020, 2), ("BBB", "CCC")),
        _game("w3", (2020, 3), ("AAA", "BBB")),
    )

    # Act
    byes = context.bye_weeks(games)

    # Assert
    assert _by_game(byes, "home_bye") == {"w1": False, "w2": False, "w3": True}
    assert _by_game(byes, "away_bye") == {"w1": False, "w2": False, "w3": False}


def test_bye_weeks_never_flag_a_teams_first_game_of_a_season() -> None:
    # Arrange
    games = _schedule(
        _game("old", (2019, 15), ("AAA", "BBB")),
        _game("new", (2020, 3), ("AAA", "BBB")),
    )

    # Act
    byes = context.bye_weeks(games)

    # Assert
    assert _by_game(byes, "home_bye") == {"old": False, "new": False}


def test_bye_weeks_flag_a_playoff_team_that_skipped_the_wild_card_round() -> None:
    # Arrange
    games = _schedule(
        _game("finale", (2020, 17), ("AAA", "BBB")),
        _game("wild_card", (2020, 18), ("BBB", "CCC"), game_type="WC"),
        _game("divisional", (2020, 19), ("AAA", "BBB"), game_type="DIV"),
    )

    # Act
    byes = context.bye_weeks(games)

    # Assert
    assert _by_game(byes, "home_bye")["divisional"] is True
    assert _by_game(byes, "away_bye")["divisional"] is False


def test_home_stadiums_pick_the_venue_of_most_non_neutral_home_games() -> None:
    # Arrange
    games = _schedule(
        _game("g1", (2005, 1), ("NO", "AAA"), stadium_id="SAN00"),
        _game("g2", (2005, 2), ("NO", "BBB"), stadium_id="BRG00"),
        _game("g3", (2005, 3), ("NO", "CCC"), stadium_id="BRG00"),
        _game("g4", (2005, 4), ("NO", "DDD"), stadium_id="LON00", neutral=1),
        _game("g5", (2005, 5), ("NO", "EEE"), stadium_id="LON00", neutral=1),
    )

    # Act
    stadiums = context.home_stadiums(games)

    # Assert
    assert stadiums.filter(pl.col("team_abbr") == "NO")["stadium_id"].to_list() == ["BRG00"]


def test_travel_miles_measure_each_side_from_its_own_home_stadium() -> None:
    # Arrange
    games = _schedule(
        _game("bos_home", (2020, 1), ("NE", "KC"), stadium_id="BOS00"),
        _game("kc_home", (2020, 2), ("KC", "NE"), stadium_id="KAN00"),
        _game("london", (2020, 3), ("KC", "NE"), stadium_id="LON00", neutral=1),
    )
    boston = constants.STADIUM_COORDINATES["BOS00"]
    kansas_city = constants.STADIUM_COORDINATES["KAN00"]
    london = constants.STADIUM_COORDINATES["LON00"]

    # Act
    miles = context.travel_miles(games)

    # Assert
    away = _by_game(miles, "away_miles")
    home = _by_game(miles, "home_miles")
    assert home["bos_home"] == 0.0
    assert away["bos_home"] == pytest.approx(context.great_circle_miles(kansas_city, boston))
    assert home["london"] == pytest.approx(context.great_circle_miles(kansas_city, london))
    assert away["london"] == pytest.approx(context.great_circle_miles(boston, london))


def test_travel_miles_are_null_for_a_venue_without_coordinates() -> None:
    # Arrange
    games = _schedule(_game("g1", (2020, 1), ("AAA", "BBB"), stadium_id="NOWHERE"))

    # Act
    miles = context.travel_miles(games)

    # Assert
    assert miles["away_miles"].to_list() == [None]


def test_home_edges_with_every_switch_off_are_65_at_home_and_0_at_a_neutral_site() -> None:
    # Arrange
    games = _schedule(
        _game("home", (2020, 1), ("AAA", "BBB")),
        _game("neutral", (2020, 2), ("AAA", "BBB"), neutral=1),
    )

    # Act
    edges = context.home_edges(games, rest=False, travel=False, rolling=False)

    # Assert
    assert _by_game(edges, "home_edge") == {"home": 65.0, "neutral": 0.0}


def test_home_edges_rest_credits_25_points_to_a_team_off_a_bye() -> None:
    # Arrange
    games = _schedule(
        _game("w1", (2020, 1), ("AAA", "BBB")),
        _game("w2", (2020, 2), ("BBB", "CCC")),
        _game("home_bye", (2020, 3), ("AAA", "BBB")),
        _game("away_bye", (2020, 4), ("BBB", "CCC")),
    )

    # Act
    edges = context.home_edges(games, rest=True, travel=False, rolling=False)

    # Assert
    edge = _by_game(edges, "home_edge")
    assert edge["home_bye"] == 65.0 + 25.0
    assert edge["away_bye"] == 65.0 - 25.0


def test_home_edges_travel_is_55_plus_4_points_per_1000_visitor_miles() -> None:
    # Arrange
    games = _schedule(
        _game("ne_home", (2020, 1), ("NE", "LAC"), stadium_id="BOS00"),
        _game("lac_home", (2020, 2), ("LAC", "NE"), stadium_id="LAX01"),
    )
    miles = context.great_circle_miles(
        constants.STADIUM_COORDINATES["LAX01"], constants.STADIUM_COORDINATES["BOS00"]
    )

    # Act
    edges = context.home_edges(games, rest=False, travel=True, rolling=False)

    # Assert
    assert _by_game(edges, "home_edge")["ne_home"] == pytest.approx(55.0 + 4.0 * miles / 1000.0)


def test_home_edges_travel_counts_only_at_home_or_the_super_bowl() -> None:
    # Arrange
    games = _schedule(
        _game("kc_home", (2020, 1), ("KC", "SF"), stadium_id="KAN00"),
        _game("sf_home", (2020, 2), ("SF", "KC"), stadium_id="SFO01"),
        _game("london", (2020, 3), ("KC", "SF"), stadium_id="LON00", neutral=1),
        _game(
            "super_bowl", (2020, 21), ("KC", "SF"), stadium_id="MIA00", neutral=1, game_type="SB"
        ),
    )
    coordinates = constants.STADIUM_COORDINATES
    kc_trip = context.great_circle_miles(coordinates["KAN00"], coordinates["MIA00"])
    sf_trip = context.great_circle_miles(coordinates["SFO01"], coordinates["MIA00"])

    # Act
    edges = context.home_edges(games, rest=False, travel=True, rolling=False)

    # Assert
    edge = _by_game(edges, "home_edge")
    assert edge["london"] == 0.0
    assert edge["super_bowl"] == pytest.approx(4.0 * (sf_trip - kc_trip) / 1000.0)


def _home_wins_by(
    margin: int, seasons: range, *, teams: tuple[str, str] = ("AAA", "BBB")
) -> list[dict[str, object]]:
    """Return one completed week-1 game per season that the home team wins by ``margin``."""
    return [_game(f"s{season}", (season, 1), teams, (20 + margin, 20)) for season in seasons]


def test_rolling_home_field_is_null_in_the_frames_first_season() -> None:
    # Arrange
    games = _schedule(*_home_wins_by(3, range(2000, 2002)))

    # Act
    rolling = context.rolling_home_field(games, travel=False)

    # Assert
    by_season = dict(zip(rolling["season"].to_list(), rolling["home_field"].to_list(), strict=True))
    assert by_season[2000] is None
    assert by_season[2001] == pytest.approx(75.0)


def test_rolling_home_field_averages_25_points_per_point_of_home_margin() -> None:
    # Arrange
    games = _schedule(
        _game("a", (2000, 1), ("AAA", "BBB"), (24, 20)),
        _game("b", (2000, 2), ("BBB", "AAA"), (20, 22)),
        _game("c", (2001, 1), ("AAA", "BBB")),
    )

    # Act
    rolling = context.rolling_home_field(games, travel=False)

    # Assert
    row = rolling.filter((pl.col("season") == 2001) & (pl.col("week") == 1))
    assert row["home_field"].item() == pytest.approx(25.0 * (4 + -2) / 2)


def test_rolling_home_field_window_keeps_ten_seasons_before_the_week() -> None:
    # Arrange
    games = _schedule(
        _game("first", (2000, 1), ("AAA", "BBB")),
        _game("dropped", (2000, 2), ("AAA", "BBB"), (40, 0)),
        *_home_wins_by(2, range(2001, 2010)),
        _game("kept", (2000, 3), ("AAA", "BBB"), (22, 20)),
        _game("target", (2010, 3), ("AAA", "BBB")),
    )

    # Act
    rolling = context.rolling_home_field(games, travel=False)

    # Assert
    row = rolling.filter((pl.col("season") == 2010) & (pl.col("week") == 3))
    assert row["home_field"].item() == pytest.approx(50.0)


def test_rolling_home_field_ignores_neutral_sites_and_the_playoffs() -> None:
    # Arrange
    games = _schedule(
        _game("regular", (2000, 1), ("AAA", "BBB"), (23, 20)),
        _game("neutral", (2000, 2), ("AAA", "BBB"), (40, 0), neutral=1),
        _game("playoff", (2000, 19), ("AAA", "BBB"), (40, 0), game_type="DIV"),
        _game("target", (2001, 1), ("AAA", "BBB")),
    )

    # Act
    rolling = context.rolling_home_field(games, travel=False)

    # Assert
    row = rolling.filter(pl.col("season") == 2001)
    assert row["home_field"].item() == pytest.approx(75.0)


def test_rolling_home_field_with_travel_nets_out_the_mean_travel_term() -> None:
    # Arrange
    games = _schedule(
        _game("a", (2000, 1), ("NE", "LAC"), (23, 20), stadium_id="BOS00"),
        _game("b", (2000, 2), ("LAC", "NE"), (23, 20), stadium_id="LAX01"),
        _game("target", (2001, 1), ("NE", "LAC"), stadium_id="BOS00"),
    )
    miles = context.great_circle_miles(
        constants.STADIUM_COORDINATES["LAX01"], constants.STADIUM_COORDINATES["BOS00"]
    )

    # Act
    rolling = context.rolling_home_field(games, travel=True)

    # Assert
    row = rolling.filter(pl.col("season") == 2001)
    assert row["home_field"].item() == pytest.approx(75.0 - 4.0 * miles / 1000.0)


def test_home_edges_rolling_home_field_falls_back_to_the_fixed_value_in_the_first_season() -> None:
    # Arrange
    games = _schedule(
        _game("first", (2000, 1), ("AAA", "BBB"), (30, 20)),
        _game("second", (2001, 1), ("AAA", "BBB")),
    )

    # Act
    edges = context.home_edges(games, rest=False, travel=False, rolling=True)

    # Assert
    assert _by_game(edges, "home_edge") == {"first": 65.0, "second": 250.0}


def test_home_edges_travel_warns_and_skips_a_venue_without_coordinates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)
    games = _schedule(_game("g1", (2020, 1), ("NE", "KC"), stadium_id="NOWHERE"))

    # Act
    edges = context.home_edges(games, rest=False, travel=True, rolling=False)

    # Assert
    assert edges["home_edge"].to_list() == [55.0]
    assert any("NOWHERE" in message for message in caplog.messages)
