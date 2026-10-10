"""Tests for repairing player team codes in the games nflverse credits to one team."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import player_team_repair

# A game nflverse credits wholly to PIT (jax_1 plays for JAX, pit_1 for PIT), and a normal one.
BROKEN = "2001_01_PIT_JAX"
NORMAL = "2001_01_DEN_KC"
# The one affected game whose visiting code differs from this repository's canonical code.
WASHINGTON_AT_JACKSONVILLE = "2002_10_WAS_JAX"


def _tackle_rows(game: str, teams: list[str]) -> pl.DataFrame:
    """Return one play per team code, each crediting a solo tackle to that team."""
    return pl.DataFrame(
        {
            "game_id": [game] * len(teams),
            "solo_tackle_1_team": teams,
            "solo_tackle_1_player_id": ["X"] * len(teams),
        }
    )


def _rosters(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    """Return roster rows of ``(season, player_id, team)``."""
    return pl.DataFrame(rows, schema=["season", "player_id", "team"], orient="row")


def test_one_team_games_finds_games_whose_player_teams_name_one_team() -> None:
    # Arrange
    minimum = constants.ONE_TEAM_GAME_MIN_VALUES
    plays = pl.concat(
        [
            _tackle_rows(BROKEN, ["PIT"] * minimum),
            _tackle_rows(NORMAL, ["DEN", "KC"] * (minimum // 2)),
        ]
    )

    # Act
    games = player_team_repair.one_team_games(plays, ["solo_tackle_1_team"])

    # Assert
    assert games == [BROKEN]


def test_one_team_games_ignores_games_with_too_few_team_values_to_tell() -> None:
    # Arrange
    plays = _tackle_rows(BROKEN, ["PIT"] * (constants.ONE_TEAM_GAME_MIN_VALUES - 1))

    # Act
    games = player_team_repair.one_team_games(plays, ["solo_tackle_1_team", "missing_team_column"])

    # Assert
    assert games == []


def test_one_team_games_without_a_game_id_finds_nothing() -> None:
    # Arrange
    plays = _tackle_rows(BROKEN, ["PIT"] * constants.ONE_TEAM_GAME_MIN_VALUES).drop("game_id")

    # Act
    games = player_team_repair.one_team_games(plays, ["solo_tackle_1_team"])

    # Assert
    assert games == []


def test_game_sides_read_the_season_and_the_canonical_teams_from_the_game_id() -> None:
    # Arrange
    games = [BROKEN, WASHINGTON_AT_JACKSONVILLE, "2016_03_LA_SEA"]

    # Act
    sides = player_team_repair.game_sides(games)

    # Assert
    assert sides.rows() == [
        (BROKEN, 2001, "PIT", "JAX"),
        (WASHINGTON_AT_JACKSONVILLE, 2002, "WSH", "JAX"),
        ("2016_03_LA_SEA", 2016, "LAR", "SEA"),
    ]


@pytest.mark.parametrize("game", ["2001_01_PITJAX", "season_01_PIT_JAX"])
def test_game_sides_reject_an_id_that_is_not_season_week_away_home(game: str) -> None:
    # Act & Assert
    with pytest.raises(ValueError, match=game):
        player_team_repair.game_sides([game])


def test_game_player_teams_keeps_the_roster_team_that_plays_in_the_game() -> None:
    # Arrange
    rosters = _rosters(
        [
            (2001, "jax_1", "JAX"),
            (2001, "pit_1", "PIT"),
            (2001, "ten_1", "TEN"),
            (2001, "ten_1", "JAX"),
            (2001, "both_1", "JAX"),
            (2001, "both_1", "PIT"),
        ]
    )

    # Act
    teams = player_team_repair.game_player_teams([BROKEN], rosters)

    # Assert
    assert teams.rows() == [
        (BROKEN, "jax_1", "JAX"),
        (BROKEN, "pit_1", "PIT"),
        (BROKEN, "ten_1", "JAX"),
    ]


def test_game_player_teams_reads_each_games_own_season_roster() -> None:
    # Arrange
    rosters = _rosters([(2001, "moved_1", "PIT"), (2002, "moved_1", "JAX")])

    # Act
    teams = player_team_repair.game_player_teams([BROKEN, "2002_13_PIT_JAX"], rosters)

    # Assert
    assert teams.rows() == [(BROKEN, "moved_1", "PIT"), ("2002_13_PIT_JAX", "moved_1", "JAX")]


def test_game_player_teams_match_a_visitor_the_game_id_writes_under_another_code() -> None:
    # Arrange
    rosters = _rosters([(2002, "was_1", "WSH"), (2002, "jax_1", "JAX")])

    # Act
    teams = player_team_repair.game_player_teams([WASHINGTON_AT_JACKSONVILLE], rosters)

    # Assert
    assert teams.rows() == [
        (WASHINGTON_AT_JACKSONVILLE, "jax_1", "JAX"),
        (WASHINGTON_AT_JACKSONVILLE, "was_1", "WSH"),
    ]


def test_repair_pbp_player_teams_rebuilds_team_codes_from_player_ids() -> None:
    """Each team column takes its player's team; a penalty without a player reads the play text.

    Columns with no player behind them (``return_team``) become null in a repaired game, and other
    games keep their values.
    """
    # Arrange
    plays = pl.DataFrame(
        {
            "game_id": [BROKEN, BROKEN, BROKEN, NORMAL],
            "fumbled_1_team": ["PIT", None, None, "KC"],
            "fumbled_1_player_id": ["jax_1", None, None, "kc_1"],
            "td_team": [None, "PIT", None, "DEN"],
            "td_player_id": [None, "jax_1", None, "den_1"],
            "penalty_team": [None, "PIT", "PIT", "KC"],
            "penalty_player_id": [None, "pit_1", None, None],
            "return_team": ["PIT", None, None, "KC"],
            "desc": ["muff", "TD", "PENALTY on JAX, Delay of Game, 5 yards", "x"],
        }
    )
    player_teams = pl.DataFrame(
        {"game_id": [BROKEN, BROKEN], "player_id": ["jax_1", "pit_1"], "team": ["JAX", "PIT"]}
    )

    # Act
    repaired = player_team_repair.repair_pbp_player_teams(plays, [BROKEN], player_teams)

    # Assert
    assert repaired.columns == plays.columns
    assert repaired.select("fumbled_1_team", "td_team", "penalty_team", "return_team").rows() == [
        ("JAX", None, None, None),
        (None, "JAX", "PIT", None),
        (None, None, "JAX", None),
        ("KC", "DEN", "KC", "KC"),
    ]


def test_repair_pbp_player_teams_reads_a_penalty_text_code_as_the_canonical_team() -> None:
    # Arrange
    plays = pl.DataFrame(
        {
            "game_id": [WASHINGTON_AT_JACKSONVILLE, WASHINGTON_AT_JACKSONVILLE],
            "penalty_team": ["WAS", "WAS"],
            "penalty_player_id": [None, None],
            "desc": ["PENALTY on WAS, False Start, 5 yards", "PENALTY on NYG, Offside, 5 yards"],
        },
        schema_overrides={"penalty_player_id": pl.String},
    )
    player_teams = pl.DataFrame(
        schema={"game_id": pl.String, "player_id": pl.String, "team": pl.String}
    )

    # Act
    repaired = player_team_repair.repair_pbp_player_teams(
        plays, [WASHINGTON_AT_JACKSONVILLE], player_teams
    )

    # Assert
    assert repaired.get_column("penalty_team").to_list() == ["WSH", None]


def test_repair_player_stats_teams_rebuilds_team_and_opponent() -> None:
    # Arrange
    stats = pl.DataFrame(
        {
            "game_id": [BROKEN, BROKEN, BROKEN, NORMAL],
            "player_id": ["jax_1", "pit_1", "unknown_1", "den_1"],
            "team": ["PIT", "PIT", "PIT", "DEN"],
            "opponent_team": ["JAX", "JAX", "JAX", "KC"],
        }
    )
    player_teams = pl.DataFrame(
        {"game_id": [BROKEN, BROKEN], "player_id": ["jax_1", "pit_1"], "team": ["JAX", "PIT"]}
    )

    # Act
    repaired = player_team_repair.repair_player_stats_teams(stats, [BROKEN], player_teams)

    # Assert
    assert repaired.columns == stats.columns
    assert repaired.select("player_id", "team", "opponent_team").rows() == [
        ("jax_1", "JAX", "PIT"),
        ("pit_1", "PIT", "JAX"),
        ("unknown_1", None, None),
        ("den_1", "DEN", "KC"),
    ]


def test_repair_player_stats_teams_names_a_renamed_visitor_by_its_canonical_code() -> None:
    # Arrange
    stats = pl.DataFrame(
        {
            "game_id": [WASHINGTON_AT_JACKSONVILLE] * 2,
            "player_id": ["was_1", "jax_1"],
            "team": ["WSH", "WSH"],
            "opponent_team": ["JAX", "JAX"],
        }
    )
    player_teams = pl.DataFrame(
        {
            "game_id": [WASHINGTON_AT_JACKSONVILLE] * 2,
            "player_id": ["was_1", "jax_1"],
            "team": ["WSH", "JAX"],
        }
    )

    # Act
    repaired = player_team_repair.repair_player_stats_teams(
        stats, [WASHINGTON_AT_JACKSONVILLE], player_teams
    )

    # Assert
    assert repaired.select("team", "opponent_team").rows() == [("WSH", "JAX"), ("JAX", "WSH")]
