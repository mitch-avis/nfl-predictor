"""Tests for the season quarterback identity built from the players file and weekly rosters."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils.polars import player_identity

PLAYERS_SCHEMA = {
    "gsis_id": pl.String,
    "display_name": pl.String,
    "position": pl.String,
    "pfr_id": pl.String,
    "espn_id": pl.String,
}
ROSTER_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "game_type": pl.String,
    "team": pl.String,
    "position": pl.String,
    "full_name": pl.String,
    "gsis_id": pl.String,
    "pfr_id": pl.String,
}


def _players(rows: list[tuple[str | None, str, str, str | None]]) -> pl.DataFrame:
    """Return players-file rows of ``(gsis_id, display_name, position, pfr_id)``."""
    return pl.DataFrame([(*row, None) for row in rows], schema=PLAYERS_SCHEMA, orient="row")


def _roster(rows: list[tuple[int, str, str, str, str | None]]) -> pl.DataFrame:
    """Return weekly roster rows of ``(week, position, full_name, gsis_id, pfr_id)``."""
    return pl.DataFrame(
        [
            (2022, week, "REG", "NO", position, name, gsis, pfr)
            for week, position, name, gsis, pfr in rows
        ],
        schema=ROSTER_SCHEMA,
        orient="row",
    )


def _no_roster() -> pl.DataFrame:
    """Return an empty weekly roster, as for a season before nflverse publishes one."""
    return pl.DataFrame(schema=ROSTER_SCHEMA)


def test_a_season_without_a_weekly_roster_reads_the_players_file_alone() -> None:
    # Arrange
    players = _players([("00-0031234", "John Doe", "QB", "DoeJo00")])

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, _no_roster())

    # Assert
    assert crosswalk.to_dicts() == [
        {
            "qb_id": "00-0031234",
            "snap_player_id": "DoeJo00",
            "qb_name": "John Doe",
            "qb_position": "QB",
        }
    ]


def test_a_player_the_weekly_roster_lists_at_qb_is_a_quarterback_that_season() -> None:
    """The players file holds the latest position, so a quarterback later at tight end stays one."""
    # Arrange
    players = _players([("00-0033357", "Taysom Hill", "TE", "HillTa00")])
    roster = _roster(
        [
            (1, "TE", "Taysom Hill", "00-0033357", "HillTa00"),
            (2, "QB", "Taysom Hill", "00-0033357", "HillTa00"),
        ]
    )

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, roster)

    # Assert
    assert crosswalk.get_column("qb_position").to_list() == ["QB"]


def test_a_player_no_source_lists_at_qb_keeps_his_position() -> None:
    """A receiver who throws a trick-play pass is not made a quarterback."""
    # Arrange
    players = _players([("00-0020001", "A. Receiver", "WR", None)])
    roster = _roster([(1, "WR", "A. Receiver", "00-0020001", None)])

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, roster)

    # Assert
    assert crosswalk.get_column("qb_position").to_list() == ["WR"]


def test_players_are_matched_by_id_never_by_name() -> None:
    # Arrange
    players = _players(
        [("00-0000001", "Josh Allen", "QB", "AlleJo02"), ("00-0000002", "Josh Allen", "LB", None)]
    )
    roster = _roster([(1, "LB", "Josh Allen", "00-0000002", None)])

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, roster)

    # Assert
    assert crosswalk.select("qb_id", "qb_position").rows() == [
        ("00-0000001", "QB"),
        ("00-0000002", "LB"),
    ]


def test_the_players_file_wins_and_the_roster_fills_what_it_lacks() -> None:
    # Arrange
    players = _players([("00-0000003", "Players Name", "RB", None)])
    roster = _roster(
        [
            (1, "RB", "Roster Name", "00-0000003", "RostPl00"),
            (2, "RB", "Later Name", "00-0000003", "LateRo00"),
        ]
    )

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, roster)

    # Assert
    assert crosswalk.to_dicts() == [
        {
            "qb_id": "00-0000003",
            "snap_player_id": "RostPl00",
            "qb_name": "Players Name",
            "qb_position": "RB",
        }
    ]


def test_a_rostered_player_missing_from_the_players_file_comes_from_the_roster() -> None:
    # Arrange
    players = _players([])
    roster = _roster([(3, "QB", "New Rookie", "00-0099999", "RookNe00")])

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, roster)

    # Assert
    assert crosswalk.rows() == [("00-0099999", "RookNe00", "New Rookie", "QB")]


def test_rows_without_a_player_id_are_left_out() -> None:
    # Arrange
    players = _players([(None, "No Id", "QB", "NoIdXx00"), ("00-0000004", "Has Id", "K", None)])

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(players, _no_roster())

    # Assert
    assert crosswalk.get_column("qb_id").to_list() == ["00-0000004"]


def test_no_players_and_no_roster_give_an_empty_typed_crosswalk() -> None:
    # Act
    crosswalk = player_identity.qb_identity_crosswalk(_players([]), _no_roster())

    # Assert
    assert crosswalk.is_empty()
    assert crosswalk.schema == pl.Schema(player_identity.QB_IDENTITY_SCHEMA)


def test_a_source_without_an_id_column_contributes_no_snap_id() -> None:
    # Arrange
    roster = _roster([(1, "QB", "No Pfr Id", "00-0000008", None)]).drop("pfr_id")

    # Act
    crosswalk = player_identity.qb_identity_crosswalk(_players([]), roster)

    # Assert
    assert crosswalk.rows() == [("00-0000008", None, "No Pfr Id", "QB")]


def test_season_quarterback_ids_are_the_sorted_ids_listed_at_qb() -> None:
    # Arrange
    crosswalk = player_identity.qb_identity_crosswalk(
        _players(
            [
                ("00-0000009", "Late Starter", "QB", None),
                ("00-0000005", "Converted", "TE", None),
                ("00-0000006", "Kicker", "K", None),
            ]
        ),
        _roster([(1, "QB", "Converted", "00-0000005", None)]),
    )

    # Act
    quarterbacks = player_identity.season_quarterback_ids(crosswalk)

    # Assert
    assert quarterbacks == ["00-0000005", "00-0000009"]
