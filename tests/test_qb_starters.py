"""Tests for the play-by-play starter rule shared by every quarterback feature."""

from __future__ import annotations

from typing import Any

import polars as pl

from nfl_predictor.utils.polars import qb_starters, qb_stats

_GAME = "2024_01_NE_NYJ"
_DEFAULTS: dict[str, Any] = {
    "game_id": _GAME,
    "season": 2024,
    "week": 1,
    "season_type": "REG",
    "posteam": "NE",
    "defteam": "NYJ",
    "qtr": 1.0,
    "play_type": "pass",
    "qb_dropback": 0.0,
    "rush": 0.0,
    "rush_attempt": 0.0,
    "two_point_attempt": 0.0,
    "passer_player_id": None,
    "rusher_player_id": None,
}
_DTYPES: dict[str, Any] = {
    "qtr": pl.Float64,
    "play_id": pl.Float64,
    "passer_player_id": pl.String,
    "rusher_player_id": pl.String,
}


def _pass(play_id: float, passer: str, **overrides: Any) -> dict[str, Any]:
    """Return a dropback by ``passer``."""
    return (
        _DEFAULTS | {"play_id": play_id, "qb_dropback": 1.0, "passer_player_id": passer} | overrides
    )


def _run(play_id: float, rusher: str, **overrides: Any) -> dict[str, Any]:
    """Return a designed run by ``rusher``."""
    run = {"play_id": play_id, "play_type": "run", "rush": 1.0, "rush_attempt": 1.0}
    return _DEFAULTS | run | {"rusher_player_id": rusher} | overrides


def _positions(rows: list[tuple[str, str]], season: int = 2024) -> pl.DataFrame:
    """Return the season positions of ``(qb_id, qb_position)`` pairs."""
    return pl.DataFrame(
        [(season, qb_id, position) for qb_id, position in rows],
        schema=qb_starters.POSITIONS_SCHEMA,
        orient="row",
    )


def _starters(plays: list[dict[str, Any]], positions: pl.DataFrame) -> pl.DataFrame:
    """Return the rule's starters for ``plays``, with the quarterback games built from them."""
    frame = pl.DataFrame(plays, schema_overrides=_DTYPES)
    return qb_starters.play_by_play_starters(
        frame, qb_games=qb_stats.aggregate_qb_game_stats(frame), positions=positions
    )


def test_the_starter_is_the_quarterback_on_the_first_first_quarter_snap() -> None:
    # Arrange
    plays = [_run(10, "RB"), _pass(20, "A"), _pass(30, "B"), _pass(40, "B")]
    positions = _positions([("A", "QB"), ("B", "QB"), ("RB", "RB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.rows() == [(_GAME, 2024, 1, "NE", "A")]


def test_a_quarterback_who_ran_on_the_first_snap_starts() -> None:
    # Arrange
    plays = [_run(10, "A"), _pass(20, "B"), _pass(30, "A")]
    positions = _positions([("A", "QB"), ("B", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_a_passer_listed_at_another_position_that_season_is_skipped() -> None:
    # Arrange
    plays = [_pass(10, "X"), _pass(20, "A")]
    positions = _positions([("A", "QB"), ("X", "RB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_a_passer_without_a_dropback_in_the_game_is_skipped() -> None:
    # Arrange
    plays = [_pass(10, "G", two_point_attempt=1.0), _pass(20, "A")]
    positions = _positions([("A", "QB"), ("G", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_a_passer_no_identity_source_lists_can_start() -> None:
    # Arrange
    plays = [_pass(10, "U"), _pass(20, "A")]
    positions = _positions([("A", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["U"]


def test_a_team_game_without_a_first_quarter_quarterback_snap_has_no_starter() -> None:
    # Arrange
    plays = [_run(10, "RB"), _pass(20, "A", qtr=2.0)]
    positions = _positions([("A", "QB"), ("RB", "RB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.height == 0


def test_the_first_snap_is_found_by_play_id_not_by_row_order() -> None:
    # Arrange
    plays = [_pass(30, "B"), _pass(20, "A"), _pass(40, "B")]
    positions = _positions([("A", "QB"), ("B", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_a_play_with_two_quarterbacks_credits_its_passer() -> None:
    # Arrange
    plays = [_pass(10, "A", rusher_player_id="B"), _pass(20, "B")]
    positions = _positions([("A", "QB"), ("B", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_each_team_of_a_game_gets_its_own_starter() -> None:
    # Arrange
    plays = [
        _pass(10, "A"),
        _pass(20, "J", posteam="NYJ", defteam="NE"),
        _pass(30, "A"),
    ]
    positions = _positions([("A", "QB"), ("J", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.select("team_abbr", "qb_id").rows() == [("NE", "A"), ("NYJ", "J")]


def test_postseason_and_teamless_plays_never_name_a_starter() -> None:
    # Arrange
    plays = [
        _pass(10, "P", season_type="POST"),
        _pass(15, "T", posteam=""),
        _pass(20, "A"),
    ]
    positions = _positions([("A", "QB"), ("P", "QB"), ("T", "QB")])

    # Act
    starters = _starters(plays, positions)

    # Assert
    assert starters.get_column("qb_id").to_list() == ["A"]


def test_season_positions_stack_each_season_crosswalk() -> None:
    # Arrange
    crosswalk = pl.DataFrame(
        {
            "qb_id": ["A", "X"],
            "snap_player_id": [None, None],
            "qb_name": ["Al", "Xa"],
            "qb_position": ["QB", "TE"],
        }
    )

    # Act
    positions = qb_starters.season_positions({2023: crosswalk, 2024: crosswalk.head(1)})

    # Assert
    assert positions.rows() == [(2023, "A", "QB"), (2023, "X", "TE"), (2024, "A", "QB")]
