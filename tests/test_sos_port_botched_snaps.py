"""Tests for the botched-snap charge rule measurement (``.agents/sos_port/botched_snaps.py``)."""

from typing import Any

import polars as pl
import pytest

from tests.sos_port_scripts import load_sos_port_script

botched = load_sos_port_script("botched_snaps")

_GAME = "2010_01_AAA_BBB"
_QB = "00-QB"
_CENTER = "00-C"
_RB = "00-RB"


def _aborted(play_id: int, epa: float, **values: object) -> dict[str, object]:
    """Return one aborted play of AAA's offense; ``values`` sets players and flags."""
    row: dict[str, object] = {
        "game_id": _GAME,
        "play_id": float(play_id),
        "posteam": "AAA",
        "play_type": "pass",
        "aborted_play": 1.0,
        "two_point_attempt": 0.0,
        "qb_dropback": 0.0,
        "epa": epa,
        "qb_epa": epa,
        "fumbled_1_player_id": None,
        "rusher_player_id": None,
        "passer_player_id": None,
    }
    row.update(values)
    return row


def _plays() -> pl.DataFrame:
    """Return aborted plays of every kind the rule tells apart, and two it ignores."""
    return pl.DataFrame(
        [
            _aborted(1, -2.5, play_type="run", fumbled_1_player_id=_QB, rusher_player_id=_QB),
            _aborted(2, -0.4, fumbled_1_player_id=_CENTER, passer_player_id=_QB, qb_dropback=1.0),
            _aborted(3, -1.5, play_type="run", fumbled_1_player_id=_CENTER, rusher_player_id=_QB),
            _aborted(4, -0.9, play_type="run", fumbled_1_player_id=_RB, rusher_player_id=_RB),
            _aborted(5, 0.1, play_type="field_goal"),
            _aborted(
                6, -0.2, fumbled_1_player_id=_CENTER, passer_player_id=_QB, two_point_attempt=1.0
            ),
            _aborted(
                7, -0.3, play_type="no_play", fumbled_1_player_id=_CENTER, passer_player_id=_QB
            ),
        ],
        infer_schema_length=None,
    )


def _qb_rows() -> pl.DataFrame:
    """Return AAA's starter's row in the test game."""
    return pl.DataFrame(
        {
            "game_id": [_GAME],
            "week": [1],
            "team": ["AAA"],
            "opponent_team": ["BBB"],
            "is_home": [False],
            "qb_id": [_QB],
            "qb_name": ["Starter"],
            "qb_dropbacks": [30],
            "qb_dropback_epa": [3.0],
            "qb_attempts": [28],
        }
    )


def _positions() -> pl.DataFrame:
    """Return the players file's positions of the test game's players."""
    return pl.DataFrame({"player_id": [_QB, _CENTER, _RB], "position": ["QB", "C", "RB"]})


def _classified() -> pl.DataFrame:
    """Return the test plays classified by charged player and rusher."""
    return botched.classify_aborted(_plays(), _qb_rows(), _positions())


def test_classify_aborted_leaves_out_two_point_tries_and_penalty_wiped_plays() -> None:
    # Act
    classified = _classified()

    # Assert
    assert sorted(classified.get_column("play_id").to_list()) == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    ("play_id", "roles"),
    [
        (1.0, ("qb", "qb")),
        (2.0, ("line", "none")),
        (3.0, ("line", "qb")),
        (4.0, ("other", "other")),
        (5.0, ("none", "none")),
    ],
)
def test_classify_aborted_names_the_charged_player_and_the_rusher(
    play_id: float, roles: tuple[str, str]
) -> None:
    # Act
    classified = _classified()

    # Assert
    row = classified.filter(pl.col("play_id") == play_id).row(0, named=True)
    assert (row["charged"], row["rusher"]) == roles


def test_lineman_dropbacks_keep_only_a_linemans_snap_that_went_on_as_a_dropback() -> None:
    # Act
    removals = botched.lineman_dropbacks(_classified(), _qb_rows())

    # Assert
    assert removals.select("game_id", "team", "qb_id", "plays").rows() == [(_GAME, "AAA", _QB, 1)]
    assert removals.get_column("qb_epa").to_list() == [pytest.approx(-0.4)]


def test_remove_plays_takes_one_dropback_and_its_epa_from_the_passers_row() -> None:
    # Arrange
    removals = botched.lineman_dropbacks(_classified(), _qb_rows())

    # Act
    rows = botched.remove_plays(_qb_rows(), removals)

    # Assert
    assert rows.select("qb_dropbacks", "qb_dropback_epa").row(0) == (29, pytest.approx(3.4))
    assert rows.schema == _qb_rows().schema


def test_remove_plays_without_removals_returns_the_rows_unchanged() -> None:
    # Arrange
    removals = botched.lineman_dropbacks(_classified().head(0), _qb_rows())

    # Act
    rows = botched.remove_plays(_qb_rows(), removals)

    # Assert
    assert rows.equals(_qb_rows())


def _league_rows() -> Any:
    """Return two quarterbacks' rows over four weeks against two defenses, two games a week."""
    rows = []
    for week in range(1, 5):
        for qb_id, team, opponent, epa in (
            ("00-A", "AAA", "BBB", 6.0),
            ("00-B", "BBB", "AAA", 2.0),
        ):
            rows.append(
                {
                    "game_id": f"2010_{week:02d}_{team}",
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": team == "AAA",
                    "qb_id": qb_id,
                    "qb_name": qb_id,
                    "qb_dropbacks": 30,
                    "qb_dropback_epa": epa + week * 0.5,
                    "qb_attempts": 28,
                }
            )
    return pl.DataFrame(rows)


def test_rating_movement_reports_the_moved_passer_and_the_penalty() -> None:
    # Arrange
    before = _league_rows()
    after = before.with_columns(
        pl.when(pl.col("qb_id") == "00-A")
        .then(pl.col("qb_dropback_epa") + 0.5)
        .otherwise(pl.col("qb_dropback_epa"))
        .alias("qb_dropback_epa")
    )

    # Act
    movement = botched.rating_movement(before, after, qualified=["00-A", "00-B"])

    # Assert
    assert movement["passers"] == 2
    assert movement["moved"] >= 1
    assert movement["largest"] > 0.0
    assert set(movement) >= {
        "penalty_before",
        "penalty_after",
        "rank_changes",
        "largest_rank_change",
    }


def test_season_table_writes_a_dash_for_a_part_the_season_lacks() -> None:
    # Arrange
    row = {"season": 2001, "plays": 0, "epa": 0.0, "qb_epa": 0.0, "pre_week_largest": None}

    # Act
    lines = botched._season_table([row])

    # Assert
    cells = [cell.strip() for cell in lines[-1].strip("|").split("|")]
    assert cells[:4] == ["2001", "0", "+0.00", "+0.00"]
    assert cells[4:] == ["-"] * 9


def test_plays_table_strips_the_play_text_and_replaces_pipes() -> None:
    # Arrange
    plays = pl.DataFrame(
        {
            "season": [2010],
            "game_id": [_GAME],
            "play_id": [55.0],
            "passer_player_id": [_QB],
            "fumbled_1_player_id": [_CENTER],
            "epa": [-1.25],
            "qb_epa": [-1.0],
            "desc": ["  (Shotgun) C.Center fumbles | recovered  "],
        }
    )

    # Act
    lines = botched._plays_table(plays)

    # Assert
    assert lines[-1].endswith("| (Shotgun) C.Center fumbles / recovered |")
