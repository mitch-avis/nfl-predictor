"""Tests for the shared play-by-play expressions: dropbacks and scrimmage snaps, who lost a fumble,
and who gave the ball away."""

import polars as pl
import pytest

from nfl_predictor.season_stats.pbp_expressions import (
    dropback_expr,
    giveaway_team_expr,
    lost_fumble_team_expr,
    scrimmage_snap_expr,
)


def _play(**overrides: object) -> dict[str, object]:
    """Return one DEN-possession play with no turnover."""
    play: dict[str, object] = {
        "posteam": "DEN",
        "defteam": "KC",
        "punt_attempt": 0,
        "interception": 0,
        "fumble_lost": 0,
        "fumbled_1_team": None,
    }
    play.update(overrides)
    return play


@pytest.mark.parametrize(
    ("play", "expected"),
    [
        (_play(), None),
        (_play(fumble_lost=1, fumbled_1_team="DEN"), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(interception=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(fumble_lost=1), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1), "KC"),
        (_play(interception=1, fumble_lost=1), "KC"),
    ],
    ids=[
        "no lost fumble",
        "offense fumble",
        "returner muff",
        "interception fumbled back",
        "fumbler unknown on a scrimmage play",
        "fumbler unknown on a punt",
        "fumbler unknown after an interception",
    ],
)
def test_lost_fumble_team_names_the_fumbling_team(
    play: dict[str, object], expected: str | None
) -> None:
    # Arrange
    plays = pl.DataFrame([play], schema_overrides={"fumbled_1_team": pl.String})

    # Act
    team = plays.select(lost_fumble_team_expr(plays.columns)).item()

    # Assert
    assert team == expected


def test_lost_fumble_team_falls_back_without_a_fumbler_column() -> None:
    # Arrange
    plays = pl.DataFrame([_play(fumble_lost=1)]).drop("fumbled_1_team")

    # Act
    team = plays.select(lost_fumble_team_expr(plays.columns)).item()

    # Assert
    assert team == "DEN"


@pytest.mark.parametrize(
    ("play", "expected"),
    [
        (_play(), None),
        (_play(interception=1), "DEN"),
        (_play(interception=1, fumble_lost=1, fumbled_1_team="KC"), "DEN"),
        (_play(punt_attempt=1, fumble_lost=1, fumbled_1_team="KC"), "KC"),
        (_play(fumble_lost=1, fumbled_1_team="DEN"), "DEN"),
    ],
    ids=[
        "no giveaway",
        "interception",
        "interception fumbled back",
        "returner muff",
        "offense fumble",
    ],
)
def test_giveaway_team_names_the_first_team_to_give_the_ball_away(
    play: dict[str, object], expected: str | None
) -> None:
    # Arrange
    plays = pl.DataFrame([play], schema_overrides={"fumbled_1_team": pl.String})

    # Act
    team = plays.select(giveaway_team_expr(plays.columns)).item()

    # Assert
    assert team == expected


def _snap(**overrides: object) -> dict[str, object]:
    """Return one play with every snap flag off, as nflverse codes them."""
    play: dict[str, object] = {
        "play_type": "run",
        "qb_dropback": 0,
        "qb_scramble": 0,
        "rush": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
    }
    play.update(overrides)
    return play


_DROPBACK_CASES = [
    (_snap(play_type="pass", qb_dropback=1), True),
    (_snap(qb_dropback=1, qb_scramble=1), True),
    (_snap(qb_scramble=1), True),
    (_snap(play_type="no_play", qb_scramble=1), False),
    (_snap(rush=1), False),
]
_DROPBACK_IDS = [
    "pass attempt",
    "scramble flagged as a dropback (2006 on)",
    "scramble without the dropback flag (before 2006)",
    "scramble a penalty wiped out",
    "designed run",
]


@pytest.mark.parametrize(("play", "expected"), _DROPBACK_CASES, ids=_DROPBACK_IDS)
def test_dropback_counts_every_scramble_that_stood(
    play: dict[str, object], *, expected: bool
) -> None:
    # Arrange
    plays = pl.DataFrame([play])

    # Act
    dropback = plays.select(dropback_expr(plays.columns)).item()

    # Assert
    assert dropback is expected


def test_dropback_without_scramble_columns_reads_the_dropback_flag() -> None:
    # Arrange
    plays = pl.DataFrame([{"qb_dropback": 1}, {"qb_dropback": 0}])

    # Act
    dropbacks = plays.select(dropback_expr(plays.columns)).to_series().to_list()

    # Assert
    assert dropbacks == [True, False]


@pytest.mark.parametrize(
    ("play", "expected"),
    [
        (_snap(qb_scramble=1), True),
        (_snap(play_type="no_play", qb_scramble=1), False),
        (_snap(rush=1), True),
        (_snap(play_type="qb_kneel", qb_kneel=1), True),
        (_snap(play_type="qb_spike", qb_spike=1), True),
        (_snap(), False),
    ],
    ids=[
        "scramble without the dropback flag",
        "scramble a penalty wiped out",
        "designed run",
        "kneel-down",
        "spike",
        "no flag",
    ],
)
def test_scrimmage_snap_counts_dropbacks_runs_kneels_and_spikes(
    play: dict[str, object], *, expected: bool
) -> None:
    # Arrange
    plays = pl.DataFrame([play])

    # Act
    snap = plays.select(scrimmage_snap_expr(plays.columns)).item()

    # Assert
    assert snap is expected
