"""Tests for the confidence pool sheet built from a week's predictions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from nfl_predictor.api.pool.sheet import (
    ConfidenceSheet,
    SubmittedPicks,
    build_pre_lock_sheet,
    build_sheet,
)


def _frame(**overrides: list[Any]) -> pl.DataFrame:
    """Return a three-game predictions frame; ``overrides`` replace whole columns."""
    columns: dict[str, list[Any]] = {
        "game_id": ["2026_05_TB_DAL", "2026_05_HOU_TEN", "2026_05_IND_PIT"],
        "away_abbr": ["TB", "HOU", "IND"],
        "home_abbr": ["DAL", "TEN", "PIT"],
        "game_datetime": [
            "2026-10-08T20:15:00.000000",
            "2026-10-11T13:00:00.000000",
            "2026-10-11T16:25:00.000000",
        ],
        "home_win_prob": [0.7928, 0.291, 0.505],
        "away_win_prob": [0.2072, 0.709, 0.495],
        "predicted_winner": ["DAL", "HOU", "PIT"],
        "confidence_rank": [3, 2, 1],
        "away_score": [None, None, None],
        "home_score": [None, None, None],
    }
    columns.update(overrides)
    return pl.DataFrame(columns, strict=False)


def test_sheet_lists_picks_most_points_first() -> None:
    """Rows come out by points, each with its pick, opponent and the pick's win chance."""
    # Act
    sheet = build_sheet(_frame(), {}, None)

    # Assert
    assert [(row.points, row.pick, row.opponent, row.pick_is_home) for row in sheet.rows] == [
        (3, "DAL", "TB", True),
        (2, "HOU", "TEN", False),
        (1, "PIT", "IND", True),
    ]
    assert [row.pick_win_prob for row in sheet.rows] == [0.7928, 0.709, 0.505]


def test_sheet_totals_the_points_on_offer_and_expected() -> None:
    """The week offers N(N+1)/2 points; expected points weigh each rank by its pick's chance."""
    # Act
    sheet = build_sheet(_frame(), {}, None)

    # Assert
    assert sheet.games == 3
    assert sheet.max_points == 6
    assert sheet.expected_points == pytest.approx(3 * 0.7928 + 2 * 0.709 + 1 * 0.505)
    assert sheet.rank_problem is None


def test_sheet_scores_a_tie_as_incorrect_for_both_sides() -> None:
    """A win scores its points, a loss scores none, and a tie scores none for either pick."""
    # Arrange
    finals = {"2026_05_TB_DAL": (10, 31), "2026_05_HOU_TEN": (17, 20), "2026_05_IND_PIT": (23, 23)}

    # Act
    sheet = build_sheet(_frame(), finals, None)

    # Assert
    assert [(row.pick, row.result, row.points_won) for row in sheet.rows] == [
        ("DAL", "win", 3),
        ("HOU", "loss", 0),
        ("PIT", "tie", 0),
    ]
    assert sheet.games_final == 3
    assert sheet.points_won == 3
    assert sheet.points_possible == 3


def test_sheet_counts_unplayed_games_as_still_possible() -> None:
    """Points still possible are the points won plus the points of games not yet final."""
    # Arrange
    finals = {"2026_05_TB_DAL": (31, 10)}

    # Act
    sheet = build_sheet(_frame(), finals, None)

    # Assert
    assert sheet.games_final == 1
    assert sheet.points_won == 0
    assert sheet.points_possible == 3


def test_sheet_without_a_final_game_has_no_points_won() -> None:
    """Before any game ends there is nothing won yet, and every point is still possible."""
    # Act
    sheet = build_sheet(_frame(), {}, None)

    # Assert
    assert sheet.games_final == 0
    assert sheet.points_won is None
    assert sheet.points_possible == 6


def test_sheet_prefers_the_scores_in_the_predictions_file() -> None:
    """Final scores already in the predictions file win over the dataset's."""
    # Arrange
    frame = _frame(away_score=[35, None, None], home_score=[3, None, None])
    finals = {"2026_05_TB_DAL": (3, 35)}

    # Act
    sheet = build_sheet(frame, finals, None)

    # Assert
    first = sheet.rows[0]
    assert (first.away_score, first.home_score, first.result) == (35, 3, "loss")


def test_sheet_reports_ranks_that_are_not_one_to_n() -> None:
    """A repeated or missing rank is reported, and points still possible are left unknown."""
    # Arrange
    frame = _frame(confidence_rank=[3, 3, 1])

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    assert (
        sheet.rank_problem
        == "Confidence ranks should run 1 to 3 once each: 3 repeats, 2 is missing."
    )
    assert sheet.points_possible is None


def test_sheet_reports_a_file_without_ranks() -> None:
    """A predictions file without confidence ranks says so instead of inventing them."""
    # Arrange
    frame = _frame().drop("confidence_rank")

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    assert sheet.rank_problem == "The predictions file has no confidence ranks."
    assert [row.points for row in sheet.rows] == [None, None, None]
    assert sheet.expected_points is None


def test_sheet_locks_at_the_first_kickoff_in_eastern_time() -> None:
    """Picks are single-shot before the week's first game, so the sheet names that kickoff."""
    # Act
    sheet = build_sheet(_frame(), {}, None)

    # Assert
    assert sheet.lock_at == "2026-10-08T20:15:00-04:00"
    assert sheet.rows[0].kickoff == "2026-10-08T20:15:00-04:00"


@pytest.mark.parametrize(
    ("generated_at", "expected"),
    [
        (datetime(2026, 10, 9, 0, 14, tzinfo=UTC), False),
        (datetime(2026, 10, 9, 0, 16, tzinfo=UTC), True),
    ],
    ids=["a-minute-before-kickoff", "a-minute-after-kickoff"],
)
def test_sheet_flags_predictions_written_after_the_lock(
    generated_at: datetime, *, expected: bool
) -> None:
    """Predictions written after the first kickoff are flagged: the pool's picks were already in."""
    # Act
    sheet = build_sheet(_frame(), {}, generated_at)

    # Assert
    assert sheet.generated_after_lock is expected


def test_sheet_reads_the_kickoff_from_date_and_time_without_a_datetime() -> None:
    """Files without ``game_datetime`` take the kickoff from the date and the Eastern time."""
    # Arrange
    frame = (
        _frame()
        .drop("game_datetime")
        .with_columns(
            pl.Series("date", ["2026-10-08", "2026-10-11", "2026-10-11"]),
            pl.Series("gametime", ["20:15", "13:00", "16:25"]),
        )
    )

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    assert sheet.lock_at == "2026-10-08T20:15:00-04:00"


def test_sheet_reports_games_without_a_rank() -> None:
    """Games the file left unranked are counted in the problem."""
    # Arrange
    frame = _frame(confidence_rank=[3, None, 1])

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    assert sheet.rank_problem == "1 of 3 games have no confidence rank."


def test_sheet_takes_an_away_pick_chance_from_the_home_chance() -> None:
    """Without the away column, an away pick's chance is one minus the home chance."""
    # Arrange
    frame = _frame().drop("away_win_prob")

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    hou = next(row for row in sheet.rows if row.pick == "HOU")
    assert hou.pick_win_prob == pytest.approx(0.709)


def test_sheet_row_without_a_pick_or_a_chance_has_no_pick() -> None:
    """A game with neither a published pick nor a win probability is listed without a pick."""
    # Arrange
    frame = _frame(predicted_winner=[None, "HOU", "PIT"], home_win_prob=[None, 0.291, 0.505])

    # Act
    sheet = build_sheet(frame, {"2026_05_TB_DAL": (10, 31)}, None)

    # Assert
    dal = next(row for row in sheet.rows if row.game_id == "2026_05_TB_DAL")
    assert (dal.pick, dal.opponent, dal.pick_win_prob, dal.result) == (None, None, None, None)
    assert sheet.expected_points is None


def test_sheet_without_a_pick_column_picks_by_the_win_probability() -> None:
    """Without ``predicted_winner`` the pick is the home side when its chance is at least half."""
    # Arrange
    frame = _frame().drop("predicted_winner")

    # Act
    sheet = build_sheet(frame, {}, None)

    # Assert
    assert [row.pick for row in sheet.rows] == ["DAL", "HOU", "PIT"]


def _pre_lock_picks() -> pl.DataFrame:
    """Return the picks file the weekly run wrote before the lock: DAL for 3, HOU 2, PIT 1."""
    return pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [5, 5, 5],
            "date": ["2026-10-08", "2026-10-11", "2026-10-11"],
            "game_id": ["2026_05_TB_DAL", "2026_05_HOU_TEN", "2026_05_IND_PIT"],
            "away_abbr": ["TB", "HOU", "IND"],
            "home_abbr": ["DAL", "TEN", "PIT"],
            "predicted_winner": ["DAL", "HOU", "PIT"],
            "home_win_prob": [0.7928, 0.291, 0.505],
            "away_win_prob": [0.2072, 0.709, 0.495],
            "confidence_rank": [3, 2, 1],
        }
    )


def _newer_predictions() -> pl.DataFrame:
    """Return predictions rewritten after the lock: IND now picked, and the ranks reshuffled."""
    return _frame(
        home_win_prob=[0.7928, 0.291, 0.45],
        away_win_prob=[0.2072, 0.709, 0.55],
        predicted_winner=["DAL", "HOU", "IND"],
        confidence_rank=[3, 1, 2],
    )


def _build_pre_lock(finals: dict[str, tuple[int, int]]) -> ConfidenceSheet:
    """Build the pre-lock sheet of the fixture week, the picks written on 2026-10-07."""
    submitted = SubmittedPicks(
        _pre_lock_picks(),
        "season_2026_week_05_confidence_picks.csv",
        datetime(2026, 10, 7, tzinfo=UTC),
    )
    return build_pre_lock_sheet(
        _newer_predictions(), submitted, finals, datetime(2026, 10, 10, tzinfo=UTC)
    )


def test_pre_lock_sheet_scores_the_picks_made_before_the_lock() -> None:
    """After the lock the sheet and its points come from the picks entered before it."""
    # Arrange
    finals = {"2026_05_TB_DAL": (10, 31), "2026_05_IND_PIT": (17, 24)}

    # Act
    sheet = _build_pre_lock(finals)

    # Assert
    assert [(row.points, row.pick, row.result, row.points_won) for row in sheet.rows] == [
        (3, "DAL", "win", 3),
        (2, "HOU", None, None),
        (1, "PIT", "win", 1),
    ]
    assert (sheet.points_won, sheet.generated_after_lock) == (4, False)
    assert sheet.picks_file == "season_2026_week_05_confidence_picks.csv"


def test_pre_lock_sheet_notes_the_newer_ranks() -> None:
    """Games whose pick or points changed after the lock carry the newer ones as a note."""
    # Act
    sheet = _build_pre_lock({})

    # Assert
    notes = {row.game_id: (row.newer_points, row.newer_pick) for row in sheet.rows}
    assert notes == {
        "2026_05_TB_DAL": (None, None),
        "2026_05_HOU_TEN": (1, "HOU"),
        "2026_05_IND_PIT": (2, "IND"),
    }
    assert sheet.changed_after_lock == 2
    assert sheet.newer_predictions_at == "2026-10-10T00:00:00+00:00"


def test_pre_lock_sheet_takes_the_kickoffs_from_the_predictions() -> None:
    """The picks file has no kickoff times, so the lock and kickoffs come from the predictions."""
    # Act
    sheet = _build_pre_lock({})

    # Assert
    assert sheet.lock_at == "2026-10-08T20:15:00-04:00"
    assert sheet.rows[0].kickoff == "2026-10-08T20:15:00-04:00"
