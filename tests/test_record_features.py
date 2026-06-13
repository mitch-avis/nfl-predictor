"""Unit tests for season-to-date record feature engineering."""

from __future__ import annotations

import polars as pl

from nfl_predictor.utils import polars_utils


def _row_by_team(records_df: pl.DataFrame) -> dict[str, dict[str, int | float]]:
    """Return a dict mapping team -> record values for easy assertions."""
    out: dict[str, dict[str, int | float]] = {}
    for row in records_df.iter_rows(named=True):
        team = row["team_abbr"]
        values: dict[str, int | float] = {}
        for k, v in row.items():
            if k == "team_abbr":
                continue
            if v is None:
                raise AssertionError(f"Unexpected null record value for {team}.{k}")
            if isinstance(v, (int, bool)):
                values[k] = int(v)
            else:
                values[k] = float(v)
        out[team] = values
    return out


def test_compute_team_records_before_week_week1_empty() -> None:
    """Week 1 should have no prior games, so record table is empty."""
    schedule_df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [20],
            "home_score": [17],
        }
    )

    records = polars_utils.compute_team_records_before_week(schedule_df, season=2024, week=1)
    assert records.height == 0


def test_compute_team_records_before_week_overall_div_conf() -> None:
    """Records should be computed from prior weeks only and split by division/conference."""
    schedule_df = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 2],
            "game_type": ["REG", "REG", "REG"],
            "away_abbr": ["BUF", "PHI", "KC"],
            "home_abbr": ["KC", "DAL", "BUF"],
            "away_score": [20, 10, 14],
            "home_score": [17, 21, 14],
        }
    )

    # Before week 2, only week 1 games count.
    records_w2 = polars_utils.compute_team_records_before_week(schedule_df, season=2024, week=2)
    by_team_w2 = _row_by_team(records_w2)

    assert by_team_w2["BUF"]["wins"] == 1
    assert by_team_w2["BUF"]["losses"] == 0
    assert by_team_w2["BUF"]["ties"] == 0
    assert by_team_w2["BUF"]["games_played"] == 1
    assert by_team_w2["BUF"]["win_pct"] == 1.0

    assert by_team_w2["KC"]["wins"] == 0
    assert by_team_w2["KC"]["losses"] == 1
    assert by_team_w2["KC"]["ties"] == 0
    assert by_team_w2["KC"]["games_played"] == 1
    assert by_team_w2["KC"]["win_pct"] == 0.0

    # BUF vs KC is same conference (AFC), different divisions.
    assert by_team_w2["BUF"]["conference_wins"] == 1
    assert by_team_w2["BUF"]["conference_losses"] == 0
    assert by_team_w2["BUF"]["conference_ties"] == 0
    assert by_team_w2["BUF"]["division_wins"] == 0
    assert by_team_w2["BUF"]["division_losses"] == 0
    assert by_team_w2["BUF"]["division_ties"] == 0

    # PHI at DAL is a divisional game (NFC East).
    assert by_team_w2["DAL"]["wins"] == 1
    assert by_team_w2["DAL"]["division_wins"] == 1
    assert by_team_w2["DAL"]["conference_wins"] == 1
    assert by_team_w2["DAL"]["games_played"] == 1
    assert by_team_w2["DAL"]["win_pct"] == 1.0

    assert by_team_w2["PHI"]["losses"] == 1
    assert by_team_w2["PHI"]["division_losses"] == 1
    assert by_team_w2["PHI"]["conference_losses"] == 1
    assert by_team_w2["PHI"]["games_played"] == 1
    assert by_team_w2["PHI"]["win_pct"] == 0.0

    # Before week 3, week 1 and week 2 games count.
    records_w3 = polars_utils.compute_team_records_before_week(schedule_df, season=2024, week=3)
    by_team_w3 = _row_by_team(records_w3)

    # BUF and KC tied in week 2.
    assert by_team_w3["BUF"]["wins"] == 1
    assert by_team_w3["BUF"]["ties"] == 1
    assert by_team_w3["BUF"]["games_played"] == 2
    assert by_team_w3["BUF"]["win_pct"] == 0.5
    assert by_team_w3["KC"]["losses"] == 1
    assert by_team_w3["KC"]["ties"] == 1
    assert by_team_w3["KC"]["games_played"] == 2
    assert by_team_w3["KC"]["win_pct"] == 0.0


def test_compute_team_records_before_week_validates_schema() -> None:
    """Helper should raise a clear error if the schedule schema is incomplete."""
    bad_schedule_df = pl.DataFrame({"season": [2024], "week": [1]})

    try:
        polars_utils.compute_team_records_before_week(bad_schedule_df, season=2024, week=2)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required columns")
