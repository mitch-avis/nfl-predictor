"""Tests for offline dataset validation utilities."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor.utils import validation_utils


def test_validation_result_is_valid_reflects_error_presence() -> None:
    """ValidationResult validity is driven strictly by the error list."""
    assert validation_utils.ValidationResult(errors=[], warnings=[]).is_valid() is True
    assert validation_utils.ValidationResult(errors=["bad"], warnings=[]).is_valid() is False


def test_validate_required_columns_missing() -> None:
    """Missing required columns are reported."""
    df = pl.DataFrame({"season": [2024], "week": [1]})
    missing = validation_utils.validate_required_columns(df, ["season", "week", "away_abbr"])
    assert missing == ["away_abbr"]


def test_validate_team_abbrs() -> None:
    """Invalid team abbreviations are flagged."""
    df = pl.DataFrame({"away_abbr": ["BUF", "XXX"], "home_abbr": ["KC", "NYJ"]})
    invalid = validation_utils.validate_team_abbrs(df)
    assert "away_abbr:XXX" in invalid
    assert "home_abbr:NYJ" not in invalid


def test_validate_team_abbrs_skips_missing_columns_and_null_values() -> None:
    """Missing abbreviation columns and null rows should be ignored cleanly."""
    df = pl.DataFrame({"away_abbr": [None, "BUF"]})

    assert validation_utils.validate_team_abbrs(df, columns=("away_abbr", "home_abbr")) == []


def test_validate_week_range() -> None:
    """Out-of-range weeks are flagged."""
    df = pl.DataFrame({"season": [2024, 2024], "week": [1, 25]})
    issues = validation_utils.validate_week_range(df)
    assert any("week 25" in issue for issue in issues)


def test_validate_week_range_ignores_missing_columns_and_null_rows() -> None:
    """Missing week/season columns or null values should not produce issues."""
    assert validation_utils.validate_week_range(pl.DataFrame({"season": [2024]})) == []

    df = pl.DataFrame({"season": [2024, None], "week": [1, 5]})
    assert validation_utils.validate_week_range(df) == []


def test_validate_unique_games() -> None:
    """Duplicate game ids with conflicting rows are flagged."""
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_BUF_KC"],
            "away_score": [21, 24],
            "home_score": [17, 17],
        }
    )
    issues = validation_utils.validate_unique_games(df)
    assert issues


def test_validate_unique_games_identical_rows() -> None:
    """Duplicate game ids with identical rows are allowed."""
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_BUF_KC"],
            "away_score": [21, 21],
            "home_score": [17, 17],
        }
    )
    issues = validation_utils.validate_unique_games(df)
    assert not issues


def test_validate_unique_games_with_unique_game_ids_returns_empty() -> None:
    """A game_id column without duplicate ids should not produce issues."""
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_MIA_NE"],
            "away_score": [21, 14],
            "home_score": [17, 10],
        }
    )

    assert validation_utils.validate_unique_games(df) == []


def test_validate_unique_games_returns_empty_for_non_duplicate_or_incomplete_key_rows() -> None:
    """Missing key columns or non-duplicate rows should not produce duplicate issues."""
    assert validation_utils.validate_unique_games(pl.DataFrame({"season": [2024]})) == []

    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )
    assert validation_utils.validate_unique_games(df) == []


def test_validate_unique_games_identical_composite_key_rows_are_allowed() -> None:
    """Composite-key duplicates with identical payloads should be accepted."""
    df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "away_abbr": ["BUF", "BUF"],
            "home_abbr": ["KC", "KC"],
            "away_score": [21, 21],
            "home_score": [17, 17],
        }
    )

    assert validation_utils.validate_unique_games(df) == []


def test_compare_latest_week_scores() -> None:
    """Latest-week score mismatches vs schedule are detected."""
    all_data = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [20],
        }
    )

    mismatches = validation_utils.compare_latest_week_scores(all_data, schedule)
    assert mismatches.height == 1


def test_validate_scores_and_unique_keys() -> None:
    """Invalid scores and duplicate games are flagged."""
    df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "away_abbr": ["BUF", "BUF"],
            "home_abbr": ["KC", "KC"],
            "away_score": [-1, 3],
            "home_score": [10, 10],
        }
    )

    score_issues = validation_utils.validate_scores(df)
    assert score_issues

    dup_issues = validation_utils.validate_unique_games(df)
    assert dup_issues


def test_validate_scores_ignores_missing_score_columns() -> None:
    """Missing score columns should be ignored rather than treated as an error."""
    df = pl.DataFrame({"away_score": [7, 10]})
    assert validation_utils.validate_scores(df) == []


def test_validate_dataframe_collects_errors() -> None:
    """DataFrame validation collects multiple error types."""
    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [25],
            "away_abbr": ["XXX"],
            "home_abbr": ["KC"],
            "away_score": [3],
            "home_score": [7],
        }
    )
    result = validation_utils.validate_dataframe(df)
    assert not result.is_valid()
    assert result.errors


def test_validate_dataframe_valid_returns_no_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid frame should return an empty ValidationResult."""
    monkeypatch.setattr(validation_utils.constants, "METADATA_COLUMNS", ["season", "week"])
    monkeypatch.setattr(validation_utils.constants, "LINES_COLUMNS", ["away_abbr", "home_abbr"])
    monkeypatch.setattr(
        validation_utils.constants,
        "RESULT_COLUMNS",
        ["away_score", "home_score"],
    )

    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )

    result = validation_utils.validate_dataframe(df)
    assert result.is_valid()
    assert result.errors == []
    assert result.warnings == []


def test_validate_dataframe_collects_duplicate_and_score_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DataFrame validation should surface duplicate-game and score issue summaries."""
    monkeypatch.setattr(validation_utils.constants, "METADATA_COLUMNS", ["season", "week"])
    monkeypatch.setattr(validation_utils.constants, "LINES_COLUMNS", ["away_abbr", "home_abbr"])
    monkeypatch.setattr(
        validation_utils.constants,
        "RESULT_COLUMNS",
        ["away_score", "home_score"],
    )

    df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "away_abbr": ["BUF", "BUF"],
            "home_abbr": ["KC", "KC"],
            "away_score": [-1, 10],
            "home_score": [17, 20],
        }
    )

    result = validation_utils.validate_dataframe(df)

    assert any(error.startswith("duplicate games:") for error in result.errors)
    assert any(error.startswith("score issues:") for error in result.errors)


def test_compare_latest_week_scores_missing_columns() -> None:
    """Missing score columns are handled gracefully."""
    df = pl.DataFrame({"season": [2024], "week": [1]})
    mismatches = validation_utils.compare_latest_week_scores(df, None)
    assert mismatches.height == 0


def test_compare_latest_week_scores_without_completed_games_returns_empty() -> None:
    """A schedule comparison is skipped when there are no completed games yet."""
    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [None],
            "home_score": [None],
        },
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "away_abbr": pl.String,
            "home_abbr": pl.String,
            "away_score": pl.Int64,
            "home_score": pl.Int64,
        },
    )

    assert validation_utils.compare_latest_week_scores(df).height == 0


def test_compare_latest_week_scores_load_failure(monkeypatch) -> None:
    """Schedule load failure is handled gracefully."""
    df = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )

    monkeypatch.setattr(
        validation_utils.polars_utils,
        "load_schedule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    mismatches = validation_utils.compare_latest_week_scores(df, None)
    assert mismatches.height == 0


def test_compare_latest_week_scores_missing_schedule_week_returns_empty() -> None:
    """A missing schedule row for the latest completed week should return no mismatches."""
    all_data = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )

    assert validation_utils.compare_latest_week_scores(all_data, schedule).height == 0


def test_compare_latest_week_scores_without_matching_game_join_returns_empty() -> None:
    """Completed and schedule data with no matching matchup should return no mismatches."""
    all_data = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [21],
            "home_score": [17],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["MIA"],
            "home_abbr": ["NE"],
            "away_score": [14],
            "home_score": [10],
        }
    )

    assert validation_utils.compare_latest_week_scores(all_data, schedule).height == 0
