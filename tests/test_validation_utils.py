"""Tests for offline dataset validation utilities."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor.utils import validation_utils


@pytest.mark.parametrize(("errors", "expected"), [([], True), (["bad"], False)])
def test_validation_result_is_valid_reflects_error_presence(
    errors: list[str], *, expected: bool
) -> None:
    """ValidationResult validity is driven strictly by the error list."""
    # Arrange
    result = validation_utils.ValidationResult(errors=errors, warnings=[])

    # Act
    valid = result.is_valid()

    # Assert
    assert valid is expected


def test_validate_required_columns_missing() -> None:
    """Missing required columns are reported."""
    # Arrange
    df = pl.DataFrame({"season": [2024], "week": [1]})

    # Act
    missing = validation_utils.validate_required_columns(df, ["season", "week", "away_abbr"])

    # Assert
    assert missing == ["away_abbr"]


def test_validate_team_abbrs() -> None:
    """Invalid team abbreviations are flagged."""
    # Arrange
    df = pl.DataFrame({"away_abbr": ["BUF", "XXX"], "home_abbr": ["KC", "NYJ"]})

    # Act
    invalid = validation_utils.validate_team_abbrs(df)

    # Assert
    assert "away_abbr:XXX" in invalid
    assert "home_abbr:NYJ" not in invalid


def test_validate_team_abbrs_skips_missing_columns_and_null_values() -> None:
    """Missing abbreviation columns and null rows should be ignored cleanly."""
    # Arrange
    df = pl.DataFrame({"away_abbr": [None, "BUF"]})

    # Act
    invalid = validation_utils.validate_team_abbrs(df, columns=("away_abbr", "home_abbr"))

    # Assert
    assert invalid == []


def test_validate_week_range() -> None:
    """Out-of-range weeks are flagged."""
    # Arrange
    df = pl.DataFrame({"season": [2024, 2024], "week": [1, 25]})

    # Act
    issues = validation_utils.validate_week_range(df)

    # Assert
    assert any("week 25" in issue for issue in issues)


@pytest.mark.parametrize(
    "data",
    [{"season": [2024]}, {"season": [2024, None], "week": [1, 5]}],
    ids=["missing_week_column", "null_season"],
)
def test_validate_week_range_ignores_missing_columns_and_null_rows(
    data: dict[str, list[int | None]],
) -> None:
    """Missing week/season columns or null values should not produce issues."""
    # Arrange
    df = pl.DataFrame(data)

    # Act
    issues = validation_utils.validate_week_range(df)

    # Assert
    assert issues == []


def test_validate_unique_games() -> None:
    """Duplicate game ids with conflicting rows are flagged."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_BUF_KC"],
            "away_score": [21, 24],
            "home_score": [17, 17],
        }
    )

    # Act
    issues = validation_utils.validate_unique_games(df)

    # Assert
    assert issues


def test_validate_unique_games_identical_rows() -> None:
    """Duplicate game ids with identical rows are allowed."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_BUF_KC"],
            "away_score": [21, 21],
            "home_score": [17, 17],
        }
    )

    # Act
    issues = validation_utils.validate_unique_games(df)

    # Assert
    assert not issues


def test_validate_unique_games_with_unique_game_ids_returns_empty() -> None:
    """A game_id column without duplicate ids should not produce issues."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["2024_01_BUF_KC", "2024_01_MIA_NE"],
            "away_score": [21, 14],
            "home_score": [17, 10],
        }
    )

    # Act
    issues = validation_utils.validate_unique_games(df)

    # Assert
    assert issues == []


@pytest.mark.parametrize(
    "data",
    [
        {"season": [2024]},
        {"season": [2024], "week": [1], "away_abbr": ["BUF"], "home_abbr": ["KC"]},
    ],
    ids=["incomplete_key", "non_duplicate"],
)
def test_validate_unique_games_returns_empty_for_non_duplicate_or_incomplete_key_rows(
    data: dict[str, list[int] | list[str]],
) -> None:
    """Missing key columns or non-duplicate rows should not produce duplicate issues."""
    # Arrange
    df = pl.DataFrame(data)

    # Act
    issues = validation_utils.validate_unique_games(df)

    # Assert
    assert issues == []


def test_validate_unique_games_identical_composite_key_rows_are_allowed() -> None:
    """Composite-key duplicates with identical payloads should be accepted."""
    # Arrange
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

    # Act
    issues = validation_utils.validate_unique_games(df)

    # Assert
    assert issues == []


def test_compare_latest_week_scores() -> None:
    """Latest-week score mismatches vs schedule are detected."""
    # Arrange
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

    # Act
    mismatches = validation_utils.compare_latest_week_scores(all_data, schedule)

    # Assert
    assert mismatches.height == 1


def _negative_score_duplicate_frame() -> pl.DataFrame:
    """Return two rows for one game that disagree, one with a negative score."""
    return pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "away_abbr": ["BUF", "BUF"],
            "home_abbr": ["KC", "KC"],
            "away_score": [-1, 3],
            "home_score": [10, 10],
        }
    )


def test_validate_scores_and_unique_keys() -> None:
    """Invalid scores are flagged."""
    # Arrange
    df = _negative_score_duplicate_frame()

    # Act
    score_issues = validation_utils.validate_scores(df)

    # Assert
    assert score_issues


def test_validate_unique_keys_flags_the_conflicting_duplicate() -> None:
    """Duplicate games are flagged."""
    # Arrange
    df = _negative_score_duplicate_frame()

    # Act
    dup_issues = validation_utils.validate_unique_games(df)

    # Assert
    assert dup_issues


def test_validate_scores_ignores_missing_score_columns() -> None:
    """Missing score columns should be ignored rather than treated as an error."""
    # Arrange
    df = pl.DataFrame({"away_score": [7, 10]})

    # Act
    issues = validation_utils.validate_scores(df)

    # Assert
    assert issues == []


def test_validate_dataframe_collects_errors() -> None:
    """DataFrame validation collects multiple error types."""
    # Arrange
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

    # Act
    result = validation_utils.validate_dataframe(df)

    # Assert
    assert not result.is_valid()
    assert result.errors


def test_validate_dataframe_valid_returns_no_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid frame should return an empty ValidationResult."""
    # Arrange
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

    # Act
    result = validation_utils.validate_dataframe(df)

    # Assert
    assert result.is_valid()
    assert result.errors == []
    assert result.warnings == []


def test_validate_dataframe_collects_duplicate_and_score_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DataFrame validation should surface duplicate-game and score issue summaries."""
    # Arrange
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

    # Act
    result = validation_utils.validate_dataframe(df)

    # Assert
    assert any(error.startswith("duplicate games:") for error in result.errors)
    assert any(error.startswith("score issues:") for error in result.errors)


def test_compare_latest_week_scores_missing_columns() -> None:
    """Missing score columns are handled gracefully."""
    # Arrange
    df = pl.DataFrame({"season": [2024], "week": [1]})

    # Act
    mismatches = validation_utils.compare_latest_week_scores(df, None)

    # Assert
    assert mismatches.height == 0


def test_compare_latest_week_scores_without_completed_games_returns_empty() -> None:
    """A schedule comparison is skipped when there are no completed games yet."""
    # Arrange
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

    # Act
    mismatches = validation_utils.compare_latest_week_scores(df)

    # Assert
    assert mismatches.height == 0


def test_compare_latest_week_scores_load_failure(monkeypatch) -> None:
    """Schedule load failure is handled gracefully."""
    # Arrange
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("fail")),
    )

    # Act
    mismatches = validation_utils.compare_latest_week_scores(df, None)

    # Assert
    assert mismatches.height == 0


def test_compare_latest_week_scores_missing_schedule_week_returns_empty() -> None:
    """A missing schedule row for the latest completed week should return no mismatches."""
    # Arrange
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

    # Act
    mismatches = validation_utils.compare_latest_week_scores(all_data, schedule)

    # Assert
    assert mismatches.height == 0


def test_compare_latest_week_scores_without_matching_game_join_returns_empty() -> None:
    """Completed and schedule data with no matching matchup should return no mismatches."""
    # Arrange
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

    # Act
    mismatches = validation_utils.compare_latest_week_scores(all_data, schedule)

    # Assert
    assert mismatches.height == 0


def test_compare_latest_week_scores_surfaces_unexpected_load_errors(monkeypatch) -> None:
    """Only I/O and data errors from the schedule load degrade; a bug still surfaces."""
    # Arrange
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    # Act & Assert
    with pytest.raises(RuntimeError, match="bug"):
        validation_utils.compare_latest_week_scores(df, None)
