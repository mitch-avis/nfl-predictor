"""Tests for offline dataset validation utilities."""

import polars as pl

from nfl_predictor.utils import validation_utils


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


def test_validate_week_range() -> None:
    """Out-of-range weeks are flagged."""
    df = pl.DataFrame({"season": [2024, 2024], "week": [1, 25]})
    issues = validation_utils.validate_week_range(df)
    assert any("week 25" in issue for issue in issues)


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


def test_compare_latest_week_scores_missing_columns() -> None:
    """Missing score columns are handled gracefully."""
    df = pl.DataFrame({"season": [2024], "week": [1]})
    mismatches = validation_utils.compare_latest_week_scores(df, None)
    assert mismatches.height == 0


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
