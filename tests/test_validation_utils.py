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
