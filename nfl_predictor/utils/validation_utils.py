"""
Validation utilities for NFL predictor datasets.

These helpers focus on offline checks for schema, ranges, and internal consistency,
with optional hooks to compare latest results against an external schedule source.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log


@dataclass
class ValidationResult:
    """Simple container for validation issues."""

    errors: list[str]
    warnings: list[str]

    def is_valid(self) -> bool:
        """Return True if no errors were found."""
        return not self.errors


def validate_required_columns(df: pl.DataFrame, required_cols: Iterable[str]) -> list[str]:
    """Return a list of missing required columns."""
    required = list(required_cols)
    missing = [col for col in required if col not in df.columns]
    return missing


def validate_team_abbrs(
    df: pl.DataFrame,
    columns: Iterable[str] = ("away_abbr", "home_abbr"),
) -> list[str]:
    """Return a list of invalid team abbreviations in the given columns."""
    valid = set(constants.TEAM_ABBR)
    invalid = []
    for col in columns:
        if col not in df.columns:
            continue
        values = df.select(pl.col(col).unique()).to_series().to_list()
        for value in values:
            if value is None:
                continue
            if value not in valid:
                invalid.append(f"{col}:{value}")
    return invalid


def validate_week_range(
    df: pl.DataFrame,
    season_col: str = "season",
    week_col: str = "week",
) -> list[str]:
    """Return a list of week values outside the expected range for their season."""
    if season_col not in df.columns or week_col not in df.columns:
        return []

    issues = []
    for row in df.select([season_col, week_col]).iter_rows(named=True):
        season = row.get(season_col)
        week = row.get(week_col)
        if season is None or week is None:
            continue
        max_week = constants.get_regular_season_weeks(int(season)) + 4
        if week < 1 or week > max_week:
            issues.append(f"season {season} has out-of-range week {week}")
    return issues


def validate_unique_games(df: pl.DataFrame) -> list[str]:
    """Return a list of duplicate game identifiers with conflicting data."""
    if "game_id" in df.columns:
        duplicates = (
            df.filter(pl.col("game_id").is_not_null())
            .group_by("game_id")
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicates.height == 0:
            return []

        issues = []
        for game_id in duplicates.select("game_id").to_series().to_list():
            subset = df.filter(pl.col("game_id") == game_id)
            if subset.unique().height > 1:
                issues.append(f"duplicate game_id {game_id}")
        return issues

    key_cols = ["season", "week", "away_abbr", "home_abbr"]
    if not all(col in df.columns for col in key_cols):
        return []

    duplicates = df.group_by(key_cols).len().filter(pl.col("len") > 1)
    if duplicates.height == 0:
        return []

    issues = []
    for row in duplicates.iter_rows(named=True):
        subset = df.filter(
            (pl.col("season") == row["season"])
            & (pl.col("week") == row["week"])
            & (pl.col("away_abbr") == row["away_abbr"])
            & (pl.col("home_abbr") == row["home_abbr"])
        )
        if subset.unique().height > 1:
            issues.append(
                "duplicate game "
                f"{row['season']}-W{row['week']} {row['away_abbr']}@{row['home_abbr']}"
            )
    return issues


def validate_scores(df: pl.DataFrame) -> list[str]:
    """Return a list of score-related issues."""
    issues = []
    for col in ("away_score", "home_score"):
        if col not in df.columns:
            continue
        negatives = df.filter(pl.col(col).is_not_null() & (pl.col(col) < 0))
        if negatives.height > 0:
            issues.append(f"negative values in {col}: {negatives.height}")
    return issues


def validate_dataframe(df: pl.DataFrame) -> ValidationResult:
    """Run a standard validation suite for a collected dataset."""
    errors = []
    warnings = []

    required = (
        constants.POLARS_METADATA_COLUMNS
        + constants.POLARS_LINES_COLUMNS
        + constants.POLARS_RESULT_COLUMNS
    )
    missing = validate_required_columns(df, required)
    if missing:
        errors.append(f"missing required columns: {missing}")

    invalid_teams = validate_team_abbrs(df)
    if invalid_teams:
        errors.append(f"invalid team abbreviations: {invalid_teams[:10]}")

    week_issues = validate_week_range(df)
    if week_issues:
        errors.append(f"out-of-range weeks: {week_issues[:10]}")

    dup_issues = validate_unique_games(df)
    if dup_issues:
        errors.append(f"duplicate games: {dup_issues[:10]}")

    score_issues = validate_scores(df)
    if score_issues:
        errors.append(f"score issues: {score_issues}")

    return ValidationResult(errors=errors, warnings=warnings)


def compare_latest_week_scores(
    all_data_df: pl.DataFrame,
    schedule_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Compare the latest completed week in all_data against a schedule source.

    If schedule_df is None, loads it via polars_utils.load_schedule() for the
    detected latest season. This may require network access depending on the
    nflreadpy backend.

    Returns:
        DataFrame of mismatched games (empty if none or if schedule unavailable).
    """
    required_cols = {"season", "week", "away_abbr", "home_abbr", "away_score", "home_score"}
    if not required_cols.issubset(all_data_df.columns):
        log.warning("all_data_df missing required score columns for comparison")
        return pl.DataFrame()

    completed = all_data_df.filter(
        pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null()
    )
    if completed.height == 0:
        return pl.DataFrame()

    latest_week = (
        completed.select(["season", "week"])
        .unique()
        .sort(["season", "week"], descending=True)
        .head(1)
    )
    latest_season = int(latest_week.select("season").item())
    latest_week_num = int(latest_week.select("week").item())

    if schedule_df is None:
        try:
            schedule_df = polars_utils.load_schedule([latest_season])
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Failed to load schedule for validation: %s", exc)
            return pl.DataFrame()

    schedule_week = schedule_df.filter(
        (pl.col("season") == latest_season) & (pl.col("week") == latest_week_num)
    )

    if schedule_week.height == 0:
        return pl.DataFrame()

    schedule_week = schedule_week.select(list(required_cols))
    completed_week = completed.filter(
        (pl.col("season") == latest_season) & (pl.col("week") == latest_week_num)
    ).select(list(required_cols))

    joined = completed_week.join(
        schedule_week,
        on=["season", "week", "away_abbr", "home_abbr"],
        how="inner",
        suffix="_schedule",
    )

    if joined.height == 0:
        return pl.DataFrame()

    mismatches = joined.filter(
        (pl.col("away_score") != pl.col("away_score_schedule"))
        | (pl.col("home_score") != pl.col("home_score_schedule"))
    )

    return mismatches
