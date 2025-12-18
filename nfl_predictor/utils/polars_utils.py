"""
Polars-based utility functions for NFL data processing.

This module provides functions for loading, transforming, and processing NFL game data
using nflreadpy as the primary data source and Polars for high-performance data manipulation.

Key functionality includes:
    - Loading NFL schedules with game metadata, lines/odds, and stadium information
    - Loading per-game team statistics from nflreadpy
    - Loading ELO and QB ELO ratings from local CSV files
    - Loading TeamRankings data from scraped CSV files
    - Aggregating per-game stats into rolling averages for ML features
    - Computing derived metrics (yards per point, penalty efficiency, etc.)
    - Normalizing team abbreviations across different data sources

Data Flow:
    1. load_schedule() -> Schedule with lines, venue info, city/state
    2. load_team_stats() -> Per-game offensive/defensive stats
    3. add_scoring_data_to_team_stats() -> Adds points scored/allowed from schedule
    4. aggregate_team_stats_to_week() -> Rolling averages for prediction
    5. merge_schedule_with_team_stats() -> Combines schedule with aggregated stats
    6. calculate_stat_differentials() -> Computes away - home differences

Example:
    >>> from nfl_predictor.utils import polars_utils
    >>> schedule = polars_utils.load_schedule([2023, 2024])
    >>> team_stats = polars_utils.load_team_stats([2023, 2024])
    >>> agg_stats = polars_utils.aggregate_team_stats_to_week(team_stats, week=10, season=2024)
"""

import math
import os
import re
from typing import Optional

import nflreadpy as nfl
import polars as pl
import requests
from bs4 import BeautifulSoup
from polars.datatypes import DataType

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _is_numeric_dtype(dtype: DataType) -> bool:
    """Return True if dtype is numeric."""
    return isinstance(dtype, pl.Decimal) or dtype in NUMERIC_DTYPES


def normalize_team_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Normalize team abbreviations in a column to canonical form.

    Args:
        df: Polars DataFrame
        column: Name of the column containing team abbreviations

    Returns:
        DataFrame with normalized team abbreviations
    """
    return df.with_columns(pl.col(column).replace(constants.ALIAS_TO_CANONICAL).alias(column))


def load_schedule(seasons: list[int]) -> pl.DataFrame:
    """
    Load NFL schedule data for specified seasons using nflreadpy.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with schedule data including lines/odds
    """
    log.info("Loading schedule for seasons: %s", seasons)
    schedule_df = nfl.load_schedules(seasons=seasons)

    # Select only the columns we need (if they exist)
    available_cols = set(schedule_df.columns)
    cols_to_select = [c for c in constants.NFLREADPY_SCHEDULE_COLUMNS if c in available_cols]
    schedule_df = schedule_df.select(cols_to_select)

    # Rename columns to match our internal naming
    rename_mapping = {
        k: v for k, v in constants.NFLREADPY_SCHEDULE_RENAME.items() if k in schedule_df.columns
    }
    schedule_df = schedule_df.rename(rename_mapping)

    # Normalize team abbreviations
    if "away_abbr" in schedule_df.columns:
        schedule_df = normalize_team_column(schedule_df, "away_abbr")
    if "home_abbr" in schedule_df.columns:
        schedule_df = normalize_team_column(schedule_df, "home_abbr")

    # Transform neutral column: "Home" -> 0, "Neutral" -> 1
    if "neutral" in schedule_df.columns:
        schedule_df = schedule_df.with_columns(
            pl.when(pl.col("neutral") == "Neutral")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("neutral")
        )

    # Calculate away_spread from home_spread (nflreadpy uses home perspective)
    if "home_spread" in schedule_df.columns:
        schedule_df = schedule_df.with_columns((-pl.col("home_spread")).alias("away_spread"))

    # Parse date column
    if "date" in schedule_df.columns:
        schedule_df = schedule_df.with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))

    # Add stadium city and state from stadium_id
    if "stadium_id" in schedule_df.columns:
        schedule_df = _add_stadium_location(schedule_df)

    return schedule_df


def _add_stadium_location(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add stadium city and state columns based on stadium_id.

    Uses the STADIUM_LOCATIONS mapping in constants to look up
    city and state for each stadium.

    Args:
        df: DataFrame with stadium_id column

    Returns:
        DataFrame with stadium_city and stadium_state columns added
    """
    # Create mapping dictionaries for city and state
    city_map = {k: v["city"] for k, v in constants.STADIUM_LOCATIONS.items()}
    state_map = {k: v["state"] for k, v in constants.STADIUM_LOCATIONS.items()}

    # Add city and state columns using replace
    df = df.with_columns(
        [
            pl.col("stadium_id").replace(city_map, default=None).alias("stadium_city"),
            pl.col("stadium_id").replace(state_map, default=None).alias("stadium_state"),
        ]
    )

    return df


def load_team_stats(seasons: list[int], regular_season_only: bool = True) -> pl.DataFrame:
    """
    Load team statistics for specified seasons using nflreadpy.

    Args:
        seasons: List of season years to load
        regular_season_only: If True, filter to only regular season games

    Returns:
        Polars DataFrame with team statistics per game
    """
    log.info("Loading team stats for seasons: %s", seasons)
    team_stats_df = nfl.load_team_stats(seasons=seasons)

    # Filter to regular season only (exclude preseason and postseason)
    if regular_season_only and "season_type" in team_stats_df.columns:
        team_stats_df = team_stats_df.filter(pl.col("season_type") == "REG")
        log.debug("Filtered to regular season only: %d records", team_stats_df.height)

    # Normalize team abbreviations
    if "team" in team_stats_df.columns:
        team_stats_df = normalize_team_column(team_stats_df, "team")
        team_stats_df = team_stats_df.rename({"team": "team_abbr"})

    if "opponent_team" in team_stats_df.columns:
        team_stats_df = normalize_team_column(team_stats_df, "opponent_team")
        team_stats_df = team_stats_df.rename({"opponent_team": "opponent_abbr"})

    # Rename columns using our mapping
    available_cols = set(team_stats_df.columns)
    rename_mapping = {
        k: v for k, v in constants.NFLREADPY_TEAM_STATS_MAPPING.items() if k in available_cols
    }
    team_stats_df = team_stats_df.rename(rename_mapping)

    # Combine stats as specified
    team_stats_df = combine_stats(team_stats_df)

    # Add per-game opponent stats (opponent's stats in each game)
    # This allows aggregating "stats of teams this team has faced"
    team_stats_df = add_per_game_opponent_stats(team_stats_df)

    return team_stats_df


def add_scoring_data_to_team_stats(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add points scored and points allowed from schedule to team stats.

    The schedule has per-game scores (away_score, home_score). This function
    extracts that into per-team format and merges with team_stats so we can
    compute scoring-related metrics.

    Args:
        team_stats_df: Per-team per-game stats DataFrame
        schedule_df: Schedule DataFrame with scores

    Returns:
        team_stats_df with points_scored and points_allowed columns added
    """
    if schedule_df.height == 0:
        return team_stats_df

    # Extract scoring for away teams
    away_scores = schedule_df.select(
        [
            pl.col("season"),
            pl.col("week"),
            pl.col("away_abbr").alias("team_abbr"),
            pl.col("away_score").alias("points_scored"),
            pl.col("home_score").alias("points_allowed"),
        ]
    )

    # Extract scoring for home teams
    home_scores = schedule_df.select(
        [
            pl.col("season"),
            pl.col("week"),
            pl.col("home_abbr").alias("team_abbr"),
            pl.col("home_score").alias("points_scored"),
            pl.col("away_score").alias("points_allowed"),
        ]
    )

    # Combine into per-team scoring
    all_scores = pl.concat([away_scores, home_scores])

    # Compute scoring margin
    all_scores = all_scores.with_columns(
        [
            (pl.col("points_scored") - pl.col("points_allowed")).alias("scoring_margin"),
        ]
    )

    # Merge with team_stats
    result = team_stats_df.join(
        all_scores,
        on=["season", "week", "team_abbr"],
        how="left",
    )

    return result


def combine_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine related stats into single columns and remove originals.

    Combines:
    - sack + rushing + receiving fumbles -> fumbles
    - sack + rushing + receiving fumbles_lost -> fumbles_lost
    - passing + rushing + receiving first_downs -> first_downs
    - passing + rushing + receiving 2pt_conversions -> 2pt_conversions
    - fumble_recovery_own + fumble_recovery_opp -> fumble_recoveries

    Args:
        df: DataFrame with raw stats

    Returns:
        DataFrame with combined stats
    """
    combine_operations = []

    # Fumbles (sack + rushing + receiving)
    fumble_cols = ["sack_fumbles", "rushing_fumbles", "receiving_fumbles"]
    if all(c in df.columns for c in fumble_cols):
        combine_operations.append(
            (
                pl.col("sack_fumbles") + pl.col("rushing_fumbles") + pl.col("receiving_fumbles")
            ).alias("fumbles")
        )

    # Fumbles lost (sack + rushing + receiving)
    fumble_lost_cols = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
    if all(c in df.columns for c in fumble_lost_cols):
        combine_operations.append(
            (
                pl.col("sack_fumbles_lost")
                + pl.col("rushing_fumbles_lost")
                + pl.col("receiving_fumbles_lost")
            ).alias("fumbles_lost")
        )

    # First downs (passing + rushing only - receiving first downs overlap with passing)
    first_down_cols = ["passing_first_downs", "rushing_first_downs"]
    if all(c in df.columns for c in first_down_cols):
        combine_operations.append(
            (pl.col("passing_first_downs") + pl.col("rushing_first_downs")).alias("first_downs")
        )

    # 2pt conversions (passing + rushing + receiving)
    two_pt_cols = [
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
    ]
    if all(c in df.columns for c in two_pt_cols):
        combine_operations.append(
            (
                pl.col("passing_2pt_conversions")
                + pl.col("rushing_2pt_conversions")
                + pl.col("receiving_2pt_conversions")
            ).alias("2pt_conversions")
        )

    # Fumble recoveries (own + opponent)
    fumble_recovery_cols = ["fumble_recovery_own", "fumble_recovery_opp"]
    if all(c in df.columns for c in fumble_recovery_cols):
        combine_operations.append(
            (pl.col("fumble_recovery_own") + pl.col("fumble_recovery_opp")).alias(
                "fumble_recoveries"
            )
        )

    # Computed: Turnover margin = turnovers gained - turnovers lost
    # turnovers_gained = def_interceptions + fumble_recovery_opp
    # turnovers_lost = interceptions_thrown + fumbles_lost (after combining)
    # Note: passing_interceptions is renamed to interceptions_thrown before this function
    turnover_gain_cols = ["def_interceptions", "fumble_recovery_opp"]
    turnover_loss_cols = ["interceptions_thrown"]
    fumbles_lost_available = any(
        c in df.columns for c in ["sack_fumbles_lost", "rushing_fumbles_lost", "fumbles_lost"]
    )
    has_turnover_cols = all(c in df.columns for c in turnover_gain_cols + turnover_loss_cols)
    if has_turnover_cols and fumbles_lost_available:
        # We'll compute this after fumbles_lost is created, so use a placeholder
        pass  # Will be computed after with_columns

    # Apply combinations
    if combine_operations:
        df = df.with_columns(combine_operations)

    # Now compute turnover_margin (after fumbles_lost exists)
    has_fumbles_lost = "fumbles_lost" in df.columns
    has_all_turnover = all(c in df.columns for c in turnover_gain_cols + turnover_loss_cols)
    if has_fumbles_lost and has_all_turnover:
        df = df.with_columns(
            (
                (pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
                - (pl.col("interceptions_thrown") + pl.col("fumbles_lost"))
            ).alias("turnover_margin")
        )

    # Compute total yards (passing + rushing - sack yards lost)
    yards_cols = ["pass_yards", "rush_yards"]
    if all(c in df.columns for c in yards_cols):
        total_yards_expr = pl.col("pass_yards") + pl.col("rush_yards")
        # Subtract sack yards lost if available
        if "yards_lost_from_sacks" in df.columns:
            total_yards_expr = total_yards_expr - pl.col("yards_lost_from_sacks")
        df = df.with_columns(total_yards_expr.alias("total_yards"))

    # Drop original columns that were combined
    cols_to_drop = (
        fumble_cols + fumble_lost_cols + first_down_cols + two_pt_cols + fumble_recovery_cols
    )
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    return df


def add_per_game_opponent_stats(team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add opponent's stats for each game to the team's record.

    For each team-game record, looks up the opponent's stats for that same game
    and adds them as opponent_* columns. This allows aggregating "stats of teams
    this team has faced" when computing rolling averages.

    Args:
        team_stats_df: DataFrame with per-game team statistics

    Returns:
        DataFrame with added opponent_* columns for each game
    """
    # Identify stat columns to copy from opponent (exclude identifiers)
    exclude_cols = {"season", "week", "team_abbr", "opponent_abbr", "season_type", "games_played"}
    stat_cols = [col for col in team_stats_df.columns if col not in exclude_cols]

    # Create a lookup table with opponent stats
    # Key: (season, week, team_abbr) -> opponent's stats for that game
    opponent_lookup = team_stats_df.select(
        [pl.col("season"), pl.col("week"), pl.col("team_abbr")]
        + [pl.col(c).alias(f"opponent_{c}") for c in stat_cols]
    )

    # Join: for each team's game, look up the opponent's record using opponent_abbr
    # The join key is: this team's opponent_abbr = lookup's team_abbr (same game)
    result = team_stats_df.join(
        opponent_lookup,
        left_on=["season", "week", "opponent_abbr"],
        right_on=["season", "week", "team_abbr"],
        how="left",
        suffix="_opp_lookup",
    )

    return result


def load_pbp(seasons: list[int]) -> pl.DataFrame:
    """
    Load play-by-play data for specified seasons using nflreadpy.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with play-by-play data
    """
    log.info("Loading play-by-play for seasons: %s", seasons)
    pbp_df = nfl.load_pbp(seasons=seasons)

    # Normalize team abbreviations in relevant columns
    team_cols = ["home_team", "away_team", "posteam", "defteam"]
    for col in team_cols:
        if col in pbp_df.columns:
            pbp_df = normalize_team_column(pbp_df, col)

    return pbp_df


def aggregate_pbp_stats(
    pbp_df: pl.DataFrame,
    seasons: list[int],
) -> pl.DataFrame:
    """
    Aggregate play-by-play data into per-team per-week statistics.

    Computes statistics that aren't directly available in load_team_stats:
    - Third down attempts and conversions
    - Fourth down attempts and conversions
    - Red zone attempts and touchdowns
    - Two-point conversion attempts and successes
    - Total plays (for points per play calculations)

    Args:
        pbp_df: Play-by-play DataFrame from load_pbp
        seasons: List of seasons to process

    Returns:
        DataFrame with columns: season, week, team_abbr, and computed stats
    """
    if pbp_df.height == 0:
        return pl.DataFrame()

    # Filter to real plays (exclude nulls, penalties, etc.)
    # Filter to requested seasons and regular season
    plays = pbp_df.filter(
        pl.col("season").is_in(seasons)
        & pl.col("season_type").eq("REG")
        & pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["run", "pass", "qb_kneel", "qb_spike"])
    )

    if plays.height == 0:
        log.warning("No valid plays found in PBP data for aggregation")
        return pl.DataFrame()

    # Aggregate by team (posteam = team on offense)
    team_stats = plays.group_by(["season", "week", "posteam"]).agg(
        [
            # Third down stats
            pl.col("third_down_converted").sum().alias("third_down_conversions"),
            pl.col("third_down_failed").sum().alias("third_down_fails"),
            # Fourth down stats
            pl.col("fourth_down_converted").sum().alias("fourth_down_conversions"),
            pl.col("fourth_down_failed").sum().alias("fourth_down_fails"),
            # Red zone stats (plays inside opponent's 20)
            ((pl.col("yardline_100").le(20)) & (pl.col("td_team").is_not_null()))
            .sum()
            .alias("red_zone_tds"),
            (pl.col("yardline_100").le(20)).sum().alias("red_zone_plays"),
            # Two-point attempts
            pl.col("two_point_attempt").sum().alias("two_point_attempts"),
            (pl.col("two_point_conv_result") == "success").sum().alias("two_point_successes"),
            # Total plays for points per play
            pl.len().alias("total_plays"),
        ]
    )

    # Rename posteam to team_abbr
    team_stats = team_stats.rename({"posteam": "team_abbr"})

    # Calculate third and fourth down attempts
    team_stats = team_stats.with_columns(
        [
            (pl.col("third_down_conversions") + pl.col("third_down_fails")).alias(
                "third_down_attempts"
            ),
            (pl.col("fourth_down_conversions") + pl.col("fourth_down_fails")).alias(
                "fourth_down_attempts"
            ),
        ]
    )

    return team_stats


def load_elo_ratings(seasons: list[int]) -> pl.DataFrame:
    """
    Load ELO ratings from qb_elos.csv file.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with ELO ratings per game
    """
    elo_path = os.path.join(constants.DATA_PATH, "qb_elos.csv")

    if not os.path.exists(elo_path):
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    log.info("Loading ELO ratings from %s", elo_path)
    elo_df = pl.read_csv(elo_path)

    # Filter to requested seasons
    if "season" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("season").is_in(seasons))

    # Cast week to integer and filter out nulls
    if "week" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
        # Week may be like "19.0" so cast to float first, then int
        elo_df = elo_df.with_columns(pl.col("week").cast(pl.Float64).cast(pl.Int64))

    # Normalize team abbreviations
    if "team1" in elo_df.columns:
        elo_df = elo_df.with_columns(
            pl.col("team1").replace(constants.ALIAS_TO_CANONICAL).alias("team1")
        )
    if "team2" in elo_df.columns:
        elo_df = elo_df.with_columns(
            pl.col("team2").replace(constants.ALIAS_TO_CANONICAL).alias("team2")
        )

    # Select and rename columns for away/home format
    # In qb_elos.csv: team1 = home, team2 = away
    cols_to_keep = [
        "season",
        "week",
        "team1",
        "team2",
        "elo1_pre",
        "elo2_pre",
        "qb1",
        "qb2",
        "qb1_value_pre",
        "qb2_value_pre",
        "qbelo1_pre",
        "qbelo2_pre",
    ]
    available = [c for c in cols_to_keep if c in elo_df.columns]
    elo_df = elo_df.select(available)

    # Cast numeric string columns to floats
    numeric_cols = ["qb1_value_pre", "qb2_value_pre", "qbelo1_pre", "qbelo2_pre"]
    for col in numeric_cols:
        if col in elo_df.columns:
            elo_df = elo_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Rename to away/home format (team1=home, team2=away in ELO data)
    rename_map = {
        "team1": "home_abbr",
        "team2": "away_abbr",
        "elo1_pre": "home_elo_pre",
        "elo2_pre": "away_elo_pre",
        "qb1": "home_qb",
        "qb2": "away_qb",
        "qb1_value_pre": "home_qb_value_pre",
        "qb2_value_pre": "away_qb_value_pre",
        "qbelo1_pre": "home_qb_elo_pre",
        "qbelo2_pre": "away_qb_elo_pre",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in elo_df.columns}
    elo_df = elo_df.rename(rename_map)

    return elo_df


def load_raw_elo_data() -> pl.DataFrame:
    """
    Load raw ELO data from qb_elos.csv file without transformations.

    This is used for QB-specific lookups where we need the original
    column names (qb1, qb2, qb1_value_pre, etc.).

    Returns:
        Raw Polars DataFrame with ELO data
    """
    elo_path = os.path.join(constants.DATA_PATH, "qb_elos.csv")

    if not os.path.exists(elo_path):
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    elo_df = pl.read_csv(elo_path)

    # Cast numeric columns that may be strings
    numeric_cols = ["qb1_value_pre", "qb2_value_pre", "qbelo1_pre", "qbelo2_pre"]
    for col in numeric_cols:
        if col in elo_df.columns:
            elo_df = elo_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    return elo_df


def get_latest_elo_by_team(elo_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """
    Get the most recent ELO ratings for each team from a given season.

    This is used for future games that don't yet have specific week ELO data.
    For each team, finds their most recent ELO rating from the season.

    Args:
        elo_df: Full ELO DataFrame with season, week, and per-game ratings
        season: The season to get latest ratings from

    Returns:
        DataFrame with columns: team_abbr, elo_pre, qb_value_pre, qb_elo_pre
        One row per team with their most recent ELO values
    """
    if elo_df.height == 0:
        return pl.DataFrame()

    # Filter to the target season
    season_elo = elo_df.filter(pl.col("season") == season)

    if season_elo.height == 0:
        return pl.DataFrame()

    # We need to extract per-team ELO values. The data has away/home format.
    # Create a "melted" view with team_abbr and their ELO values
    away_elo = season_elo.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("away_abbr").alias("team_abbr"),
        pl.col("away_elo_pre").alias("elo_pre"),
        pl.col("away_qb_value_pre").alias("qb_value_pre"),
        pl.col("away_qb_elo_pre").alias("qb_elo_pre"),
    )

    home_elo = season_elo.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("home_abbr").alias("team_abbr"),
        pl.col("home_elo_pre").alias("elo_pre"),
        pl.col("home_qb_value_pre").alias("qb_value_pre"),
        pl.col("home_qb_elo_pre").alias("qb_elo_pre"),
    )

    # Combine and sort by week descending, then take first per team
    all_team_elo = pl.concat([away_elo, home_elo])
    all_team_elo = all_team_elo.sort("week", descending=True)

    # Group by team and take the first (most recent) row
    latest_elo = all_team_elo.group_by("team_abbr").agg(
        pl.col("elo_pre").first(),
        pl.col("qb_value_pre").first(),
        pl.col("qb_elo_pre").first(),
    )

    return latest_elo


def load_team_rankings(season: int) -> pl.DataFrame:
    """
    Load TeamRankings data for a season from CSV files.

    Args:
        season: Season year to load

    Returns:
        Polars DataFrame with TeamRankings ratings/stats per team per week
    """
    tr_path = os.path.join(constants.DATA_PATH, str(season), f"{season}_team_rankings.csv")

    if not os.path.exists(tr_path):
        log.debug("TeamRankings file not found: %s", tr_path)
        return pl.DataFrame()

    log.debug("Loading TeamRankings for season %d", season)
    tr_df = pl.read_csv(tr_path)

    # Drop unnamed index column if present
    if "" in tr_df.columns:
        tr_df = tr_df.drop("")

    # Rename 'abbr' to 'team_abbr' for consistency
    if "abbr" in tr_df.columns:
        tr_df = tr_df.rename({"abbr": "team_abbr"})

    # Normalize team abbreviations
    if "team_abbr" in tr_df.columns:
        tr_df = normalize_team_column(tr_df, "team_abbr")

    return tr_df


def get_latest_team_rankings(tr_df: pl.DataFrame) -> pl.DataFrame:
    """
    Get the most recent TeamRankings values for each team.

    This is used for:
    - Playoff games (use end-of-regular-season TR values)
    - Future weeks without specific TR data

    Args:
        tr_df: TeamRankings DataFrame for a season

    Returns:
        DataFrame with latest TR values per team (week column removed)
    """
    if tr_df.height == 0:
        return pl.DataFrame()

    if "week" not in tr_df.columns:
        return tr_df

    # Sort by week descending and take first per team
    tr_sorted = tr_df.sort("week", descending=True)

    # Get non-week columns for aggregation
    non_key_cols = [c for c in tr_df.columns if c not in ["week", "team_abbr"]]

    # Group by team and take first (most recent) value for each column
    agg_exprs = [pl.col(c).first() for c in non_key_cols]
    latest_tr = tr_sorted.group_by("team_abbr").agg(agg_exprs)

    return latest_tr


def aggregate_team_stats_to_week(
    team_stats_df: pl.DataFrame,
    target_week: int,
    season: int,
) -> pl.DataFrame:
    """
    Aggregate team statistics up to (but not including) a target week.

    This creates rolling averages of team performance metrics that can be used
    as features for predicting the target week's games.

    For regular season games, uses stats from prior weeks within the season.
    For playoff games (after regular season), uses the full regular season stats.

    Note: Before 2021, the NFL had 17-week regular seasons (playoffs started week 18).
          From 2021 onwards, the NFL has 18-week regular seasons (playoffs start week 19).

    Args:
        team_stats_df: DataFrame with per-game team statistics
        target_week: The week we're predicting (stats up to week-1 are used)
        season: The season year

    Returns:
        DataFrame with aggregated team statistics
    """
    # Get the number of regular season weeks for this season
    regular_season_weeks = constants.get_regular_season_weeks(season)

    # For playoff games (week > regular season weeks), use full regular season stats
    # For regular season, use only games before target_week
    if target_week > regular_season_weeks:
        max_week_to_include = regular_season_weeks
    else:
        max_week_to_include = target_week - 1

    # Filter to games in the specified week range
    prior_games = team_stats_df.filter(
        (pl.col("season") == season) & (pl.col("week") <= max_week_to_include)
    )

    if prior_games.height == 0:
        # Use debug level for week 1 (expected) - this is handled by regression logic
        log.debug("No prior games found for season %s week %s", season, target_week)
        return pl.DataFrame()

    # Identify numeric columns for aggregation (excluding identifiers)
    exclude_cols = {"season", "week", "team_abbr", "opponent_abbr", "season_type"}
    numeric_cols = [
        col
        for col in prior_games.columns
        if col not in exclude_cols and prior_games[col].dtype in [pl.Float64, pl.Int32, pl.Int64]
    ]

    # Aggregate by team
    agg_exprs = [pl.col(c).mean().alias(c) for c in numeric_cols]
    agg_exprs.append(pl.len().alias("games_played"))

    agg_df = prior_games.group_by("team_abbr").agg(agg_exprs)

    # Compute derived ratio metrics from the aggregated averages
    agg_df = _compute_derived_metrics(agg_df)

    return agg_df


def _compute_derived_metrics(agg_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute derived ratio metrics from aggregated team statistics.

    These metrics require division and should be computed after averaging
    raw stats to avoid ratio-of-averages issues.

    Args:
        agg_df: DataFrame with averaged team statistics

    Returns:
        DataFrame with additional derived metric columns
    """
    derived_cols = []

    # Yards per point = total_yards / points_scored
    if "total_yards" in agg_df.columns and "points_scored" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("points_scored") > 0)
            .then(pl.col("total_yards") / pl.col("points_scored"))
            .otherwise(pl.lit(0.0))
            .alias("yards_per_point")
        )

    # Opponent yards per point (from opponent_total_yards and points_allowed)
    if "opponent_total_yards" in agg_df.columns and "points_allowed" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("points_allowed") > 0)
            .then(pl.col("opponent_total_yards") / pl.col("points_allowed"))
            .otherwise(pl.lit(0.0))
            .alias("opponent_yards_per_point")
        )

    # Penalty yards per penalty = penalty_yards / penalties
    if "penalty_yards" in agg_df.columns and "penalties" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("penalties") > 0)
            .then(pl.col("penalty_yards") / pl.col("penalties"))
            .otherwise(pl.lit(0.0))
            .alias("penalty_yards_per_penalty")
        )

    # Opponent penalty yards per penalty
    opp_pen_yards = "opponent_penalty_yards"
    opp_pens = "opponent_penalties"
    if opp_pen_yards in agg_df.columns and opp_pens in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col(opp_pens) > 0)
            .then(pl.col(opp_pen_yards) / pl.col(opp_pens))
            .otherwise(pl.lit(0.0))
            .alias("opponent_penalty_yards_per_penalty")
        )

    if derived_cols:
        agg_df = agg_df.with_columns(derived_cols)

    return agg_df


def calculate_league_means(team_stats_df: pl.DataFrame, season: int) -> dict[str, float]:
    """
    Calculate league-wide mean statistics for a given season.

    Args:
        team_stats_df: DataFrame with per-game team statistics
        season: The season year to calculate means for

    Returns:
        Dictionary mapping stat names to their league-wide mean values
    """
    season_stats = team_stats_df.filter(pl.col("season") == season)

    if season_stats.height == 0:
        return {}

    # Identify numeric columns
    exclude_cols = {"season", "week", "team_abbr", "opponent_abbr", "season_type"}
    numeric_cols = [
        col
        for col in season_stats.columns
        if col not in exclude_cols and season_stats[col].dtype in [pl.Float64, pl.Int32, pl.Int64]
    ]

    # Calculate means
    means = {}
    for col in numeric_cols:
        mean_val = season_stats.select(pl.col(col).mean()).item()
        if mean_val is not None:
            means[col] = float(mean_val)

    return means


def regress_to_mean(
    team_stats: pl.DataFrame,
    league_means: dict[str, float],
    regression_factor: float = 1 / 3,
) -> pl.DataFrame:
    """
    Regress team statistics toward league mean.

    For week-1 games, we regress prior season stats toward the league mean
    to account for roster changes, coaching changes, and mean reversion.

    The formula is: regressed_stat = team_stat * (1 - factor) + league_mean * factor

    Args:
        team_stats: DataFrame with team statistics
        league_means: Dictionary of league-wide mean values
        regression_factor: Fraction to regress toward mean (default 1/3)

    Returns:
        DataFrame with regressed statistics
    """
    if not league_means:
        return team_stats

    regression_exprs = []
    for col in team_stats.columns:
        if col in league_means and col not in {"team_abbr", "games_played"}:
            # regressed = team_value * (1 - factor) + league_mean * factor
            regressed = (
                pl.col(col) * (1 - regression_factor)
                + pl.lit(league_means[col]) * regression_factor
            ).alias(col)
            regression_exprs.append(regressed)
        else:
            regression_exprs.append(pl.col(col))

    return team_stats.select(regression_exprs)


def get_stat_columns() -> list[str]:
    """
    Get the nflreadpy stat column names to use.

    Returns:
        List of stat column names
    """
    return constants.POLARS_NFLREADPY_STATS.copy()


def get_elo_columns() -> list[str]:
    """
    Get the ELO column names.

    Returns:
        List of ELO column names
    """
    return constants.POLARS_ELO_COLUMNS.copy()


def get_tr_columns() -> list[str]:
    """
    Get all TeamRankings column names (ratings + stats).

    Returns:
        List of TeamRankings column names
    """
    return constants.POLARS_TR_RATINGS.copy() + constants.POLARS_TR_STATS.copy()


def calculate_game_result(row_dict: dict) -> Optional[float]:
    """
    Calculate game result from away team perspective.

    Args:
        row_dict: Dictionary with away_score and home_score

    Returns:
        1.0 if away team won, 0.0 if lost, 0.5 if tie, None if scores are missing
    """
    away_score = row_dict.get("away_score")
    home_score = row_dict.get("home_score")

    if away_score is None or home_score is None:
        return None

    if away_score > home_score:
        return 1.0
    elif away_score < home_score:
        return 0.0
    else:
        return 0.5


def merge_schedule_with_team_stats(
    schedule_df: pl.DataFrame,
    agg_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge schedule data with aggregated team statistics.

    Args:
        schedule_df: Schedule DataFrame with game matchups
        agg_stats_df: Aggregated team statistics DataFrame

    Returns:
        DataFrame with schedule and team stats merged for both away and home teams
    """
    # Prepare away team stats with prefix
    away_stats = agg_stats_df.rename(
        {col: f"away_{col}" for col in agg_stats_df.columns if col != "team_abbr"}
    ).rename({"team_abbr": "away_abbr"})

    # Prepare home team stats with prefix
    home_stats = agg_stats_df.rename(
        {col: f"home_{col}" for col in agg_stats_df.columns if col != "team_abbr"}
    ).rename({"team_abbr": "home_abbr"})

    # Join with schedule
    merged = schedule_df.join(away_stats, on="away_abbr", how="left")
    merged = merged.join(home_stats, on="home_abbr", how="left")

    return merged


def calculate_stat_differentials(
    df: pl.DataFrame,
    stats_to_diff: list[str],
) -> pl.DataFrame:
    """
    Calculate differentials between away and home team statistics.

    Args:
        df: DataFrame with away_* and home_* prefixed stat columns
        stats_to_diff: List of stat names (without prefix) to calculate differences for

    Returns:
        DataFrame with added *_diff columns
    """
    for stat in stats_to_diff:
        away_col = f"away_{stat}"
        home_col = f"home_{stat}"

        if away_col in df.columns and home_col in df.columns:
            away_dtype = df.schema.get(away_col)
            home_dtype = df.schema.get(home_col)

            # Only compute differentials on numeric columns; skip others to avoid type errors
            if away_dtype is None or home_dtype is None:
                continue

            if not (_is_numeric_dtype(away_dtype) and _is_numeric_dtype(home_dtype)):
                log.debug(
                    "Skipping diff for %s (non-numeric types: %s vs %s)",
                    stat,
                    away_dtype,
                    home_dtype,
                )
                continue

            df = df.with_columns((pl.col(away_col) - pl.col(home_col)).alias(f"{stat}_diff"))

    return df


def get_team_name(abbr: str) -> Optional[str]:
    """
    Get the full team name from an abbreviation.

    Args:
        abbr: Team abbreviation (canonical or alias)

    Returns:
        Full team name or None if not found
    """
    canonical = constants.normalize_team_abbr(abbr)
    team_info = constants.TEAM_MAPPING.get(canonical)
    return team_info["name"] if team_info else None


def filter_completed_games(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter schedule to only completed games (those with scores).

    Args:
        schedule_df: Schedule DataFrame

    Returns:
        DataFrame with only completed games
    """
    return schedule_df.filter(
        pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null()
    )


def filter_upcoming_games(
    schedule_df: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """
    Filter schedule to upcoming games for a specific week.

    Args:
        schedule_df: Schedule DataFrame
        season: Season year
        week: Week number

    Returns:
        DataFrame with upcoming games (no scores yet)
    """
    return schedule_df.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("away_score").is_null() | pl.col("home_score").is_null())
    )


def remove_diff_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove all differential columns (ending in _diff) from a DataFrame.

    This creates a non-ML version of the data for faster local usage.

    Args:
        df: DataFrame with differential columns

    Returns:
        DataFrame without _diff columns
    """
    non_diff_cols = [c for c in df.columns if not c.endswith("_diff")]
    return df.select(non_diff_cols)


def polars_to_pandas(df: pl.DataFrame):
    """
    Convert a Polars DataFrame to pandas for compatibility with existing code.

    Args:
        df: Polars DataFrame

    Returns:
        pandas DataFrame
    """
    return df.to_pandas()


def pandas_to_polars(df) -> pl.DataFrame:
    """
    Convert a pandas DataFrame to Polars.

    Args:
        df: pandas DataFrame

    Returns:
        Polars DataFrame
    """
    return pl.from_pandas(df)


def get_stats_for_diff() -> list[str]:
    """
    Get list of stats that should have differentials calculated.

    Returns:
        List of stat names (without prefix) to calculate diffs for
    """
    all_stats = []

    # nflreadpy stats
    all_stats.extend(get_stat_columns())

    # Opponent stats
    all_stats.extend([f"opponent_{s}" for s in get_stat_columns()])

    # ELO columns
    all_stats.extend(get_elo_columns())

    # TeamRankings columns
    all_stats.extend(get_tr_columns())

    return all_stats


def build_final_column_order() -> list[str]:
    """
    Build the final column order according to specification.

    Order:
    1. Metadata columns
    2. ALL away_ columns (ELO, TR ratings, TR stats, nflreadpy stats, opponent stats)
    3. ALL home_ columns (ELO, TR ratings, TR stats, nflreadpy stats, opponent stats)
    4. ALL diff columns
    5. Lines/odds columns
    6. Result columns

    Note: Deduplicates columns that appear in multiple source lists
    (e.g., penalty_yards_per_penalty in both TR and computed stats).

    Returns:
        Ordered list of column names
    """
    columns = []

    # 1. Metadata
    columns.extend(constants.POLARS_METADATA_COLUMNS)

    # Build the per-team column list (ELO, TR, stats, opponent stats)
    per_team_cols = []

    # ELO columns
    per_team_cols.extend(get_elo_columns())

    # TeamRankings ratings and stats
    per_team_cols.extend(get_tr_columns())

    # nflreadpy stats (may overlap with TR - will be deduplicated)
    per_team_cols.extend(get_stat_columns())

    # Opponent stats (from nflreadpy)
    per_team_cols.extend([f"opponent_{s}" for s in get_stat_columns()])

    # Deduplicate while preserving order
    seen = set()
    unique_per_team = []
    for col in per_team_cols:
        if col not in seen:
            seen.add(col)
            unique_per_team.append(col)
    per_team_cols = unique_per_team

    # 2. ALL away_ columns
    columns.extend([f"away_{s}" for s in per_team_cols])

    # 3. ALL home_ columns
    columns.extend([f"home_{s}" for s in per_team_cols])

    # 4. ALL diff columns (same order as per_team_cols)
    columns.extend([f"{s}_diff" for s in per_team_cols])

    # 5. Lines/odds
    columns.extend(constants.POLARS_LINES_COLUMNS)

    # 6. Results
    columns.extend(constants.POLARS_RESULT_COLUMNS)

    return columns


def select_final_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Select and order final columns according to specification.

    Args:
        df: DataFrame with all computed columns

    Returns:
        DataFrame with only specified columns in correct order
    """
    final_order = build_final_column_order()

    # Filter to columns that exist in the DataFrame
    available_cols = [c for c in final_order if c in df.columns]

    # Log any expected columns that are missing
    missing_cols = [c for c in final_order if c not in df.columns]
    if missing_cols:
        log.debug("Missing expected columns: %s", missing_cols[:10])

    return df.select(available_cols)


def spread_to_moneyline(spread: float, vig: float = 0.05) -> int:
    """
    Convert an NFL point spread to a moneyline, including the effect of vig.

    Uses the normal distribution to model score differentials and convert
    spreads to implied probabilities, then to moneyline odds.

    Args:
        spread: The point spread (negative for favorites, positive for underdogs)
        vig: The vig percentage as a decimal (default is 0.05 for 5%)

    Returns:
        The moneyline corresponding to the given spread
    """
    # Use the standard deviation of NFL score differences
    std_dev = constants.SCORE_DIFF_STD_DEV

    # Calculate the implied probability using normal CDF
    # This is equivalent to: stats.norm.cdf(-spread, 0, std_dev)
    z_score = -spread / std_dev
    implied_probability = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    # Scale probabilities to include the vig
    adjusted_implied_probability = implied_probability * (1 + vig)

    # Clamp probability to avoid division by zero
    adjusted_implied_probability = max(0.01, min(0.99, adjusted_implied_probability))

    # Convert probabilities back to moneyline odds
    if spread < 0:
        # Favorite moneyline (negative)
        moneyline = -100 * (adjusted_implied_probability / (1 - adjusted_implied_probability))
    elif spread > 0:
        # Underdog moneyline (positive)
        moneyline = 100 * ((1 - adjusted_implied_probability) / adjusted_implied_probability)
    else:
        # Pick'em (spread = 0): slight underdog due to vig
        moneyline = 100 * ((1 - adjusted_implied_probability) / adjusted_implied_probability)

    return int(round(moneyline))


def fill_missing_moneylines(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill in missing moneylines by calculating them from spreads.

    For games where moneyline is missing but spread is available,
    calculates the moneyline using spread_to_moneyline conversion.

    Args:
        df: DataFrame with home_spread, away_spread, home_moneyline, away_moneyline columns

    Returns:
        DataFrame with moneylines filled in where possible
    """
    if "home_spread" not in df.columns:
        return df

    # Check if we have missing moneylines but valid spreads
    has_home_ml = "home_moneyline" in df.columns
    has_away_ml = "away_moneyline" in df.columns

    # For rows with spread but missing moneyline, calculate it
    if has_home_ml and has_away_ml:
        # Count missing moneylines with valid spreads
        missing_count = df.filter(
            pl.col("home_spread").is_not_null()
            & (pl.col("home_moneyline").is_null() | pl.col("away_moneyline").is_null())
        ).height

        if missing_count > 0:
            log.debug("Calculating %d missing moneylines from spreads", missing_count)

            # Calculate moneylines for all rows, then fill only where missing
            df = df.with_columns(
                [
                    pl.when(
                        pl.col("home_moneyline").is_null() & pl.col("home_spread").is_not_null()
                    )
                    .then(
                        pl.col("home_spread").map_elements(
                            lambda s: spread_to_moneyline(s) if s is not None else None,
                            return_dtype=pl.Int64,
                        )
                    )
                    .otherwise(pl.col("home_moneyline"))
                    .alias("home_moneyline"),
                    pl.when(
                        pl.col("away_moneyline").is_null() & pl.col("away_spread").is_not_null()
                    )
                    .then(
                        pl.col("away_spread").map_elements(
                            lambda s: spread_to_moneyline(s) if s is not None else None,
                            return_dtype=pl.Int64,
                        )
                    )
                    .otherwise(pl.col("away_moneyline"))
                    .alias("away_moneyline"),
                ]
            )

    return df


def get_latest_qb_by_team(elo_df: pl.DataFrame) -> pl.DataFrame:
    """
    Get the most recent starting QB for each team from ELO data.

    Uses the qb_elos.csv data to find the most recent game where each team
    had a recorded starting QB.

    Args:
        elo_df: Raw ELO DataFrame with qb1, qb2, team1, team2 columns

    Returns:
        DataFrame with columns: team_abbr, qb_name, qb_value_pre, qb_elo_pre
    """
    required_cols = ["qb1", "qb2", "team1", "team2"]
    if not all(c in elo_df.columns for c in required_cols):
        return pl.DataFrame()

    # Normalize team abbreviations if needed
    elo_df = elo_df.with_columns(
        [
            pl.col("team1").replace(constants.ALIAS_TO_CANONICAL).alias("team1"),
            pl.col("team2").replace(constants.ALIAS_TO_CANONICAL).alias("team2"),
        ]
    )

    # Get home team QBs (team1 = home, qb1 = home QB)
    home_qbs = elo_df.filter(pl.col("qb1").is_not_null() & (pl.col("qb1") != "")).select(
        [
            pl.col("date"),
            pl.col("team1").alias("team_abbr"),
            pl.col("qb1").alias("qb_name"),
            pl.col("qb1_value_pre").alias("qb_value_pre"),
            pl.col("qbelo1_pre").alias("qb_elo_pre"),
        ]
    )

    # Get away team QBs (team2 = away, qb2 = away QB)
    away_qbs = elo_df.filter(pl.col("qb2").is_not_null() & (pl.col("qb2") != "")).select(
        [
            pl.col("date"),
            pl.col("team2").alias("team_abbr"),
            pl.col("qb2").alias("qb_name"),
            pl.col("qb2_value_pre").alias("qb_value_pre"),
            pl.col("qbelo2_pre").alias("qb_elo_pre"),
        ]
    )

    # Combine and sort by date descending
    all_qbs = pl.concat([home_qbs, away_qbs])
    all_qbs = all_qbs.sort("date", descending=True)

    # Get most recent QB per team
    latest_qbs = all_qbs.group_by("team_abbr").agg(
        [
            pl.col("qb_name").first(),
            pl.col("qb_value_pre").first(),
            pl.col("qb_elo_pre").first(),
        ]
    )

    return latest_qbs


def get_qb_elo_by_name(elo_df: pl.DataFrame, qb_name: str) -> dict:
    """
    Get the most recent ELO values for a specific QB by name.

    Searches both qb1 and qb2 columns to find the most recent game
    the QB started, then returns their pre-game ELO values.

    Args:
        elo_df: Full ELO DataFrame
        qb_name: Name of the QB to search for

    Returns:
        Dict with qb_value_pre and qb_elo_pre, or empty dict if not found
    """
    if elo_df.height == 0 or not qb_name:
        return {}

    # Search in qb1 column (home QB)
    qb1_games = elo_df.filter(pl.col("qb1") == qb_name)
    if qb1_games.height > 0:
        # Sort by date descending, take first
        latest = qb1_games.sort("date", descending=True).head(1)
        qb_value = latest.select("qb1_value_pre").item()
        qb_elo = latest.select("qbelo1_pre").item()
        return {"qb_value_pre": qb_value, "qb_elo_pre": qb_elo}

    # Search in qb2 column (away QB)
    qb2_games = elo_df.filter(pl.col("qb2") == qb_name)
    if qb2_games.height > 0:
        latest = qb2_games.sort("date", descending=True).head(1)
        qb_value = latest.select("qb2_value_pre").item()
        qb_elo = latest.select("qbelo2_pre").item()
        return {"qb_value_pre": qb_value, "qb_elo_pre": qb_elo}

    return {}


def fill_future_qb_data(
    df: pl.DataFrame,
    elo_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Fill in QB data for future games using each team's most recent starter.

    For games where away_qb/home_qb are null, looks up the most recent QB
    for each team from the ELO data and fills in their name and ELO values.

    Args:
        df: Combined data DataFrame with potential null QBs
        elo_df: Raw ELO DataFrame (to look up QB-specific ELO values)

    Returns:
        DataFrame with QB data filled in for future games
    """
    required_cols = ["away_abbr", "home_abbr"]
    if not all(c in df.columns for c in required_cols):
        return df

    # Check if away_qb column exists; if not, all games need QB data
    has_away_qb = "away_qb" in df.columns
    has_home_qb = "home_qb" in df.columns

    if has_away_qb:
        null_away = df.filter(pl.col("away_qb").is_null())
    else:
        null_away = df  # All rows need QB data

    if has_home_qb:
        null_home = df.filter(pl.col("home_qb").is_null())
    else:
        null_home = df

    if null_away.height == 0 and null_home.height == 0:
        return df

    log.debug(
        "Filling QB data for %d away nulls and %d home nulls",
        null_away.height,
        null_home.height,
    )

    # Get latest QB per team from ELO data
    latest_qbs = get_latest_qb_by_team(elo_df)

    if latest_qbs.height == 0:
        return df

    # Build lookup dict for QBs
    qb_lookup = {}
    for row in latest_qbs.iter_rows(named=True):
        team = row["team_abbr"]
        qb_lookup[team] = {
            "qb_name": row["qb_name"],
            "qb_value_pre": row["qb_value_pre"],
            "qb_elo_pre": row["qb_elo_pre"],
        }

    # Fill in away QB data
    if null_away.height > 0:
        new_cols = [
            pl.when(pl.col("away_qb").is_null() if has_away_qb else pl.lit(True))
            .then(
                pl.col("away_abbr").map_elements(
                    lambda t: qb_lookup.get(t, {}).get("qb_name"), return_dtype=pl.Utf8
                )
            )
            .otherwise(pl.col("away_qb") if has_away_qb else pl.lit(None))
            .alias("away_qb"),
        ]
        # Only fill ELO values if columns exist and are null
        if "away_qb_value_pre" in df.columns:
            new_cols.append(
                pl.when(pl.col("away_qb_value_pre").is_null())
                .then(
                    pl.col("away_abbr").map_elements(
                        lambda t: qb_lookup.get(t, {}).get("qb_value_pre"), return_dtype=pl.Float64
                    )
                )
                .otherwise(pl.col("away_qb_value_pre"))
                .alias("away_qb_value_pre")
            )
        if "away_qb_elo_pre" in df.columns:
            new_cols.append(
                pl.when(pl.col("away_qb_elo_pre").is_null())
                .then(
                    pl.col("away_abbr").map_elements(
                        lambda t: qb_lookup.get(t, {}).get("qb_elo_pre"), return_dtype=pl.Float64
                    )
                )
                .otherwise(pl.col("away_qb_elo_pre"))
                .alias("away_qb_elo_pre")
            )
        df = df.with_columns(new_cols)

    # Fill in home QB data
    if null_home.height > 0:
        new_cols = [
            pl.when(pl.col("home_qb").is_null() if has_home_qb else pl.lit(True))
            .then(
                pl.col("home_abbr").map_elements(
                    lambda t: qb_lookup.get(t, {}).get("qb_name"), return_dtype=pl.Utf8
                )
            )
            .otherwise(pl.col("home_qb") if has_home_qb else pl.lit(None))
            .alias("home_qb"),
        ]
        if "home_qb_value_pre" in df.columns:
            new_cols.append(
                pl.when(pl.col("home_qb_value_pre").is_null())
                .then(
                    pl.col("home_abbr").map_elements(
                        lambda t: qb_lookup.get(t, {}).get("qb_value_pre"), return_dtype=pl.Float64
                    )
                )
                .otherwise(pl.col("home_qb_value_pre"))
                .alias("home_qb_value_pre")
            )
        if "home_qb_elo_pre" in df.columns:
            new_cols.append(
                pl.when(pl.col("home_qb_elo_pre").is_null())
                .then(
                    pl.col("home_abbr").map_elements(
                        lambda t: qb_lookup.get(t, {}).get("qb_elo_pre"), return_dtype=pl.Float64
                    )
                )
                .otherwise(pl.col("home_qb_elo_pre"))
                .alias("home_qb_elo_pre")
            )
        df = df.with_columns(new_cols)

    return df


def scrape_survivor_grid_spreads() -> dict[str, dict[int, float]]:
    """
    Scrape weekly spreads from SurvivorGrid.com for future games.

    The site provides spreads for remaining weeks in the current NFL season.
    Each team row shows the team name and spreads for upcoming weeks.

    Returns:
        Dictionary mapping team abbreviation to dict of week -> spread.
        Example: {"BUF": {16: -10.5, 17: -3.0, 18: -14.0}, ...}
        Returns empty dict if scraping fails.
    """
    try:
        response = requests.get(constants.SURVIVOR_GRID_URL, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch SurvivorGrid data: %s", e)
        return {}

    soup = BeautifulSoup(response.text, "lxml")

    # Find the main data table
    tables = soup.find_all("table")
    if not tables:
        log.warning("No tables found on SurvivorGrid page")
        return {}

    # The main grid table is typically the first/largest table
    data_table = None
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) >= 32:  # Should have all 32 teams
            data_table = table
            break

    if not data_table:
        log.warning("Could not find SurvivorGrid data table")
        return {}

    # Parse header row to get week numbers
    header_row = data_table.find("tr")
    if not header_row:
        return {}

    headers = []
    for th in header_row.find_all(["th", "td"]):
        text = th.get_text(strip=True)
        headers.append(text)

    # Find which columns contain week numbers and which has Team
    week_columns = {}  # index -> week number
    team_col_idx = None
    for i, header in enumerate(headers):
        if header.isdigit():
            week_columns[i] = int(header)
        elif header == "Team":
            team_col_idx = i

    if not week_columns:
        log.warning("No week columns found in SurvivorGrid table")
        return {}

    if team_col_idx is None:
        log.warning("No Team column found in SurvivorGrid table")
        return {}

    log.debug("Found SurvivorGrid week columns: %s", list(week_columns.values()))

    # Parse team rows
    spreads = {}
    rows = data_table.find_all("tr")[1:]  # Skip header

    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) <= team_col_idx:
            continue

        # Get team abbreviation from the Team column
        team_cell = cells[team_col_idx].get_text(strip=True)

        # Extract team abbr (may have record in parentheses like "BUF(10-4)")
        if "(" in team_cell:
            team_abbr_raw = team_cell.split("(")[0].strip()
        else:
            team_abbr_raw = team_cell.strip()

        # Normalize to canonical abbreviation
        team_abbr = constants.normalize_team_abbr(team_abbr_raw)
        if team_abbr not in constants.TEAM_ABBR:
            continue  # Not a valid team

        team_spreads = {}

        for col_idx, week in week_columns.items():
            if col_idx >= len(cells):
                continue

            cell = cells[col_idx]
            # Get cell text - format is like "@CLE-10.5" or "LV-14" or "@KC+3.5"
            cell_text = cell.get_text(strip=True)

            # Skip bye weeks and empty cells
            if not cell_text or cell_text == "BYE":
                continue

            # Extract spread from cell text
            # The spread is at the end, after the opponent indicator
            # Patterns: "@CLE-10.5", "LV+3", "@KC-7", "PK"
            # Look for spread pattern: optional sign followed by number or PK
            match = re.search(r"([+-]?\d+\.?\d*|PK)$", cell_text)
            if not match:
                continue

            spread_text = match.group(1)
            if spread_text == "PK":
                team_spreads[week] = 0.0
            else:
                try:
                    spread_val = float(spread_text)
                    team_spreads[week] = spread_val
                except ValueError:
                    continue

        if team_spreads:
            spreads[team_abbr] = team_spreads

    log.info("Scraped SurvivorGrid spreads for %d teams", len(spreads))
    return spreads


def fill_future_game_lines(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill in lines (spreads, moneylines, totals) for future games using SurvivorGrid data.

    Scrapes current spreads from SurvivorGrid.com and applies them to future games
    that don't have lines data. Also calculates moneylines from spreads and uses
    the historical average for total lines.

    Args:
        df: DataFrame with games, some of which may be missing lines data
        current_week: Current week number (auto-detected from schedule if not provided)

    Returns:
        DataFrame with lines filled in for future games
    """
    # Only process if we have the required columns
    required_cols = ["week", "away_abbr", "home_abbr"]
    if not all(c in df.columns for c in required_cols):
        return df

    # Identify future games (no score data)
    if "away_score" in df.columns:
        future_mask = pl.col("away_score").is_null()
    else:
        # Fall back to checking if lines are missing
        if "home_spread" in df.columns:
            future_mask = pl.col("home_spread").is_null()
        else:
            return df

    future_games = df.filter(future_mask)
    if future_games.height == 0:
        return df

    log.debug("Found %d future games to fill lines for", future_games.height)

    # Scrape current spreads
    spreads_data = scrape_survivor_grid_spreads()

    if not spreads_data:
        log.warning("No spreads data available from SurvivorGrid")
        return df

    # Build lookup function for home spread
    # SurvivorGrid shows spread from perspective of the team listed
    # Positive = underdog, Negative = favorite
    # We need to convert to home_spread perspective
    def get_home_spread(week: int, home_abbr: str, away_abbr: str) -> float | None:
        home_spread = spreads_data.get(home_abbr, {}).get(week)
        if home_spread is not None:
            return home_spread
        # Try getting from away perspective (negate)
        away_spread = spreads_data.get(away_abbr, {}).get(week)
        if away_spread is not None:
            return -away_spread
        return None

    # Apply spreads to future games
    updates = []
    for row in df.iter_rows(named=True):
        week = row.get("week")
        home_abbr = row.get("home_abbr")
        away_abbr = row.get("away_abbr")
        away_score = row.get("away_score")

        # Only update future games (no score) with missing spreads
        is_future = away_score is None
        has_spread = row.get("home_spread") is not None

        if is_future and not has_spread and week and home_abbr and away_abbr:
            home_spread = get_home_spread(week, home_abbr, away_abbr)
            if home_spread is not None:
                updates.append(
                    {
                        "game_id": row.get("game_id"),
                        "new_home_spread": home_spread,
                        "new_away_spread": -home_spread,
                        "new_total_line": constants.DEFAULT_TOTAL_LINE,
                    }
                )

    if not updates:
        return df

    log.info("Filling lines for %d future games from SurvivorGrid", len(updates))

    # Create updates DataFrame
    updates_df = pl.DataFrame(updates)

    # Join and update
    df = df.join(updates_df, on="game_id", how="left")

    # Apply updates where we have new values
    if "new_home_spread" in df.columns:
        df = df.with_columns(
            [
                pl.when(pl.col("new_home_spread").is_not_null())
                .then(pl.col("new_home_spread"))
                .otherwise(pl.col("home_spread") if "home_spread" in df.columns else pl.lit(None))
                .alias("home_spread"),
                pl.when(pl.col("new_away_spread").is_not_null())
                .then(pl.col("new_away_spread"))
                .otherwise(pl.col("away_spread") if "away_spread" in df.columns else pl.lit(None))
                .alias("away_spread"),
                pl.when(
                    pl.col("new_total_line").is_not_null()
                    & (
                        pl.col("total_line").is_null()
                        if "total_line" in df.columns
                        else pl.lit(True)
                    )
                )
                .then(pl.col("new_total_line"))
                .otherwise(pl.col("total_line") if "total_line" in df.columns else pl.lit(None))
                .alias("total_line"),
            ]
        )

        # Drop temporary columns
        df = df.drop(["new_home_spread", "new_away_spread", "new_total_line"])

    # Now calculate moneylines from the new spreads
    df = fill_missing_moneylines(df)

    return df
