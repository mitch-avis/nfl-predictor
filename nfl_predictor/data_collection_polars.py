"""
Data collection module for NFL game prediction using nflreadpy and Polars.

This module orchestrates the collection, processing, and storage of NFL game data
for use in prediction models. It uses nflreadpy as the primary data source and
Polars for high-performance data manipulation.

Key Features:
    - Collects historical game data from 2003 to present
    - Includes both regular season (weeks 1-18) and playoff games (WC, DIV, CON, SB)
    - Aggregates per-game team statistics into rolling averages
    - Merges ELO ratings and TeamRankings data for enhanced features
    - Computes statistical differentials between away and home teams
    - Produces separate ML-ready (with diffs) and analysis (without diffs) outputs

Output Files:
    - data/all_data_ml.csv: Full dataset with differential columns for ML
    - data/all_data.csv: Dataset without differential columns for analysis
    - data/completed_games_ml.csv: Only completed games with diffs
    - data/completed_games.csv: Only completed games without diffs
    - data/predict/week_XX_games_to_predict.csv: Upcoming games for prediction

Usage:
    Run directly to collect and process all data:
        python -m nfl_predictor.data_collection_polars

    Or import and call programmatically:
        from nfl_predictor.data_collection_polars import collect_all_data
        df = collect_all_data([2023, 2024])
"""

import os
from datetime import date, timedelta
from typing import Optional

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.logger import log

# Configuration: Seasons to process (inclusive range)
SEASONS_TO_PROCESS = list(range(constants.MIN_SEASON, 2026))

# Set to True to force refresh of cached data (not currently used)
FORCE_REFRESH = False


def main() -> None:
    """
    Main entry point for data collection using nflreadpy.

    Orchestrates the data collection, processing, and storage for NFL game predictions.
    """
    log.info("Starting data collection with nflreadpy...")

    # Determine current season and week
    today = date.today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1
    current_week = _determine_nfl_week(today)

    log.info("Current season: %s, week: %s", current_season, current_week)

    # Collect and process all data
    all_data_df = collect_all_data(SEASONS_TO_PROCESS)

    # Create version without diff columns (for non-ML local usage)
    no_diff_df = polars_utils.remove_diff_columns(all_data_df)

    # Save all data (ML version with diffs)
    save_dataframe(all_data_df, "all_data_ml")

    # Save all data (non-ML version without diffs)
    save_dataframe(no_diff_df, "all_data")

    # Filter and save completed games (both versions)
    completed_df = polars_utils.filter_completed_games(all_data_df)
    completed_no_diff_df = polars_utils.remove_diff_columns(completed_df)
    save_dataframe(completed_df, "completed_games_ml")
    save_dataframe(completed_no_diff_df, "completed_games")

    # Filter and save upcoming games for prediction (ML version only)
    upcoming_df = polars_utils.filter_upcoming_games(all_data_df, current_season, current_week)
    save_dataframe(upcoming_df, f"predict/week_{current_week:>02}_games_to_predict")

    log.info("Data collection complete.")


def collect_all_data(seasons: list[int]) -> pl.DataFrame:
    """
    Collect and combine all data for specified seasons.

    Args:
        seasons: List of season years to process

    Returns:
        Combined DataFrame with all game data and features
    """
    log.info("Collecting data for %d seasons: %s - %s", len(seasons), min(seasons), max(seasons))

    # Load full schedule with lines/odds directly from nflreadpy
    # Includes both regular season (REG) and playoff games (WC, DIV, CON, SB)
    schedule_df = polars_utils.load_schedule(seasons)
    log.info("Loaded schedule: %d total games (regular season + playoffs)", schedule_df.height)

    # Determine seasons to load for team stats
    # Include previous season for week 1 regression if not processing from the beginning
    stats_seasons = list(seasons)
    min_season = min(seasons)
    if min_season > constants.MIN_SEASON - 1:  # Need prior season for week 1 regression
        stats_seasons = [min_season - 1] + stats_seasons

    # Load team statistics (regular season only - used for building features)
    # Playoff games use cumulative stats from the regular season
    team_stats_df = polars_utils.load_team_stats(stats_seasons, regular_season_only=True)
    log.info("Loaded team stats: %d regular season team-game records", team_stats_df.height)

    # Add scoring data (points scored/allowed) to team stats from schedule
    # This enables computing points-related metrics like scoring margin
    # Need to also load schedule for previous season for scoring data
    if min_season > constants.MIN_SEASON - 1:
        prev_schedule = polars_utils.load_schedule([min_season - 1])
        full_schedule = pl.concat([prev_schedule, schedule_df], how="diagonal")
        team_stats_df = polars_utils.add_scoring_data_to_team_stats(team_stats_df, full_schedule)
    else:
        team_stats_df = polars_utils.add_scoring_data_to_team_stats(team_stats_df, schedule_df)

    # Add per-game opponent stats AFTER scoring data is added
    # This ensures opponent_points_scored, opponent_points_allowed, etc. are included
    team_stats_df = polars_utils.add_per_game_opponent_stats(team_stats_df)

    # Load ELO ratings
    elo_df = polars_utils.load_elo_ratings(seasons)
    if elo_df.height > 0:
        log.info("Loaded ELO ratings: %d game records", elo_df.height)
    else:
        log.warning("No ELO ratings loaded")

    # Load raw ELO data for QB lookups (needed for fill_future_qb_data)
    raw_elo_df = polars_utils.load_raw_elo_data()

    # Determine current season and week for TR scraping decisions
    current_season, current_week = polars_utils.get_current_nfl_week()
    log.info("Current season: %d, week: %d (for TR scraping)", current_season, current_week)

    # Process each season
    all_seasons_data = []

    for season in seasons:
        log.info("Processing season %d...", season)

        # Load TeamRankings for this season (will scrape if current/future week)
        tr_df = polars_utils.load_team_rankings(season, current_season, current_week)

        # Load previous season's TeamRankings for week 1 regression
        prev_tr_df = None
        if season > min(seasons):
            prev_tr_df = polars_utils.load_team_rankings(season - 1, current_season, current_week)

        season_data = process_season(season, schedule_df, team_stats_df, elo_df, tr_df, prev_tr_df)
        if season_data.height > 0:
            all_seasons_data.append(season_data)

    # Combine all seasons
    if all_seasons_data:
        combined_df = pl.concat(all_seasons_data, how="diagonal")
        # Sort by date, newest first
        if "date" in combined_df.columns:
            combined_df = combined_df.sort("date", descending=True)
        # Fill in QB data for future games using most recent starters
        combined_df = game_utils.fill_future_qb_data(combined_df, raw_elo_df)
        # Fill in lines for future games from SurvivorGrid
        combined_df = game_utils.fill_future_game_lines(combined_df)
        # Fill missing moneylines by calculating from spreads
        combined_df = game_utils.fill_missing_moneylines(combined_df)
        # Select final columns in correct order
        combined_df = polars_utils.select_final_columns(combined_df)
        return combined_df

    return pl.DataFrame()


def process_season(
    season: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    elo_df: Optional[pl.DataFrame] = None,
    tr_df: Optional[pl.DataFrame] = None,
    prev_tr_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Process a single season's data.

    Args:
        season: Season year to process
        schedule_df: Full schedule DataFrame
        team_stats_df: Full team stats DataFrame
        elo_df: ELO ratings DataFrame
        tr_df: TeamRankings DataFrame for this season
        prev_tr_df: TeamRankings DataFrame for previous season (for week 1)

    Returns:
        Processed DataFrame for the season
    """
    # Filter to this season
    season_schedule = schedule_df.filter(pl.col("season") == season)

    if season_schedule.height == 0:
        log.warning("No schedule data for season %d", season)
        return pl.DataFrame()

    # Get unique weeks in the schedule
    weeks = sorted(season_schedule.select("week").unique().to_series().to_list())

    # Process each week
    weekly_data = []
    for week in weeks:
        week_data = process_week(
            season, week, season_schedule, team_stats_df, elo_df, tr_df, prev_tr_df
        )
        if week_data.height > 0:
            weekly_data.append(week_data)

    if weekly_data:
        return pl.concat(weekly_data, how="diagonal")

    return pl.DataFrame()


def process_week(
    season: int,
    week: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    elo_df: Optional[pl.DataFrame] = None,
    tr_df: Optional[pl.DataFrame] = None,
    prev_tr_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Process a single week's games with aggregated stats from prior weeks.

    For teams that have no prior games in the current season (e.g., Week 1, or
    teams whose games were postponed like MIA/TB in 2017), we fall back to using
    the previous season's stats with regression to mean.

    Args:
        season: Season year
        week: Week number
        schedule_df: Season schedule DataFrame
        team_stats_df: Full team stats DataFrame (all seasons for week-1 lookback)
        elo_df: ELO ratings DataFrame
        tr_df: TeamRankings DataFrame for this season
        prev_tr_df: TeamRankings DataFrame for previous season (for week 1)

    Returns:
        DataFrame with week's games and features
    """
    # Skip the very first week of the first processed season (no prior data to aggregate)
    if season == min(SEASONS_TO_PROCESS) and week == 1:
        log.info("Skipping season %d week %d (no prior games to build features)", season, week)
        return pl.DataFrame()

    # Get this week's games
    week_games = schedule_df.filter((pl.col("season") == season) & (pl.col("week") == week))

    if week_games.height == 0:
        return pl.DataFrame()

    # Get season-specific stats
    season_stats = team_stats_df.filter(pl.col("season") == season)

    # Aggregate stats from prior weeks
    agg_stats = polars_utils.aggregate_team_stats_to_week(season_stats, week, season)

    # Get teams playing this week
    teams_this_week = set(
        week_games.select("away_abbr").to_series().to_list()
        + week_games.select("home_abbr").to_series().to_list()
    )

    # Check which teams have prior stats
    teams_with_stats = set()
    if agg_stats.height > 0 and "team_abbr" in agg_stats.columns:
        teams_with_stats = set(agg_stats.select("team_abbr").to_series().to_list())

    # Find teams that need fallback to previous season
    teams_needing_fallback = teams_this_week - teams_with_stats

    # If any teams need fallback (week 1, or teams with postponed first games like MIA/TB 2017)
    if teams_needing_fallback and season > min(SEASONS_TO_PROCESS):
        log.debug(
            "Teams needing previous season fallback for season %d week %d: %s",
            season,
            week,
            teams_needing_fallback,
        )

        prev_season_stats = team_stats_df.filter(pl.col("season") == season - 1)

        if prev_season_stats.height > 0:
            # Use full previous season for teams that need fallback
            # Use a high week number to get all regular season games
            prev_agg = polars_utils.aggregate_team_stats_to_week(prev_season_stats, 99, season - 1)

            if prev_agg.height > 0:
                # Calculate league means from previous season
                league_means = polars_utils.calculate_league_means(team_stats_df, season - 1)

                # Regress toward league mean
                prev_agg = polars_utils.regress_to_mean(
                    prev_agg,
                    league_means,
                    constants.WEEK1_REGRESSION_FACTOR,
                )

                # Filter to only teams that need fallback
                fallback_stats = prev_agg.filter(
                    pl.col("team_abbr").is_in(list(teams_needing_fallback))
                )

                if fallback_stats.height > 0:
                    if agg_stats.height > 0:
                        # Combine current season stats with fallback stats
                        agg_stats = pl.concat([agg_stats, fallback_stats])
                    else:
                        agg_stats = fallback_stats

    if agg_stats.height == 0:
        log.debug("No aggregated stats available for season %d week %d", season, week)
        return pl.DataFrame()

    # Merge schedule with aggregated team stats
    merged = polars_utils.merge_schedule_with_team_stats(week_games, agg_stats)

    # Merge with ELO ratings
    if elo_df is not None and elo_df.height > 0:
        season_elo = elo_df.filter(pl.col("season") == season)
        if "week" in season_elo.columns:
            week_elo = season_elo.filter(pl.col("week") == week)
            if week_elo.height > 0:
                # Exact week match - drop season/week from ELO before merge
                elo_cols = [c for c in week_elo.columns if c not in ["season", "week"]]
                week_elo = week_elo.select(elo_cols)
                merged = merged.join(
                    week_elo,
                    on=["away_abbr", "home_abbr"],
                    how="left",
                )
            else:
                # No ELO for this specific week - use most recent ELO per team
                # This handles future weeks and playoff games
                latest_elo = polars_utils.get_latest_elo_by_team(elo_df, season)
                if latest_elo.height > 0:
                    # Join for away team
                    away_elo = latest_elo.rename(
                        {
                            "team_abbr": "away_abbr",
                            "elo_pre": "away_elo_pre",
                            "qb_value_pre": "away_qb_value_pre",
                            "qb_elo_pre": "away_qb_elo_pre",
                        }
                    )
                    merged = merged.join(away_elo, on="away_abbr", how="left")

                    # Join for home team
                    home_elo = latest_elo.rename(
                        {
                            "team_abbr": "home_abbr",
                            "elo_pre": "home_elo_pre",
                            "qb_value_pre": "home_qb_value_pre",
                            "qb_elo_pre": "home_qb_elo_pre",
                        }
                    )
                    merged = merged.join(home_elo, on="home_abbr", how="left")

    # Merge with TeamRankings
    merged = _merge_team_rankings(merged, season, week, tr_df, prev_tr_df)

    # Calculate stat differentials
    stats_to_diff = polars_utils.get_stats_for_diff()
    merged = polars_utils.calculate_stat_differentials(merged, stats_to_diff)

    return merged


def _merge_team_rankings(
    merged: pl.DataFrame,
    season: int,
    week: int,
    tr_df: Optional[pl.DataFrame],
    prev_tr_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """
    Merge TeamRankings data into the game DataFrame.

    Handles three scenarios:
    1. Regular season (week 2+): Use current season's TR for that week
    2. Week 1: Use previous season's final TR values
    3. Playoffs (week > regular season weeks): Use TR data for that specific playoff week
       (which should be freshly scraped during the playoff week)

    Note: Before 2021, playoffs started in week 18 (17-week season).
          From 2021 onwards, playoffs start in week 19 (18-week season).

    Args:
        merged: Game DataFrame to merge TR data into
        season: Season year (used to determine regular season length)
        week: Week number
        tr_df: TeamRankings DataFrame for current season
        prev_tr_df: TeamRankings DataFrame for previous season

    Returns:
        DataFrame with TR columns merged
    """
    tr_to_use = None
    regular_season_weeks = constants.get_regular_season_weeks(season)

    # Determine which TR data to use based on week
    if week == 1 and prev_tr_df is not None and prev_tr_df.height > 0:
        # Week 1: Use previous season's final TR values
        tr_to_use = polars_utils.get_latest_team_rankings(prev_tr_df)
        log.debug("Using previous season TR for week 1")
    elif tr_df is not None and tr_df.height > 0 and "week" in tr_df.columns:
        # Try to get specific week's TR data (works for regular season AND playoffs)
        week_tr = tr_df.filter(pl.col("week") == week)
        if week_tr.height > 0:
            # Drop week column since we're joining on team only
            tr_to_use = week_tr.drop("week")
            if week > regular_season_weeks:
                log.debug("Using scraped TR for playoff week %d", week)
            else:
                log.debug("Using TR for regular season week %d", week)
        elif week > regular_season_weeks:
            # Fallback for playoffs: use most recent available TR data
            tr_to_use = polars_utils.get_latest_team_rankings(tr_df)
            log.debug("Using latest available TR for playoff week %d (fallback)", week)

    # Merge TR data if available
    if tr_to_use is not None and tr_to_use.height > 0:
        # Join for away team
        away_tr = tr_to_use.rename(
            {c: f"away_{c}" for c in tr_to_use.columns if c != "team_abbr"}
        ).rename({"team_abbr": "away_abbr"})
        merged = merged.join(away_tr, on="away_abbr", how="left")

        # Join for home team
        home_tr = tr_to_use.rename(
            {c: f"home_{c}" for c in tr_to_use.columns if c != "team_abbr"}
        ).rename({"team_abbr": "home_abbr"})
        merged = merged.join(home_tr, on="home_abbr", how="left")

    return merged


def save_dataframe(df: pl.DataFrame, name: str) -> None:
    """
    Save a Polars DataFrame to CSV.

    Args:
        df: DataFrame to save
        name: Base name for the file (without extension)
    """
    file_path = f"{constants.DATA_PATH}/{name}.csv"

    # Create directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to CSV
    df.write_csv(file_path)
    log.info("Saved %s (%d rows) to %s", name, df.height, file_path)


def load_dataframe(name: str) -> Optional[pl.DataFrame]:
    """
    Load a Polars DataFrame from CSV.

    Args:
        name: Base name for the file (without extension)

    Returns:
        DataFrame or None if file doesn't exist
    """
    file_path = f"{constants.DATA_PATH}/{name}.csv"

    if not os.path.isfile(file_path):
        log.warning("File not found: %s", file_path)
        return None

    return pl.read_csv(file_path)


def _determine_nfl_week(given_date: date) -> int:
    """
    Determine the current NFL week for a given date.

    Args:
        given_date: Date to check

    Returns:
        Week number (1-18)
    """

    def get_season_start(year: int) -> date:
        sept_first = date(year, 9, 1)
        first_monday = sept_first + timedelta((7 - sept_first.weekday()) % 7)
        return first_monday + timedelta(days=3)

    def adjust_to_tuesday(start_date: date) -> date:
        return start_date - timedelta(days=(start_date.weekday() - 1) % 7)

    season_start = adjust_to_tuesday(get_season_start(given_date.year))

    if given_date < season_start:
        if given_date.month in (1, 2):
            previous_season_start = adjust_to_tuesday(get_season_start(given_date.year - 1))
            week_number = ((given_date - previous_season_start).days // 7) + 1
            return max(1, min(week_number, 18))
        previous_season_start = adjust_to_tuesday(get_season_start(given_date.year - 1))
        if given_date >= previous_season_start:
            return 1
        return 0

    week_number = ((given_date - season_start).days // 7) + 1
    return max(1, min(week_number, 18))


if __name__ == "__main__":
    main()
