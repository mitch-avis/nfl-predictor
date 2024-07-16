"""
This module is responsible for the collection, processing, and storage of data relevant to NFL
game predictions. It includes functionalities for scraping NFL game data, fetching ELO ratings,
and preparing datasets for analysis. The module leverages pandas for data manipulation and
sportsipy for data scraping, along with custom utilities for data I/O and logging.

Main Functionalities:
- Scraping historical NFL game data across specified seasons.
- Fetching the latest ELO ratings for NFL teams.
- Processing and cleaning the collected data to prepare it for analysis.
- Storing the processed data in a structured format for easy access and analysis.

Usage:
This module is typically used to refresh the underlying data used for NFL game predictions,
including both historical game data and ELO ratings. It supports conditional data refreshes based
on specific flags to avoid unnecessary processing.

Dependencies:
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- sportsipy: For scraping sports data from the web.
- custom utilities: For logging, data I/O, and additional NFL-specific operations.
"""

from datetime import date, datetime
from time import sleep

import numpy as np
import pandas as pd
from sportsipy.nfl.boxscore import Boxscore, Boxscores

from nfl_predictor import constants
from nfl_predictor.utils.csv_utils import read_write_data
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.nfl_utils import (
    calculate_stats,
    compute_game_outcomes,
    create_stats_dfs_from_boxscore,
    determine_nfl_week_by_date,
    determine_weeks_to_scrape,
    fetch_nfl_elo_ratings,
    init_team_stats_dfs,
    merge_and_finalize,
    merge_and_format_df,
    prepare_data_for_week,
)

SEASONS_TO_SCRAPE = [
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]

REFRESH_ELO = False
REFRESH_SEASON_DATA = True
REFRESH_GAME_DATA = False
REFRESH_SCHEDULE = False

SEASON_END_MONTH = constants.SEASON_END_MONTH
ELO_DATA_URL = constants.ELO_DATA_URL
BASE_COLUMNS = constants.BASE_COLUMNS
BOXSCORE_STATS = constants.BOXSCORE_STATS
AGG_STATS = constants.AGG_STATS
AGG_DROP_COLS = constants.AGG_DROP_COLS
ELO_DROP_COLS = constants.ELO_DROP_COLS
ELO_TEAMS = constants.ELO_TEAMS
STD_TEAMS = constants.STD_TEAMS
TEAMS = constants.TEAMS


def main() -> None:
    """
    Orchestrates the data collection, processing, and storage for NFL game predictions.

    This function performs the following steps:
    1. Consolidates data from multiple seasons.
    2. Filters and stores data for completed games.
    3. Identifies and stores upcoming games for predictions.
    """
    # Step 1: Collect and consolidate data from multiple seasons
    combined_data_df = read_write_data("all_data", collect_data, force_refresh=True)

    # Step 2: Filter and store data for completed games
    read_write_data("completed_games", parse_completed_games, combined_data_df, force_refresh=True)

    # Determine today's date and the current NFL week
    today = date.today()
    current_week = determine_nfl_week_by_date(today)

    # Step 3: Identify and store upcoming games for the current week for predictions
    read_write_data(
        f"predict/week_{current_week:>02}_games_to_predict",
        parse_upcoming_games_to_predict,
        combined_data_df,
        force_refresh=True,
    )


def collect_data() -> pd.DataFrame:
    """
    Collects, processes, and cleans NFL game data from multiple seasons.

    Fetches the latest ELO ratings, processes game data for each season using these ratings, and
    aggregates the cleaned data into a single DataFrame.

    Returns:
        pd.DataFrame: Combined and cleaned data from all processed seasons.
    """
    # Step 1: Fetch and save the latest ELO ratings for NFL teams
    elo_df = read_write_data("nfl_elo", fetch_nfl_elo_ratings, force_refresh=REFRESH_ELO)

    # Step 2: Process game data for each season, using the ELO ratings for enhancement
    combined_data_list = process_seasons(elo_df)

    # Step 3: Clean and aggregate the processed data from all seasons
    # Remove columns that are entirely empty or contain only NA values
    cleaned_data_list = [df.dropna(axis=1, how="all") for df in combined_data_list]
    # Combine cleaned data into a single DataFrame
    combined_data_df = pd.concat(cleaned_data_list, ignore_index=True)

    return combined_data_df


def process_seasons(elo_df: pd.DataFrame) -> list:
    """
    Processes game data for each specified NFL season.

    Collects game data, aggregates it, fetches ELO ratings for the season, and combines this
    information.

    Args:
        elo_df (pd.DataFrame): DataFrame containing ELO ratings.

    Returns:
        list: List of DataFrames, each representing combined game data and ELO ratings for a season.
    """
    log.info(
        "Collecting game data for the following [%s] season(s): %s",
        len(SEASONS_TO_SCRAPE),
        SEASONS_TO_SCRAPE,
    )

    combined_data_list = []  # Holds combined data for each season

    for season in SEASONS_TO_SCRAPE:
        weeks = determine_weeks_to_scrape(season)  # Determine weeks to scrape for the season

        # Collect game data for the season
        log.info("Collecting game data for the %s season...", season)
        season_games_df = read_write_data(
            f"{season}/{season}_season_games",
            scrape_season_data,
            season,
            weeks,
            force_refresh=REFRESH_SEASON_DATA,
        )

        log.info("Collecting schedule for %s...", season)
        schedule_df = read_write_data(
            f"{season}/{season}_schedule",
            get_schedule,
            season,
            force_refresh=REFRESH_SCHEDULE,
        )

        # Aggregate game data for the season
        log.info("Aggregating data for the %s season...", season)
        agg_games_df = read_write_data(
            f"{season}/{season}_agg_games",
            aggregate_season_data,
            weeks,
            season_games_df,
            schedule_df,
            force_refresh=True,
        )

        # Fetch ELO ratings for the season
        log.info("Collecting ELO ratings for the %s season...", season)
        season_elo_df = read_write_data(
            f"{season}/{season}_elo",
            get_season_elo,
            elo_df,
            season,
            force_refresh=True,
        )

        # Combine game data and ELO ratings
        log.info("Combining all data for the %s season...", season)
        combined_data_df = read_write_data(
            f"{season}/{season}_combined_data",
            combine_data,
            agg_games_df,
            season_elo_df,
            force_refresh=True,
        )

        combined_data_list.append(combined_data_df)  # Append combined data for the season

    return combined_data_list


def scrape_season_data(season: int, weeks: list) -> pd.DataFrame:
    """
    Scrapes game data for a given NFL season and specified weeks.

    Args:
        season (int): NFL season year.
        weeks (list): Weeks within the season to scrape data for.

    Returns:
        pd.DataFrame: Aggregated game data for the specified season and weeks.
    """
    log.info("Scraping game data for %s season...", season)
    season_games_list = []

    # Early return if no weeks are specified to avoid unnecessary processing
    if not weeks:
        log.info("No weeks specified for season %s. Returning empty DataFrame.", season)
        return pd.DataFrame()

    for week in weeks:
        log.info("Collecting game data for Week %s...", week)
        # Scrape and save game data for the week
        week_games_df = read_write_data(
            f"{season}/{season}_week_{week:>02}_game_data",
            scrape_weekly_game_data,
            season,
            week,
            force_refresh=REFRESH_GAME_DATA,
        )
        # Append the week's game data to the season list if it's not empty
        if not week_games_df.empty:
            season_games_list.append(week_games_df)

    # Concatenate all weekly game data into a single DataFrame, if any data was collected
    if season_games_list:
        return pd.concat(season_games_list, ignore_index=True)
    # Return an empty DataFrame if no game data was collected
    log.info("No game data collected for season %s. Returning empty DataFrame.", season)
    return pd.DataFrame()


def scrape_weekly_game_data(season: int, week: int) -> pd.DataFrame:
    # pylint: disable=broad-except
    """
    Scrapes game data for a specific week and season.

    Handles retries for network or data access issues and attempts to fetch box scores for all
    games in the specified week and season up to three times in case of exceptions.

    Args:
        season (int): NFL season year to scrape data for.
        week (int): Week within the specified season.

    Returns:
        pd.DataFrame: DataFrame containing detailed game data for the week and season.
    """
    log.info("Scraping game data for Week %s of the %s season...", week, season)
    # Initialize a list to store game data DataFrames
    games_data = []

    # Attempt to retrieve the box scores for all games in the specified week and season, with
    # retries
    for attempt in range(3):
        try:
            week_scores = Boxscores(week, season)
            break
        except Exception as e:
            log.warning("Failed to fetch week scores on attempt %s: %s", attempt + 1, e)
            if attempt < 2:  # Retry logic
                sleep(2**attempt)  # Exponential backoff
            else:
                return pd.DataFrame()  # Return empty DataFrame after 3 failed attempts

    # Check if there are games for the week-season combination to avoid unnecessary processing
    game_key = f"{week}-{season}"
    if game_key not in week_scores.games:
        log.info("No games found for Week %s of the %s season.", week, season)
        return pd.DataFrame()

    # Iterate through each game in the week, scraping data
    for game_info in week_scores.games[game_key]:
        if game_info["home_score"] is None and game_info["away_score"] is None:
            log.info("Game %s has not finished yet.", game_info["boxscore"])
            game_stats = None  # Set game_stats to None for unfinished games
        else:
            # Retry logic for fetching individual game stats
            for attempt in range(3):
                try:
                    game_stats = Boxscore(game_info["boxscore"])
                    break
                except Exception as e:
                    log.warning(
                        "Failed to fetch game stats for %s on attempt %s: %s",
                        game_info["boxscore"],
                        attempt + 1,
                        e,
                    )
                    game_stats = None
                    if attempt < 2:
                        sleep(2**attempt)  # Exponential backoff
                    else:
                        continue  # Skip this game after 3 failed attempts

        log.info("Scraping game data for %s...", game_info["boxscore"])
        # Call parse_game_data even if the game has not finished
        away_team_df, home_team_df = extract_team_statistics_from_game(
            pd.DataFrame(game_info, index=[0]), game_stats
        )

        # Append team data with week and season information
        for team_df in (away_team_df, home_team_df):
            team_df["week"] = week
            team_df["season"] = season
            games_data.append(team_df)

    # Concatenate all game data into a single DataFrame, if any was collected
    return pd.concat(games_data, ignore_index=True) if games_data else pd.DataFrame()


def extract_team_statistics_from_game(
    game_df: pd.DataFrame, boxscore: Boxscore
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses game data and boxscore information, returning DataFrames for away and home team stats.

    Args:
        game_df (pd.DataFrame): DataFrame containing basic game information.
        boxscore (Boxscore): Boxscore object with detailed game statistics.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames for away and home team statistics.
    """
    # Prepare team data frames
    away_team_df, home_team_df = init_team_stats_dfs(game_df)

    # Calculate win, loss, and tie outcomes
    away_team_df, home_team_df = compute_game_outcomes(away_team_df, home_team_df)

    # Handle detailed game statistics
    away_stats_df, home_stats_df = create_stats_dfs_from_boxscore(boxscore)

    # Merge team stats with opponent stats and format DataFrames
    away_team_df = merge_and_format_df(away_team_df, away_stats_df, home_stats_df)
    home_team_df = merge_and_format_df(home_team_df, home_stats_df, away_stats_df)

    return away_team_df, home_team_df


def get_schedule(season: int) -> pd.DataFrame:
    # pylint: disable=broad-except
    """
    Scrapes and returns the NFL game schedule for a given season as a DataFrame.

    Args:
        season (int): The NFL season year to scrape the schedule for.

    Returns:
        pd.DataFrame: A DataFrame containing the schedule for the specified NFL season. Each row
                      represents a game with columns for away and home team names and abbreviations,
                      winning team name and abbreviation (if available), and the week of the season.
    """
    weeks = determine_weeks_to_scrape(season)  # Determine weeks to scrape for the season
    all_games_data = []  # Initialize a list to store game data for all weeks

    log.info("Scraping %s schedule...", season)
    for week in weeks:
        log.info("Week %s...", week)
        week_scores = None
        for attempt in range(3):
            try:
                week_scores = Boxscores(week, season)
                if week_scores.games:  # Check if Boxscores returned data
                    break
            except Exception as e:
                log.warning(
                    "Attempt %d to fetch week %s of season %s failed: %s",
                    attempt + 1,
                    week,
                    season,
                    e,
                )
                sleep(2**attempt)  # Exponential backoff
        if (
            not week_scores or not week_scores.games
        ):  # If still no data after retries, skip this week
            log.warning(
                "Failed to fetch data for week %s of season %s after 3 attempts.", week, season
            )
            continue

        date_string = f"{week}-{season}"
        # Check if there are games for the week to avoid KeyError
        if date_string not in week_scores.games:
            continue

        for game in week_scores.games[date_string]:
            game_data = {
                "away_name": game["away_name"],
                "away_abbr": game["away_abbr"],
                "home_name": game["home_name"],
                "home_abbr": game["home_abbr"],
                "season": season,
                "week": week,
            }
            all_games_data.append(game_data)

    # Convert the list of dictionaries to a DataFrame in one operation
    schedule_df = pd.DataFrame(all_games_data)
    return schedule_df


def aggregate_season_data(
    weeks: list, season_games_df: pd.DataFrame, schedule_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates game data for specified weeks, calculating team performance metrics.

    This function processes game and schedule data for an NFL season, calculating metrics like
    win percentages and conversion rates. It prepares data by computing differences in stats
    between opposing teams for the specified weeks.

    Args:
        weeks (list): Weeks within the season to aggregate data for.
        season_games_df (pd.DataFrame): Detailed game data for the season.
        schedule_df (pd.DataFrame): Schedule and outcomes for the season.

    Returns:
        pd.DataFrame: Aggregated game data with calculated stats for specified weeks.
    """
    # Initialize an empty list to hold aggregated data for each week
    agg_games_list = []

    # Loop through each week to aggregate data
    for week in weeks:
        log.info("Aggregating game data for Week %s...", week)
        # Prepare data for the current week
        week_games_df, results_df = prepare_data_for_week(week, schedule_df, season_games_df)
        # Calculate stats for the current week, considering data from previous weeks if applicable
        if week == 1:
            # For Week 1, initialize agg_weekly_df with no prior data
            agg_weekly_df = season_games_df.head(0)
            agg_weekly_df = calculate_stats(agg_weekly_df, week)
        else:
            # For subsequent weeks, calculate stats based on previous weeks' data
            previous_weeks_df = season_games_df[season_games_df["week"] < week]
            agg_weekly_df = calculate_stats(previous_weeks_df, week)
        # Merge current week's data with results and add to the list
        merged_df = merge_and_finalize(week_games_df, agg_weekly_df, results_df)
        agg_games_list.append(merged_df)

    # Concatenate all weekly aggregated data into a single DataFrame
    final_agg_df = pd.concat(agg_games_list, ignore_index=True)
    return final_agg_df


def get_season_elo(elo_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Filters and adjusts ELO ratings for a specific NFL season.

    Args:
        elo_df (pd.DataFrame): DataFrame containing ELO ratings for all teams and seasons.
        season (int): The NFL season year to filter the ELO data for.

    Returns:
        pd.DataFrame: Adjusted ELO ratings for the specified season.
    """
    # Copy the DataFrame to avoid modifying the original
    yearly_elo_df = elo_df.copy()
    # Drop columns not needed for analysis
    yearly_elo_df = yearly_elo_df.drop(columns=constants.ELO_DROP_COLS)
    # Convert date column to datetime format
    yearly_elo_df["date"] = pd.to_datetime(yearly_elo_df["date"])
    # Define the start and end dates for the NFL season
    start_date = datetime(season, 9, 1)
    end_date = datetime(season + 1, 3, 1)
    # Create a mask to filter rows within the NFL season dates
    mask = (yearly_elo_df["date"] >= start_date) & (yearly_elo_df["date"] <= end_date)
    # Apply the mask to filter the DataFrame
    yearly_elo_df = yearly_elo_df.loc[mask]
    # Create a mapping from ELO team names to standard team names
    team_name_mapping = dict(zip(constants.ELO_TEAMS, constants.STD_TEAMS))
    # Replace team names in the dataframe using the mapping
    yearly_elo_df["team1"] = yearly_elo_df["team1"].map(team_name_mapping)
    yearly_elo_df["team2"] = yearly_elo_df["team2"].map(team_name_mapping)
    return yearly_elo_df


def combine_data(agg_games_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhances aggregated game data with ELO ratings, focusing on differences in ELO and QB values.

    This function enriches NFL game data by merging it with ELO ratings, calculating the differences
    in ELO ratings and QB values between the home and away teams. It standardizes team abbreviations
    and reorders columns to facilitate future data cleaning and analysis, particularly for machine
    learning models aimed at predicting game outcomes and scores.

    Args:
        agg_games_df (pd.DataFrame): Aggregated game data for the season.
        elo_df (pd.DataFrame): ELO ratings for the season.

    Returns:
        pd.DataFrame: Combined DataFrame with calculated differences, standardized abbreviations,
                      and columns reordered for analysis, including final scores and result.
    """
    # Merge game data with ELO ratings on team abbreviations; drop redundant 'team' columns
    combined_df = pd.merge(
        agg_games_df,
        elo_df,
        how="inner",
        left_on=["home_abbr", "away_abbr"],
        right_on=["team1", "team2"],
    ).drop(columns=["team1", "team2"])

    # Calculate differences in pre-game ELO and QB values between home and away teams
    combined_df["elo_dif"] = combined_df["elo2_pre"] - combined_df["elo1_pre"]
    combined_df["qb_dif"] = combined_df["qb2_value_pre"] - combined_df["qb1_value_pre"]
    combined_df["qb_elo_dif"] = combined_df["qbelo2_pre"] - combined_df["qbelo1_pre"]

    # Drop columns not needed after calculating differences
    combined_df = combined_df.drop(
        columns=[
            "date",
            "elo1_pre",
            "elo2_pre",
            "qb1_value_pre",
            "qb2_value_pre",
            "qbelo1_pre",
            "qbelo2_pre",
        ]
    )

    # Standardize team abbreviations for consistency
    combined_df["home_abbr"] = combined_df["home_abbr"].replace(
        constants.STD_TEAMS, constants.TEAMS
    )
    combined_df["away_abbr"] = combined_df["away_abbr"].replace(
        constants.STD_TEAMS, constants.TEAMS
    )

    # Reorder columns for readability, ensuring 'away_score', 'home_score', and 'result' are last
    result_cols = [
        col for col in combined_df.columns if col not in ["away_score", "home_score", "result"]
    ]
    combined_df = combined_df[result_cols + ["away_score", "home_score", "result"]]

    return combined_df


def parse_completed_games(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for completed games based on the presence of a result.

    Args:
        combined_data_df (pd.DataFrame): DataFrame containing combined game and ELO data.

    Returns:
        pd.DataFrame: DataFrame filtered for completed games.
    """
    # Filter for completed games based on the presence of a result
    completed_games_df = combined_data_df[combined_data_df["result"].notna()]

    return completed_games_df


def parse_upcoming_games_to_predict(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for upcoming games to predict based on the current date and season.

    Args:
        combined_data_df (pd.DataFrame): DataFrame containing combined game and ELO data.

    Returns:
        pd.DataFrame: DataFrame filtered for upcoming games lacking results.
    """
    # Determine the current season based on today's date and SEASON_END_MONTH
    today = date.today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1

    # Determine the current week for the current season
    current_week = determine_nfl_week_by_date(today)

    # Filter for games to predict in the current week of the current season
    games_to_predict_df = combined_data_df[
        (combined_data_df["season"] == current_season)
        & (combined_data_df["week"] == current_week)
        & combined_data_df["result"].isna()
    ]

    return games_to_predict_df


if __name__ == "__main__":
    main()
