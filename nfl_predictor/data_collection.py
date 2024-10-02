"""
This module, data_collection.py, is designed for the collection, processing, and storage of NFL game
data, crucial for making accurate game predictions. It offers a wide range of functionalities,
including scraping historical NFL game data, fetching the latest ELO ratings for teams, and
preparing datasets for analysis. It leverages pandas for data manipulation, sportsipy for scraping,
and custom utilities for efficient data I/O and logging, making it a core component for keeping NFL
data current and analytically valuable.

Main Functionalities:
- Scraping historical NFL game data to provide a rich historical dataset.
- Fetching current ELO ratings for teams to assess performance.
- Processing and cleaning data to ready it for analysis.
- Storing data in a structured format for easy access and analysis.

Dependencies:
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- sportsipy: For scraping sports data.
- custom utilities: For logging, data I/O, and NFL-specific tasks.

Usage:
This module is primarily used for updating the data foundation for NFL game predictions, handling
both historical data and current ELO ratings. It supports conditional data refreshes for efficient
processing.
"""

from datetime import date
from io import StringIO
from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sportsipy.nfl.boxscore import Boxscore

from nfl_predictor import constants
from nfl_predictor.utils.csv_utils import read_df_from_csv, read_write_data, write_df_to_csv
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.nfl_utils import (
    calculate_stats,
    compute_game_outcomes,
    create_stats_dfs_from_boxscore,
    determine_nfl_week_by_date,
    determine_weeks_to_scrape,
    fetch_game_boxscore,
    fetch_nfl_elo_ratings,
    fetch_week_boxscores,
    get_week_dates,
    init_team_stats_dfs,
    is_division_game,
    merge_and_finalize,
    merge_and_format_df,
    prepare_data_for_week,
)

SEASONS_TO_SCRAPE = [
    # 2003,
    # 2004,
    # 2005,
    # 2006,
    # 2007,
    # 2008,
    # 2009,
    # 2010,
    # 2011,
    # 2012,
    # 2013,
    # 2014,
    # 2015,
    # 2016,
    # 2017,
    # 2018,
    # 2019,
    # 2020,
    # 2021,
    2022,
    2023,
    2024,
]
REFRESH_SEASON_DATA = False
REFRESH_WEEKLY_DATA = False
REFRESH_SCHEDULE = False
REFRESH_AGGREGATE_DATA = False
REFRESH_SEASON_TEAM_RANKINGS = False
REFRESH_WEEKLY_TEAM_RANKINGS = False
REFRESH_ELO_SEASON = False


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

    log.info("Data collection and processing complete.")


def collect_data() -> pd.DataFrame:
    """
    Collects, processes, and cleans NFL game data from multiple seasons.

    Fetches the latest ELO ratings, processes game data for each season using these ratings, and
    aggregates the cleaned data into a single DataFrame.

    Returns:
        pd.DataFrame: Combined and cleaned data from all processed seasons.
    """
    # Step 1: Fetch and save the latest ELO ratings for NFL teams
    elo_df = read_write_data("nfl_elo", fetch_nfl_elo_ratings, force_refresh=True)

    # Step 2: Process game data for each season, using the ELO ratings for enhancement
    combined_data_list = process_seasons(elo_df)

    # Step 3: Clean and aggregate the processed data from all seasons
    # Remove columns that are entirely empty or contain only NA values
    cleaned_data_list = [df.dropna(axis=1, how="all") for df in combined_data_list]
    # Combine cleaned data into a single DataFrame
    combined_data_df = pd.concat(cleaned_data_list, ignore_index=True)
    # Ensure the 'date' column is of datetime type
    combined_data_df["date"] = pd.to_datetime(combined_data_df["date"], errors="coerce")
    # Sort by date column, newest to oldest
    combined_data_df = combined_data_df.sort_values(by=["date"], ascending=False).reset_index(
        drop=True
    )

    return combined_data_df


def process_seasons(elo_df: pd.DataFrame) -> list:
    """
    Processes and combines game data, team rankings, and ELO ratings for specified NFL seasons.

    This function orchestrates the collection, aggregation, and combination of game data, team
    rankings, and ELO ratings for each season specified in SEASONS_TO_SCRAPE. The result is a list
    of DataFrames, each representing the combined data for a single season, which is suitable for
    further analysis or modeling.

    Args:
        elo_df (pd.DataFrame): DataFrame containing ELO ratings for all teams across seasons.

    Returns:
        list of pd.DataFrame: A list where each DataFrame contains combined data for a season.
    """
    log.info(
        "Collecting game data for the following [%s] season(s): %s",
        len(SEASONS_TO_SCRAPE),
        SEASONS_TO_SCRAPE,
    )

    # Holds the combined data for each season
    combined_data_list = []
    # Determine the current season year
    current_season = date.today().year

    for season in SEASONS_TO_SCRAPE:
        # Determine which weeks of the season to scrape
        weeks = determine_weeks_to_scrape(season)

        # Set force_refresh to True if the season is the current season
        force_refresh = season == current_season

        # Collect game data for the season
        log.info("Collecting game data for the %s season...", season)
        season_games_df = read_write_data(
            f"{season}/{season}_season_games",
            scrape_season_data,
            season,
            weeks,
            force_refresh=force_refresh or REFRESH_SEASON_DATA,
        )

        # Collect the schedule for the season
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
            season,
            weeks,
            season_games_df,
            schedule_df,
            force_refresh=force_refresh or REFRESH_AGGREGATE_DATA,
        )

        # Fetch team rankings for the season
        log.info("Collecting team rankings for the %s season...", season)
        team_rankings_df = read_write_data(
            f"{season}/{season}_team_rankings",
            scrape_team_rankings_for_season,
            season,
            force_refresh=force_refresh or REFRESH_SEASON_TEAM_RANKINGS,
        )

        # Fetch ELO ratings for the season
        log.info("Collecting ELO ratings for the %s season...", season)
        season_elo_df = read_write_data(
            f"{season}/{season}_elo",
            get_season_elo,
            elo_df,
            season,
            force_refresh=force_refresh or REFRESH_ELO_SEASON,
        )

        # Combine all collected and processed data for the season into a single DataFrame
        log.info("Combining all data for the %s season...", season)
        combined_data_df = read_write_data(
            f"{season}/{season}_combined_data",
            combine_data,
            agg_games_df,
            team_rankings_df,
            season_elo_df,
            force_refresh=True,
        )

        # Append the combined data for the season to the list
        combined_data_list.append(combined_data_df)

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
    # Early return if no weeks are specified to avoid unnecessary processing
    if not weeks:
        log.info("No weeks specified for season %s. Returning empty DataFrame.", season)
        return pd.DataFrame()

    log.info("Scraping game data for %s season...", season)
    season_games_list = []

    # Determine the current season year and NFL week
    today = date.today()
    current_season = today.year
    current_week = determine_nfl_week_by_date(today)

    for week in weeks:
        # Set force_refresh to True if the season is the current season and the week is the current
        # week, plus/minus one week to account for potential delays in data availability
        force_refresh = season == current_season and abs(week - current_week) <= 1

        log.info("Collecting game data for Week %s...", week)
        # Scrape and save game data for the week
        week_games_df = read_write_data(
            f"{season}/{season}_week_{week:>02}_game_data",
            scrape_weekly_game_data,
            season,
            week,
            force_refresh=force_refresh or REFRESH_WEEKLY_DATA,
            # force_refresh=False,
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

    week_scores, error = fetch_week_boxscores(season, week)
    if error or week_scores is None:
        log.warning("Failed to fetch week scores for Week %s of the %s season!", week, season)
        return pd.DataFrame()  # Return empty DataFrame if fetching week scores fails

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
            game_stats, error = fetch_game_boxscore(game_info["boxscore"])
            if error or game_stats is None:
                log.warning(
                    "Failed to fetch game stats for %s in Week %s!", game_info["boxscore"], week
                )
                continue  # Skip to the next game if fetching game stats fails

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
    """
    Scrapes and returns the NFL game schedule for a given season as a DataFrame.

    Args:
        season (int): The NFL season year to scrape the schedule for.

    Returns:
        pd.DataFrame:   A DataFrame containing the schedule for the specified NFL season. Each row
                        represents a game with columns for away and home team names and
                        abbreviations, winning team name and abbreviation (if available), and the
                        week of the season.
    """
    weeks = determine_weeks_to_scrape(season)  # Determine weeks to scrape for the season
    all_games_data = []  # Initialize a list to store game data for all weeks

    log.info("Scraping %s schedule...", season)
    for week in weeks:
        log.info("Week %s...", week)
        week_scores, error = fetch_week_boxscores(season, week)
        if error or week_scores is None:
            continue  # Skip to the next week if fetching week scores fails

        game_key = f"{week}-{season}"
        # Check if there are games for the week to avoid KeyError
        if game_key not in week_scores.games:
            continue

        for game in week_scores.games[game_key]:
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

    # Create a combined DataFrame to calculate game numbers
    away_games = schedule_df[["away_abbr", "week"]].rename(columns={"away_abbr": "team_abbr"})
    home_games = schedule_df[["home_abbr", "week"]].rename(columns={"home_abbr": "team_abbr"})
    combined_games = pd.concat([away_games, home_games]).sort_values("week").reset_index(drop=True)

    # Calculate the game number for each team
    combined_games["game_number"] = combined_games.groupby("team_abbr").cumcount() + 1

    # Merge the game numbers back into the original schedule DataFrame
    schedule_df = schedule_df.merge(
        combined_games.rename(
            columns={"team_abbr": "away_abbr", "game_number": "away_game_number"}
        ),
        on=["away_abbr", "week"],
        how="left",
    )
    schedule_df = schedule_df.merge(
        combined_games.rename(
            columns={"team_abbr": "home_abbr", "game_number": "home_game_number"}
        ),
        on=["home_abbr", "week"],
        how="left",
    )

    return schedule_df


def aggregate_season_data(
    season: int, weeks: list, season_games_df: pd.DataFrame, schedule_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates game data for specified weeks, calculating team performance metrics.

    This function processes game and schedule data for an NFL season, calculating metrics like
    win percentages and conversion rates. It prepares data by computing differences in stats
    between opposing teams for the specified weeks.

    Args:
        season (int): NFL season year.
        weeks (list): Weeks within the season to aggregate data for.
        season_games_df (pd.DataFrame): Detailed game data for the season.
        schedule_df (pd.DataFrame): Schedule and outcomes for the season.

    Returns:
        pd.DataFrame: Aggregated game data with calculated stats for specified weeks.
    """
    # pylint: disable=too-many-locals
    # Initialize an empty list to hold aggregated data for each week
    agg_games_list = []

    # Loop through each week to aggregate data
    for week in weeks:
        log.info("Aggregating game data for Week %s...", week)
        # Prepare data for the current week
        week_games_df, results_df = prepare_data_for_week(week, schedule_df, season_games_df)
        if week == 1:
            # Load previous season's game data if not the first season in SEASONS_TO_SCRAPE
            if season != SEASONS_TO_SCRAPE[0]:
                prev_season = season - 1
                prev_season_file = (
                    f"{constants.DATA_PATH}/{prev_season}/{prev_season}_season_games.csv"
                )
                previous_weeks_df = read_df_from_csv(prev_season_file)
                previous_weeks_df["team_abbr"] = previous_weeks_df["team_abbr"].replace(
                    constants.PFR_TEAM_ABBR, constants.TEAM_ABBR
                )
            else:
                continue
        else:
            previous_weeks_df = season_games_df[season_games_df["week"] < week]

            # Check for teams with missing data
            current_teams = set(previous_weeks_df["team_abbr"])
            all_teams = set(schedule_df["away_abbr"]).union(set(schedule_df["home_abbr"]))
            missing_teams = all_teams - current_teams

            if missing_teams:
                # Load previous season's game data if not the first season in SEASONS_TO_SCRAPE
                if season != SEASONS_TO_SCRAPE[0]:
                    prev_season = season - 1
                    prev_season_file = (
                        f"{constants.DATA_PATH}/{prev_season}/{prev_season}_season_games.csv"
                    )
                    previous_season_df = read_df_from_csv(prev_season_file)
                    previous_season_df["team_abbr"] = previous_season_df["team_abbr"].replace(
                        constants.PFR_TEAM_ABBR, constants.TEAM_ABBR
                    )

                    # Append missing teams' data from the previous season to previous_weeks_df
                    missing_teams_df = previous_season_df[
                        previous_season_df["team_abbr"].isin(missing_teams)
                    ]
                    previous_weeks_df = pd.concat(
                        [previous_weeks_df, missing_teams_df], ignore_index=True
                    )

        # Merge game_number values from week_games_df into previous_weeks_df
        previous_weeks_df = previous_weeks_df.merge(
            week_games_df[["away_abbr", "away_game_number"]].rename(
                columns={"away_abbr": "team_abbr", "away_game_number": "game_number"}
            ),
            on="team_abbr",
            how="left",
        )
        previous_weeks_df = previous_weeks_df.merge(
            week_games_df[["home_abbr", "home_game_number"]].rename(
                columns={"home_abbr": "team_abbr", "home_game_number": "game_number"}
            ),
            on="team_abbr",
            how="left",
        )

        # Fill NaN values in game_number column
        previous_weeks_df["game_number"] = previous_weeks_df["game_number_x"].combine_first(
            previous_weeks_df["game_number_y"]
        )
        previous_weeks_df = previous_weeks_df.drop(columns=["game_number_x", "game_number_y"])

        agg_weekly_df = calculate_stats(previous_weeks_df)
        # Merge current week's data with results and add to the list
        merged_df = merge_and_finalize(week_games_df, agg_weekly_df, results_df)
        agg_games_list.append(merged_df)
    # Concatenate all weekly aggregated data into a single DataFrame
    final_agg_df = pd.concat(agg_games_list, ignore_index=True)
    return final_agg_df


def scrape_team_rankings_for_season(season: int) -> pd.DataFrame:
    """
    Scrapes and compiles team rankings for each week of a specified NFL season into a single
    DataFrame.

    Iterates through each week of the season, scraping team rankings and compiling them. It ensures
    that only non-empty weekly data frames are included in the final season-wide DataFrame.

    Args:
        season (int): The NFL season year for which to scrape team rankings.

    Returns:
        pd.DataFrame: A DataFrame containing the compiled team rankings for the entire season.
    """
    # Get list of dates for each week in the season
    week_dates = get_week_dates(season)

    # Determine the current season year and NFL week
    today = date.today()
    current_season = today.year
    current_week = determine_nfl_week_by_date(today)

    # Initialize list to hold weekly rankings DataFrames
    season_rankings = []

    for week_number, week_date in enumerate(week_dates, start=1):
        # Set force_refresh to True if the season is the current season and the week is the current
        # week, plus/minus one week to account for potential delays in data availability
        force_refresh = season == current_season and abs(week_number - current_week) <= 1
        log.info("Collecting team rankings for Week %s...", week_number)

        if season == current_season and week_number >= current_week + 1:
            # Copy the current week's rankings to future weeks
            data_path = f"{constants.DATA_PATH}/{season}"
            current_week_file = f"{data_path}/{season}_week_{current_week:>02}_team_rankings.csv"
            future_week_file = f"{data_path}/{season}_week_{week_number:>02}_team_rankings.csv"
            log.info("Copying rankings from Week %s to Week %s...", current_week, week_number)
            week_rankings_df = read_df_from_csv(current_week_file, check_exists=True)
            write_df_to_csv(week_rankings_df, future_week_file)
        else:
            # Attempt to scrape and retrieve team rankings for the week
            week_rankings_df = read_write_data(
                f"{season}/{season}_week_{week_number:>02}_team_rankings",
                scrape_team_rankings_for_week,
                week_number,
                week_date,
                force_refresh=force_refresh or REFRESH_WEEKLY_TEAM_RANKINGS,
            )

        # Append the week's rankings to the season list if data is present
        if not week_rankings_df.empty:
            week_rankings_df = week_rankings_df.fillna(0.0)
            season_rankings.append(week_rankings_df)

    # Concatenate all weekly rankings into a single DataFrame for the season
    season_rankings_df = pd.concat(season_rankings, ignore_index=True)
    return season_rankings_df


def scrape_team_rankings_for_week(week_number: int, week_date: date) -> pd.DataFrame:
    """
    Scrapes team rankings for a specific week and date, returning a DataFrame of the rankings.

    This function iterates over each team ranking type defined in constants, constructs URLs to
    scrape data, parses the HTML to extract ranking information, and compiles it into a DataFrame.
    It ensures data consistency by handling missing data and converting team names to abbreviations
    according to a predefined mapping. A delay is introduced between requests to avoid overloading
    the server.

    Args:
        week_number (int): The week number for which to scrape rankings.
        week_date (date): The date corresponding to the week of interest.

    Returns:
        pd.DataFrame:   A DataFrame containing the team rankings for the specified week, with teams
                        represented by their abbreviations and including rankings for each rating
                        type.
    """
    # Initialize DataFrame to hold rankings with a column for team abbreviations
    week_rankings_df = pd.DataFrame(columns=["abbr"])

    for rating, rating_name in constants.TEAM_RANKINGS_RATINGS.items():
        log.info("Scraping team rankings: %s for Week %s...", rating_name, week_number)
        # Construct the URL for scraping
        url = f"{constants.TEAM_RANKINGS_URL}/ranking/{rating}?date={week_date}"
        log.debug("Scraping data from %s...", url)
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")

        if table:
            # Parse the HTML table, convert ratings to numeric, and rename columns
            rating_df = pd.read_html(StringIO(str(table)))[0]
            rating_df["Rating"] = pd.to_numeric(rating_df["Rating"], errors="coerce").fillna(0.0)
            rating_df = rating_df.iloc[:, 1:3].rename(
                columns={"Team": "abbr", "Rating": rating_name}
            )
            # Strip win-loss-tie records from team abbreviations and map to standard abbreviations
            rating_df["abbr"] = rating_df["abbr"].str.replace(
                r"\s+\(\d+-\d+(-\d+)*\)$", "", regex=True
            )
            rating_df["abbr"] = rating_df["abbr"].apply(lambda x: constants.TEAMS_TO_ABBR.get(x, x))
            # Merge the current rating data with the week's DataFrame
            week_rankings_df = pd.merge(week_rankings_df, rating_df, on="abbr", how="outer")
        else:
            log.warning("No data found for %s on %s", rating_name, week_date)

        # Sleep to avoid overloading the server
        sleep(constants.TEAM_RANKINGS_SLEEP)

    for statistic, statistic_name in constants.TEAM_RANKINGS_STATS.items():
        log.info("Scraping team statistics: %s for Week %s...", statistic_name, week_number)
        # Construct the URL for scraping
        url = f"{constants.TEAM_RANKINGS_URL}/stat/{statistic}?date={week_date}"
        log.debug("Scraping data from %s...", url)
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")

        if table:
            # Parse the HTML table, convert stats to numeric, and rename columns
            stat_df = pd.read_html(StringIO(str(table)))[0]
            stat_df["Stat"] = pd.to_numeric(stat_df.iloc[:, 2], errors="coerce").fillna(0.0)
            stat_df = stat_df.iloc[:, [1, -1]].rename(
                columns={"Team": "abbr", stat_df.columns[-1]: statistic_name}
            )
            # Map team names to standard abbreviations
            stat_df["abbr"] = stat_df["abbr"].apply(lambda x: constants.TEAMS_TO_ABBR.get(x, x))
            # Merge the current rating data with the week's DataFrame
            week_rankings_df = pd.merge(week_rankings_df, stat_df, on="abbr", how="outer")
        else:
            log.warning("No data found for %s on %s", statistic_name, week_date)

        # Sleep to avoid overloading the server
        sleep(constants.TEAM_RANKINGS_SLEEP)

    # Add the week number to the DataFrame
    week_rankings_df["week"] = week_number
    return week_rankings_df


def get_season_elo(elo_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Filters and adjusts ELO ratings for a specific NFL season from a DataFrame containing ELO
    ratings across multiple seasons. This function optimizes data handling by dropping unnecessary
    columns early, using vectorized operations for date filtering, and minimizing data copying.
    It also handles neutral games by creating and appending swapped rows to account for all game
    scenarios.

    Args:
        elo_df (pd.DataFrame): DataFrame containing ELO ratings for all teams across seasons.
        season (int): The NFL season year for which to filter and adjust ELO ratings.

    Returns:
        pd.DataFrame:   Adjusted DataFrame with ELO ratings for the specified season, including
                        handling of neutral games and re-mapping of team names.
    """
    # Drop columns not needed for analysis to reduce memory usage
    elo_df = elo_df.drop(columns=constants.ELO_DROP_COLS)

    # Convert 'date' column to datetime, then to date for efficient filtering
    elo_df["date"] = pd.to_datetime(elo_df["date"]).dt.date
    # Get the start and end dates for the specified season
    week_dates = get_week_dates(season)
    # Extend the date range to include the end of the last week
    week_dates.append((week_dates[-1] + pd.DateOffset(weeks=1)).date())
    # Create a mask to filter rows within the season date range
    mask = (elo_df["date"] >= week_dates[0]) & (elo_df["date"] <= week_dates[-1])
    filtered_elo_df = elo_df.loc[mask].copy()
    # Add week number to the DataFrame for consistency with other data
    filtered_elo_df["week"] = filtered_elo_df["date"].apply(determine_nfl_week_by_date)

    # Map ELO team names to standard team names using a predefined mapping
    team_name_mapping = dict(zip(constants.ELO_TEAM_ABBR, constants.TEAM_ABBR))
    filtered_elo_df["team1"] = filtered_elo_df["team1"].map(team_name_mapping)
    filtered_elo_df["team2"] = filtered_elo_df["team2"].map(team_name_mapping)

    # Create swapped rows to account for neutral games
    duplicate_rows = filtered_elo_df.copy()
    # Swap columns for team names and related ELO ratings according to constants.ELO_SWAP_COLS
    swapped_rows = duplicate_rows.rename(columns=constants.ELO_SWAP_COLS)

    # Concatenate original and swapped rows, then clean up the DataFrame
    final_elo_df = (
        pd.concat([swapped_rows, filtered_elo_df], ignore_index=True)
        .sort_values(by=["date"])  # Sort by date for chronological order
        .reset_index(drop=True)  # Reset index for a clean DataFrame
    )

    return final_elo_df


def combine_data(
    agg_games_df: pd.DataFrame, team_rankings_df: pd.DataFrame, elo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines aggregated game data with team rankings and ELO ratings for enhanced analysis.

    This function merges aggregated game data with team rankings and ELO ratings to provide a
    comprehensive dataset that includes differences in ELO ratings, QB values, and other relevant
    metrics between the home and away teams. It also reorders columns to facilitate analysis,
    especially for predictive modeling of game outcomes.

    Args:
        agg_games_df (pd.DataFrame): Aggregated game data for the season.
        team_rankings_df (pd.DataFrame): Team rankings for the season.
        elo_df (pd.DataFrame): ELO ratings for the season.

    Returns:
        pd.DataFrame:   A DataFrame that combines all the provided data, with calculated differences
                        in ELO and QB values, standardized team abbreviations, and reordered columns
                        for easier analysis.
    """
    # Rename columns in team_rankings_df to differentiate between home and away teams
    team_rankings_df_away = team_rankings_df.rename(
        lambda x: f"away_{x}" if x != "week" else x, axis=1
    )
    team_rankings_df_home = team_rankings_df.rename(
        lambda x: f"home_{x}" if x != "week" else x, axis=1
    )

    # Merge aggregated game data with team rankings for away and then for home teams
    combined_df = pd.merge(
        agg_games_df, team_rankings_df_away, how="left", on=["away_abbr", "week"]
    ).merge(team_rankings_df_home, how="left", on=["home_abbr", "week"])

    # Merge the combined DataFrame with ELO ratings, focusing on team abbreviations
    combined_df = pd.merge(
        combined_df,
        elo_df,
        how="inner",
        left_on=["home_abbr", "away_abbr", "week"],
        right_on=["team1", "team2", "week"],
    ).drop(
        columns=["team1", "team2"]
    )  # Remove redundant columns after merge

    # Calculate differences in pre-game ELO and QB values between home and away teams
    combined_df["elo_dif"] = combined_df["elo2_pre"] - combined_df["elo1_pre"]
    combined_df["qb_dif"] = combined_df["qb2_value_pre"] - combined_df["qb1_value_pre"]
    combined_df["qb_elo_dif"] = combined_df["qbelo2_pre"] - combined_df["qbelo1_pre"]

    # Rename ELO columns to away/home format for consistency with other data
    combined_df = combined_df.rename(columns=constants.ELO_RENAME_COLS)

    # Add a column to indicate if the game is a divisional matchup
    combined_df["division"] = combined_df.apply(is_division_game, axis=1)

    # Reorder columns to improve readability, placing scores and result at the end
    first_columns = [
        "away_name",
        "away_abbr",
        "home_name",
        "home_abbr",
        "away_qb",
        "home_qb",
        "season",
        "week",
        "date",
        "away_game_number",
        "home_game_number",
        "neutral",
        "division",
    ]
    result_columns = ["away_score", "home_score", "result"]
    data_columns = sorted(
        [col for col in combined_df.columns if col not in first_columns + result_columns]
    )
    combined_df = combined_df[first_columns + data_columns + result_columns]

    # Rename legacy teams to modern team names using constants.MODERN_TEAM_NAMES
    combined_df["away_name"] = combined_df["away_name"].replace(constants.MODERN_TEAM_NAMES)
    combined_df["home_name"] = combined_df["home_name"].replace(constants.MODERN_TEAM_NAMES)

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
