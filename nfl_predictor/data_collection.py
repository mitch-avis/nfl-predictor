"""
This module orchestrates the collection, processing, and storage of NFL game data across multiple
seasons.  It includes functionality to:

1. Collect and aggregate game data and ELO ratings for NFL teams across multiple seasons.
2. Extract and save completed games data.
3. Identify and save upcoming games for the current week that require predictions.

The module ensures data freshness by incorporating a `force_refresh` mechanism, which can be used to
bypass cached data and fetch the latest information available.

Functions within this module handle various steps of the data collection and processing workflow,
including:
- Fetching the latest ELO ratings for NFL teams.
- Scraping game data for specified seasons and weeks.
- Aggregating collected data into a comprehensive dataset.
- Parsing completed games and games to predict for the current week.

This module is designed to support predictive modeling efforts by providing a rich dataset that
includes historical game outcomes and team performance metrics.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sportsipy.nfl.boxscore import Boxscore, Boxscores

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.utils import determine_nfl_week_by_date, read_write_data

STARTING_SEASON = 2000
NUM_SEASONS = 25

REFRESH_ELO = False
REFRESH_GAME_DATA = True
REFRESH_SCHEDULE = True

SEASON_END_MONTH = 2  # NFL season typically ends in February
WEEKS_BEFORE_2021 = 17  # Weeks in seasons before 2021
WEEKS_FROM_2021_ONWARDS = 18  # Weeks from 2021 onwards

ELO_DATA_URL = constants.ELO_DATA_URL
AWAY_STATS = constants.AWAY_STATS
AWAY_STATS_RENAME = constants.AWAY_STATS_RENAME
HOME_STATS = constants.HOME_STATS
HOME_STATS_RENAME = constants.HOME_STATS_RENAME
AGG_RENAME_AWAY = constants.AGG_RENAME_AWAY
AGG_RENAME_HOME = constants.AGG_RENAME_HOME
AGG_MERGE_ON = constants.AGG_MERGE_ON
AGG_DROP_COLS = constants.AGG_DROP_COLS
ELO_DROP_COLS = constants.ELO_DROP_COLS
ELO_TEAMS = constants.ELO_TEAMS
STD_TEAMS = constants.STD_TEAMS
TEAMS = constants.TEAMS


def main() -> None:
    """
    Orchestrates the data collection and processing workflow for NFL game data.

    This function executes a series of steps to ensure the latest NFL game data is collected,
    processed, and saved for further analysis.  It specifically:
    1.  Collects and processes data across multiple seasons, saving the consolidated data for all
        games.
    2.  Filters and saves data for games that have been completed.
    3.  Identifies and saves data for upcoming games in the current week that require predictions.

    Each step involves reading from or writing to a data store, with an option to force a refresh of
    the data to ensure the latest information is used.

    Note:
    The `force_refresh` parameter, when set to True, indicates that the data should be fetched anew
    rather than using any cached versions.  This ensures that the most current data is always used.
    """
    # Collect and save all data for multiple seasons
    combined_data_df = read_write_data("all_data", collect_data, force_refresh=True)

    # Extract and save completed games
    read_write_data("completed_games", parse_completed_games, combined_data_df, force_refresh=True)

    today = date.today()
    current_week = determine_nfl_week_by_date(today)
    # Extract and save upcoming games for prediction
    read_write_data(
        f"predict/week_{current_week:>02}_games_to_predict",
        parse_upcoming_games_to_predict,
        combined_data_df,
        force_refresh=True,
    )


def collect_data() -> pd.DataFrame:
    """
    Collects and aggregates NFL game data across multiple seasons into a single DataFrame.

    This function orchestrates the data collection process by:
    1. Fetching the latest ELO ratings for NFL teams.
    2. Processing game data for each season using the ELO ratings.
    3. Aggregating the processed data from all seasons into a single DataFrame.

    The ELO ratings are used to enhance the game data with team performance metrics.

    Returns:
        pd.DataFrame:   A DataFrame containing aggregated game data across multiple seasons.
    """
    # Get and save latest ELO spreadsheet
    elo_df = read_write_data("nfl_elo", get_elo, force_refresh=REFRESH_ELO)
    # Get and save all season data
    combined_data_list = process_seasons(elo_df)
    # Combine all season data into one DataFrame
    combined_data_df = pd.concat(combined_data_list, ignore_index=True)
    return combined_data_df


def get_elo() -> pd.DataFrame:
    """
    Fetches the latest ELO ratings from a specified URL and returns them as a DataFrame.

    This function directly accesses a CSV file containing ELO ratings for NFL teams from a
    predefined URL (`ELO_DATA_URL`).

    It uses pandas to read the CSV file and load it into a DataFrame, which is then returned for
    further processing.

    Returns:
        pd.DataFrame:   A DataFrame containing the ELO ratings for NFL teams.
    """
    # Read ELO ratings from the specified URL into a DataFrame
    elo_df = pd.read_csv(ELO_DATA_URL)
    return elo_df


def process_seasons(elo_df: pd.DataFrame) -> list:
    """
    Processes and aggregates NFL game data and ELO ratings for multiple seasons.

    This function iterates through each season within a specified date range, performing several
    steps:
    1. Determines the weeks to scrape for each season.
    2. Collects game data for each determined week.
    3. Aggregates the collected game data.
    4. Fetches ELO ratings for the season.
    5. Combines the game data and ELO ratings into a single DataFrame.
    6. Appends the combined data for each season to a list.

    Args:
        elo_df (pd.DataFrame):  A DataFrame containing ELO ratings for NFL teams.

    Returns:
        list:   A list of DataFrames, each representing combined game data and ELO ratings for a
                season.
    """
    # Calculate the start and end dates for the seasons to process
    start_date, end_date = get_season_dates()
    log.info(
        "Collecting game data for the following [%s] season(s): %s",
        end_date.year - start_date.year,
        list(range(start_date.year, end_date.year)),
    )

    # Initialize list to hold combined data for each season
    combined_data_list = []

    # Iterate through each season in the specified range
    for season in range(start_date.year, end_date.year):
        # Determine weeks to scrape for the season
        weeks = determine_weeks_to_scrape(season)

        # Collect and save game data for the season
        log.info("Collecting game data for the %s season...", season)
        season_games_df = read_write_data(
            f"{season}/season_games",
            scrape_season_data,
            season,
            weeks,
            force_refresh=REFRESH_GAME_DATA,
        )

        # Aggregate and save game data for the season
        log.info("Aggregating data for the %s season...", season)
        agg_games_df = read_write_data(
            f"{season}/agg_games",
            aggregate_season_data,
            season,
            weeks,
            season_games_df,
            force_refresh=True,
        )

        # Collect and save ELO ratings for the season
        log.info("Collecting ELO ratings for the %s season...", season)
        season_elo_df = read_write_data(
            f"{season}/elo",
            get_season_elo,
            elo_df,
            season,
            force_refresh=True,
        )

        # Combine game data and ELO ratings, then save
        log.info("Combining all data for the %s season...", season)
        combined_data_df = read_write_data(
            f"{season}/combined_data",
            combine_data,
            season,
            agg_games_df,
            season_elo_df,
            force_refresh=True,
        )

        # Append combined data for the season to the list
        combined_data_list.append(combined_data_df)

    return combined_data_list


def get_season_dates() -> tuple[date, date]:
    """
    Calculates the start and end dates for NFL season data collection.

    This function determines the start and end dates for collecting NFL season data based on
    predefined starting season and number of seasons to collect.  The NFL season typically starts
    in September and ends in February of the following year, hence the addition of six months to the
    end date.

    Returns:
        tuple[date, date]:  A tuple containing the calculated start and end dates for the data
                            collection period.
    """
    # Set the start date as the beginning of September of the starting season
    start_date = date(STARTING_SEASON, 9, 1)
    # Calculate the end date by adding the number of seasons (minus one) and six months to cover the
    # season end
    end_date = start_date + relativedelta(years=NUM_SEASONS - 1) + relativedelta(months=6)
    return start_date, end_date


def determine_weeks_to_scrape(season: int) -> list:
    """
    Determines the weeks of the NFL season to scrape based on the given season year.

    The NFL season structure changed in 2021, increasing from 17 to 18 weeks.  This function
    accounts for this change by returning the appropriate number of weeks for the given season.
    For the current season, it returns weeks up to the current week to ensure data is not
    requested for games that have not yet occurred.

    Args:
        season (int):   The NFL season year for which to determine the weeks to scrape.

    Returns:
        list:   A list of integers representing the weeks of the season to scrape.
    """
    today = date.today()
    # Determine the current NFL season based on today's date
    current_season = today.year if today.month > SEASON_END_MONTH else today.year - 1

    if season < current_season:
        weeks_to_scrape = (
            list(range(1, WEEKS_BEFORE_2021 + 1))
            if season < 2021
            else list(range(1, WEEKS_FROM_2021_ONWARDS + 1))
        )
    elif season == current_season:
        # Calculate the current week only for the current season
        current_week = determine_nfl_week_by_date(today)
        weeks_to_scrape = list(range(1, current_week + 1))
    else:
        # Future seasons should not have data to scrape yet
        weeks_to_scrape = []
    return weeks_to_scrape


def scrape_season_data(season: int, weeks: list) -> pd.DataFrame:
    """
    Scrapes and aggregates game data for a given NFL season across specified weeks.

    This function iterates over a list of weeks for the specified NFL season, scraping game data for
    each week.  If it's the beginning of a new season with only one week of data available, it
    returns an empty DataFrame immediately to avoid unnecessary processing.  For each week with
    available data, it scrapes the game data, saves it, and then aggregates all weekly data into a
    single DataFrame.

    Args:
        season (int):   The NFL season year to scrape data for.
        weeks (list):   A list of integers representing the weeks of the season to scrape data for.

    Returns:
        pd.DataFrame:   A DataFrame containing aggregated game data for the specified weeks of the
                        season.
    """
    # Check for a new season with only one week of data and return an empty DataFrame
    if len(weeks) == 1:
        return pd.DataFrame()

    log.info("Scraping game data for %s season...", season)
    season_games_list = []
    # Iterate over each week to scrape game data
    for week in weeks:
        log.info("Collecting game data for Week %s...", week)
        # Scrape and save game data for the week
        week_games_df = read_write_data(
            f"{season}/week_{week:>02}_game_data",
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
    return pd.DataFrame()


def scrape_weekly_game_data(season: int, week: int) -> pd.DataFrame:
    """
    Scrapes game data for a specific week and season, returning it as a DataFrame.

    This function retrieves game data for each game in the specified week of the NFL season.  It
    iterates through all games in the week, scraping data for both the away and home teams.  The
    data for each game is then aggregated into a single DataFrame.  If no games are found or data is
    unavailable, an empty DataFrame is returned.

    Args:
        season (int):   The NFL season year.
        week (int):     The week number within the NFL season.

    Returns:
        pd.DataFrame:   A DataFrame containing the scraped game data for the week. If no data is
                        available, an empty DataFrame is returned.
    """
    log.info("Scraping game data for Week %s of the %s season...", week, season)
    # Retrieve the box scores for all games in the specified week and season
    week_scores = Boxscores(week, season)
    # Initialize a list to store game data DataFrames
    games_data = []

    # Iterate through each game in the week, scraping data
    for game_info in week_scores.games[f"{week}-{season}"]:
        game_df = pd.DataFrame(game_info, index=[0])
        if game_info["winning_name"] is None or game_info["losing_name"] is None:
            log.info("Game %s has not finished yet.", game_info["boxscore"])
            game_stats = None
        else:
            game_str = game_info["boxscore"]
            log.info("Scraping game data for %s...", game_str)
            game_stats = Boxscore(game_str)

        away_team_df, home_team_df = parse_game_data(game_df, game_stats)
        # Append team data with week and season information
        for team_df in (away_team_df, home_team_df):
            team_df["week"] = week
            team_df["season"] = season
            games_data.append(team_df)

    # Concatenate all game data into a single DataFrame, if any was collected
    return pd.concat(games_data, ignore_index=True) if games_data else pd.DataFrame()


def parse_game_data(
    game_df: pd.DataFrame, game_stats: Boxscore
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses game data from a DataFrame and Boxscore object into structured team data.

    This function takes raw game data and statistics, separates it into away and home team data, and
    formats it for further analysis.  It calculates win/loss status based on scores and converts
    time of possession to seconds.  The function handles missing scores by assigning NaN values to
    game outcomes.  It returns two DataFrames, one for each team, with consistent column names for
    easy comparison and analysis.

    Args:
        game_df (pd.DataFrame): DataFrame containing basic game information.
        game_stats (Boxscore):  Boxscore object containing detailed game statistics.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:  A tuple containing two DataFrames, one for the away team
                                            and one for the home team, with parsed and formatted
                                            game data.
    """
    # Extract and rename columns for away and home team data
    away_team_df = game_df[["away_name", "away_abbr", "away_score"]].copy()
    home_team_df = game_df[["home_name", "home_abbr", "home_score"]].copy()
    away_team_df.columns = home_team_df.columns = ["team_name", "team_abbr", "points_scored"]

    # Calculate win, loss, and tie outcomes
    if game_df["away_score"].isna().any() or game_df["home_score"].isna().any():
        # Handle missing scores by assigning NaN
        away_team_df["game_won"] = home_team_df["game_won"] = np.nan
        away_team_df["game_lost"] = home_team_df["game_lost"] = np.nan
    else:
        # Calculate outcomes based on scores
        away_win = (away_team_df["points_scored"] > home_team_df["points_scored"]).astype(float)
        home_win = (away_team_df["points_scored"] < home_team_df["points_scored"]).astype(float)
        tie = (away_team_df["points_scored"] == home_team_df["points_scored"]).astype(float) * 0.5
        away_team_df["game_won"], home_team_df["game_won"] = away_win + tie, home_win + tie
        away_team_df["game_lost"], home_team_df["game_lost"] = (
            1 - away_team_df["game_won"],
            1 - home_team_df["game_won"],
        )

    # Merge game statistics with team data
    if game_stats is not None:
        away_stats_df = (
            game_stats.dataframe[AWAY_STATS]
            .reset_index(drop=True)
            .rename(columns=AWAY_STATS_RENAME)
        )
        home_stats_df = (
            game_stats.dataframe[HOME_STATS]
            .reset_index(drop=True)
            .rename(columns=HOME_STATS_RENAME)
        )
    else:
        away_stats_df = pd.DataFrame(columns=AWAY_STATS_RENAME.values()).astype(float)
        home_stats_df = pd.DataFrame(columns=HOME_STATS_RENAME.values()).astype(float)
    away_team_df = pd.concat([away_team_df, away_stats_df], axis=1)
    home_team_df = pd.concat([home_team_df, home_stats_df], axis=1)

    # Convert time of possession to seconds
    away_team_df["time_of_possession"] = away_team_df.get("time_of_possession", np.nan).apply(
        convert_top_to_seconds
    )
    home_team_df["time_of_possession"] = home_team_df.get("time_of_possession", np.nan).apply(
        convert_top_to_seconds
    )

    return away_team_df, home_team_df


def convert_top_to_seconds(top: str) -> float:
    """
    Converts a time of possession string to seconds.

    This function takes a time of possession (ToP) string formatted as "MM:SS" and converts it into
    the total number of seconds.  If the input is null or not in the expected format, it returns
    NaN.

    Args:
        top (str):  The time of possession string in "MM:SS" format.

    Returns:
        float:  The time of possession in seconds, or NaN if input is null or improperly formatted.
    """
    # Check if the input is not null and properly formatted
    if pd.notnull(top) and ":" in top:
        # Split the string into minutes and seconds, convert to integers, and calculate total
        # seconds
        minutes, seconds = map(int, top.split(":"))
        return minutes * 60 + seconds
    # Return NaN for null or improperly formatted inputs
    return np.nan


def aggregate_season_data(season: int, weeks: list, season_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates game data for a specific NFL season.

    This function compiles game data across specified weeks of an NFL season into a single
    DataFrame.  It is designed to work with data that has already been collected and formatted into
    a preliminary DataFrame.  The aggregation process may involve summing statistics, averaging
    performance metrics, or any other method of combining data points across the weeks of the
    season.

    Args:
        season (int):                   The NFL season year for which data is being aggregated.
        weeks (list):                   A list of integers representing the weeks of the season to
                                        include in the aggregation.
        season_games_df (pd.DataFrame): A DataFrame containing game data for the season.

    Returns:
        pd.DataFrame:   A DataFrame containing aggregated game data for the specified weeks of the
                        season.
    """
    # Get and save season's schedule
    log.info("Collecting schedule for %s...", season)
    schedule_df_name = f"{season}/schedule"
    schedule_df = read_write_data(
        schedule_df_name,
        get_schedule,
        season,
        force_refresh=REFRESH_SCHEDULE,
    )
    agg_games_list = []
    for week in weeks:
        games_df = schedule_df[schedule_df["week"] == week]
        if week == 1:
            agg_weekly_df = (
                season_games_df[season_games_df["week"] <= week]
                .drop(columns=["week", "season", "game_won", "game_lost"])
                .groupby(by=["team_name", "team_abbr"])
                .mean()
                .reset_index()
            )
            win_loss_df = (
                season_games_df[season_games_df["week"] <= week][
                    ["team_name", "team_abbr", "game_won", "game_lost"]
                ]
                .groupby(by=["team_name", "team_abbr"])
                .sum()
                .reset_index()
            )
        else:
            agg_weekly_df = (
                season_games_df[season_games_df["week"] < week]
                .drop(columns=["week", "season", "game_won", "game_lost"])
                .groupby(by=["team_name", "team_abbr"])
                .mean()
                .reset_index()
            )
            win_loss_df = (
                season_games_df[season_games_df["week"] < week][
                    ["team_name", "team_abbr", "game_won", "game_lost"]
                ]
                .groupby(by=["team_name", "team_abbr"])
                .sum()
                .reset_index()
            )
        agg_weekly_df["fourth_down_perc"] = agg_weekly_df["fourth_down_conversions"].div(
            agg_weekly_df["fourth_down_attempts"]
        )
        agg_weekly_df["third_down_perc"] = agg_weekly_df["third_down_conversions"].div(
            agg_weekly_df["third_down_attempts"]
        )
        agg_weekly_df.loc[~np.isfinite(agg_weekly_df["fourth_down_perc"]), "fourth_down_perc"] = 0
        agg_weekly_df.loc[~np.isfinite(agg_weekly_df["third_down_perc"]), "third_down_perc"] = 0
        agg_weekly_df["fourth_down_perc"] = agg_weekly_df["fourth_down_perc"].fillna(0)
        agg_weekly_df["third_down_perc"] = agg_weekly_df["third_down_perc"].fillna(0)
        if week == 1:
            agg_weekly_df["fourth_down_perc"] = np.nan
            agg_weekly_df["third_down_perc"] = np.nan
        agg_weekly_df = agg_weekly_df.drop(
            columns=[
                "fourth_down_attempts",
                "fourth_down_conversions",
                "third_down_attempts",
                "third_down_conversions",
            ]
        )

        win_loss_df["win_perc"] = win_loss_df["game_won"].div(
            win_loss_df["game_won"] + win_loss_df["game_lost"]
        )
        win_loss_df.loc[~np.isfinite(win_loss_df["win_perc"]), "win_perc"] = 0
        if week == 1:
            win_loss_df["win_perc"] = np.nan
        win_loss_df = win_loss_df.drop(columns=["game_won", "game_lost"])

        agg_weekly_df = pd.merge(
            win_loss_df,
            agg_weekly_df,
            left_on=["team_name", "team_abbr"],
            right_on=["team_name", "team_abbr"],
        )

        away_df = (
            pd.merge(
                games_df,
                agg_weekly_df,
                how="inner",
                left_on=["away_name", "away_abbr"],
                right_on=["team_name", "team_abbr"],
            )
            .drop(columns=["team_name", "team_abbr"])
            .rename(columns=AGG_RENAME_AWAY)
        )
        home_df = (
            pd.merge(
                games_df,
                agg_weekly_df,
                how="inner",
                left_on=["home_name", "home_abbr"],
                right_on=["team_name", "team_abbr"],
            )
            .drop(columns=["team_name", "team_abbr"])
            .rename(columns=AGG_RENAME_HOME)
        )
        agg_weekly_df = pd.merge(away_df, home_df, left_on=AGG_MERGE_ON, right_on=AGG_MERGE_ON)
        agg_weekly_df["win_perc_dif"] = (
            agg_weekly_df["away_win_perc"] - agg_weekly_df["home_win_perc"]
        )
        agg_weekly_df["first_downs_dif"] = (
            agg_weekly_df["away_first_downs"] - agg_weekly_df["home_first_downs"]
        )
        agg_weekly_df["fumbles_dif"] = agg_weekly_df["away_fumbles"] - agg_weekly_df["home_fumbles"]
        agg_weekly_df["interceptions_dif"] = (
            agg_weekly_df["away_interceptions"] - agg_weekly_df["home_interceptions"]
        )
        agg_weekly_df["net_pass_yards_dif"] = (
            agg_weekly_df["away_net_pass_yards"] - agg_weekly_df["home_net_pass_yards"]
        )
        agg_weekly_df["pass_attempts_dif"] = (
            agg_weekly_df["away_pass_attempts"] - agg_weekly_df["home_pass_attempts"]
        )
        agg_weekly_df["pass_completions_dif"] = (
            agg_weekly_df["away_pass_completions"] - agg_weekly_df["home_pass_completions"]
        )
        agg_weekly_df["pass_touchdowns_dif"] = (
            agg_weekly_df["away_pass_touchdowns"] - agg_weekly_df["home_pass_touchdowns"]
        )
        agg_weekly_df["pass_yards_dif"] = (
            agg_weekly_df["away_pass_yards"] - agg_weekly_df["home_pass_yards"]
        )
        agg_weekly_df["penalties_dif"] = (
            agg_weekly_df["away_penalties"] - agg_weekly_df["home_penalties"]
        )
        agg_weekly_df["points_scored_dif"] = (
            agg_weekly_df["away_points_scored"] - agg_weekly_df["home_points_scored"]
        )
        agg_weekly_df["points_allowed_dif"] = (
            agg_weekly_df["away_points_allowed"] - agg_weekly_df["home_points_allowed"]
        )
        agg_weekly_df["rush_attempts_dif"] = (
            agg_weekly_df["away_rush_attempts"] - agg_weekly_df["home_rush_attempts"]
        )
        agg_weekly_df["rush_touchdowns_dif"] = (
            agg_weekly_df["away_rush_touchdowns"] - agg_weekly_df["home_rush_touchdowns"]
        )
        agg_weekly_df["rush_yards_dif"] = (
            agg_weekly_df["away_rush_yards"] - agg_weekly_df["home_rush_yards"]
        )
        agg_weekly_df["time_of_possession_dif"] = (
            agg_weekly_df["away_time_of_possession"] - agg_weekly_df["home_time_of_possession"]
        )
        agg_weekly_df["times_sacked_dif"] = (
            agg_weekly_df["away_times_sacked"] - agg_weekly_df["home_times_sacked"]
        )
        agg_weekly_df["total_yards_dif"] = (
            agg_weekly_df["away_total_yards"] - agg_weekly_df["home_total_yards"]
        )
        agg_weekly_df["turnovers_dif"] = (
            agg_weekly_df["away_turnovers"] - agg_weekly_df["home_turnovers"]
        )
        agg_weekly_df["yards_from_penalties_dif"] = (
            agg_weekly_df["away_yards_from_penalties"] - agg_weekly_df["home_yards_from_penalties"]
        )
        agg_weekly_df["yards_lost_from_sacks_dif"] = (
            agg_weekly_df["away_yards_lost_from_sacks"]
            - agg_weekly_df["home_yards_lost_from_sacks"]
        )
        agg_weekly_df["fourth_down_perc_dif"] = (
            agg_weekly_df["away_fourth_down_perc"] - agg_weekly_df["home_fourth_down_perc"]
        )
        agg_weekly_df["third_down_perc_dif"] = (
            agg_weekly_df["away_third_down_perc"] - agg_weekly_df["home_third_down_perc"]
        )
        agg_weekly_df = agg_weekly_df.drop(columns=AGG_DROP_COLS, errors="ignore")
        if agg_weekly_df["winning_name"].isna().values.any():
            if agg_weekly_df["winning_name"].isna().sum() == agg_weekly_df.shape[0]:
                agg_weekly_df["result"] = np.nan
                log.info("Week %s games have not finished yet.", week)
            else:
                agg_weekly_df.loc[agg_weekly_df["winning_name"].isna(), "result"] = 0
                agg_weekly_df["result"] = agg_weekly_df["result"].astype("float")
        else:
            agg_weekly_df["result"] = agg_weekly_df["winning_name"] == agg_weekly_df["away_name"]
            agg_weekly_df["result"] = agg_weekly_df["result"].astype("float")
        agg_weekly_df = agg_weekly_df.drop(columns=["winning_name", "winning_abbr"])
        agg_games_list.append(agg_weekly_df)
    agg_games_df = pd.concat(agg_games_list, ignore_index=True).reset_index(drop=True)
    return agg_games_df


def get_schedule(season: int) -> pd.DataFrame:
    """
    Scrapes and compiles the NFL schedule for a given season into a DataFrame.

    This function scrapes the NFL schedule for a specified season, accommodating the change in
    season length starting from 2021.  It iterates through each week of the season, scraping game
    data and compiling it into a single DataFrame.  The resulting DataFrame includes details for
    each game, such as the names and abbreviations of the away and home teams, as well as the
    winning team's name and abbreviation.

    Args:
        season (int): The NFL season year to scrape the schedule for.

    Returns:
        pd.DataFrame:   A DataFrame containing the schedule, with each row representing a game and
                        columns for away and home team names and abbreviations, winning team name
                        and abbreviation, and the week of the season.
    """
    # Determine the number of weeks in the season
    weeks = list(range(1, 18)) if season < 2021 else list(range(1, 19))
    schedule_df = pd.DataFrame()
    log.info("Scraping %s schedule...", season)
    for week in weeks:
        log.info("Week %s...", week)
        date_string = f"{week}-{season}"
        week_scores = Boxscores(week, season)
        week_games_df = pd.DataFrame()
        for game in week_scores.games[date_string]:
            game_df = pd.DataFrame(game, index=[0])[
                ["away_name", "away_abbr", "home_name", "home_abbr", "winning_name", "winning_abbr"]
            ]
            game_df["week"] = week
            week_games_df = pd.concat([week_games_df, game_df])
        schedule_df = pd.concat([schedule_df, week_games_df]).reset_index(drop=True)
    return schedule_df


def get_season_elo(elo_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Filters and transforms ELO ratings DataFrame for a specific NFL season.

    This function processes a DataFrame containing ELO ratings by filtering it for a specific NFL
    season, dropping unnecessary columns, converting date strings to datetime objects, and
    standardizing team names.  The NFL season is considered to start in September of the given year
    and end in March of the following year.

    Args:
        elo_df (pd.DataFrame):  The DataFrame containing ELO ratings for multiple seasons.
        season (int):           The NFL season year to filter the DataFrame for.

    Returns:
        pd.DataFrame:   A DataFrame containing filtered and processed ELO ratings for the specified
                        season.
    """
    # Copy the DataFrame to avoid modifying the original
    yearly_elo_df = elo_df.copy()
    # Drop columns not needed for analysis
    yearly_elo_df = yearly_elo_df.drop(columns=ELO_DROP_COLS)
    # Convert date column to datetime format
    yearly_elo_df["date"] = pd.to_datetime(yearly_elo_df["date"])
    # Define the start and end dates for the NFL season
    start_date = datetime(season, 9, 1)
    end_date = datetime(season + 1, 3, 1)
    # Create a mask to filter rows within the NFL season dates
    mask = (yearly_elo_df["date"] >= start_date) & (yearly_elo_df["date"] <= end_date)
    # Apply the mask to filter the DataFrame
    yearly_elo_df = yearly_elo_df.loc[mask]
    # Replace ELO team names with standard team names for consistency
    yearly_elo_df["team1"] = yearly_elo_df["team1"].replace(ELO_TEAMS, STD_TEAMS)
    yearly_elo_df["team2"] = yearly_elo_df["team2"].replace(ELO_TEAMS, STD_TEAMS)
    return yearly_elo_df


def combine_data(season: int, agg_games_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines aggregated game data with ELO ratings for a specific NFL season.

    This function merges an aggregated games DataFrame with an ELO ratings DataFrame based on team
    abbreviations.  It calculates the difference in ELO ratings and quarterback values between the
    home and away teams, adds a column for the season, and standardizes team abbreviations.  The
    'result' column is moved to the end of the DataFrame for readability.

    Args:
        season (int):                   The NFL season year for which the data is being combined.
        agg_games_df (pd.DataFrame):    DataFrame containing aggregated game statistics.
        elo_df (pd.DataFrame):          DataFrame containing ELO ratings for teams.

    Returns:
        pd.DataFrame:   A DataFrame with combined game statistics and ELO ratings, including
                        calculated differences and standardized team abbreviations.
    """
    # Merge aggregated game data with ELO ratings on team abbreviations
    combined_df = pd.merge(
        agg_games_df,
        elo_df,
        how="inner",
        left_on=["home_abbr", "away_abbr"],
        right_on=["team1", "team2"],
    ).drop(
        columns=["date", "team1", "team2"]
    )  # Drop redundant columns after merge

    # Calculate differences in ELO ratings and quarterback values
    combined_df["elo_dif"] = combined_df["elo2_pre"] - combined_df["elo1_pre"]
    combined_df["qb_dif"] = combined_df["qb2_value_pre"] - combined_df["qb1_value_pre"]

    # Drop columns now redundant after calculating differences
    combined_df = combined_df.drop(
        columns=["elo1_pre", "elo2_pre", "qb1_value_pre", "qb2_value_pre"]
    )

    # Insert a column for the season
    combined_df.insert(loc=4, column="season", value=season)

    # Standardize team abbreviations back to their original format
    combined_df["home_abbr"] = combined_df["home_abbr"].replace(STD_TEAMS, TEAMS)
    combined_df["away_abbr"] = combined_df["away_abbr"].replace(STD_TEAMS, TEAMS)

    # Reorder columns to move 'result' to the end for readability
    result_col = ["result"] if "result" in combined_df else []  # Ensure 'result' column exists
    cols_except_result = [col for col in combined_df if col not in result_col]
    combined_df = combined_df[cols_except_result + result_col]

    return combined_df


def parse_completed_games(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and returns a DataFrame containing only the rows from the input DataFrame where the
    "result" column is not null, indicating the games have been completed.

    Args:
        combined_data_df (pd.DataFrame):    The input DataFrame containing game data, including a
                                            "result" column that indicates the outcome of each game.

    Returns:
        pd.DataFrame:   A DataFrame consisting of only the rows from the input DataFrame where the
                        "result" column is not null, indicating completed games.
    """
    # Filter for completed games based on the presence of a result
    completed_games_df = combined_data_df[combined_data_df["result"].notna()]

    return completed_games_df


def parse_upcoming_games_to_predict(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the input DataFrame to include only rows representing upcoming games for the current
    week and season.

    This function identifies the current NFL season and week based on today's date.  It then filters
    the input DataFrame for games that are scheduled for the current week of the current season and
    have a null 'result', indicating that the games have not yet been played.

    Args:
        combined_data_df (pd.DataFrame):    The input DataFrame containing game data for multiple
                                            seasons and weeks, including a 'result' column that
                                            indicates the outcome of each game.

    Returns:
        pd.DataFrame:   A DataFrame consisting of only the rows from the input DataFrame that
                        represent games scheduled to be played in the current week of the current
                        season, as determined by the system's current date.
    """
    # Determine the current season based on today's date and SEASON_END_MONTH
    today = date.today()
    current_season = today.year if today.month > SEASON_END_MONTH else today.year - 1

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
