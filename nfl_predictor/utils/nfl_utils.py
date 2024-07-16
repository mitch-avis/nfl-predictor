"""
Provides utility functions for the NFL predictor project, facilitating operations such as game
outcome computation, stats DataFrame creation, and NFL week determination. These utilities support
the data collection and analysis processes by offering specialized operations tailored to NFL data.

Functions:
- compute_game_outcomes: Calculates the outcomes of games based on team stats.
- create_stats_dfs_from_boxscore: Generates pandas DataFrames from boxscore data.
- determine_nfl_week_by_date: Determines the NFL week number based on a given date.
- determine_weeks_to_scrape: Identifies the weeks of NFL games to scrape data for.
- fetch_nfl_elo_ratings: Retrieves ELO ratings for NFL teams.
- init_team_stats_dfs: Initializes DataFrames for team statistics.
- merge_and_format_df: Merges and formats different DataFrames into a coherent structure.

Dependencies:
- pandas: For creating and manipulating DataFrames.
- numpy: Used for numerical operations.
"""

from datetime import date, timedelta
from time import sleep

import numpy as np
import pandas as pd
from sportsipy.nfl.boxscore import Boxscore, Boxscores

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

DATA_PATH = constants.DATA_PATH
SEASON_END_MONTH = constants.SEASON_END_MONTH
WEEKS_BEFORE_2021 = constants.WEEKS_BEFORE_2021
WEEKS_FROM_2021_ONWARDS = constants.WEEKS_FROM_2021_ONWARDS
ELO_DATA_URL = constants.ELO_DATA_URL
BASE_COLUMNS = constants.BASE_COLUMNS
BOXSCORE_STATS = constants.BOXSCORE_STATS
AGG_STATS = constants.AGG_STATS
AGG_DROP_COLS = constants.AGG_DROP_COLS


def fetch_nfl_elo_ratings() -> pd.DataFrame:
    """
    Fetches the latest ELO ratings for NFL teams from a specified URL.

    This function uses pandas to read a CSV file containing ELO ratings directly from a URL into
    a DataFrame. The ELO ratings are used to assess the relative skill levels of teams.

    Returns:
        pd.DataFrame: A DataFrame containing the ELO ratings for NFL teams.
    """
    # Utilize pandas to directly read the ELO ratings CSV from the URL into a DataFrame
    elo_df = pd.read_csv(ELO_DATA_URL)
    return elo_df


def get_season_start(year: int) -> date:
    """
    Calculates the NFL season start date for a given year, traditionally the Thursday following the
    first Monday of September.

    Args:
        year (int): The year for which the season start date is calculated.

    Returns:
        date: The start date of the NFL season for the given year.
    """
    # Calculate the date for the first of September of the given year
    sept_first = date(year, 9, 1)
    # Calculate the date for the first Monday of September
    first_monday = sept_first + timedelta((7 - sept_first.weekday()) % 7)
    # The season starts on the Thursday following the first Monday of September
    season_start = first_monday + timedelta(days=3)
    return season_start


def determine_nfl_week_by_date(given_date: date) -> int:
    """
    Determines the NFL week number for a given date, accounting for the season's structure and
    handling dates outside the regular season by assigning them to the closest season's start or
    treating them as offseason.

    Args:
        given_date (date): The date for which the NFL week number is determined.

    Returns:
        int: The NFL week number for the given date, with 1 for dates before the season start or
             offseason, and up to 18 for the regular season weeks.
    """
    # Determine the given season's start date based on the year of the given date
    season_start = get_season_start(given_date.year)

    # If the given date is before the season's start, determine if it's closer to the previous
    # season's start or should be considered as offseason/preseason for the current year
    if given_date < season_start:
        previous_season_start = get_season_start(given_date.year - 1)
        # If the given date is after the previous season's start, consider it as the start of the
        # new season (offseason/preseason period)
        if given_date >= previous_season_start:
            return 1  # Treat as the beginning of the new season
        return 0  # Indicates the date is before the previous season's start (deep offseason)

    # Calculate the week number by dividing the difference in days by 7 and adding 1 to adjust
    # for the first week. This calculation assumes a week starts from the season start date.
    week_number = ((given_date - season_start).days // 7) + 1

    # Ensure the week number is within the NFL regular season range (1 to 18), adjusting for
    # dates that fall outside this range to treat them as offseason/preseason.
    return max(1, min(week_number, 18)) if week_number > 0 else 1


def determine_weeks_to_scrape(season: int, include_future_weeks: bool = False) -> list:
    """
    Determines the weeks to scrape for a given NFL season, adjusting for changes in the season
    structure over time and whether to include future weeks in the scraping process.

    Args:
        season (int): The NFL season year.
        include_future_weeks (bool): If True, includes all weeks for the current or future seasons.

    Returns:
        list: A list of weeks to scrape for the specified season.
    """
    today = date.today()
    # Determine the current NFL season based on today's date, considering the NFL season typically
    # ends in February.
    current_season = today.year if today.month > SEASON_END_MONTH else today.year - 1

    if season < 2021:
        # For seasons before 2021, the NFL had 17 weeks. This accounts for the schedule prior to
        # the expansion to an 18-week season.
        weeks_to_scrape = list(range(1, WEEKS_BEFORE_2021 + 1))
    elif season <= current_season or include_future_weeks:
        # For past seasons from 2021 onwards, including the current season if include_future_weeks
        # is True, and for future seasons if include_future_weeks is True, the NFL has 18 weeks.
        weeks_to_scrape = list(range(1, WEEKS_FROM_2021_ONWARDS + 1))
    else:
        # For the current season without including future weeks, scrape up to the current week.
        # This ensures data is only collected for weeks that have potentially completed.
        if season == current_season:
            current_week = determine_nfl_week_by_date(today)
            weeks_to_scrape = list(range(1, current_week + 1))
        else:
            # For future seasons without including future weeks, there are no weeks to scrape
            # since the season hasn't started or we're avoiding future data collection.
            weeks_to_scrape = []

    return weeks_to_scrape


def get_week_dates(season: int) -> list[date]:
    """
    Generate a list of dates representing the start of each week to scrape for a given season.

    This function calculates the start date for each week of the season to scrape, based on the
    season's start date and the number of weeks determined to scrape.

    Args:
        season (int): The season number to calculate week start dates for.

    Returns:
        list: A list of datetime.date objects representing the start of each week to scrape.
    """
    # Calculate the initial date to start from, which is 8 days before the season start date
    season_start_date = get_season_start(season) - timedelta(days=8)
    # Determine the number of weeks to scrape for the given season
    weeks_to_scrape = determine_weeks_to_scrape(season)
    # Use list comprehension to generate the list of week start dates
    week_dates = [season_start_date + timedelta(weeks=week) for week in weeks_to_scrape]
    return week_dates


def fetch_week_boxscores(season: int, week: int) -> tuple[Boxscores, Exception]:
    """
    Fetches box scores for all games in a specified week and season, retrying on failure.

    This function attempts to retrieve box scores for a given week and season. If the initial
    attempt fails, it retries up to two more times, with an increasing delay between attempts.
    It returns a tuple containing the Boxscores object (or None if unsuccessful) and an Exception
    object if an error occurred on the final attempt (or None if successful).

    Args:
        season (int): The NFL season year.
        week (int): The week number within the NFL season.

    Returns:
        tuple[Boxscores, Exception]: A tuple containing the Boxscores object and an Exception
                                     object if an error occurred, or None for both if successful.
    """
    # pylint: disable=broad-except
    for attempt in range(3):
        try:
            return Boxscores(week, season), None
        except Exception as e:
            log.warning("Failed to fetch week scores on attempt %s: %s", attempt + 1, e)
            if attempt == 2:
                return None, e  # Return the exception on the final attempt
            sleep(2**attempt)  # Exponential backoff between retries
    return None, Exception("Unable to fetch week scores after multiple attempts.")


def fetch_game_boxscore(game_info: str) -> tuple[Boxscore, Exception]:
    """
    Fetches individual game stats with retries on failure.

    Attempts to retrieve the boxscore for a single game identified by its boxscore URL segment.
    If the initial attempt fails, it retries up to two more times, with an increasing delay
    between attempts. It returns a tuple containing the Boxscore object (or None if unsuccessful)
    and an Exception object if an error occurred on the final attempt (or None if successful).

    Args:
        game_info (str): The boxscore URL segment for the game.

    Returns:
        tuple[Boxscore, Exception]: A tuple containing the Boxscore object and an Exception
                                    object if an error occurred, or None for both if successful.
    """
    # pylint: disable=broad-except
    for attempt in range(3):
        try:
            return Boxscore(game_info), None
        except Exception as e:
            log.warning(
                "Failed to fetch game stats for %s on attempt %s: %s",
                game_info,
                attempt + 1,
                e,
            )
            if attempt == 2:
                return None, e  # Return the exception on the final attempt
            sleep(2**attempt)  # Exponential backoff between retries
    return None, Exception("Unable to fetch game stats after multiple attempts.")


def init_team_stats_dfs(game_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares DataFrames for away and home teams based on game data, focusing on team names,
    abbreviations, and scores. This function is essential for separating and standardizing game
    data for further analysis or processing.

    Args:
        game_df (pd.DataFrame): The DataFrame containing basic game information.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames for away and home teams, respectively,
                                           with standardized column names.
    """
    # Select and rename columns for the away team to standardize the structure. The renaming
    # aligns with a base format for easier comparison and analysis of team performance.
    away_team_df = game_df.loc[:, ["away_name", "away_abbr", "away_score", "home_score"]].copy()
    away_team_df.columns = BASE_COLUMNS[:-2]

    # Perform a similar selection and renaming process for the home team, ensuring both DataFrames
    # have a consistent format for direct comparison and further statistical analysis.
    home_team_df = game_df.loc[:, ["home_name", "home_abbr", "home_score", "away_score"]].copy()
    home_team_df.columns = BASE_COLUMNS[:-2]

    return away_team_df, home_team_df


def compute_game_outcomes(
    away_team_df: pd.DataFrame, home_team_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the outcomes of games based on points scored and updates the DataFrames with
    'game_won' and 'game_lost' columns. Handles unplayed games by setting relevant fields to NaN.
    In the event of a tie, both 'game_won' and 'game_lost' are set to 0.5.

    Args:
        away_team_df (pd.DataFrame): DataFrame containing the away team's game statistics,
                                     including 'points_scored' and 'points_allowed'.
        home_team_df (pd.DataFrame): DataFrame containing the home team's game statistics,
                                     including 'points_scored' and 'points_allowed'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated away and home team
                                           DataFrames with new columns for game outcomes.
    """
    # Identify unplayed games by checking for NaN in 'points_scored'
    unplayed_games = away_team_df["points_scored"].isna() | home_team_df["points_scored"].isna()

    # Determine wins and ties based on points scored and allowed
    away_wins = away_team_df["points_scored"] > away_team_df["points_allowed"]
    home_wins = home_team_df["points_scored"] > home_team_df["points_allowed"]
    ties = away_team_df["points_scored"] == home_team_df["points_scored"]

    # Update 'game_won' and 'game_lost' for away team, handling unplayed games and ties
    away_team_df["game_won"] = np.where(
        unplayed_games, np.nan, np.where(away_wins | ties, 1.0, 0.0)
    )
    away_team_df["game_lost"] = np.where(unplayed_games, np.nan, np.where(away_wins, 0.0, 1.0))

    # Update 'game_won' and 'game_lost' for home team, similar to away team
    home_team_df["game_won"] = np.where(
        unplayed_games, np.nan, np.where(home_wins | ties, 1.0, 0.0)
    )
    home_team_df["game_lost"] = np.where(unplayed_games, np.nan, np.where(home_wins, 0.0, 1.0))

    # For tied games, set 'game_won' and 'game_lost' to 0.5 for both teams
    away_team_df.loc[ties, ["game_won", "game_lost"]] = 0.5
    home_team_df.loc[ties, ["game_won", "game_lost"]] = 0.5

    # For unplayed games, set 'points_scored' and 'points_allowed' to NaN
    for df in [away_team_df, home_team_df]:
        df.loc[unplayed_games, ["points_scored", "points_allowed"]] = np.nan

    return away_team_df, home_team_df


def create_stats_dfs_from_boxscore(
    boxscore: Boxscore,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares DataFrames for away and home team statistics from a Boxscore object.

    This function extracts detailed game statistics for both the away and home teams from a given
    Boxscore object, organizing the data into separate DataFrames for each team. If the Boxscore
    object is missing or lacks a dataframe attribute, it returns empty DataFrames structured
    according to predefined statistics.

    Args:
        boxscore (Boxscore): The Boxscore object containing detailed game statistics.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing statistics for away and home
                                           teams.
    """
    # Check if the Boxscore object is valid and contains a dataframe with game statistics. This
    # step ensures that the function can handle cases where the Boxscore might be incomplete or
    # missing.
    if boxscore is not None and hasattr(boxscore, "dataframe") and boxscore.dataframe is not None:
        # Extract and format the statistics for both the away and home teams from the Boxscore's
        # dataframe. This involves selecting relevant columns and standardizing the data format.
        away_stats_df = create_stats_df(boxscore.dataframe, "away")
        home_stats_df = create_stats_df(boxscore.dataframe, "home")
    else:
        # Prepare empty DataFrames with a consistent structure for cases where the Boxscore data
        # is unavailable. This ensures that the function's return value remains consistent and
        # predictable, even in the absence of data.
        empty_df_structure = pd.DataFrame(
            {stat: pd.Series(dtype="float") for stat in BOXSCORE_STATS}
        )
        away_stats_df = empty_df_structure.copy()
        home_stats_df = empty_df_structure.copy()

    return away_stats_df, home_stats_df


def create_stats_df(game_stats_df: pd.DataFrame, team_prefix: str) -> pd.DataFrame:
    """
    Transforms game statistics into a team-specific DataFrame by selecting and renaming columns
    based on a specified prefix ('away' or 'home'). This standardization facilitates easier
    comparison and analysis of team performance across games.

    Args:
        game_stats_df (pd.DataFrame): DataFrame containing detailed game statistics.
        team_prefix (str): Prefix indicating whether the statistics are for the away team ('away')
                           or the home team ('home'), used to select relevant columns.

    Returns:
        pd.DataFrame: A DataFrame with standardized and correctly labeled columns, facilitating
                      uniform data analysis.
    """
    # Create a mapping of prefixed column names (e.g., 'away_points_scored') to standard column
    # names (e.g., 'points_scored'). This step is crucial for ensuring that data from both home
    # and away teams can be analyzed in a consistent format.
    column_mapping = {f"{team_prefix}_{stat}": stat for stat in BOXSCORE_STATS}

    # Use the column mapping to rename the selected columns in the DataFrame. This involves
    # filtering out columns based on the prefix and then renaming them to a standardized format,
    # which is essential for subsequent data processing and analysis steps.
    transformed_df = game_stats_df.rename(columns=column_mapping)[list(column_mapping.values())]

    # Reset the index of the transformed DataFrame. This is a common practice when creating a new
    # DataFrame based on selected or transformed data to ensure the index starts at 0 and is
    # continuous, which can be important for data concatenation, iteration, and other operations.
    return transformed_df.reset_index(drop=True)


def merge_and_format_df(
    team_df: pd.DataFrame, team_stats_df: pd.DataFrame, opponent_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges team and opponent statistics into a single DataFrame and applies formatting for
    consistency and analysis readiness. This function is crucial for consolidating various
    statistics into a unified structure, facilitating easier comparison and analysis.

    Args:
        team_df (pd.DataFrame): DataFrame containing basic team data such as names and scores.
        team_stats_df (pd.DataFrame): DataFrame containing detailed statistics for the team.
        opponent_stats_df (pd.DataFrame): DataFrame containing detailed statistics for the opponent.

    Returns:
        pd.DataFrame: A comprehensive DataFrame that combines team and opponent statistics, with
                      specific formatting applied to ensure data consistency and readability.
    """
    # Rename opponent's statistics columns for clarity, prefixing them to distinguish from the
    # team's stats
    opponent_stats_df = opponent_stats_df.rename(columns=lambda x: f"opponent_{x}")

    # Merge the team's DataFrame with its own and the opponent's statistics along columns
    merged_df = pd.concat([team_df, team_stats_df, opponent_stats_df], axis=1)

    # Apply formatting functions to the merged DataFrame for further analysis:
    # - Convert time of possession to seconds for uniformity
    # - Reorder columns for readability and drop unnecessary columns for clarity
    merged_df = convert_top_to_seconds_in_df(merged_df)
    merged_df = reorder_and_drop_columns(merged_df)

    return merged_df


def convert_top_to_seconds_in_df(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts time of possession (TOP) from "MM:SS" format to seconds in a DataFrame, handling
    cases where TOP might be NaN. This conversion is essential for enabling numerical analysis
    and facilitating comparisons of TOP across different games.

    Args:
        team_df (pd.DataFrame): DataFrame containing team data, including TOP in "MM:SS" format.

    Returns:
        pd.DataFrame: Updated DataFrame with TOP converted from "MM:SS" to seconds, ensuring
                      consistency in data format for analysis.
    """
    # Check for 'time_of_possession' column and convert non-NaN TOP values from "MM:SS" to seconds.
    # NaN values are left unchanged, preserving data integrity for entries without TOP information.
    if "time_of_possession" in team_df.columns:
        team_df["time_of_possession"] = team_df["time_of_possession"].apply(
            lambda x: (
                float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if pd.notna(x) else np.nan
            )
        )
    return team_df


def reorder_and_drop_columns(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders and selectively drops columns in a DataFrame to streamline data structure for analysis.
    This function focuses on maintaining essential columns while removing those deemed unnecessary
    for the current analytical context, enhancing data clarity and processing efficiency.

    Args:
        team_df (pd.DataFrame): DataFrame to be reordered and modified.

    Returns:
        pd.DataFrame: Modified DataFrame with columns reordered and unnecessary columns dropped.
    """
    # Identify additional columns to include, excluding specific opponent-related columns
    # that are not needed for the current analysis. This step ensures that only relevant data
    # is retained, making the DataFrame easier to work with and understand.
    other_columns = [
        col
        for col in team_df.columns
        if col not in BASE_COLUMNS
        and col not in ("opponent_points_allowed", "opponent_time_of_possession")
    ]
    # Combine the base columns with the additional, relevant columns for the final column order.
    # This reordering aligns the DataFrame structure with expected formats for downstream
    # analysis or reporting, ensuring consistency and readability.
    ordered_columns = BASE_COLUMNS + other_columns
    # Return the DataFrame with columns reordered according to the specified order. This final
    # DataFrame is streamlined for analysis, with unnecessary columns removed and relevant
    # columns properly organized.
    return team_df[ordered_columns]


def prepare_data_for_week(
    week: int, schedule_df: pd.DataFrame, season_games_df: pd.DataFrame
) -> tuple:
    """
    Prepares weekly game and results data from season schedules and game outcomes.

    This function filters the provided season schedule and game outcomes data for a specific week,
    returning two DataFrames: one with the week's games and another with the results, focusing on
    essential columns defined in BASE_COLUMNS.

    Args:
        week (int): Target week number for data extraction.
        schedule_df (pd.DataFrame): Season's schedule data.
        season_games_df (pd.DataFrame): Detailed outcomes of season's games.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - DataFrame of the week's games from the schedule.
            - DataFrame of the week's game results, limited to essential columns.
    """
    # Filter for the specified week's games and results
    week_games_df = schedule_df[schedule_df["week"] == week]
    results_df = season_games_df[season_games_df["week"] == week][BASE_COLUMNS]

    return week_games_df, results_df


def calculate_stats(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """
    Optimizes statistics calculation for NFL games by week, handling initialization for week 1 and
    aggregating metrics for subsequent weeks.

    Args:
        df (pd.DataFrame): Season games data.
        week (int): Target week for statistics calculation.

    Returns:
        pd.DataFrame: DataFrame with updated or calculated statistics for the specified week.
    """
    # For week 1, initialize specific columns with NaN
    if week == 1:
        for col in [
            "win_perc",
            "third_down_perc",
            "fourth_down_perc",
            "opponent_third_down_perc",
            "opponent_fourth_down_perc",
        ]:
            df[col] = np.nan
        return df

    # Define metrics for aggregation
    metrics = {
        "game_won": "sum",
        "game_lost": "sum",
        "third_down_conversions": "sum",
        "third_down_attempts": "sum",
        "fourth_down_conversions": "sum",
        "fourth_down_attempts": "sum",
        "opponent_third_down_conversions": "sum",
        "opponent_third_down_attempts": "sum",
        "opponent_fourth_down_conversions": "sum",
        "opponent_fourth_down_attempts": "sum",
    }

    # Aggregate metrics and calculate win/conversion rates
    agg_metrics_df = df.groupby(["team_name", "team_abbr"], as_index=False).agg(metrics)
    agg_metrics_df = calc_win_and_conversion_rates(agg_metrics_df)

    # Calculate mean for columns not included in metrics, excluding specific columns
    excluded_columns = set(metrics) | {"team_name", "team_abbr", "week", "season"}
    mean_columns = df.columns.difference(excluded_columns)
    agg_metrics_df[mean_columns] = df.groupby(["team_name", "team_abbr"])[mean_columns].transform(
        "mean"
    )

    return agg_metrics_df


def merge_and_finalize(week_games_df, agg_weekly_df, results_df):
    """
    Prepares and merges weekly game data with aggregated statistics and results.

    Drops unnecessary columns from aggregated data, prefixes columns to distinguish between home
    and away teams, merges the dataframes, calculates statistical differences, and merges final
    scores.

    Args:
        week_games_df (pd.DataFrame): DataFrame containing the week's games schedule.
        agg_weekly_df (pd.DataFrame): DataFrame with aggregated weekly statistics.
        results_df (pd.DataFrame): DataFrame with the week's game results.

    Returns:
        pd.DataFrame: The finalized DataFrame ready for analysis or storage.
    """
    # Drop specified columns from aggregated data
    agg_weekly_df = agg_weekly_df.drop(columns=AGG_DROP_COLS, errors="ignore")

    # Prepare away and home team data with appropriate prefixes and renames
    agg_weekly_away_df = agg_weekly_df.add_prefix("away_").rename(
        columns={"away_team_name": "away_name", "away_team_abbr": "away_abbr"}
    )
    agg_weekly_home_df = agg_weekly_df.add_prefix("home_").rename(
        columns={"home_team_name": "home_name", "home_team_abbr": "home_abbr"}
    )

    # Merge aggregated stats with the week's game schedule
    merged_df = merge_aggregated_stats(week_games_df, agg_weekly_away_df, agg_weekly_home_df)

    # Calculate statistical differences between teams
    merged_df = calc_stat_diffs(merged_df)

    # Merge in the game results
    return merge_scores(merged_df, results_df)


def calc_win_and_conversion_rates(agg_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates win percentages and down conversion rates for aggregated weekly data.

    This function takes a DataFrame of aggregated weekly data and calculates the win percentage,
    third down conversion rate, fourth down conversion rate, and opponent's third and fourth down
    conversion rates for each team.

    Args:
        agg_weekly_df (pd.DataFrame): DataFrame containing aggregated weekly data for teams.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for each of the calculated rates.
    """
    # Calculate win percentage; handle cases with missing data using np.where to avoid division by
    # zero
    agg_weekly_df["win_perc"] = np.where(
        (agg_weekly_df["game_won"].notna() & agg_weekly_df["game_lost"].notna()),
        agg_weekly_df["game_won"] / (agg_weekly_df["game_won"] + agg_weekly_df["game_lost"]),
        np.nan,
    )

    # Calculate third down conversion percentage; handle missing data to avoid division errors
    agg_weekly_df["third_down_perc"] = np.where(
        (
            agg_weekly_df["third_down_conversions"].notna()
            & agg_weekly_df["third_down_attempts"].notna()
        ),
        agg_weekly_df["third_down_conversions"] / agg_weekly_df["third_down_attempts"],
        np.nan,
    )

    # Calculate fourth down conversion percentage; similarly handle missing data
    agg_weekly_df["fourth_down_perc"] = np.where(
        (
            agg_weekly_df["fourth_down_conversions"].notna()
            & agg_weekly_df["fourth_down_attempts"].notna()
        ),
        agg_weekly_df["fourth_down_conversions"] / agg_weekly_df["fourth_down_attempts"],
        np.nan,
    )

    # Calculate opponent's third down conversion percentage; handle missing data
    agg_weekly_df["opponent_third_down_perc"] = np.where(
        (
            agg_weekly_df["opponent_third_down_conversions"].notna()
            & agg_weekly_df["opponent_third_down_attempts"].notna()
        ),
        agg_weekly_df["opponent_third_down_conversions"]
        / agg_weekly_df["opponent_third_down_attempts"],
        np.nan,
    )

    # Calculate opponent's fourth down conversion percentage; handle missing data
    agg_weekly_df["opponent_fourth_down_perc"] = np.where(
        (
            agg_weekly_df["opponent_fourth_down_conversions"].notna()
            & agg_weekly_df["opponent_fourth_down_attempts"].notna()
        ),
        agg_weekly_df["opponent_fourth_down_conversions"]
        / agg_weekly_df["opponent_fourth_down_attempts"],
        np.nan,
    )

    return agg_weekly_df


def merge_aggregated_stats(
    games_df: pd.DataFrame, agg_weekly_away_df: pd.DataFrame, agg_weekly_home_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges aggregated stats with game data for both away and home teams.

    This function merges the aggregated statistics for away and home teams with the main game
    DataFrame. It ensures that each game record is enriched with the corresponding team stats for
    both the away and home teams, facilitating comprehensive analysis.

    Args:
        games_df (pd.DataFrame): DataFrame containing the main game data.
        agg_weekly_away_df (pd.DataFrame): DataFrame with aggregated stats for away teams.
        agg_weekly_home_df (pd.DataFrame): DataFrame with aggregated stats for home teams.

    Returns:
        pd.DataFrame: The merged DataFrame including game data and corresponding team stats.
    """
    # Merge the main game DataFrame with the aggregated stats for away teams
    merged_df = pd.merge(
        games_df,
        agg_weekly_away_df,
        how="left",  # Use left join to keep all records from the main game DataFrame
        on=["away_name", "away_abbr"],  # Join on away team name and abbreviation
    )
    # Merge the result with the aggregated stats for home teams
    merged_df = pd.merge(
        merged_df,
        agg_weekly_home_df,
        how="left",  # Use left join to ensure no game data is lost
        on=["home_name", "home_abbr"],  # Join on home team name and abbreviation
    )
    return merged_df


def calc_stat_diffs(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates differences in stats between away and home teams for analysis.

    This function computes the differences in various statistics between the away and home teams
    for each game, facilitating comparison and analysis of team performance.

    Args:
        merged_df (pd.DataFrame): DataFrame containing merged game data and team stats.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for each of the stat differences.
    """
    # Iterate over predefined aggregate stats to calculate differences between away and home teams
    for stat in AGG_STATS:
        # Calculate and store the difference for each stat in a new column
        merged_df[f"{stat}_diff"] = merged_df[f"away_{stat}"] - merged_df[f"home_{stat}"]

    # Define stats not applicable for opponent comparison
    excluded_stats = ["win_perc", "points_scored", "points_allowed", "time_of_possession"]
    # Filter out excluded stats to focus on opponent-specific stats
    opponent_stats = [stat for stat in AGG_STATS if stat not in excluded_stats]

    # Calculate differences in opponent stats between away and home teams
    for stat in opponent_stats:
        # Calculate and store the difference in opponent stats in a new column
        merged_df[f"opponent_{stat}_diff"] = (
            merged_df[f"away_opponent_{stat}"] - merged_df[f"home_opponent_{stat}"]
        )

    return merged_df


def merge_scores(merged_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges game scores and results into the aggregated data DataFrame.

    This function integrates the game scores and outcomes with the existing merged DataFrame that
    contains detailed game data and team statistics. It aligns the data based on team names and
    abbreviations to ensure consistency across the dataset.

    Args:
        merged_df (pd.DataFrame): DataFrame with merged game data and team stats.
        results_df (pd.DataFrame): DataFrame with game scores and outcomes.

    Returns:
        pd.DataFrame: Enhanced DataFrame with both game details and outcomes.
    """
    # Drop the 'game_lost' column from the results DataFrame as it's not needed for the merge
    results_df.drop(columns=["game_lost"], inplace=True)
    # Rename columns in the results DataFrame for consistency and to facilitate the merge
    # This aligns 'team_name' and 'team_abbr' with 'away_name' and 'away_abbr' respectively
    # It also renames scoring columns to reflect whether they belong to away or home teams
    results_df = results_df.rename(
        columns={
            "team_name": "away_name",  # Align team name for away team
            "team_abbr": "away_abbr",  # Align team abbreviation for away team
            "points_scored": "away_score",  # Rename to indicate away team score
            "points_allowed": "home_score",  # Rename to indicate home team score
            "game_won": "result",  # Rename to indicate the game result from the away team's POV
        }
    )
    # Merge the modified results DataFrame with the merged DataFrame containing game data and stats
    # The merge is based on away team name and abbreviation
    merged_df = pd.merge(
        merged_df,
        results_df,
        how="left",  # Use left join to ensure all records in 'merged_df' are retained
        on=["away_name", "away_abbr"],  # Join on away team name and abbreviation
    )

    return merged_df
