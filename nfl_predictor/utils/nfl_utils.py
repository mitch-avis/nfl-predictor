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

import numpy as np
import pandas as pd
from sportsipy.nfl.boxscore import Boxscore

from nfl_predictor import constants

DATA_PATH = constants.DATA_PATH
SEASON_END_MONTH = constants.SEASON_END_MONTH
WEEKS_BEFORE_2021 = constants.WEEKS_BEFORE_2021
WEEKS_FROM_2021_ONWARDS = constants.WEEKS_FROM_2021_ONWARDS
ELO_DATA_URL = constants.ELO_DATA_URL
BOXSCORE_STATS = constants.BOXSCORE_STATS


def fetch_nfl_elo_ratings() -> pd.DataFrame:
    """
    Fetches the latest ELO ratings for NFL teams from a specified URL and returns them as a
    DataFrame.

    Returns:
        pd.DataFrame: The ELO ratings for NFL teams.
    """
    # Utilize pandas to directly read the ELO ratings CSV from the URL into a DataFrame
    elo_df = pd.read_csv(ELO_DATA_URL)
    return elo_df


def get_season_start(year: int) -> date:
    """
    Calculates the NFL season start date for a given year, which is the Thursday following the
    first Monday of September.

    Args:
        year (int): The year for which the season start date is calculated.

    Returns:
        date: The calculated start date of the NFL season for the given year.
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
    Determines the NFL week number for a given date, considering the season start date and the
    structure of the NFL season.

    Args:
        given_date (date): The date for which the NFL week number is determined.

    Returns:
        int: The NFL week number for the given date.
    """
    # Determine the given season's start date
    season_start = get_season_start(given_date.year)

    # If given_date is before season's start, check if it's closer to the previous season's start
    if given_date < season_start:
        previous_season_start = get_season_start(given_date.year - 1)
        # If today is after the previous season start, use it
        if given_date >= previous_season_start:
            # Treat the offseason after the previous season as the start of the new season
            return 1
        # Indicates the date is before the previous season's start
        return 0

    # Calculate the week number, adjusting for the season structure
    week_number = ((given_date - season_start).days // 7) + 1

    # Ensure week number falls within the 1 to 18 range; otherwise, return 1 for offseason
    return max(1, min(week_number, 18)) if week_number > 0 else 1


def determine_weeks_to_scrape(season: int, include_future_weeks: bool = False) -> list:
    """
    Determines the weeks to scrape for a given NFL season based on the current date, the
    structure of the NFL season, and whether to include future weeks.

    Args:
        season (int): The NFL season year.
        include_future_weeks (bool):    If True, includes all weeks for the current or future
                                        seasons, ignoring the current date.

    Returns:
        list: A list of weeks to scrape for the specified season.
    """
    today = date.today()
    # Determine the current NFL season based on today's date.
    current_season = today.year if today.month > SEASON_END_MONTH else today.year - 1

    if season < 2021:
        # For seasons before 2021, the NFL had 17 weeks.
        weeks_to_scrape = list(range(1, WEEKS_BEFORE_2021 + 1))
    elif season <= current_season or include_future_weeks:
        # For past seasons from 2021 onwards and the current season if include_future_weeks is True,
        # the NFL has 18 weeks. Also applies to future seasons if include_future_weeks is True.
        weeks_to_scrape = list(range(1, WEEKS_FROM_2021_ONWARDS + 1))
    else:
        # For the current season without including future weeks, scrape up to the current week.
        # This case is only relevant if include_future_weeks is False.
        if season == current_season:
            current_week = determine_nfl_week_by_date(today)
            weeks_to_scrape = list(range(1, current_week + 1))
        else:
            # For future seasons without including future weeks, no weeks to scrape.
            weeks_to_scrape = []

    return weeks_to_scrape


def init_team_stats_dfs(
    game_df: pd.DataFrame, base_columns: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares DataFrames for away and home teams based on game data and specified base columns.

    Args:
        game_df (pd.DataFrame): The DataFrame containing basic game information.
        base_columns (list): The list of base column names to include in the team DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames for away and home teams, respectively.
    """
    # Select and rename columns for the away team
    away_team_df = game_df.loc[:, ["away_name", "away_abbr", "away_score"]].copy()
    away_team_df.columns = base_columns

    # Select and rename columns for the home team in a similar manner
    home_team_df = game_df.loc[:, ["home_name", "home_abbr", "home_score"]].copy()
    home_team_df.columns = base_columns

    return away_team_df, home_team_df


def compute_game_outcomes(
    away_team_df: pd.DataFrame, home_team_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates game outcomes (win, loss, tie) for away and home teams and updates the DataFrames
    with these outcomes.

    Args:
        away_team_df (pd.DataFrame): The DataFrame for the away team.
        home_team_df (pd.DataFrame): The DataFrame for the home team.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The updated DataFrames for away and home teams with
        game outcomes.
    """
    # Determine if the game has been played based on the presence of NaN in 'points_scored'
    game_not_played = away_team_df["points_scored"].isna() | home_team_df["points_scored"].isna()

    # Calculate 'game_won': 1 for win, 0.5 for tie, 0 for loss, NaN if not played
    away_team_df["game_won"] = np.where(
        game_not_played,
        np.nan,
        np.where(
            away_team_df["points_scored"] > home_team_df["points_scored"],
            1,
            np.where(away_team_df["points_scored"] < home_team_df["points_scored"], 0, 0.5),
        ),
    )
    home_team_df["game_won"] = 1 - away_team_df["game_won"].fillna(0.5)  # Reflects opposite outcome

    # Calculate 'game_lost': Opposite of 'game_won', with NaN preserved for games not played
    away_team_df["game_lost"] = np.where(game_not_played, np.nan, 1 - away_team_df["game_won"])
    home_team_df["game_lost"] = np.where(game_not_played, np.nan, 1 - home_team_df["game_won"])

    return away_team_df, home_team_df


def create_stats_dfs_from_boxscore(
    boxscore: Boxscore,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares DataFrames for away and home team statistics from a Boxscore object.

    Args:
        boxscore (Boxscore): The Boxscore object containing detailed game statistics.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing statistics for away and home
        teams, respectively.
    """
    # Define the columns for the DataFrames, adjusting 'points' to 'points_allowed'
    columns = [col if col != "points" else "points_allowed" for col in BOXSCORE_STATS]

    if boxscore is not None:
        # Extract and format detailed statistics for both teams from the boxscore
        away_stats_df = create_stats_df(boxscore.dataframe, "away")
        home_stats_df = create_stats_df(boxscore.dataframe, "home")
    else:
        # Create empty DataFrames with the defined columns if boxscore is None
        away_stats_df = pd.DataFrame(columns=columns).astype(float)
        home_stats_df = pd.DataFrame(columns=columns).astype(float)

    return away_stats_df, home_stats_df


def create_stats_df(game_stats_df: pd.DataFrame, team_prefix: str) -> pd.DataFrame:
    """
    Creates a DataFrame of team statistics from game statistics, selecting and renaming columns
    based on a team prefix.

    Args:
        game_stats_df (pd.DataFrame): The DataFrame containing game statistics.
        team_prefix (str): The prefix indicating whether the team is 'away' or 'home'.

    Returns:
        pd.DataFrame: The DataFrame containing selected and renamed statistics for the team.
    """
    # Define a mapping from original to desired column names, including the workaround for the
    # Boxscore bug by renaming "points" to "points_allowed".
    column_mapping = {
        f"{team_prefix}_{stat}": (stat if stat != "points" else "points_allowed")
        for stat in BOXSCORE_STATS
    }

    # Apply the column mapping to select and rename columns, ensuring a consistent structure
    # and addressing the Boxscore bug by correctly representing defensive statistics.
    transformed_df = game_stats_df[list(column_mapping.keys())].rename(columns=column_mapping)

    return transformed_df.reset_index(drop=True)


def merge_and_format_df(
    team_df: pd.DataFrame, team_stats_df: pd.DataFrame, opponent_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges team and opponent statistics into a single DataFrame and applies formatting.

    Args:
        team_df (pd.DataFrame): The DataFrame containing team data.
        team_stats_df (pd.DataFrame): The DataFrame containing the team's statistics.
        opponent_stats_df (pd.DataFrame): The DataFrame containing the opponent's statistics.

    Returns:
        pd.DataFrame: The merged and formatted DataFrame containing team and opponent statistics.
    """
    # Rename opponent's statistics columns for clarity
    opponent_stats_df = opponent_stats_df.rename(columns=lambda x: f"opponent_{x}")

    # Merge the team's DataFrame with its own and the opponent's statistics
    merged_df = pd.concat([team_df, team_stats_df, opponent_stats_df], axis=1)

    # Apply formatting functions
    merged_df = convert_top_to_seconds_in_df(merged_df)  # Convert time of possession to seconds
    merged_df = reorder_and_drop_columns(merged_df)  # Reorder and drop columns as needed

    return merged_df


def convert_top_to_seconds_in_df(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts time of possession from "MM:SS" format to seconds in a DataFrame.

    Args:
        team_df (pd.DataFrame): The DataFrame containing team data with time of possession.

    Returns:
        pd.DataFrame: The DataFrame with time of possession converted to seconds.
    """
    if "time_of_possession" in team_df.columns:
        # Convert "MM:SS" format to seconds, handling NaN values appropriately.
        team_df["time_of_possession"] = team_df["time_of_possession"].apply(
            lambda x: (
                (int(x.split(":")[0]) * 60 + int(x.split(":")[1])) if pd.notnull(x) else np.nan
            )
        )
    return team_df


def reorder_and_drop_columns(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders and drops specific columns in a DataFrame based on predefined criteria.

    Args:
        team_df (pd.DataFrame): The DataFrame to be reordered and modified.

    Returns:
        pd.DataFrame: The modified DataFrame with columns reordered and unnecessary columns
        dropped.
    """
    # Define the base columns to prioritize in the DataFrame
    base_columns = [
        "team_name",
        "team_abbr",
        "game_won",
        "game_lost",
        "points_scored",
        "points_allowed",
    ]
    # Identify additional columns to include, excluding specific opponent-related columns
    other_columns = [
        col
        for col in team_df.columns
        if col not in base_columns
        and col not in ("opponent_points_allowed", "opponent_time_of_possession")
    ]
    # Combine the base and additional columns for the final order
    ordered_columns = base_columns + other_columns
    # Return the DataFrame with columns reordered according to the specified order
    return team_df[ordered_columns]
