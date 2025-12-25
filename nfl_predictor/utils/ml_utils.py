"""
Module for machine learning utilities in the NFL predictor project.

This module contains functions for displaying predictions, flattening nested dictionaries, and
converting nested dictionaries to pandas DataFrames. These utilities support the analysis and
presentation of machine learning model predictions, as well as the manipulation of complex data
structures.
"""

import numpy as np
import pandas as pd

from nfl_predictor.utils.logger import log


def display_predictions(y_pred: np.ndarray, x_test: pd.DataFrame) -> None:
    """
    Displays the predictions for NFL games in a readable format.

    This function takes an array of predictions and the corresponding test dataset, then logs
    the predicted probabilities for each game along with game details extracted from the test
    dataset.

    Args:
        y_pred (np.ndarray):    The array of predictions, with each element being the probability
                                of the away team winning.
        x_test (pd.DataFrame):  The test dataset containing details of the games, including week
                                number and team names.
    """
    # Reset the index of x_test once to avoid repeated operations
    x_test_reset = x_test.reset_index().drop(columns="index")

    for idx, game in enumerate(y_pred):
        # Calculate probabilities
        away_win_prob = round(game * 100, 2)
        home_win_prob = round((1 - game) * 100, 2)

        # Extract game details
        season = x_test_reset.loc[idx, "season"]
        week = x_test_reset.loc[idx, "week"]
        away_team = x_test_reset.loc[idx, "away_name"]
        home_team = x_test_reset.loc[idx, "home_name"]

        # Format and log the display string
        display_string = (
            f"Season: {season} {'Week ' + str(week):<7}: "
            f"{away_team:<21} ({str(away_win_prob) + '%)':<8} at "
            f"{home_team:<21} ({str(home_win_prob) + '%)':<8}"
        )
        log.info(display_string)


def display_weekly_predictions(predictions: pd.DataFrame) -> None:
    """
    Displays weekly score predictions with confidence ranks and win probabilities.

    Args:
        predictions (pd.DataFrame): DataFrame containing predicted scores, win probabilities,
            and confidence ranks.
    """
    if predictions.empty:
        log.info("No predictions to display.")
        return

    team_columns = None
    for candidates in (("away_abbr", "home_abbr"), ("away_name", "home_name")):
        if all(col in predictions.columns for col in candidates):
            team_columns = candidates
            break

    if team_columns is None:
        log.info("Missing team columns for pretty output.")
        return

    away_col, home_col = team_columns
    sorted_df = predictions.copy()
    if "confidence_rank" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("confidence_rank", ascending=False)
    elif "confidence_strength" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("confidence_strength", ascending=False)

    season = sorted_df["season"].iloc[0] if "season" in sorted_df.columns else None
    week = sorted_df["week"].iloc[0] if "week" in sorted_df.columns else None
    if season is not None and week is not None:
        log.info("Weekly predictions (season %s week %s)", season, week)
    else:
        log.info("Weekly predictions")

    for _, row in sorted_df.iterrows():
        away_team = str(row[away_col])
        home_team = str(row[home_col])
        away_score = row.get("predicted_away_score", np.nan)
        home_score = row.get("predicted_home_score", np.nan)
        rank = row.get("confidence_rank", np.nan)
        winner = row.get("predicted_winner")

        if winner is None or pd.isna(winner):
            winner = home_team if home_score >= away_score else away_team

        home_prob = row.get("home_win_prob")
        away_prob = row.get("away_win_prob")
        win_prob = None
        if winner == home_team and pd.notna(home_prob):
            win_prob = home_prob
        if winner == away_team and pd.notna(away_prob):
            win_prob = away_prob

        rank_str = f"{int(rank):>2}" if pd.notna(rank) else "--"
        away_score_str = f"{away_score:>5.1f}" if pd.notna(away_score) else "  n/a"
        home_score_str = f"{home_score:>5.1f}" if pd.notna(home_score) else "  n/a"
        prob_str = f"{win_prob * 100:>5.1f}%" if win_prob is not None else "  n/a"

        log.info(
            "#%s (%s) %s %s @ %s %s | win %s",
            rank_str,
            winner,
            away_team,
            away_score_str,
            home_team,
            home_score_str,
            prob_str,
        )


def flatten_dict(nested_dict):
    """
    Recursively flattens a nested dictionary.

    Each key in the resulting dictionary is a tuple representing the path to the corresponding
    value in the nested dictionary. This function is useful for converting complex, nested data
    structures into a flat form that can be more easily manipulated or analyzed.

    Args:
        nested_dict (dict): The nested dictionary to be flattened.

    Returns:
        dict:   A flat dictionary where keys are tuples representing paths in the original nested
                dictionary.
    """
    res = {}
    # Check if the input is a dictionary
    if isinstance(nested_dict, dict):
        # Iterate through each item in the dictionary
        for k, v in nested_dict.items():
            # Recursively flatten the dictionary
            flattened_dict = flatten_dict(v)
            for key, val in flattened_dict.items():
                # Prepend the current key to the tuple key from the nested dictionary
                res[(k,) + key] = val
    else:
        # Base case: if it's not a dictionary, return it wrapped in a tuple
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    """
    Converts a nested dictionary into a pandas DataFrame.

    This function first flattens the nested dictionary, then converts it into a DataFrame. The
    DataFrame's columns represent the last level of keys in the original nested dictionary, and
    the index represents the hierarchical structure of the original keys.

    Args:
        values_dict (dict): The nested dictionary to be converted.

    Returns:
        pd.DataFrame:   A DataFrame representation of the nested dictionary, with hierarchical
                        indices and columns based on the original dictionary's structure.
    """
    # Flatten the nested dictionary
    flat_dict = flatten_dict(values_dict)
    # Convert the flat dictionary to a DataFrame
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    # Convert the index to a MultiIndex
    df.index = pd.MultiIndex.from_tuples(df.index)
    # Unstack the DataFrame based on the last level of the tuple keys
    df = df.unstack(level=-1)
    # Format the column names to only include the last level of the tuple keys
    df.columns = df.columns.map(lambda x: f"{x[1]}")
    return df
