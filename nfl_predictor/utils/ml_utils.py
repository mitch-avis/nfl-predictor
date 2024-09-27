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
