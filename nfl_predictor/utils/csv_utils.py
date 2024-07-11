"""
This module provides a comprehensive suite of utilities for handling CSV file operations, tailored
specifically for the NFL predictor project. It encapsulates functionalities for efficiently
reading from, writing to, and updating CSV files, with a particular focus on integration with
pandas DataFrames. These utilities are designed to streamline the import and export processes of
data, ensuring efficient data storage and retrieval mechanisms that are vital for the management
of the project's dataset.

Key Functions:
- read_write_data: Orchestrates the reading of data from a CSV file or its generation using a
  specified function, followed by writing the data back to a CSV file. This function is
  instrumental in managing datasets that require periodic updates or refreshes.
- read_df_from_csv: Facilitates the reading of data from a CSV file into a pandas DataFrame,
  including optional checks for file existence to prevent runtime errors.
- write_df_to_csv: Handles the writing of a pandas DataFrame to a CSV file, ensuring the creation
  of necessary directories and the preservation of DataFrame indices.

Dependencies:
- os: Utilized for file system interactions, such as checking for the existence of files and
  directories.
- pandas (pd): The primary library used for all operations involving reading from and writing to
  CSV files, as well as for manipulating the data within DataFrames.
- constants: Provides access to project-wide constants, including the data storage path, which is
  essential for locating CSV files within the project's directory structure.
- logger (log): Employed for logging informational messages and errors, facilitating debugging and
  operational monitoring.

Usage Scenario:
This module is integral to the NFL predictor project, finding application across various stages
where data needs to be ingested from or persisted to CSV files. It supports critical operations
such as loading historical game data, storing processed data for future analysis, and exporting
data for external use or sharing.

By abstracting the complexities of CSV file handling and integrating closely with pandas
DataFrames, this module significantly contributes to the project's data management efficiency and
reliability.
"""

import os
import sys
from typing import Callable

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

DATA_PATH = constants.DATA_PATH


def read_write_data(
    data_name: str, func: Callable, *args, force_refresh: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Reads data from a CSV file or generates it using a specified function, then writes it back.

    This function checks if a CSV file with the given data name exists. If it does and
    force_refresh is False, it reads the data from the file. Otherwise, it generates the data by
    calling the provided function and writes the new data to a CSV file.

    Args:
        data_name (str): The base name of the data file (without extension).
        func (Callable): The function to generate data if needed.
        *args: Positional arguments to pass to the data generation function.
        force_refresh (bool, optional): If True, forces data regeneration. Defaults to False.
        **kwargs: Keyword arguments to pass to the data generation function.

    Returns:
        pd.DataFrame: The data as a pandas DataFrame.
    """
    # Initialize an empty DataFrame
    dataframe = pd.DataFrame()
    file_path = f"{DATA_PATH}/{data_name}.csv"

    # Check if the CSV file exists and read it if force_refresh is not True
    if os.path.isfile(file_path) and not force_refresh:
        dataframe = read_df_from_csv(file_path, check_exists=False)

    # If the DataFrame is empty (file doesn't exist) or force_refresh is True, generate the data
    if dataframe.empty or force_refresh:
        log.debug("* Calling %s()", func.__name__)
        dataframe = pd.DataFrame(func(*args, **kwargs))
        # Write the generated DataFrame to a CSV file
        write_df_to_csv(dataframe, file_path)

    return dataframe


def read_df_from_csv(file_path: str, check_exists: bool = True) -> pd.DataFrame:
    """
    Reads a DataFrame from a CSV file.

    If check_exists is True, the function first checks if the file exists. If it does not, logs an
    error message and exits the program.

    Args:
        file_path (str): The path to the CSV file.
        check_exists (bool, optional):  Whether to check if the file exists before reading. Defaults
                                        to True.

    Returns:
        pd.DataFrame: The data read from the CSV file.
    """
    # Check if the file exists, if required
    if check_exists and not os.path.isfile(file_path):
        # Log an error message and exit if the file does not exist
        log.info("%s not found!", os.path.basename(file_path))
        sys.exit(1)

    # Read the CSV file into a DataFrame, using the first column as the index
    dataframe = pd.read_csv(file_path, index_col=0)
    return dataframe


def write_df_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Writes a DataFrame to a CSV file.

    If the directory for the file does not exist, it is created. The DataFrame is then written to
    the file, including the index.

    Args:
        dataframe (pd.DataFrame): The DataFrame to write.
        file_path (str): The path to the CSV file where the data should be written.
    """
    # Construct the directory path for the CSV file
    directory_path = os.path.dirname(file_path)

    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Write the DataFrame to the CSV file, including the index
    dataframe.to_csv(file_path, index=True)
