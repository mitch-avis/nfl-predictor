import os

import pandas as pd
import sqlalchemy as db
from definitions import DATA_PATH
from logger import log
from numpy import ndarray

DB_TYPE = os.getenv("DB_TYPE", "postgresql")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1:5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "nfl")
DB_PATH = f"{DB_TYPE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
ENGINE = db.create_engine(DB_PATH)
INSP = db.inspect(ENGINE)


def read_write_data(data_name: str, func, *args, **kwargs) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    # Get dataframe from CSV if it exists
    if os.path.isfile(f"{DATA_PATH}/{data_name}.csv"):
        dataframe = read_df_from_csv(f"{data_name}.csv")
    # Otherwise,
    if dataframe.empty:
        # Get dataframe from DB table if it exists
        if INSP.has_table(data_name):
            dataframe = read_df_from_sql(data_name)
        # Otherwise, call function to get/create dataframe
        else:
            log.debug(f"Calling {func.__name__} with args [{args}]...")
            dataframe = pd.DataFrame(func(*args, **kwargs))
        # Write dataframe to CSV file
        write_df_to_csv(dataframe, f"{data_name}.csv")
    # Write dataframe to DB
    write_df_to_sql(dataframe, f"{data_name}")
    return dataframe


def get_dataframe(data_name: str) -> pd.DataFrame:
    if os.path.isfile(f"{DATA_PATH}/{data_name}.csv"):
        dataframe = read_df_from_csv(f"{data_name}.csv")
    elif INSP.has_table(data_name):
        dataframe = read_df_from_sql(data_name)
    else:
        log.info(f"{data_name} not found!")
        exit(1)
    return dataframe


def read_df_from_csv(file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(f"{DATA_PATH}/{file_name}")
    return dataframe


def write_df_to_csv(dataframe: pd.DataFrame, file_name: str) -> pd.DataFrame:
    dataframe.to_csv(f"{DATA_PATH}/{file_name}", index=False)


def read_df_from_sql(table_name: str) -> pd.DataFrame:
    dataframe = pd.read_sql(table_name, con=ENGINE, index_col="id")
    return dataframe


def write_df_to_sql(dataframe: pd.DataFrame, table_name: str) -> pd.DataFrame:
    dataframe.to_sql(name=table_name, con=ENGINE, index=True, index_label="id", if_exists="replace")


def display(y_pred: ndarray, x_test: pd.DataFrame) -> None:
    for game in range(len(y_pred)):
        win_prob = round(y_pred[game] * 100, 2)
        week = x_test.reset_index().drop(columns="index").loc[game, "week"]
        away_team = x_test.reset_index().drop(columns="index").loc[game, "away_name"]
        home_team = x_test.reset_index().drop(columns="index").loc[game, "home_name"]
        print(
            f"Week {week}: The {away_team} have a probability of {win_prob}% of beating the "
            f"{home_team}."
        )
