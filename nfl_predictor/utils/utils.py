import os
import sys

import pandas as pd
import sqlalchemy as db
from numpy import ndarray

from nfl_predictor.constants import Constants
from nfl_predictor.utils.logger import log

DATA_PATH = Constants.DATA_PATH
DB_TYPE = os.getenv("DB_TYPE", "postgresql")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1:5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "nfl")
DB_PATH = f"{DB_TYPE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
ENGINE = db.create_engine(DB_PATH)
INSP = db.inspect(ENGINE)


# def read_write_data(data_name: str, func, *args, **kwargs) -> pd.DataFrame:
#     dataframe = pd.DataFrame()
#     # Get dataframe from CSV if it exists
#     if os.path.isfile(f"{DATA_PATH}/{data_name}.csv"):
#         dataframe = read_df_from_csv(f"{data_name}.csv")
#     # Otherwise,
#     if dataframe.empty:
#         # # Get dataframe from DB table if it exists
#         # if INSP.has_table(data_name):
#         #     dataframe = read_df_from_sql(data_name)
#         # # Otherwise, call function to get/create dataframe
#         # else:
#         log.debug(f" * Calling {func.__name__}()")
#         dataframe = pd.DataFrame(func(*args, **kwargs))
#         # Write dataframe to CSV file
#         write_df_to_csv(dataframe, f"{data_name}.csv")
#         # # Write dataframe to DB
#         # write_df_to_sql(dataframe, f"{data_name}")
#     return dataframe


def read_write_data(data_name: str, func, *args, **kwargs) -> pd.DataFrame:
    log.debug(" * Calling %s()", func.__name__)
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
        log.info("%s not found!", data_name)
        sys.exit(1)
    return dataframe


def read_df_from_csv(file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(f"{DATA_PATH}/{file_name}", low_memory=False)
    return dataframe


def write_df_to_csv(dataframe: pd.DataFrame, file_name: str) -> pd.DataFrame:
    dataframe.to_csv(f"{DATA_PATH}/{file_name}", index=True)


def read_df_from_sql(table_name: str) -> pd.DataFrame:
    dataframe = pd.read_sql(table_name, con=ENGINE, index_col="id")
    return dataframe


def write_df_to_sql(dataframe: pd.DataFrame, table_name: str) -> pd.DataFrame:
    dataframe.to_sql(name=table_name, con=ENGINE, index=True, index_label="id", if_exists="replace")


def display(y_pred: ndarray, x_test: pd.DataFrame) -> None:
    for idx, game in enumerate(y_pred):
        away_win_prob = round(game * 100, 2)
        home_win_prob = round((1 - game) * 100, 2)
        week = x_test.reset_index().drop(columns="index").loc[idx, "week"]
        away_team = x_test.reset_index().drop(columns="index").loc[idx, "away_name"]
        home_team = x_test.reset_index().drop(columns="index").loc[idx, "home_name"]
        display_string = (
            f"{'Week ' + str(week):<7}: "
            f"{away_team:<21} ({str(away_win_prob) + '%)':<8} at "
            f"{home_team:<21} ({str(home_win_prob) + '%)':<8}"
        )
        log.info(display_string)


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)  # pylint: disable=C0209
    return df
