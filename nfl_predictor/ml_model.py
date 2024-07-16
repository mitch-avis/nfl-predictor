"""
_summary_
"""

from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from nfl_predictor import constants
from nfl_predictor.utils.csv_utils import read_df_from_csv
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.ml_utils import display_predictions
from nfl_predictor.utils.nfl_utils import determine_nfl_week_by_date

NUM_ROUNDS = 1000


def main() -> None:
    """
    _summary_
    """
    today = date.today()
    current_week = determine_nfl_week_by_date(today)
    completed_games_df_name = f"{constants.DATA_PATH}/completed_games.csv"
    completed_games_df = read_df_from_csv(completed_games_df_name, check_exists=True)
    pred_games_name = f"{constants.DATA_PATH}/predict/week_{current_week:>02}_games_to_predict.csv"
    games_to_predict_df = read_df_from_csv(pred_games_name, check_exists=True)
    predict_games(completed_games_df, games_to_predict_df)


def predict_games(comp_games_df: pd.DataFrame, pred_games_df: pd.DataFrame) -> None:
    """
    _summary_
    """
    msk = np.random.rand(len(comp_games_df)) < 0.8
    train_df = comp_games_df[msk]
    test_df = comp_games_df[~msk]

    x_train = train_df.drop(columns=constants.ML_DROP_COLS)
    y_train = train_df[["result"]]
    x_test = test_df.drop(columns=constants.ML_DROP_COLS)
    y_test = test_df[["result"]]

    clf = LogisticRegression(
        penalty="l1",
        dual=False,
        tol=0.001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight="balanced",
        random_state=None,
        solver="liblinear",
        max_iter=1000,
        multi_class="ovr",
        verbose=0,
    )

    clf.fit(x_train, np.ravel(y_train.values))
    y_pred = clf.predict_proba(x_test)
    y_pred = y_pred[:, 1]

    display_predictions(y_pred, test_df)
    dtest = xgb.DMatrix(x_test, y_test, feature_names=x_test.columns)
    dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns)
    params = {
        "booster": "gblinear",
        "verbosity": 1,
        "nthread": -1,
        "learning_rate": 0.01,
        "feature_selector": "shuffle",
        "objective": "binary:hinge",
        "eval_metric": "error",
    }
    watchlist = [(dtest, "eval"), (dtrain, "train")]
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=NUM_ROUNDS, evals=watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    error = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(
        len(preds)
    )
    log.debug("error=%s", error)
    x_test = pred_games_df.drop(columns=constants.ML_DROP_COLS)
    y_pred = clf.predict_proba(x_test)
    y_pred = y_pred[:, 1]

    display_predictions(y_pred, pred_games_df)


if __name__ == "__main__":
    main()
