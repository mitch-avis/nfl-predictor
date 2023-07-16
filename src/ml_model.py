#!/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from utils.logger import log
from utils.utils import display, get_dataframe

NUM_ROUNDS = 1000


def main():
    year = 2022
    current_week = 18
    comp_games_name = f"{year}_comp_games"
    comp_games_df = get_dataframe(comp_games_name)
    pred_games_name = f"{year}_{current_week}_pred_games"
    pred_games_df = get_dataframe(pred_games_name)
    predict_games(comp_games_df, pred_games_df)


def predict_games(comp_games_df: pd.DataFrame, pred_games_df: pd.DataFrame) -> None:
    msk = np.random.rand(len(comp_games_df)) < 0.8
    train_df = comp_games_df[msk]
    test_df = comp_games_df[~msk]

    X_train = train_df.drop(
        columns=["away_name", "away_abbr", "home_name", "home_abbr", "week", "result"]
    )
    y_train = train_df[["result"]]
    X_test = test_df.drop(
        columns=["away_name", "away_abbr", "home_name", "home_abbr", "week", "result"]
    )
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

    clf.fit(X_train, np.ravel(y_train.values))
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    display(y_pred, test_df)
    dtest = xgb.DMatrix(X_test, y_test, feature_names=X_test.columns)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns)
    param = {
        "booster": "gblinear",
        "verbosity": 1,
        "nthread": -1,
        "learning_rate": 0.01,
        "feature_selector": "shuffle",
        "objective": "binary:hinge",
        "eval_metric": "error",
    }
    watchlist = [(dtest, "eval"), (dtrain, "train")]
    bst = xgb.train(param, dtrain, NUM_ROUNDS, watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    error = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(
        len(preds)
    )
    log.debug(f"error={error}")
    X_test = pred_games_df.drop(
        columns=["away_name", "away_abbr", "home_name", "home_abbr", "week", "result"]
    )
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    display(y_pred, pred_games_df)


if __name__ == "__main__":
    main()
