#!/bin/env python

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


def predict_games(pred_games_df, test_df, x_train, y_train, x_test, y_test, rounds):
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
    display(y_pred, test_df)
    # accuracy = accuracy_score(y_test, np.round(y_pred))

    dtest = xgb.DMatrix(x_test, y_test, feature_names=x_test.columns)
    dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns)
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
    bst = xgb.train(param, dtrain, rounds, watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    print(
        "error=%f"
        % (
            sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
            / float(len(preds))
        )
    )
    X_test = pred_games_df.drop(
        columns=["away_name", "away_abbr", "home_name", "home_abbr", "week", "result"]
    )
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    display(y_pred, pred_games_df)


def display(y_pred, x_test):
    for game in range(len(y_pred)):
        win_prob = round(y_pred[game] * 100, 2)
        week = x_test.reset_index().drop(columns="index").loc[game, "week"]
        away_team = x_test.reset_index().drop(columns="index").loc[game, "away_name"]
        home_team = x_test.reset_index().drop(columns="index").loc[game, "home_name"]
        print(
            f"Week {week}: The {away_team} had a probability of {win_prob}% of beating the "
            f"{home_team}."
        )


def main():
    return


if __name__ == "__main__":
    main()
