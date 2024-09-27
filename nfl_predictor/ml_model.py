"""
_summary_
"""

import time
from datetime import date
from typing import Tuple

# import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.pylab import rcParams
from scipy import stats
from sklearn import metrics, preprocessing
from sklearn.model_selection import (  # GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from nfl_predictor import constants
from nfl_predictor.utils.csv_utils import read_df_from_csv
from nfl_predictor.utils.logger import log

# from nfl_predictor.utils.ml_utils import display_predictions
from nfl_predictor.utils.nfl_utils import determine_nfl_week_by_date

# Set up matplotlib
rcParams["figure.figsize"] = 12, 4

# Initialize scalers
STANDARD_SCALER = preprocessing.StandardScaler()
POWER_TRANSFORMER = preprocessing.PowerTransformer()


def main() -> None:
    """
    Run the main function to predict the outcomes of upcoming NFL games.

    This function reads the completed games data and the upcoming week's games data from CSV files,
    preprocesses the data, trains XGBoost regression models for predicting the away and home scores,
    evaluates the models, and predicts the scores for the upcoming games.

    Args:
        None

    Returns:
        None
    """
    # Get today's date and determine the current NFL week
    today = date.today()
    current_week = determine_nfl_week_by_date(today)

    # Read the completed games data from CSV
    completed_games_df_name = f"{constants.DATA_PATH}/completed_games.csv"
    completed_games_df = read_df_from_csv(completed_games_df_name, check_exists=True)

    # Read the upcoming week's games data from CSV
    pred_games_name = f"{constants.DATA_PATH}/predict/week_{current_week:>02}_games_to_predict.csv"
    games_to_predict_df = read_df_from_csv(pred_games_name, check_exists=True)

    # Predict the outcomes of the upcoming games
    predict_games_new(completed_games_df, games_to_predict_df)


def predict_games_new(completed_games: pd.DataFrame, week_games_to_predict: pd.DataFrame) -> None:
    """Predict the outcomes of upcoming NFL games.

    Args:
        completed_games (pd.DataFrame): DataFrame containing historical game data.
        week_games_to_predict (pd.DataFrame):   DataFrame containing data for the upcoming week's
                                                games.
    """
    # pylint: disable=too-many-locals
    # Preprocessing
    features = completed_games.drop(columns=constants.ML_DROP_COLS)
    target_regression_away = completed_games["away_score"].astype(int)
    target_regression_home = completed_games["home_score"].astype(int)
    # target_regression_total = (
    #     completed_games["away_score"] + completed_games["home_score"]
    # ).astype(int)

    features_scaled, normal_features = scale_features(features)

    # Train/Test Split
    x_train_away, x_test_away, y_train_away, y_test_away = train_test_split(
        features_scaled, target_regression_away, test_size=0.3, random_state=42
    )
    x_train_home, x_test_home, y_train_home, y_test_home = train_test_split(
        features_scaled, target_regression_home, test_size=0.3, random_state=42
    )

    # Hyperparameter tuning for XGBoost
    params = {
        "n_estimators": [int(i) for i in np.linspace(200, 400, 10)],
        "eta": list(np.linspace(0.001, 0.02, 10)),
        "gamma": list(np.linspace(0.5, 0.7, 10)),
        "max_depth": list(np.linspace(1, 10, 10)),
        "min_child_weight": list(np.linspace(6.0, 8.0, 10)),
        "subsample": list(np.linspace(0.4, 0.6, 10)),
        "colsample_bytree": list(np.linspace(0.4, 0.6, 10)),
    }
    folds = 5
    param_comb = 100
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Regression Model for Away Score
    reg_away = xgb.XGBRegressor(objective="reg:squarederror", nthread=1)
    random_search_away = RandomizedSearchCV(
        reg_away,
        param_distributions=params,
        n_iter=param_comb,
        scoring="neg_mean_squared_error",
        n_jobs=2,
        cv=skf.split(x_train_away, y_train_away),
        verbose=3,
        random_state=42,
    )
    start_time = time.time()
    random_search_away.fit(x_train_away, y_train_away)
    log.info("RandomizedSearchCV took %.2f seconds.", (time.time() - start_time))
    log.info("Best Parameters for Away Score: %s", random_search_away.best_params_)

    # Regression Model for Home Score
    reg_home = xgb.XGBRegressor(objective="reg:squarederror", nthread=1)
    random_search_home = RandomizedSearchCV(
        reg_home,
        param_distributions=params,
        n_iter=param_comb,
        scoring="neg_mean_squared_error",
        n_jobs=2,
        cv=skf.split(x_train_home, y_train_home),
        verbose=3,
        random_state=42,
    )
    start_time = time.time()
    random_search_home.fit(x_train_home, y_train_home)
    log.info("RandomizedSearchCV took %.2f seconds.", (time.time() - start_time))
    log.info("Best Parameters for Home Score: %s", random_search_home.best_params_)

    # Evaluate away model
    best_reg_away = random_search_away.best_estimator_
    y_pred_away = best_reg_away.predict(x_test_away)
    mae_away = metrics.mean_absolute_error(y_test_away, y_pred_away)
    log.info("Away Score RMAE: %s", np.sqrt(mae_away))

    # Evaluate home model
    best_reg_home = random_search_home.best_estimator_
    y_pred_home = best_reg_home.predict(x_test_home)
    mae_home = metrics.mean_absolute_error(y_test_home, y_pred_home)
    log.info("Home Score RMAE: %s", mae_home)

    # Predict upcoming games
    week_games_features = week_games_to_predict.drop(columns=constants.ML_DROP_COLS)
    week_games_scaled = np.empty_like(week_games_features)
    week_games_scaled[:, normal_features] = STANDARD_SCALER.transform(
        week_games_features.iloc[:, normal_features]
    )
    week_games_scaled[:, ~normal_features] = POWER_TRANSFORMER.transform(
        week_games_features.iloc[:, ~normal_features]
    )

    predicted_away_scores = best_reg_away.predict(week_games_scaled)
    predicted_home_scores = best_reg_home.predict(week_games_scaled)

    # Add predictions to dataframe
    week_games_to_predict["predicted_away_score"] = np.round(predicted_away_scores, 1)
    week_games_to_predict["predicted_home_score"] = np.round(predicted_home_scores, 1)

    log.info(week_games_to_predict)


def scale_features(features: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Scales the features of a DataFrame using StandardScaler for normally distributed features and
    PowerTransformer for skewed features.

    Args:
        features (pd.DataFrame): DataFrame containing the features to be scaled.

    Returns:
        tuple
            - np.ndarray: Scaled features as a NumPy array.
            - pd.Series: Boolean series indicating which features are normally distributed.
    """
    # Identify normally distributed features
    p_values = features.apply(lambda x: stats.normaltest(x).pvalue)
    normal_features = p_values > 0.05

    # Apply StandardScaler to normally distributed features
    features_normal_scaled = STANDARD_SCALER.fit_transform(features.loc[:, normal_features])

    # Apply PowerTransformer to skewed features
    features_skewed_scaled = POWER_TRANSFORMER.fit_transform(features.loc[:, ~normal_features])

    # Combine the scaled features back into a single dataset
    features_scaled = np.empty_like(features)
    features_scaled[:, normal_features] = features_normal_scaled
    features_scaled[:, ~normal_features] = features_skewed_scaled

    return features_scaled, normal_features


# def modelfit(alg, dtrain, predictors, use_train_cv=True, cv_folds=5, early_stopping_rounds=50):
#     """_summary_

#     Args:
#         alg (_type_): _description_
#         dtrain (_type_): _description_
#         predictors (_type_): _description_
#         useTrainCV (bool, optional): _description_. Defaults to True.
#         cv_folds (int, optional): _description_. Defaults to 5.
#         early_stopping_rounds (int, optional): _description_. Defaults to 50.
#     """
#     if use_train_cv:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(
#             xgb_param,
#             xgtrain,
#             num_boost_round=alg.get_params()["n_estimators"],
#             nfold=cv_folds,
#             metrics="auc",
#             early_stopping_rounds=early_stopping_rounds,
#         )
#         alg.set_params(n_estimators=cvresult.shape[0])

#     # Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain["Disbursed"], eval_metric="auc")

#     # Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

#     # Print model report:
#     log.info("\nModel Report")
#     log.info(
#         "Accuracy : %.4g", metrics.accuracy_score(dtrain["Disbursed"].values, dtrain_predictions)
#     )
#     log.info("AUC Score (Train): %f", metrics.roc_auc_score(dtrain["Disbursed"], dtrain_predprob))

#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind="bar", title="Feature Importances")
#     plt.ylabel("Feature Importance Score")


if __name__ == "__main__":
    main()
