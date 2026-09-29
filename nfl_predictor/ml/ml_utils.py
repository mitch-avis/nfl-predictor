"""ML-oriented utility helpers.

This module contains helpers used by ML prediction tooling (e.g., pretty-printing weekly
predictions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.ml.metrics import confidence_ranks
from nfl_predictor.utils.logger import log


def display_predictions(y_pred: np.ndarray, x_test: pd.DataFrame) -> None:
    """Log per-game win probability predictions.

    Args:
        y_pred: Array of predicted away-team win probabilities.
        x_test: Test dataset containing game details.

    """
    x_test_reset = x_test.reset_index().drop(columns="index")

    for idx, game in enumerate(y_pred):
        away_win_prob = round(float(game) * 100, 2)
        home_win_prob = round((1 - float(game)) * 100, 2)

        season = x_test_reset.loc[idx, "season"]
        week = x_test_reset.loc[idx, "week"]
        away_team = x_test_reset.loc[idx, "away_name"]
        home_team = x_test_reset.loc[idx, "home_name"]

        display_string = (
            f"Season: {season} {'Week ' + str(week):<7}: "
            f"{away_team:<21} ({str(away_win_prob) + '%)':<8} at "
            f"{home_team:<21} ({str(home_win_prob) + '%)':<8}"
        )
        log.info(display_string)


def _team_columns(predictions: pd.DataFrame) -> tuple[str, str] | None:
    """Return the away and home team columns: abbreviations, else names."""
    for candidates in (("away_abbr", "home_abbr"), ("away_name", "home_name")):
        if all(col in predictions.columns for col in candidates):
            return candidates
    return None


def _most_confident_first(predictions: pd.DataFrame) -> pd.DataFrame:
    """Order games from the most confident pick down, ranking them when no rank is given."""
    if "confidence_rank" in predictions.columns:
        return predictions.sort_values("confidence_rank", ascending=False)
    if "home_win_prob" in predictions.columns:
        tiebreaker = predictions["game_id"].to_numpy() if "game_id" in predictions.columns else None
        ranks = confidence_ranks(predictions["home_win_prob"].to_numpy(float), tiebreaker)
        return predictions.iloc[np.argsort(-ranks)]
    return predictions


def _prediction_line(row: pd.Series, away_col: str, home_col: str) -> tuple[str, ...]:
    """Return one game's rank, pick, teams, scores and win probability, formatted."""
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
    prob_str = f"{float(win_prob) * 100:>5.1f}%" if win_prob is not None else "  n/a"
    return rank_str, winner, away_team, away_score_str, home_team, home_score_str, prob_str


def display_weekly_predictions(predictions: pd.DataFrame) -> None:
    """Log weekly score predictions with confidence ranks and win probabilities."""
    if predictions.empty:
        log.info("No predictions to display.")
        return

    team_columns = _team_columns(predictions)
    if team_columns is None:
        log.info("Missing team columns for pretty output.")
        return

    away_col, home_col = team_columns
    sorted_df = _most_confident_first(predictions.copy())

    season = sorted_df["season"].iloc[0] if "season" in sorted_df.columns else None
    week = sorted_df["week"].iloc[0] if "week" in sorted_df.columns else None
    if season is not None and week is not None:
        log.info("Weekly predictions (season %s week %s)", season, week)
    else:
        log.info("Weekly predictions")

    for _, row in sorted_df.iterrows():
        log.info("#%s (%s) %s %s @ %s %s | win %s", *_prediction_line(row, away_col, home_col))
