"""ML-oriented utility helpers.

This module contains helpers used by ML prediction tooling (e.g., pretty-printing weekly
predictions) and small dict/DataFrame utilities.

Historically this lived at `nfl_predictor.utils.ml_utils`; that path remains as a compatibility
facade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


def display_weekly_predictions(predictions: pd.DataFrame) -> None:
    """Log weekly score predictions with confidence ranks and win probabilities."""

    if predictions.empty:
        log.info("No predictions to display.")
        return

    team_columns: tuple[str, str] | None = None
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
        prob_str = f"{float(win_prob) * 100:>5.1f}%" if win_prob is not None else "  n/a"

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


def flatten_dict(nested_dict: Any) -> dict[tuple[Any, ...], Any]:
    """Recursively flatten a nested dict into tuple-keyed paths."""

    res: dict[tuple[Any, ...], Any] = {}
    if isinstance(nested_dict, dict):
        for k, v in nested_dict.items():
            flattened_dict = flatten_dict(v)
            for key, val in flattened_dict.items():
                res[(k,) + key] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict: Any) -> pd.DataFrame:
    """Convert a nested dict into a pandas DataFrame via flattening."""

    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    out = df.iloc[:, 0].unstack(level=-1)
    out.columns = out.columns.map(lambda x: f"{x}")
    return out
