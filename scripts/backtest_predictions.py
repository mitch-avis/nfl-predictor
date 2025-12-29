"""
Backtest model predictions, confidence points, and weekly power rankings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_predictor import constants, ml_model
from nfl_predictor.utils.logger import log


def _resolve_team_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    for away_col, home_col in (("away_abbr", "home_abbr"), ("away_name", "home_name")):
        if away_col in df.columns and home_col in df.columns:
            return away_col, home_col
    return None, None


def _predict_all(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, ml_model.ScoreModel):
        feature_df = ml_model.apply_feature_spec(df, model.feature_spec)
        x_games = model.preprocessor.transform(feature_df)
        pred_away = ml_model.predict_xgb(model.away_model, x_games)
        pred_home = ml_model.predict_xgb(model.home_model, x_games)
        home_win_prob = ml_model.margin_to_home_win_prob(pred_home - pred_away)
    elif isinstance(model, ml_model.MarginTotalModel):
        pred_margin, pred_total = ml_model.predict_margin_total_from_model(model, df)
        pred_away, pred_home = ml_model.derive_scores_from_margin_total(pred_margin, pred_total)
        home_win_prob = ml_model.predict_home_win_prob(pred_margin, model.calibrator)
    elif isinstance(model, ml_model.BlendedMarginTotalModel):
        team_margin, team_total = ml_model.predict_margin_total_from_model(model.team_model, df)
        if model.market_model is None:
            market_margin, market_total = ml_model.get_market_baseline(df)
        else:
            market_margin, market_total = ml_model.predict_margin_total_from_model(
                model.market_model, df
            )
        blended_margin = model.blend_layer.margin_model.predict(
            np.column_stack([team_margin, market_margin])
        )
        blended_total = model.blend_layer.total_model.predict(
            np.column_stack([team_total, market_total])
        )
        pred_away, pred_home = ml_model.derive_scores_from_margin_total(
            blended_margin, blended_total
        )
        home_win_prob = ml_model.predict_home_win_prob(blended_margin, model.calibrator)
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")

    market_prob_config = getattr(model, "market_prob_config", None)
    home_win_prob = ml_model.adjust_home_win_prob(df, home_win_prob, market_prob_config)
    output_df = ml_model.build_prediction_output(df, pred_away, pred_home, home_win_prob)
    return output_df


def _assign_weekly_confidence_ranks(df: pd.DataFrame) -> pd.DataFrame:
    if "season" not in df.columns or "week" not in df.columns:
        log.info("Missing season/week columns; skipping per-week confidence ranking.")
        return df

    ranked = df.copy()
    ranked["confidence_strength"] = (ranked["home_win_prob"] - 0.5).abs()
    tiebreaker_cols = [col for col in ("game_id", "date", "home_abbr", "away_abbr") if col in df]
    sort_cols = ["season", "week", "confidence_strength"] + tiebreaker_cols
    sorted_df = ranked.sort_values(sort_cols, ascending=True)
    ranked.loc[sorted_df.index, "confidence_rank"] = (
        sorted_df.groupby(["season", "week"]).cumcount() + 1
    )
    ranked["confidence_rank"] = ranked["confidence_rank"].astype(int)
    return ranked


def _add_ratings(df: pd.DataFrame, target_columns: tuple[str, str]) -> pd.DataFrame:
    rated = df.copy()
    rated["pregame_home_rating"] = (rated["home_win_prob"] * 10).round(2)
    rated["pregame_away_rating"] = ((1 - rated["home_win_prob"]) * 10).round(2)

    away_col, home_col = target_columns
    if away_col in rated.columns and home_col in rated.columns:
        actual_margin = rated[home_col] - rated[away_col]
        actual_home_prob = ml_model.margin_to_home_win_prob(actual_margin.to_numpy())
        rated["postgame_home_rating"] = (actual_home_prob * 10).round(2)
        rated["postgame_away_rating"] = ((1 - actual_home_prob) * 10).round(2)
    return rated


def _compute_backtest_metrics(
    df: pd.DataFrame, target_columns: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    away_col, home_col = target_columns
    eval_df = df.dropna(subset=[away_col, home_col]).copy()
    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    if "confidence_rank" not in eval_df.columns:
        eval_df["confidence_rank"] = (
            (eval_df["home_win_prob"] - 0.5).abs().rank(method="first", ascending=True).astype(int)
        )

    away_team_col, home_team_col = _resolve_team_columns(eval_df)
    if away_team_col and home_team_col:
        actual_winner = np.where(
            eval_df[home_col] > eval_df[away_col],
            eval_df[home_team_col],
            np.where(
                eval_df[home_col] < eval_df[away_col],
                eval_df[away_team_col],
                "tie",
            ),
        )
        predicted_winner = eval_df.get("predicted_winner")
        if predicted_winner is None or predicted_winner.isna().all():
            predicted_winner = np.where(
                eval_df["home_win_prob"] >= 0.5,
                eval_df[home_team_col],
                eval_df[away_team_col],
            )
    else:
        actual_winner = np.where(
            eval_df[home_col] > eval_df[away_col],
            "home",
            np.where(eval_df[home_col] < eval_df[away_col], "away", "tie"),
        )
        predicted_winner = np.where(eval_df["home_win_prob"] >= 0.5, "home", "away")

    eval_df["actual_winner"] = actual_winner
    eval_df["predicted_winner"] = predicted_winner
    eval_df["pick_correct"] = (eval_df["predicted_winner"] == eval_df["actual_winner"]) & (
        eval_df["actual_winner"] != "tie"
    )
    away_win_prob = 1 - eval_df["home_win_prob"]
    eval_df["pick_win_prob"] = np.where(
        eval_df["predicted_winner"] == "home",
        eval_df["home_win_prob"],
        away_win_prob,
    )
    eval_df["expected_points"] = eval_df["confidence_rank"] * eval_df["pick_win_prob"]
    eval_df["actual_points"] = eval_df["confidence_rank"] * eval_df["pick_correct"].astype(int)

    weekly = (
        eval_df.groupby(["season", "week"], as_index=False)
        .agg(
            expected_points=("expected_points", "sum"),
            actual_points=("actual_points", "sum"),
            picks_correct=("pick_correct", "sum"),
            games=("pick_correct", "size"),
        )
        .assign(
            pick_accuracy=lambda frame: frame["picks_correct"] / frame["games"],
            max_points=lambda frame: frame["games"] * (frame["games"] + 1) / 2,
        )
    )

    seasonal = (
        weekly.groupby("season", as_index=False)
        .agg(
            expected_points=("expected_points", "sum"),
            actual_points=("actual_points", "sum"),
            picks_correct=("picks_correct", "sum"),
            games=("games", "sum"),
            weeks=("week", "nunique"),
            max_points=("max_points", "sum"),
        )
        .assign(
            pick_accuracy=lambda frame: frame["picks_correct"] / frame["games"],
            expected_points_avg=lambda frame: frame["expected_points"] / frame["weeks"],
            actual_points_avg=lambda frame: frame["actual_points"] / frame["weeks"],
        )
    )

    overall = {
        "expected_points": float(seasonal["expected_points"].sum()),
        "actual_points": float(seasonal["actual_points"].sum()),
        "picks_correct": int(seasonal["picks_correct"].sum()),
        "games": int(seasonal["games"].sum()),
        "weeks": int(seasonal["weeks"].sum()),
        "max_points": float(seasonal["max_points"].sum()),
    }
    overall["pick_accuracy"] = (
        overall["picks_correct"] / overall["games"] if overall["games"] else 0.0
    )
    overall["expected_points_avg"] = (
        overall["expected_points"] / overall["weeks"] if overall["weeks"] else 0.0
    )
    overall["actual_points_avg"] = (
        overall["actual_points"] / overall["weeks"] if overall["weeks"] else 0.0
    )

    return weekly, seasonal, overall


def _build_pregame_power_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()

    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.upper() == "REG"]

    away_team_col, home_team_col = _resolve_team_columns(df)
    if away_team_col is None or home_team_col is None:
        return pd.DataFrame()

    required_cols = {
        "pregame_away_rating",
        "pregame_home_rating",
        "postgame_away_rating",
        "postgame_home_rating",
    }
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    home_rows = df[
        ["season", "week", home_team_col, "pregame_home_rating", "postgame_home_rating"]
    ].rename(
        columns={
            home_team_col: "team",
            "pregame_home_rating": "pregame_rating",
            "postgame_home_rating": "postgame_rating",
        }
    )
    away_rows = df[
        ["season", "week", away_team_col, "pregame_away_rating", "postgame_away_rating"]
    ].rename(
        columns={
            away_team_col: "team",
            "pregame_away_rating": "pregame_rating",
            "postgame_away_rating": "postgame_rating",
        }
    )
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.dropna(subset=["pregame_rating", "week"])
    long_df["week"] = long_df["week"].astype(int)
    if long_df.empty:
        return pd.DataFrame()

    ranking_rows: list[pd.DataFrame] = []
    for season, season_df in long_df.groupby("season"):
        season_weeks = sorted(season_df["week"].dropna().unique())
        for week in season_weeks:
            ratings = season_df.copy()
            played_mask = (ratings["week"] < week) & ratings["postgame_rating"].notna()
            ratings["rating"] = ratings["pregame_rating"]
            ratings.loc[played_mask, "rating"] = ratings.loc[played_mask, "postgame_rating"]

            team_ratings = (
                ratings.groupby("team", as_index=False)["rating"].mean().assign(season=season)
            )
            team_ratings["week"] = week
            team_ratings = team_ratings.sort_values(["rating", "team"], ascending=[False, True])
            team_ratings["rank"] = np.arange(1, len(team_ratings) + 1)
            ranking_rows.append(team_ratings)

    if not ranking_rows:
        return pd.DataFrame()
    return pd.concat(ranking_rows, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest predictions and build rankings.")
    parser.add_argument("--model-in", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--model-kind",
        choices=["score", "margin_total", "blend"],
        required=True,
        help="Model kind saved in the checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(constants.DATA_PATH) / "completed_games_ml.csv",
        help="Dataset to score and backtest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(constants.DATA_PATH) / "backtest",
        help="Output directory for CSV/JSON results.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (default: data path stem).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for running full backtests and power rankings."""
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or args.data_path.stem

    model = ml_model.load_model_checkpoint(args.model_in, args.model_kind)
    games_df = pd.read_csv(args.data_path)
    log.info("Loaded %d rows from %s", len(games_df), args.data_path)

    predictions = _predict_all(model, games_df)
    if "game_type" in predictions.columns:
        predictions = predictions[predictions["game_type"].astype(str).str.upper() == "REG"]
    predictions = _assign_weekly_confidence_ranks(predictions)

    target_columns = ml_model.get_target_columns(games_df)
    predictions = _add_ratings(predictions, target_columns)

    predictions_out = output_dir / f"{prefix}_predictions.csv"
    predictions.to_csv(predictions_out, index=False)
    log.info("Saved predictions to %s", predictions_out)

    weekly, seasonal, overall = _compute_backtest_metrics(predictions, target_columns)
    if not weekly.empty:
        weekly_out = output_dir / f"{prefix}_weekly_summary.csv"
        seasonal_out = output_dir / f"{prefix}_season_summary.csv"
        overall_out = output_dir / f"{prefix}_overall_summary.json"
        weekly.to_csv(weekly_out, index=False)
        seasonal.to_csv(seasonal_out, index=False)
        overall_out.write_text(json.dumps(overall, indent=2, sort_keys=True))
        log.info("Saved weekly summary to %s", weekly_out)
        log.info("Saved season summary to %s", seasonal_out)
        log.info("Saved overall summary to %s", overall_out)
    else:
        log.info("No completed games found; skipping backtest summaries.")

    ratings_cols = [
        col
        for col in (
            "season",
            "week",
            "game_type",
            "away_abbr",
            "home_abbr",
            "away_name",
            "home_name",
            "predicted_away_score",
            "predicted_home_score",
            "predicted_margin",
            "predicted_total",
            "pregame_away_rating",
            "pregame_home_rating",
            "postgame_away_rating",
            "postgame_home_rating",
            target_columns[0],
            target_columns[1],
        )
        if col in predictions.columns
    ]
    ratings_out = output_dir / f"{prefix}_ratings.csv"
    predictions[ratings_cols].to_csv(ratings_out, index=False)
    log.info("Saved ratings to %s", ratings_out)

    pre_rankings = _build_pregame_power_rankings(predictions)
    if not pre_rankings.empty:
        pre_out = output_dir / f"{prefix}_power_rankings.csv"
        pre_rankings.to_csv(pre_out, index=False)
        log.info("Saved pregame power rankings to %s", pre_out)


if __name__ == "__main__":
    main()
