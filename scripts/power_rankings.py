#!/usr/bin/env python
"""Generate power rankings and projected standings from a trained model.

This script is meant for fan-friendly reporting:
- A 1-10 (and 0-10) power rating per team (absolute scale vs an average team)
- Projected standings based on current record + expected wins in remaining REG games

Method summary
--------------
1) Load a trained model checkpoint.
2) Predict win probabilities for remaining regular season games (from ML feature rows).
3) Fit a simple Bradley-Terry latent-strength model using:
   - completed-game outcomes (as probability targets)
   - future-game model win probabilities (as probability targets)
4) Convert latent strengths to a stable (ultimate) power rating by mapping
    `sigmoid(rating_raw)` (win prob vs an average team on a neutral field) onto
    1-10 and 0-10 scales.

Usage example
-------------
python scripts/power_rankings.py \
  --model-in models/<run_id>/model.joblib \
  --model-kind margin_total \
  --season 2025 \
  --through-week 10 \
  --data-ml data/all_data_ml.csv \
  --data-schedule data/all_data.csv \
  --out-dir data/predict

Outputs
-------
- power_rankings_season_XXXX_week_YY.csv
- projected_standings_season_XXXX_week_YY.csv
- projected_division_standings_season_XXXX_week_YY.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

try:
    from nfl_predictor.ml import ml_model_core
    from nfl_predictor.reporting.power_rankings import (
        PowerRatingsResult,
        build_power_rankings_and_standings,
        outcome_to_home_prob,
    )
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor.ml import ml_model_core
    from nfl_predictor.reporting.power_rankings import (
        PowerRatingsResult,
        build_power_rankings_and_standings,
        outcome_to_home_prob,
    )
    from nfl_predictor.utils.logger import log


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate power rankings + projected standings")
    p.add_argument("--model-in", type=Path, required=True, help="Path to model checkpoint .joblib")
    p.add_argument(
        "--model-kind",
        type=str,
        default="margin_total",
        choices=["margin_total", "blended_margin_total", "score"],
        help="Model kind (must match the saved checkpoint).",
    )
    p.add_argument("--season", type=int, required=True, help="Season year")
    p.add_argument(
        "--through-week",
        type=int,
        required=True,
        help="Compute records through this week (inclusive); future games are week > this.",
    )
    p.add_argument(
        "--data-ml",
        type=Path,
        default=Path("data/all_data_ml.csv"),
        help="ML dataset (must include future game feature rows).",
    )
    p.add_argument(
        "--data-schedule",
        type=Path,
        default=Path("data/all_data.csv"),
        help="Schedule/results dataset (used for current records).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/predict"),
        help="Directory to write outputs.",
    )
    p.add_argument(
        "--ratings-min-season",
        type=int,
        default=None,
        help=(
            "Optional minimum season to include in the ratings fit. "
            "Default: include all historical seasons available in --data-schedule."
        ),
    )
    return p.parse_args()


def _pl_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas without requiring pyarrow.

    Polars' `to_pandas()` requires `pyarrow` in many environments; for this reporting script,
    using `to_dicts()` keeps the dependency surface smaller.
    """

    if df.is_empty():
        # Preserve the schema so downstream code can rely on column presence even when
        # there are zero rows (e.g., postseason weeks with no future REG games).
        return pd.DataFrame(columns=list(df.columns))
    return pd.DataFrame(df.to_dicts())


def _load_current_records(schedule_path: Path, *, season: int, through_week: int) -> pd.DataFrame:
    df = (
        pl.read_csv(schedule_path)
        .select(
            [
                "season",
                "week",
                "game_type",
                "away_abbr",
                "home_abbr",
                "away_score",
                "home_score",
            ]
        )
        .filter(pl.col("season") == season)
    )

    # Only count games up through the specified week that have scores.
    df = df.filter(pl.col("week") <= through_week)
    df = df.filter(pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null())

    # Compute per-team record.
    away = df.select(
        [
            pl.col("away_abbr").alias("team_abbr"),
            (pl.col("away_score") > pl.col("home_score")).cast(pl.Int32).alias("wins"),
            (pl.col("away_score") < pl.col("home_score")).cast(pl.Int32).alias("losses"),
            (pl.col("away_score") == pl.col("home_score")).cast(pl.Int32).alias("ties"),
        ]
    )
    home = df.select(
        [
            pl.col("home_abbr").alias("team_abbr"),
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("wins"),
            (pl.col("home_score") < pl.col("away_score")).cast(pl.Int32).alias("losses"),
            (pl.col("home_score") == pl.col("away_score")).cast(pl.Int32).alias("ties"),
        ]
    )

    rec = (
        pl.concat([away, home], how="vertical")
        .group_by("team_abbr")
        .agg(
            [
                pl.col("wins").sum().alias("wins"),
                pl.col("losses").sum().alias("losses"),
                pl.col("ties").sum().alias("ties"),
            ]
        )
        .with_columns((pl.col("wins") + pl.col("losses") + pl.col("ties")).alias("games_played"))
    )

    return _pl_to_pandas(rec)


def _predict_future_games(
    model: Any,
    *,
    model_kind: str,
    data_ml: Path,
    season: int,
    through_week: int,
) -> pd.DataFrame:
    # Load a season slice from the ML dataset (REG only), then predict for future games.
    # We read only the columns required by the model's FeatureSpec.
    spec = getattr(model, "feature_spec", None)
    if spec is None:
        raise ValueError("Model is missing feature_spec; cannot predict")

    available_cols = set(pl.read_csv(data_ml, n_rows=0).columns)
    base_cols = ["season", "week", "game_type", "away_abbr", "home_abbr"]
    market_raw_cols = [
        "home_spread",
        "away_spread",
        "total_line",
        "home_moneyline",
        "away_moneyline",
    ]
    base_cols = base_cols + [c for c in market_raw_cols if c in available_cols]
    feature_cols = [c for c in list(getattr(spec, "feature_columns", [])) if c in available_cols]
    usecols = sorted(set(base_cols + feature_cols))

    games = (
        pl.read_csv(data_ml, columns=usecols)
        .filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
        .filter(pl.col("week") > through_week)
    )

    games = _pl_to_pandas(games)

    if games.empty:
        return games.assign(home_win_prob=pd.Series(dtype=float))

    if model_kind == "margin_total":
        mt = cast(ml_model_core.MarginTotalModel, model)
        pred_margin, _pred_total = ml_model_core.predict_margin_total_from_model(mt, games)
        home_win_prob = ml_model_core.predict_home_win_prob(pred_margin, mt.calibrator)
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(mt, "market_prob_config", None)
        )
    elif model_kind == "blended_margin_total":
        bm = cast(ml_model_core.BlendedMarginTotalModel, model)
        # Use the model's blended margin as the win-prob driver.
        # The public helper takes pred_margin; for blended we reuse internal predict path.
        # We call build_prediction_output via the predict module would require a Path.
        team_margin, _team_total = ml_model_core.predict_margin_total_from_model(
            bm.team_model, games
        )
        if bm.market_model is None:
            market_margin, _market_total = ml_model_core.get_market_baseline(games)
        else:
            market_margin, _market_total = ml_model_core.predict_margin_total_from_model(
                bm.market_model, games
            )
        blended_margin = bm.blend_layer.margin_model.predict(
            np.column_stack([team_margin, market_margin])
        )
        home_win_prob = ml_model_core.predict_home_win_prob(blended_margin, bm.calibrator)
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(bm, "market_prob_config", None)
        )
    else:
        sm = cast(ml_model_core.ScoreModel, model)
        # ScoreModel: predict home/away scores then derive margin->prob.
        feature_df = ml_model_core.apply_feature_spec(games, sm.feature_spec)
        x = sm.preprocessor.transform(feature_df)
        pred_away = ml_model_core.predict_xgb(sm.away_model, x)
        pred_home = ml_model_core.predict_xgb(sm.home_model, x)
        home_win_prob = ml_model_core.margin_to_home_win_prob(pred_home - pred_away)
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(sm, "market_prob_config", None)
        )

    out = games[["season", "week", "away_abbr", "home_abbr"]].copy()
    out["home_win_prob"] = home_win_prob
    return out


def _build_games_for_ratings(
    *,
    schedule_path: Path,
    season: int,
    through_week: int,
    ratings_min_season: int | None,
    future_games_with_probs: pd.DataFrame,
) -> pd.DataFrame:
    sched = (
        pl.read_csv(schedule_path)
        .select(
            [
                "season",
                "week",
                "game_type",
                "away_abbr",
                "home_abbr",
                "away_score",
                "home_score",
            ]
        )
        .filter(pl.col("game_type") == "REG")
    )

    if ratings_min_season is not None:
        sched = sched.filter(pl.col("season") >= int(ratings_min_season))

    sched = _pl_to_pandas(sched)

    # Use all historical games (prior seasons) plus current-season games through `through_week`.
    # This makes the ratings more stable across weeks and comparable season-to-season.
    past = sched[
        (
            (sched["season"] < season)
            | ((sched["season"] == season) & (sched["week"] <= through_week))
        )
        & sched["away_score"].notna()
        & sched["home_score"].notna()
    ].copy()
    past["p_home"] = outcome_to_home_prob(past["home_score"], past["away_score"])

    fut = future_games_with_probs.copy()
    if not fut.empty:
        fut = fut.rename(columns={"home_win_prob": "p_home"})
        fut = fut[["season", "week", "away_abbr", "home_abbr", "p_home"]]

    games = pd.concat(
        [past[["season", "week", "away_abbr", "home_abbr", "p_home"]], fut],
        ignore_index=True,
    )
    return games.dropna(subset=["p_home", "away_abbr", "home_abbr"]).copy()


def _write_outputs(
    result: PowerRatingsResult,
    *,
    out_dir: Path,
    season: int,
    through_week: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"season_{season}_week_{through_week:02d}"

    pr_path = out_dir / f"power_rankings_{suffix}.csv"
    st_path = out_dir / f"projected_standings_{suffix}.csv"
    div_path = out_dir / f"projected_division_standings_{suffix}.csv"

    result.power_rankings.to_csv(pr_path, index=False)
    result.projected_standings.to_csv(st_path, index=False)
    result.projected_division_standings.to_csv(div_path, index=False)

    log.info("Wrote %s", pr_path)
    log.info("Wrote %s", st_path)
    log.info("Wrote %s", div_path)


def main() -> int:
    """Main script entry point."""

    args = _parse_args()

    if not args.model_in.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {args.model_in}")
    if not args.data_ml.exists():
        raise FileNotFoundError(f"Missing ML dataset: {args.data_ml}")
    if not args.data_schedule.exists():
        raise FileNotFoundError(f"Missing schedule dataset: {args.data_schedule}")

    model = ml_model_core.load_model_checkpoint(args.model_in, args.model_kind)

    current_records = _load_current_records(
        args.data_schedule, season=args.season, through_week=args.through_week
    )
    future_games = _predict_future_games(
        model,
        model_kind=args.model_kind,
        data_ml=args.data_ml,
        season=args.season,
        through_week=args.through_week,
    )
    games_for_ratings = _build_games_for_ratings(
        schedule_path=args.data_schedule,
        season=args.season,
        through_week=args.through_week,
        ratings_min_season=args.ratings_min_season,
        future_games_with_probs=future_games,
    )

    result = build_power_rankings_and_standings(
        season=args.season,
        through_week=args.through_week,
        current_records=current_records,
        games_for_ratings=games_for_ratings,
        future_games_with_probs=future_games,
    )

    _write_outputs(result, out_dir=args.out_dir, season=args.season, through_week=args.through_week)
    log.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
