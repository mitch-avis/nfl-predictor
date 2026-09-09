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
    from nfl_predictor import constants
    from nfl_predictor.ml import ml_model_core
    from nfl_predictor.reporting.power_rankings import (
        FIT_WEIGHT_COLUMN,
        PowerRatingsResult,
        build_power_rankings_and_standings,
        outcome_to_home_prob,
    )
    from nfl_predictor.utils.logger import log
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from nfl_predictor import constants
    from nfl_predictor.ml import ml_model_core
    from nfl_predictor.reporting.power_rankings import (
        FIT_WEIGHT_COLUMN,
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
    p.add_argument(
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include postseason games in records/ratings (default: regular season only).",
    )
    p.add_argument(
        "--ratings-window-seasons",
        type=int,
        default=DEFAULT_RATINGS_WINDOW_SEASONS,
        help=(
            "How many seasons the ratings fit sees, counting the current one. "
            f"Default {DEFAULT_RATINGS_WINDOW_SEASONS} (current plus previous). "
            "Use 0 for every available season."
        ),
    )
    p.add_argument(
        "--ratings-prior-season-weight",
        type=float,
        default=DEFAULT_PRIOR_SEASON_WEIGHT,
        help=(
            "Weight applied to games from seasons before the current one. "
            f"Default {DEFAULT_PRIOR_SEASON_WEIGHT}. Current-season games always weigh 1.0."
        ),
    )
    p.add_argument(
        "--ratings-target",
        choices=("margin", "binary"),
        default="margin",
        help=(
            "Target for completed games. 'margin' maps the observed point margin through "
            "the model's win-probability curve; 'binary' scores every win the same."
        ),
    )
    p.add_argument(
        "--ratings-include-future",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Feed future games' model win probabilities into the strength fit. Off by "
            "default: the fit should describe results, not the model's own forecasts. "
            "Future games always remain in projected standings."
        ),
    )
    p.add_argument(
        "--legacy-franchise-fit",
        action="store_true",
        help=(
            "Reproduce the historical ranking exactly: every season since 1999 weighted "
            "equally, binary win/loss targets, and future model probabilities in the fit. "
            "Overrides the other --ratings-* flags."
        ),
    )
    return p.parse_args()


# Seasons the ratings fit sees by default, counting the current one. Two keeps a
# full prior season of evidence for early-season weeks without letting a franchise's
# history dominate the current one.
DEFAULT_RATINGS_WINDOW_SEASONS = 2

# Weight on games from before the current season. Low enough that the current season
# dominates once a few games exist, high enough to stabilize week 1.
DEFAULT_PRIOR_SEASON_WEIGHT = 0.25

# Columns the ratings fit consumes.
_RATINGS_COLUMNS = (
    "season",
    "week",
    "away_abbr",
    "home_abbr",
    "p_home",
    FIT_WEIGHT_COLUMN,
)


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


POSTSEASON_GAME_TYPES = frozenset({"WC", "DIV", "CON", "SB", "POST"})


def _filter_by_game_type(df: pl.DataFrame, *, include_postseason: bool) -> pl.DataFrame:
    """Filter to regular season (or regular + postseason) games when game_type exists."""
    if "game_type" not in df.columns:
        return df
    game_type = pl.col("game_type").cast(pl.Utf8).str.to_uppercase()
    if include_postseason:
        allowed = ["REG", *sorted(POSTSEASON_GAME_TYPES)]
        return df.filter(game_type.is_in(allowed))
    return df.filter(game_type == "REG")


def _load_current_records(
    schedule_path: Path,
    *,
    season: int,
    through_week: int,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """Load current records through the specified week."""
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
    df = _filter_by_game_type(df, include_postseason=include_postseason)
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


def _missing_market_inputs(
    required_features: list[str],
    available_cols: set[str],
) -> list[str]:
    """Return market-derived feature names that lack required raw inputs."""
    required = set(required_features)
    missing: list[str] = []

    if (
        "market_home_margin" in required
        and "market_home_margin" not in available_cols
        and not {"home_spread", "away_spread"} & available_cols
    ):
        missing.append("market_home_margin")
    if (
        "market_total_line" in required
        and "market_total_line" not in available_cols
        and "total_line" not in available_cols
    ):
        missing.append("market_total_line")
    if (
        "home_market_prob" in required
        and "home_market_prob" not in available_cols
        and "home_moneyline" not in available_cols
    ):
        missing.append("home_market_prob")
    if (
        "away_market_prob" in required
        and "away_market_prob" not in available_cols
        and "away_moneyline" not in available_cols
    ):
        missing.append("away_market_prob")

    return missing


def _format_missing_columns(missing: list[str], *, limit: int = 10) -> str:
    """Format a missing-column list for error messages."""
    unique = sorted(set(missing))
    if len(unique) <= limit:
        return ", ".join(unique)
    shown = ", ".join(unique[:limit])
    return f"{shown} (+{len(unique) - limit} more)"


def _predict_future_games(
    model: Any,
    *,
    model_kind: str,
    data_ml: Path,
    season: int,
    through_week: int,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """Predict future games for the specified season."""
    # Load a season slice from the ML dataset (REG only by default), then predict for future games.
    # We read only the columns required by the model's FeatureSpec.
    spec = getattr(model, "feature_spec", None)
    if spec is None:
        raise ValueError("Model is missing feature_spec; cannot predict")

    available_cols = set(pl.read_csv(data_ml, n_rows=0).columns)
    derived_cols = set(ml_model_core.MARKET_DERIVED_COLUMNS)
    required_features = list(getattr(spec, "feature_columns", []))
    missing_required = set(required_features) - available_cols - derived_cols
    missing_required.update(_missing_market_inputs(required_features, available_cols))
    if missing_required:
        missing_text = _format_missing_columns(sorted(missing_required))
        raise ValueError(
            f"Missing required feature columns ({len(missing_required)}): {missing_text}"
        )
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

    games = pl.read_csv(data_ml, columns=usecols).filter(pl.col("season") == season)
    games = _filter_by_game_type(games, include_postseason=include_postseason)
    games = games.filter(pl.col("week") > through_week)

    games = _pl_to_pandas(games)

    if games.empty:
        return games.assign(home_win_prob=pd.Series(dtype=float))

    if model_kind == "margin_total":
        mt = cast(ml_model_core.MarginTotalModel, model)
        pred_margin, _pred_total = ml_model_core.predict_margin_total_from_model(mt, games)
        use_uncertainty = bool(getattr(mt, "win_prob_use_uncertainty", False))
        sigma_margin = None
        if use_uncertainty:
            margin_quantiles, _ = ml_model_core._predict_margin_total_quantiles_from_model(
                mt, games
            )
            sigma_margin = ml_model_core._resolve_margin_sigma(
                pred_margin,
                margin_quantiles,
                fallback=constants.SCORE_DIFF_STD_DEV,
            )
        home_win_prob = ml_model_core.predict_home_win_prob(
            pred_margin,
            mt.calibrator,
            sigma=sigma_margin,
            use_uncertainty=use_uncertainty,
        )
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
        x = ml_model_core._transform_matrix(sm.preprocessor, feature_df)
        pred_away = ml_model_core.predict_xgb(sm.away_model, x)
        pred_home = ml_model_core.predict_xgb(sm.home_model, x)
        pred_margin = pred_home - pred_away
        home_win_prob = ml_model_core.predict_home_win_prob(
            pred_margin,
            getattr(sm, "calibrator", None),
        )
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
    include_postseason: bool,
    window_seasons: int = DEFAULT_RATINGS_WINDOW_SEASONS,
    prior_season_weight: float = DEFAULT_PRIOR_SEASON_WEIGHT,
    target: str = "margin",
    include_future: bool = False,
) -> pd.DataFrame:
    """Assemble the games the strength fit sees, with a per-game weight.

    The fit answers "how strong is each team going into next week", so by default it
    sees a short window of recent seasons rather than the whole archive, weights games
    from before the current season down by `prior_season_weight`, and scores completed
    games by margin rather than by a flat win/loss. Future games are excluded by
    default: feeding the model's own forecasts back into the strength fit makes the
    ranking partly a picture of the model rather than of results. They still drive
    projected standings.

    Args:
        schedule_path: Schedule/results dataset.
        season: Season being ranked.
        through_week: Records and results are counted through this week inclusive.
        ratings_min_season: Optional hard floor on seasons included.
        future_games_with_probs: Future games carrying model win probabilities.
        include_postseason: Whether postseason games count.
        window_seasons: Seasons the fit sees, counting the current one; 0 means all.
        prior_season_weight: Weight for games before the current season.
        target: "margin" or "binary" target for completed games.
        include_future: Whether future model probabilities enter the fit.

    Returns:
        Games with `season`, `week`, `away_abbr`, `home_abbr`, `p_home` and `fit_weight`.

    """
    sched = pl.read_csv(schedule_path).select(
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
    sched = _filter_by_game_type(sched, include_postseason=include_postseason)

    floor_season = ratings_min_season
    if window_seasons and window_seasons > 0:
        window_floor = season - window_seasons + 1
        floor_season = (
            window_floor if floor_season is None else max(int(floor_season), window_floor)
        )
    if floor_season is not None:
        sched = sched.filter(pl.col("season") >= int(floor_season))

    sched = _pl_to_pandas(sched)

    past = sched[
        (
            (sched["season"] < season)
            | ((sched["season"] == season) & (sched["week"] <= through_week))
        )
        & sched["away_score"].notna()
        & sched["home_score"].notna()
    ].copy()
    past["p_home"] = outcome_to_home_prob(past["home_score"], past["away_score"], target=target)
    # Current-season results carry full weight; earlier seasons are evidence, not equals.
    past[FIT_WEIGHT_COLUMN] = np.where(
        past["season"].to_numpy() == season, 1.0, float(prior_season_weight)
    )

    fut = future_games_with_probs.copy()
    if include_future and not fut.empty:
        fut = fut.rename(columns={"home_win_prob": "p_home"})
        fut = fut[["season", "week", "away_abbr", "home_abbr", "p_home"]].copy()
        fut[FIT_WEIGHT_COLUMN] = 1.0
    else:
        fut = pd.DataFrame(columns=[*_RATINGS_COLUMNS])

    games = pd.concat(
        [past[list(_RATINGS_COLUMNS)], fut[list(_RATINGS_COLUMNS)]], ignore_index=True
    )
    log.info(
        "Ratings fit diagnostics: past_games=%d future_games=%d min_season=%s "
        "prior_season_weight=%s target=%s",
        len(past),
        len(fut),
        floor_season if floor_season is not None else "all",
        prior_season_weight,
        target,
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
    """Run the power rankings CLI."""
    args = _parse_args()

    if not args.model_in.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {args.model_in}")
    if not args.data_ml.exists():
        raise FileNotFoundError(f"Missing ML dataset: {args.data_ml}")
    if not args.data_schedule.exists():
        raise FileNotFoundError(f"Missing schedule dataset: {args.data_schedule}")

    model = ml_model_core.load_model_checkpoint(args.model_in, args.model_kind)

    current_records = _load_current_records(
        args.data_schedule,
        season=args.season,
        through_week=args.through_week,
        include_postseason=bool(args.include_postseason),
    )
    future_games = _predict_future_games(
        model,
        model_kind=args.model_kind,
        data_ml=args.data_ml,
        season=args.season,
        through_week=args.through_week,
        include_postseason=bool(args.include_postseason),
    )
    if args.legacy_franchise_fit:
        # The historical behavior: every season equal, binary targets, forecasts in the fit.
        window_seasons = 0
        prior_season_weight = 1.0
        ratings_target = "binary"
        include_future = True
        log.info("Legacy franchise fit requested; --ratings-* options are ignored.")
    else:
        window_seasons = int(args.ratings_window_seasons)
        prior_season_weight = float(args.ratings_prior_season_weight)
        ratings_target = str(args.ratings_target)
        include_future = bool(args.ratings_include_future)

    games_for_ratings = _build_games_for_ratings(
        schedule_path=args.data_schedule,
        season=args.season,
        through_week=args.through_week,
        ratings_min_season=args.ratings_min_season,
        future_games_with_probs=future_games,
        include_postseason=bool(args.include_postseason),
        window_seasons=window_seasons,
        prior_season_weight=prior_season_weight,
        target=ratings_target,
        include_future=include_future,
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
