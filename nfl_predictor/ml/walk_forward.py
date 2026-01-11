"""Walk-forward backtesting for margin/total models.

Implements time-aware walk-forward training/evaluation:
- trains only on games strictly before the evaluated (season, week)
- optional time-aware calibration using the last K training weeks of the eval season
- produces metrics summaries plus a calibration reliability table

This module is intentionally small and importable so tests can validate split correctness
and determinism.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.utils.logger import log

DEFAULT_CALIBRATION_WEEKS = 4
DEFAULT_RANDOM_SEED = 42
RELIABILITY_BINS = 10


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""

    eval_seasons: Optional[Sequence[int]] = None
    eval_last_n_seasons: int = 3
    wf_start_week: int = 3
    calibration: str = "platt"
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS
    random_seed: int = DEFAULT_RANDOM_SEED
    include_market: bool = True
    market_transform: Optional[bool] = None
    market_anchor: bool = True
    market_prob_weight: float = 0.0
    market_prob_clamp: float = 0.0
    include_quantiles: bool = True
    max_cardinality_ratio: float = 0.5
    feature_start: str = ml_model.DEFAULT_FEATURE_START_COLUMN
    feature_end: str = ml_model.DEFAULT_FEATURE_END_COLUMN
    early_stopping_rounds: int = ml_model.DEFAULT_EARLY_STOPPING_ROUNDS
    xgb_params_overrides: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the config."""

        return {
            "eval_seasons": list(self.eval_seasons) if self.eval_seasons else None,
            "eval_last_n_seasons": self.eval_last_n_seasons,
            "wf_start_week": self.wf_start_week,
            "calibration": self.calibration,
            "calibration_weeks": self.calibration_weeks,
            "random_seed": self.random_seed,
            "include_market": self.include_market,
            "market_transform": self.market_transform,
            "market_anchor": self.market_anchor,
            "market_prob_weight": self.market_prob_weight,
            "market_prob_clamp": self.market_prob_clamp,
            "include_quantiles": self.include_quantiles,
            "max_cardinality_ratio": self.max_cardinality_ratio,
            "feature_start": self.feature_start,
            "feature_end": self.feature_end,
            "early_stopping_rounds": self.early_stopping_rounds,
            "xgb_params_overrides": self.xgb_params_overrides,
        }


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward fold.

    Train data is strictly before (season, week); eval data is at (season, week).
    """

    season: int
    week: int
    train_df: pd.DataFrame
    eval_df: pd.DataFrame


def load_games(data_path: Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""

    df = pd.read_csv(data_path)
    log.info("Loaded %d rows from %s", len(df), data_path)
    return df


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular season games when game_type exists."""

    if "game_type" not in df.columns:
        return df
    filtered = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
    if len(filtered) != len(df):
        log.info("Filtered to regular-season games: %d -> %d rows", len(df), len(filtered))
    return filtered


def resolve_eval_seasons(
    df: pd.DataFrame, eval_seasons: Optional[Sequence[int]], eval_last_n: int
) -> list[int]:
    """Resolve which seasons to evaluate based on dataset contents and config."""

    seasons = sorted(df["season"].dropna().unique())
    if not seasons:
        raise ValueError("No seasons available in dataset.")

    if eval_seasons:
        requested = sorted({int(season) for season in eval_seasons})
        available = [season for season in requested if season in seasons]
        missing = [season for season in requested if season not in seasons]
        if missing:
            log.info("Dropping missing eval seasons: %s", missing)
        if not available:
            raise ValueError("None of the requested eval seasons exist in the dataset.")
        return sorted(available)

    if eval_last_n <= 0:
        raise ValueError("eval_last_n_seasons must be positive.")
    if len(seasons) <= eval_last_n:
        return seasons
    return seasons[-eval_last_n:]


def build_walk_forward_folds(
    df: pd.DataFrame, eval_seasons: Sequence[int], start_week: int
) -> list[WalkForwardFold]:
    """Build time-aware walk-forward folds for each eval season and week."""

    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("season and week columns are required for walk-forward splits.")

    folds: list[WalkForwardFold] = []
    for season in sorted(eval_seasons):
        max_week = constants.get_regular_season_weeks(season)
        for week in range(start_week, max_week + 1):
            eval_df = df[(df["season"] == season) & (df["week"] == week)].copy()
            if eval_df.empty:
                continue
            train_df = df[
                (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
            ].copy()
            if train_df.empty:
                log.info("Skipping season %s week %s: no training data.", season, week)
                continue
            folds.append(
                WalkForwardFold(season=season, week=week, train_df=train_df, eval_df=eval_df)
            )
    return folds


def select_calibration_data(
    train_df: pd.DataFrame, eval_season: int, eval_week: int, calibration_weeks: int
) -> pd.DataFrame:
    """Select time-aware calibration data from the training window.

    Uses the last `calibration_weeks` weeks of the eval season strictly before `eval_week`.
    Returns empty when insufficient or unavailable.
    """

    if calibration_weeks <= 0:
        return train_df.iloc[0:0].copy()
    season_df = train_df[train_df["season"] == eval_season].copy()
    if season_df.empty:
        return season_df
    eligible_weeks = sorted(season_df["week"].dropna().unique())
    eligible_weeks = [week for week in eligible_weeks if week < eval_week]
    if len(eligible_weeks) < calibration_weeks:
        return season_df.iloc[0:0].copy()
    selected_weeks = eligible_weeks[-calibration_weeks:]
    return season_df[season_df["week"].isin(selected_weeks)].copy()


def _has_market_lines(df: pd.DataFrame) -> bool:
    return any(col in df.columns for col in constants.LINES_COLUMNS)


def _can_market_anchor(df: pd.DataFrame) -> bool:
    has_spread = any(
        col in df.columns for col in ("home_spread", "away_spread", "market_home_margin")
    )
    has_total = "total_line" in df.columns or "market_total_line" in df.columns
    return has_spread and has_total


def resolve_market_settings(
    df: pd.DataFrame,
    include_market: bool,
    market_transform: Optional[bool],
    market_anchor: bool,
) -> tuple[bool, bool, bool]:
    """Resolve market feature/transform/anchor settings based on columns present."""

    has_market = _has_market_lines(df)
    resolved_transform = market_transform if market_transform is not None else has_market
    resolved_include = include_market and has_market
    resolved_anchor = market_anchor

    if market_anchor and not _can_market_anchor(df):
        log.info("Market anchor requested but spread/total lines missing; disabling anchor.")
        resolved_anchor = False

    if include_market and not has_market:
        log.info("Market columns missing; disabling market features.")
        resolved_include = False
        resolved_transform = False

    return resolved_include, resolved_transform, resolved_anchor


def _fit_calibrator(
    pred_margin: np.ndarray,
    actual_home_win: np.ndarray,
    method: str,
) -> Optional[ml_model.WinProbCalibrator]:
    method = method.lower()
    if method == "none":
        return None
    unique = np.unique(actual_home_win)
    if len(unique) < 2:
        log.info("Calibration skipped: only one outcome class present.")
        return None
    return ml_model._fit_win_prob_calibrator(pred_margin, actual_home_win, method)


def _resolve_xgb_params(config: WalkForwardConfig) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if config.xgb_params_overrides:
        overrides.update(config.xgb_params_overrides)
    overrides.setdefault("random_state", config.random_seed)
    return ml_model._resolve_xgb_params(ml_model.DEFAULT_XGB_PARAMS, overrides=overrides)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    config: WalkForwardConfig,
) -> dict[str, Any]:
    """Run walk-forward training/evaluation and return metrics plus per-game predictions."""

    np.random.seed(config.random_seed)
    df = filter_regular_season(df)
    target_columns = ml_model.get_target_columns(df)
    df = df.dropna(subset=list(target_columns)).copy()

    include_market, market_transform, market_anchor = resolve_market_settings(
        df, config.include_market, config.market_transform, config.market_anchor
    )
    resolved_settings = {
        "include_market": include_market,
        "market_transform": market_transform,
        "market_anchor": market_anchor,
        "market_prob_weight": config.market_prob_weight,
        "market_prob_clamp": config.market_prob_clamp,
        "include_quantiles": config.include_quantiles,
    }

    eval_seasons = resolve_eval_seasons(df, config.eval_seasons, config.eval_last_n_seasons)
    resolved_eval_seasons = [int(season) for season in eval_seasons]
    folds = build_walk_forward_folds(df, eval_seasons, config.wf_start_week)
    if not folds:
        raise ValueError("No walk-forward folds available with the provided settings.")

    params = _resolve_xgb_params(config)

    per_week_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_list: list[str] | None = None

    for fold in folds:
        feature_spec = ml_model._build_feature_spec(
            fold.train_df,
            include_market=include_market,
            max_cardinality_ratio=config.max_cardinality_ratio,
            feature_start=config.feature_start,
            feature_end=config.feature_end,
            market_transform=market_transform,
        )
        if feature_list is None:
            feature_list = list(feature_spec.feature_columns)
        preprocessor = ml_model._build_preprocessor(feature_spec, for_tree=True)

        x_train = preprocessor.fit_transform(
            ml_model.apply_feature_spec(fold.train_df, feature_spec)
        )
        y_margin_train, y_total_train, _, _ = ml_model._prepare_margin_total_targets_with_anchor(
            fold.train_df, target_columns, market_anchor
        )

        calibration_df = select_calibration_data(
            fold.train_df, fold.season, fold.week, config.calibration_weeks
        )
        x_calibration = None
        y_margin_calibration = None
        y_total_calibration = None
        baseline_margin_calibration = None
        if not calibration_df.empty:
            x_calibration = preprocessor.transform(
                ml_model.apply_feature_spec(calibration_df, feature_spec)
            )
            (
                y_margin_calibration,
                y_total_calibration,
                baseline_margin_calibration,
                _,
            ) = ml_model._prepare_margin_total_targets_with_anchor(
                calibration_df, target_columns, market_anchor
            )

        margin_model, total_model = ml_model._fit_margin_total_models(
            x_train,
            y_margin_train,
            y_total_train,
            params,
            x_eval=x_calibration,
            y_margin_eval=y_margin_calibration,
            y_total_eval=y_total_calibration,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        quantiles: tuple[float, ...] | None = None
        margin_quantiles: dict[float, Any] = {}
        total_quantiles: dict[float, Any] = {}
        if config.include_quantiles:
            quantiles = ml_model._validate_quantiles(ml_model.DEFAULT_QUANTILES)
            margin_quantiles = ml_model._fit_quantile_models(
                x_train,
                y_margin_train,
                params,
                quantiles,
                x_eval=x_calibration,
                y_eval=y_margin_calibration,
                early_stopping_rounds=config.early_stopping_rounds,
            )
            total_quantiles = ml_model._fit_quantile_models(
                x_train,
                y_total_train,
                params,
                quantiles,
                x_eval=x_calibration,
                y_eval=y_total_calibration,
                early_stopping_rounds=config.early_stopping_rounds,
            )

        calibrator = None
        calibration_method = config.calibration
        if config.calibration.lower() != "none":
            if calibration_df.empty or x_calibration is None:
                log.info(
                    "Calibration skipped for season %s week %s: insufficient calibration data.",
                    fold.season,
                    fold.week,
                )
                calibration_method = "none"
            else:
                pred_margin_calibration = ml_model._predict_xgb(margin_model, x_calibration)
                if market_anchor and baseline_margin_calibration is not None:
                    pred_margin_calibration = pred_margin_calibration + baseline_margin_calibration
                away_col, home_col = target_columns
                actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
                calibrator = _fit_calibrator(
                    pred_margin_calibration,
                    actual_home_win.to_numpy(),
                    config.calibration,
                )
                if calibrator is None:
                    calibration_method = "none"

        x_eval = preprocessor.transform(ml_model.apply_feature_spec(fold.eval_df, feature_spec))
        pred_margin = ml_model._predict_xgb(margin_model, x_eval)
        pred_total = ml_model._predict_xgb(total_model, x_eval)
        pred_margin_quantiles = {
            q: ml_model._predict_xgb(q_model, x_eval) for q, q_model in margin_quantiles.items()
        }
        pred_total_quantiles = {
            q: ml_model._predict_xgb(q_model, x_eval) for q, q_model in total_quantiles.items()
        }
        baseline_margin_eval = None
        baseline_total_eval = None
        if market_anchor:
            baseline_margin_eval, baseline_total_eval = ml_model.get_market_baseline(fold.eval_df)
            pred_margin = pred_margin + baseline_margin_eval
            pred_total = pred_total + baseline_total_eval
            for q in list(pred_margin_quantiles.keys()):
                pred_margin_quantiles[q] = pred_margin_quantiles[q] + baseline_margin_eval
            for q in list(pred_total_quantiles.keys()):
                pred_total_quantiles[q] = pred_total_quantiles[q] + baseline_total_eval

        pred_away, pred_home = ml_model.derive_scores_from_margin_total(pred_margin, pred_total)
        home_win_prob = ml_model.predict_home_win_prob(pred_margin, calibrator)
        if config.market_prob_weight or config.market_prob_clamp:
            market_prob_config = ml_model.MarketProbConfig(
                blend_weight=config.market_prob_weight,
                clamp_delta=config.market_prob_clamp,
            )
            home_win_prob = ml_model.adjust_home_win_prob(
                fold.eval_df, home_win_prob, market_prob_config
            )
        home_win_prob = metrics_utils.clip_probabilities(home_win_prob)

        away_col, home_col = target_columns
        away_score = fold.eval_df[away_col].to_numpy(dtype=float)
        home_score = fold.eval_df[home_col].to_numpy(dtype=float)
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        actual_home_win = (home_score > away_score).astype(int)

        tiebreaker = (
            fold.eval_df["game_id"].to_numpy() if "game_id" in fold.eval_df.columns else None
        )
        confidence_cols = metrics_utils.confidence_pool_columns(
            home_win_prob, home_score, away_score, tiebreaker=tiebreaker
        )

        fold_predictions = fold.eval_df.copy()
        fold_predictions["predicted_margin"] = pred_margin
        fold_predictions["predicted_total"] = pred_total
        for q in sorted(pred_margin_quantiles.keys()):
            column = f"predicted_margin_p{int(round(q * 100)):02d}"
            fold_predictions[column] = pred_margin_quantiles[q]
        for q in sorted(pred_total_quantiles.keys()):
            column = f"predicted_total_p{int(round(q * 100)):02d}"
            fold_predictions[column] = pred_total_quantiles[q]
        fold_predictions["predicted_home_score"] = pred_home
        fold_predictions["predicted_away_score"] = pred_away
        fold_predictions["home_win_prob"] = home_win_prob
        fold_predictions["away_win_prob"] = 1 - home_win_prob
        fold_predictions["actual_margin"] = actual_margin
        fold_predictions["actual_total"] = actual_total
        fold_predictions["actual_home_win"] = actual_home_win
        fold_predictions["confidence_rank"] = confidence_cols["confidence_rank"]
        fold_predictions["expected_points"] = confidence_cols["expected_points"]
        fold_predictions["actual_points"] = confidence_cols["actual_points"]
        fold_predictions["pick_correct"] = confidence_cols["pick_correct"]
        fold_predictions["calibration_method"] = calibration_method
        if baseline_margin_eval is not None and baseline_total_eval is not None:
            fold_predictions["market_baseline_margin"] = baseline_margin_eval
            fold_predictions["market_baseline_total"] = baseline_total_eval

        prediction_frames.append(fold_predictions)

        metrics = {
            "season": int(fold.season),
            "week": int(fold.week),
            "games": int(len(fold_predictions)),
            **metrics_utils.margin_total_metrics(
                actual_margin, actual_total, pred_margin, pred_total
            ),
            **metrics_utils.probability_metrics(actual_home_win, home_win_prob),
            **metrics_utils.confidence_pool_summary(confidence_cols),
            "calibration_method": calibration_method,
        }

        if market_anchor and baseline_margin_eval is not None and baseline_total_eval is not None:
            actual_margin_resid = actual_margin - baseline_margin_eval
            actual_total_resid = actual_total - baseline_total_eval
            pred_margin_resid = pred_margin - baseline_margin_eval
            pred_total_resid = pred_total - baseline_total_eval
            metrics["market_margin_resid_mae"] = float(
                np.mean(np.abs(actual_margin_resid - pred_margin_resid))
            )
            metrics["market_total_resid_mae"] = float(
                np.mean(np.abs(actual_total_resid - pred_total_resid))
            )
        per_week_metrics.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    sort_cols = [col for col in ("season", "week", "game_id") if col in predictions.columns]
    if sort_cols:
        predictions = predictions.sort_values(sort_cols).reset_index(drop=True)

    per_season_metrics: list[dict[str, Any]] = []
    for season in sorted(predictions["season"].dropna().unique()):
        season_df = predictions[predictions["season"] == season]
        per_season_metrics.append(_aggregate_metrics(season_df, market_anchor))

    overall_metrics = _aggregate_metrics(predictions, market_anchor)
    reliability = metrics_utils.reliability_table(
        predictions["home_win_prob"].to_numpy(),
        predictions["actual_home_win"].to_numpy(),
        bins=RELIABILITY_BINS,
    )

    return {
        "per_week": per_week_metrics,
        "per_season": per_season_metrics,
        "overall": overall_metrics,
        "reliability": reliability,
        "predictions": predictions,
        "resolved_settings": resolved_settings,
        "resolved_eval_seasons": resolved_eval_seasons,
        "feature_list": feature_list,
    }


def build_metrics_report(
    run_id: str,
    created_at: str,
    config_payload: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serializable metrics report payload."""

    return {
        "run_id": run_id,
        "created_at": created_at,
        "config": config_payload,
        "metrics": {
            "per_week": results["per_week"],
            "per_season": results["per_season"],
            "overall": results["overall"],
        },
        "calibration": {
            "bins": results["reliability"],
            "bin_count": RELIABILITY_BINS,
        },
    }


def _aggregate_metrics(frame: pd.DataFrame, market_anchor: bool) -> dict[str, Any]:
    actual_margin = frame["actual_margin"].to_numpy()
    pred_margin = frame["predicted_margin"].to_numpy()
    actual_total = frame["actual_total"].to_numpy()
    pred_total = frame["predicted_total"].to_numpy()
    actual_home_win = frame["actual_home_win"].to_numpy()
    home_win_prob = frame["home_win_prob"].to_numpy()

    season_value = None
    if "season" in frame.columns and frame["season"].nunique() == 1:
        season_value = int(frame["season"].iloc[0])
    metrics = {
        "season": season_value,
        "weeks": int(frame["week"].nunique()) if "week" in frame.columns else None,
        "games": int(len(frame)),
        **metrics_utils.margin_total_metrics(actual_margin, actual_total, pred_margin, pred_total),
        **metrics_utils.probability_metrics(actual_home_win, home_win_prob),
        "expected_points": float(frame["expected_points"].sum()),
        "actual_points": float(frame["actual_points"].sum()),
        "picks_correct": int(frame["pick_correct"].sum()),
    }
    metrics["pick_accuracy"] = (
        metrics["picks_correct"] / metrics["games"] if metrics["games"] else 0.0
    )
    if metrics.get("weeks"):
        metrics["expected_points_avg"] = metrics["expected_points"] / metrics["weeks"]
        metrics["actual_points_avg"] = metrics["actual_points"] / metrics["weeks"]

    if market_anchor and "market_baseline_margin" in frame.columns:
        baseline_margin = frame["market_baseline_margin"].to_numpy()
        baseline_total = frame["market_baseline_total"].to_numpy()
        actual_margin_resid = actual_margin - baseline_margin
        actual_total_resid = actual_total - baseline_total
        pred_margin_resid = pred_margin - baseline_margin
        pred_total_resid = pred_total - baseline_total
        metrics["market_margin_resid_mae"] = float(
            np.mean(np.abs(actual_margin_resid - pred_margin_resid))
        )
        metrics["market_total_resid_mae"] = float(
            np.mean(np.abs(actual_total_resid - pred_total_resid))
        )

    # Optional diagnostics: interval coverage (P10-P90).
    margin_p10 = "predicted_margin_p10"
    margin_p90 = "predicted_margin_p90"
    total_p10 = "predicted_total_p10"
    total_p90 = "predicted_total_p90"
    if margin_p10 in frame.columns and margin_p90 in frame.columns:
        lo = frame[margin_p10].to_numpy(dtype=float)
        hi = frame[margin_p90].to_numpy(dtype=float)
        metrics["margin_p10_p90_coverage"] = float(
            np.mean((actual_margin >= lo) & (actual_margin <= hi))
        )
    if total_p10 in frame.columns and total_p90 in frame.columns:
        lo = frame[total_p10].to_numpy(dtype=float)
        hi = frame[total_p90].to_numpy(dtype=float)
        metrics["total_p10_p90_coverage"] = float(
            np.mean((actual_total >= lo) & (actual_total <= hi))
        )

    return metrics


def dataset_fingerprint(path: Path) -> str:
    """Compute a SHA-256 fingerprint of the dataset file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_run_id(dataset_hash: str, config: WalkForwardConfig) -> str:
    """Generate a stable-ish run id from timestamp + config hash."""

    created = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = json.dumps(config.to_dict(), sort_keys=True)
    short_hash = hashlib.sha256(f"{dataset_hash}:{payload}".encode("utf-8")).hexdigest()[:8]
    return f"wf_{created}_{short_hash}"


def build_metadata(
    created_at: str, dataset_hash: str, config_payload: dict[str, Any]
) -> dict[str, Any]:
    """Build a metadata payload adjacent to the metrics report."""

    return {
        "created_at": created_at,
        "run_id": config_payload.get("run_id"),
        "git_commit_hash": _git_commit_hash(),
        "dataset_hash": dataset_hash,
        "library_versions": _library_versions(),
        "config": config_payload,
        "feature_list": config_payload.get("feature_list"),
        "splits": config_payload.get("splits"),
    }


def _git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(constants.ROOT_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _library_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for module_name in (
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "xgboost",
        "optuna",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover
            versions[module_name] = None
            continue
        versions[module_name] = getattr(module, "__version__", None)
    return versions
