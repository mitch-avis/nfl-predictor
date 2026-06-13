"""Objective model comparison helpers.

This module implements an apples-to-apples comparison protocol for two model
artifacts by evaluating them under the same time-aware walk-forward splits.

Why this exists:
- Comparing two models trained on all history by scoring them on that same
  history is biased and can hide overfitting.
- The objective approach is to evaluate with walk-forward folds where each
  fold is trained strictly on past games, then evaluated on a future week.

The comparison supports two model kinds used in this repository:
- Margin/total models (market anchoring optional)
- Blended models (team model + market baseline via a blend layer)

This module is intentionally dependency-light and reuses existing internal
training helpers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import artifacts, walk_forward
from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.ml import ml_model_core as core
from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    MarginTotalModel,
    MarketProbConfig,
)
from nfl_predictor.utils.logger import log


@dataclass(frozen=True)
class CompareConfig:
    """Configuration for objective comparison."""

    eval_seasons: list[int] | None = None
    eval_last_n_seasons: int = 3
    wf_start_week: int = 3
    calibration_weeks: int = walk_forward.DEFAULT_CALIBRATION_WEEKS
    random_seed: int = walk_forward.DEFAULT_RANDOM_SEED
    include_postseason: bool = False
    max_cardinality_ratio: float = 0.5
    feature_start: str = "away_rest"
    feature_end: str = "home_moneyline"
    early_stopping_rounds: int = 15
    include_quantiles: bool = False


@dataclass(frozen=True)
class ModelRecipe:
    """A comparable recipe extracted from a saved model artifact."""

    label: str
    kind: str  # "margin_total" | "blend"
    xgb_params: dict[str, Any]
    calibration_method: str
    market_prob_config: MarketProbConfig

    # Market behavior:
    include_market_features: bool
    market_transform: bool | None
    market_anchor: bool


def load_model(path_or_dir: Path) -> Any:
    """Load a joblib model from a file or a run directory."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*If you are loading a serialized model.*",
            category=UserWarning,
        )
        return joblib.load(path)


def _infer_include_market_from_feature_spec(model: Any) -> bool:
    cols: list[str] = []
    if hasattr(model, "feature_spec") and hasattr(model.feature_spec, "feature_columns"):
        cols = list(model.feature_spec.feature_columns)
    if hasattr(model, "team_model") and hasattr(model.team_model, "feature_spec"):
        cols = list(model.team_model.feature_spec.feature_columns)
    return any(col in cols for col in constants.LINES_COLUMNS)


def recipe_from_model(model: Any, *, label: str) -> ModelRecipe:
    """Extract a comparable recipe from a loaded model object."""
    if isinstance(model, MarginTotalModel):
        calibration_method = getattr(getattr(model, "calibrator", None), "method", "none") or "none"
        market_prob = model.market_prob_config or MarketProbConfig(
            blend_weight=0.0, clamp_delta=0.0
        )
        include_market = _infer_include_market_from_feature_spec(model)
        market_anchor = bool(getattr(model, "market_anchor", False))
        # In this repo, margin_total models typically include market features iff available.
        return ModelRecipe(
            label=label,
            kind="margin_total",
            xgb_params=dict(getattr(model, "xgb_params", {})),
            calibration_method=str(calibration_method),
            market_prob_config=market_prob,
            include_market_features=include_market,
            market_transform=None,
            market_anchor=market_anchor,
        )

    if isinstance(model, BlendedMarginTotalModel):
        calibration_method = getattr(getattr(model, "calibrator", None), "method", "none") or "none"
        market_prob = model.market_prob_config or MarketProbConfig(
            blend_weight=0.0, clamp_delta=0.0
        )
        include_market = True  # needs market baseline
        # Blended model stores xgb params under team scope.
        xgb_params = dict(getattr(model, "xgb_params", {}))
        team_params = xgb_params.get("team")
        if isinstance(team_params, dict):
            xgb_params = dict(team_params)
        return ModelRecipe(
            label=label,
            kind="blend",
            xgb_params=xgb_params,
            calibration_method=str(calibration_method),
            market_prob_config=market_prob,
            include_market_features=include_market,
            market_transform=True,
            market_anchor=False,
        )

    raise TypeError(f"Unsupported model type for comparison: {type(model)}")


def _resolve_market_settings_for_recipe(
    df: pd.DataFrame, recipe: ModelRecipe
) -> tuple[bool, bool, bool]:
    return walk_forward.resolve_market_settings(
        df,
        include_market=recipe.include_market_features,
        market_transform=recipe.market_transform,
        market_anchor=recipe.market_anchor,
    )


def _xgb_params_overrides(params: dict[str, Any], *, seed: int) -> dict[str, Any]:
    merged = dict(params or {})
    merged.setdefault("random_state", seed)
    # Ensure consistent verbosity.
    merged.setdefault("verbosity", 0)
    return merged


def _fit_margin_total_fold(
    fold: walk_forward.WalkForwardFold,
    *,
    recipe: ModelRecipe,
    cfg: CompareConfig,
    target_columns: tuple[str, str],
) -> pd.DataFrame:
    """Train/evaluate one fold for a margin/total model."""
    include_market, market_transform, market_anchor = _resolve_market_settings_for_recipe(
        fold.train_df, recipe
    )

    feature_spec = ml_model._build_feature_spec(
        fold.train_df,
        include_market=include_market,
        max_cardinality_ratio=cfg.max_cardinality_ratio,
        feature_start=cfg.feature_start,
        feature_end=cfg.feature_end,
        market_transform=market_transform,
    )
    preprocessor = ml_model._build_preprocessor(feature_spec, for_tree=True)

    x_train = ml_model._fit_transform_matrix(
        preprocessor, ml_model.apply_feature_spec(fold.train_df, feature_spec)
    )
    y_margin_train, y_total_train, _, _ = ml_model._prepare_margin_total_targets_with_anchor(
        fold.train_df, target_columns, market_anchor
    )

    calibration_df = walk_forward.select_calibration_data(
        fold.train_df, fold.season, fold.week, cfg.calibration_weeks
    )
    x_calib = None
    y_margin_calib = None
    y_total_calib = None
    baseline_margin_calib = None
    if not calibration_df.empty:
        x_calib = ml_model._transform_matrix(
            preprocessor, ml_model.apply_feature_spec(calibration_df, feature_spec)
        )
        y_margin_calib, y_total_calib, baseline_margin_calib, _ = (
            ml_model._prepare_margin_total_targets_with_anchor(
                calibration_df, target_columns, market_anchor
            )
        )

    params = _xgb_params_overrides(recipe.xgb_params, seed=cfg.random_seed)
    margin_model, total_model = ml_model._fit_margin_total_models(
        x_train,
        y_margin_train,
        y_total_train,
        params,
        x_eval=x_calib,
        y_margin_eval=y_margin_calib,
        y_total_eval=y_total_calib,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )

    calibrator = None
    calibration_method = recipe.calibration_method
    if recipe.calibration_method.lower() != "none":
        if calibration_df.empty or x_calib is None:
            calibration_method = "none"
        else:
            pred_margin_calib = ml_model._predict_xgb(margin_model, x_calib)
            if market_anchor and baseline_margin_calib is not None:
                pred_margin_calib = pred_margin_calib + baseline_margin_calib
            away_col, home_col = target_columns
            actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
            calibrator = walk_forward._fit_calibrator(
                pred_margin_calib,
                actual_home_win.to_numpy(),
                recipe.calibration_method,
            )
            if calibrator is None:
                calibration_method = "none"

    x_eval = ml_model._transform_matrix(
        preprocessor, ml_model.apply_feature_spec(fold.eval_df, feature_spec)
    )
    pred_margin = ml_model._predict_xgb(margin_model, x_eval)
    pred_total = ml_model._predict_xgb(total_model, x_eval)

    baseline_margin_eval = None
    baseline_total_eval = None
    if market_anchor:
        baseline_margin_eval, baseline_total_eval = ml_model.get_market_baseline(fold.eval_df)
        pred_margin = pred_margin + baseline_margin_eval
        pred_total = pred_total + baseline_total_eval

    pred_away, pred_home = ml_model.derive_scores_from_margin_total(pred_margin, pred_total)
    home_win_prob = ml_model.predict_home_win_prob(pred_margin, calibrator)
    if recipe.market_prob_config.blend_weight or recipe.market_prob_config.clamp_delta:
        home_win_prob = ml_model.adjust_home_win_prob(
            fold.eval_df, home_win_prob, recipe.market_prob_config
        )
    home_win_prob = metrics_utils.clip_probabilities(home_win_prob)

    out = fold.eval_df.copy()
    out["predicted_margin"] = pred_margin
    out["predicted_total"] = pred_total
    out["predicted_home_score"] = pred_home
    out["predicted_away_score"] = pred_away
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1 - home_win_prob
    out["calibration_method"] = calibration_method
    if baseline_margin_eval is not None and baseline_total_eval is not None:
        out["market_baseline_margin"] = baseline_margin_eval
        out["market_baseline_total"] = baseline_total_eval
    return out


def _fit_blended_fold(
    fold: walk_forward.WalkForwardFold,
    *,
    recipe: ModelRecipe,
    cfg: CompareConfig,
    target_columns: tuple[str, str],
) -> pd.DataFrame:
    """Train/evaluate one fold for a blended model (team model + market baseline)."""
    include_market, market_transform, _market_anchor = _resolve_market_settings_for_recipe(
        fold.train_df, recipe
    )
    if not include_market:
        raise ValueError("Blended comparison requires market baseline columns.")

    calibration_df = walk_forward.select_calibration_data(
        fold.train_df, fold.season, fold.week, cfg.calibration_weeks
    )
    if calibration_df.empty:
        raise ValueError(
            f"Insufficient calibration data for blended fold season={fold.season} week={fold.week}."
        )

    team_spec = core._build_feature_spec(
        fold.train_df,
        include_market=False,
        max_cardinality_ratio=cfg.max_cardinality_ratio,
        feature_start=cfg.feature_start,
        feature_end=cfg.feature_end,
        market_transform=market_transform,
    )
    team_preprocessor = core._build_preprocessor(team_spec, for_tree=True)

    x_train = core._fit_transform_matrix(
        team_preprocessor, core._apply_feature_spec(fold.train_df, team_spec)
    )
    y_margin_train, y_total_train = core._prepare_margin_total_targets(
        fold.train_df, target_columns
    )

    x_calib = core._transform_matrix(
        team_preprocessor, core._apply_feature_spec(calibration_df, team_spec)
    )
    y_margin_calib, y_total_calib = core._prepare_margin_total_targets(
        calibration_df, target_columns
    )
    market_margin_calib, market_total_calib = core.get_market_baseline(calibration_df)

    params = _xgb_params_overrides(recipe.xgb_params, seed=cfg.random_seed)
    team_margin_model, team_total_model = core._fit_margin_total_models(
        x_train,
        y_margin_train,
        y_total_train,
        params,
        x_eval=x_calib,
        y_margin_eval=y_margin_calib,
        y_total_eval=y_total_calib,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )

    team_margin_calib = core._predict_xgb(team_margin_model, x_calib)
    team_total_calib = core._predict_xgb(team_total_model, x_calib)

    margin_blender = core._fit_blend_ridge_constrained(
        np.column_stack([team_margin_calib, market_margin_calib]),
        y_margin_calib,
        alpha=1.0,
    )
    total_blender = core._fit_blend_ridge_constrained(
        np.column_stack([team_total_calib, market_total_calib]),
        y_total_calib,
        alpha=1.0,
    )

    blended_margin_calib = margin_blender.predict(
        np.column_stack([team_margin_calib, market_margin_calib])
    )
    away_col, home_col = target_columns
    actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)

    calibrator = None
    calibration_method = recipe.calibration_method
    if recipe.calibration_method.lower() != "none":
        calibrator = core._fit_win_prob_calibrator(
            blended_margin_calib,
            actual_home_win.to_numpy(),
            recipe.calibration_method,
        )
        if calibrator is None:
            calibration_method = "none"

    x_eval = core._transform_matrix(
        team_preprocessor, core._apply_feature_spec(fold.eval_df, team_spec)
    )
    team_margin_eval = core._predict_xgb(team_margin_model, x_eval)
    team_total_eval = core._predict_xgb(team_total_model, x_eval)
    market_margin_eval, market_total_eval = core.get_market_baseline(fold.eval_df)

    blended_margin = margin_blender.predict(np.column_stack([team_margin_eval, market_margin_eval]))
    blended_total = total_blender.predict(np.column_stack([team_total_eval, market_total_eval]))

    pred_away, pred_home = core.derive_scores_from_margin_total(blended_margin, blended_total)
    home_win_prob = core._predict_home_win_prob(blended_margin, calibrator)
    home_win_prob = core._adjust_home_win_prob(
        fold.eval_df, home_win_prob, recipe.market_prob_config
    )
    home_win_prob = metrics_utils.clip_probabilities(home_win_prob)

    out = fold.eval_df.copy()
    out["predicted_margin"] = blended_margin
    out["predicted_total"] = blended_total
    out["predicted_home_score"] = pred_home
    out["predicted_away_score"] = pred_away
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1 - home_win_prob
    out["calibration_method"] = calibration_method
    out["market_baseline_margin"] = market_margin_eval
    out["market_baseline_total"] = market_total_eval
    return out


def _aggregate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    away_score = predictions["away_score"].to_numpy(dtype=float)
    home_score = predictions["home_score"].to_numpy(dtype=float)
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    actual_home_win = (home_score > away_score).astype(int)

    pred_margin = predictions["predicted_margin"].to_numpy(dtype=float)
    pred_total = predictions["predicted_total"].to_numpy(dtype=float)
    home_win_prob = predictions["home_win_prob"].to_numpy(dtype=float)

    overall = {
        **metrics_utils.margin_total_metrics(actual_margin, actual_total, pred_margin, pred_total),
        **metrics_utils.probability_metrics(actual_home_win, home_win_prob),
        "games": int(len(predictions)),
    }
    return overall


def bootstrap_overall_metrics(
    predictions: pd.DataFrame,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap overall metrics by resampling games with replacement.

    This is a rough uncertainty estimate intended for comparing models on the
    same evaluation slice. It is not a time-series-aware block bootstrap.
    """
    if n_samples <= 0:
        return {}
    if predictions.empty:
        return {}

    rng = np.random.default_rng(int(seed))
    n = int(len(predictions))

    keys = ["brier", "log_loss", "margin_mae", "total_mae", "pick_accuracy"]
    samples: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(int(n_samples)):
        idx = rng.integers(0, n, size=n)
        sample_df = predictions.iloc[idx]
        m = _aggregate_predictions(sample_df)
        for k in keys:
            samples[k].append(float(m.get(k, float("nan"))))

    ci: dict[str, Any] = {"n_samples": int(n_samples)}
    for k, values in samples.items():
        arr = np.asarray(values, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        ci[k] = {
            "p05": float(np.nanpercentile(arr, 5)),
            "p50": float(np.nanpercentile(arr, 50)),
            "p95": float(np.nanpercentile(arr, 95)),
        }
    return ci


def run_objective_compare(
    df: pd.DataFrame,
    *,
    recipe_a: ModelRecipe,
    recipe_b: ModelRecipe,
    cfg: CompareConfig,
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    """Run an objective walk-forward comparison for two recipes."""
    np.random.seed(cfg.random_seed)  # noqa: NPY002 (legacy for reproducibility)
    df = walk_forward.filter_regular_season(df, include_postseason=cfg.include_postseason)

    target_columns = ml_model.get_target_columns(df)
    df = df.dropna(subset=list(target_columns)).copy()

    eval_seasons = walk_forward.resolve_eval_seasons(df, cfg.eval_seasons, cfg.eval_last_n_seasons)
    folds = walk_forward.build_walk_forward_folds(
        df, eval_seasons, cfg.wf_start_week, include_postseason=cfg.include_postseason
    )
    if not folds:
        raise ValueError("No folds available for objective comparison.")

    preds_a: list[pd.DataFrame] = []
    preds_b: list[pd.DataFrame] = []

    skipped_blend_folds = 0
    for fold in folds:
        # Ensure the eval set contains the targets.
        away_col, home_col = target_columns
        if away_col not in fold.eval_df.columns or home_col not in fold.eval_df.columns:
            continue

        pa = _fit_margin_total_fold(fold, recipe=recipe_a, cfg=cfg, target_columns=target_columns)
        preds_a.append(pa)

        if recipe_b.kind == "blend":
            try:
                pb = _fit_blended_fold(
                    fold, recipe=recipe_b, cfg=cfg, target_columns=target_columns
                )
            except ValueError:
                skipped_blend_folds += 1
                continue
        else:
            pb = _fit_margin_total_fold(
                fold, recipe=recipe_b, cfg=cfg, target_columns=target_columns
            )
        preds_b.append(pb)

    pred_df_a = pd.concat(preds_a, ignore_index=True)
    pred_df_b = pd.concat(preds_b, ignore_index=True)

    sort_cols = [c for c in ("season", "week", "game_id") if c in pred_df_a.columns]
    if sort_cols:
        pred_df_a = pred_df_a.sort_values(sort_cols).reset_index(drop=True)
        pred_df_b = pred_df_b.sort_values(sort_cols).reset_index(drop=True)

    # Align by key if possible.
    key_cols = [
        c for c in ("season", "week", "game_id", "away_abbr", "home_abbr") if c in pred_df_a.columns
    ]
    if key_cols:
        a_keyed = pred_df_a.set_index(key_cols)
        b_keyed = pred_df_b.set_index(key_cols)
        common = a_keyed.index.intersection(b_keyed.index)
        pred_df_a = a_keyed.loc[common].reset_index()
        pred_df_b = b_keyed.loc[common].reset_index()

    # Ensure actual score columns exist with canonical names.
    away_col, home_col = target_columns
    pred_df_a = pred_df_a.rename(columns={away_col: "away_score", home_col: "home_score"})
    pred_df_b = pred_df_b.rename(columns={away_col: "away_score", home_col: "home_score"})

    overall_a = _aggregate_predictions(pred_df_a)
    overall_b = _aggregate_predictions(pred_df_b)

    bootstrap = {}
    if bootstrap_samples:
        bootstrap = {
            recipe_a.label: bootstrap_overall_metrics(
                pred_df_a, n_samples=int(bootstrap_samples), seed=cfg.random_seed
            ),
            recipe_b.label: bootstrap_overall_metrics(
                pred_df_b, n_samples=int(bootstrap_samples), seed=cfg.random_seed
            ),
        }

    return {
        "recipe_a": recipe_a,
        "recipe_b": recipe_b,
        "overall": {
            recipe_a.label: overall_a,
            recipe_b.label: overall_b,
        },
        "bootstrap": bootstrap,
        "predictions": {
            recipe_a.label: pred_df_a,
            recipe_b.label: pred_df_b,
        },
        "resolved_eval_seasons": [int(s) for s in eval_seasons],
        "skipped_blend_folds": int(skipped_blend_folds),
    }


def write_compare_outputs(out_dir: Path, results: dict[str, Any]) -> None:
    """Write comparison outputs to a directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, dict[str, Any]] = results["overall"]
    rows: list[dict[str, Any]] = []
    for model_label, metrics in overall.items():
        row: dict[str, Any] = {"model": model_label}
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "objective_compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    for label, pred in results["predictions"].items():
        pred_path = out_dir / f"objective_compare_predictions_{label}.csv"
        pred.to_csv(pred_path, index=False)

    meta: dict[str, Any] = {
        "resolved_eval_seasons": results.get("resolved_eval_seasons"),
        "skipped_blend_folds": results.get("skipped_blend_folds"),
        "bootstrap": results.get("bootstrap") or None,
    }
    artifacts.write_json(out_dir / "objective_compare_meta.json", meta)
    log.info("Wrote comparison outputs under %s", out_dir)
