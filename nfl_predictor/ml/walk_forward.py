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
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.ml.sample_weights import combine_sample_weights, compute_recency_sample_weight
from nfl_predictor.utils.logger import log

DEFAULT_CALIBRATION_WEEKS = 4
DEFAULT_RANDOM_SEED = 42
RELIABILITY_BINS = 10

# Shared root for per-fold checkpoints. Each run writes into a subdirectory named for its
# fingerprint, so any identical walk-forward (from any script) resumes where it stopped and
# runs with different inputs never see each other's files.
DEFAULT_CHECKPOINT_DIR = Path(constants.ROOT_DIR) / "models" / "wf_checkpoints"
# Bump when the checkpoint payload layout changes, so older files are ignored, not misread.
FOLD_CHECKPOINT_VERSION = 1

SUMMARY_METRICS = (
    "margin_mae",
    "total_mae",
    "brier",
    "log_loss",
    "reliability_ece",
    "expected_points",
    "actual_points",
    "picks_correct",
    "pick_accuracy",
    "market_margin_resid_mae",
    "market_total_resid_mae",
    "margin_p10_p90_coverage",
    "total_p10_p90_coverage",
)


def _scalar_to_int(value: Any) -> int:
    """Cast a pandas/numpy scalar to a Python int."""
    return int(np.asarray(value).item())


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""

    eval_seasons: Sequence[int] | None = None
    eval_last_n_seasons: int = 3
    wf_start_week: int = 3
    calibration: str = "platt"
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS
    random_seed: int = DEFAULT_RANDOM_SEED
    include_postseason: bool = False
    exclude_incomplete_seasons: bool = False
    include_market: bool = True
    market_transform: bool | None = None
    market_anchor: bool = True
    market_prob_weight: float = 0.0
    market_prob_clamp: float = 0.0
    market_prob_source: str = "raw"
    market_prob_blend_method: str = "prob"
    win_prob_use_uncertainty: bool = False
    include_quantiles: bool = True
    max_cardinality_ratio: float = 0.5
    feature_start: str = ml_model.DEFAULT_FEATURE_START_COLUMN
    feature_end: str = ml_model.DEFAULT_FEATURE_END_COLUMN
    early_stopping_rounds: int = ml_model.DEFAULT_EARLY_STOPPING_ROUNDS
    recency_half_life_weeks: float | None = None
    recency_half_life_seasons: float | None = None
    disable_pruning: bool = False
    disabled_feature_groups: tuple[str, ...] = ()
    xgb_params_overrides: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the config."""
        return {
            "eval_seasons": list(self.eval_seasons) if self.eval_seasons else None,
            "eval_last_n_seasons": self.eval_last_n_seasons,
            "wf_start_week": self.wf_start_week,
            "calibration": self.calibration,
            "calibration_weeks": self.calibration_weeks,
            "random_seed": self.random_seed,
            "include_postseason": self.include_postseason,
            "exclude_incomplete_seasons": self.exclude_incomplete_seasons,
            "include_market": self.include_market,
            "market_transform": self.market_transform,
            "market_anchor": self.market_anchor,
            "market_prob_weight": self.market_prob_weight,
            "market_prob_clamp": self.market_prob_clamp,
            "market_prob_source": self.market_prob_source,
            "market_prob_blend_method": self.market_prob_blend_method,
            "win_prob_use_uncertainty": self.win_prob_use_uncertainty,
            "include_quantiles": self.include_quantiles,
            "max_cardinality_ratio": self.max_cardinality_ratio,
            "feature_start": self.feature_start,
            "feature_end": self.feature_end,
            "early_stopping_rounds": self.early_stopping_rounds,
            "recency_half_life_weeks": self.recency_half_life_weeks,
            "recency_half_life_seasons": self.recency_half_life_seasons,
            "disable_pruning": self.disable_pruning,
            "disabled_feature_groups": list(self.disabled_feature_groups),
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


def filter_regular_season(df: pd.DataFrame, include_postseason: bool = False) -> pd.DataFrame:
    """Filter to regular season games when game_type exists."""
    if include_postseason or "game_type" not in df.columns:
        return df
    filtered = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
    if len(filtered) != len(df):
        log.info("Filtered to regular-season games: %d -> %d rows", len(df), len(filtered))
    return filtered


def resolve_feature_group_columns(columns: Sequence[str], groups: Sequence[str]) -> list[str]:
    """Resolve which columns belong to the given feature groups for ablation.

    A column belongs to a group when any marker string configured for that group in
    ``constants.FEATURE_GROUP_COLUMN_MARKERS`` is a *substring* of the column name (not a
    prefix), so a single marker can match ``away_``/``home_``/``opponent_`` prefixed variants
    and ``_diff`` suffixed variants of the same base name at once. A group whose marker tuple
    is empty matches zero columns.

    Raises:
        ValueError: if any requested group name is not a key in
            ``constants.FEATURE_GROUP_COLUMN_MARKERS``.

    """
    valid_groups = constants.FEATURE_GROUP_COLUMN_MARKERS
    unknown = sorted({group for group in groups if group not in valid_groups})
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {unknown}. Valid groups: {sorted(valid_groups)}."
        )

    matched: set[str] = set()
    for group in groups:
        markers = valid_groups[group]
        if not markers:
            continue
        for column in columns:
            if any(marker in column for marker in markers):
                matched.add(column)

    return sorted(matched)


def resolve_eval_seasons(
    df: pd.DataFrame, eval_seasons: Sequence[int] | None, eval_last_n: int
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


def filter_incomplete_eval_seasons(
    df: pd.DataFrame, eval_seasons: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Return (kept, dropped) seasons based on regular-season completeness."""
    kept: list[int] = []
    dropped: list[int] = []
    for season in eval_seasons:
        season_value = int(season)
        season_df = df[df["season"] == season_value]
        max_week = int(season_df["week"].max()) if not season_df.empty else None
        regular_weeks = constants.get_regular_season_weeks(season_value)
        if max_week is None or max_week < regular_weeks:
            dropped.append(season_value)
        else:
            kept.append(season_value)
    return kept, dropped


def build_walk_forward_folds(
    df: pd.DataFrame,
    eval_seasons: Sequence[int],
    start_week: int,
    include_postseason: bool = False,
) -> list[WalkForwardFold]:
    """Build time-aware walk-forward folds for each eval season and week."""
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("season and week columns are required for walk-forward splits.")

    folds: list[WalkForwardFold] = []
    for season in sorted(eval_seasons):
        season_df = df[df["season"] == season]
        if include_postseason and not season_df.empty:
            max_week = int(season_df["week"].max())
        else:
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


def summarize_eval_window(
    df: pd.DataFrame,
    eval_seasons: Sequence[int],
    *,
    start_week: int,
    include_postseason: bool,
) -> dict[str, Any]:
    """Summarize the evaluation window for reporting/metadata."""
    season_rows: dict[str, dict[str, Any]] = {}
    incomplete_seasons: list[int] = []

    for season in sorted(eval_seasons):
        season_df = df[df["season"] == season]
        max_week = int(season_df["week"].max()) if not season_df.empty else None
        regular_weeks = constants.get_regular_season_weeks(int(season))
        eval_end_week = max_week if include_postseason and max_week is not None else regular_weeks
        incomplete_regular = max_week is not None and max_week < regular_weeks
        if incomplete_regular:
            incomplete_seasons.append(int(season))
        season_rows[str(season)] = {
            "regular_season_weeks": int(regular_weeks),
            "max_week_in_data": max_week,
            "eval_start_week": int(start_week),
            "eval_end_week": int(eval_end_week) if eval_end_week is not None else None,
            "incomplete_regular_season": bool(incomplete_regular),
        }

    return {
        "include_postseason": bool(include_postseason),
        "incomplete_seasons": incomplete_seasons,
        "seasons": season_rows,
    }


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
    market_transform: bool | None,
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
    sample_weight: np.ndarray | None = None,
) -> ml_model.WinProbCalibrator | None:
    method = method.lower()
    if method == "none":
        return None
    unique = np.unique(actual_home_win)
    if len(unique) < 2:
        log.info("Calibration skipped: only one outcome class present.")
        return None
    return ml_model._fit_win_prob_calibrator(
        pred_margin, actual_home_win, method, sample_weight=sample_weight
    )


def _resolve_xgb_params(config: WalkForwardConfig) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if config.xgb_params_overrides:
        overrides.update(config.xgb_params_overrides)
    overrides.setdefault("random_state", config.random_seed)
    return ml_model._resolve_xgb_params(ml_model.DEFAULT_XGB_PARAMS, overrides=overrides)


def _modelling_source_files() -> list[Path]:
    """Return the source files whose contents can change a fold's result."""
    ml_dir = Path(__file__).resolve().parent
    package_dir = ml_dir.parent
    files = sorted(ml_dir.glob("*.py"))
    files.extend(package_dir / name for name in ("constants.py", "ml_model.py"))
    return [path for path in files if path.exists()]


def fold_checkpoint_fingerprint(df: pd.DataFrame, config: WalkForwardConfig) -> str:
    """Return the key that decides whether a saved fold may be reused.

    It covers everything a fold's result depends on: the input rows and columns, the full
    config, the installed library versions, and the source of the modelling code. A saved
    fold is reused only when all of these are unchanged, so a resumed run reproduces an
    uninterrupted one instead of mixing results computed from different inputs.
    """
    digest = hashlib.sha256()
    digest.update(f"fold-checkpoint-v{FOLD_CHECKPOINT_VERSION}".encode())
    digest.update(json.dumps(config.to_dict(), sort_keys=True, default=str).encode())
    digest.update(json.dumps([str(column) for column in df.columns]).encode())
    digest.update(json.dumps([str(dtype) for dtype in df.dtypes]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    digest.update(json.dumps(_library_versions(), sort_keys=True).encode())
    for path in _modelling_source_files():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _RestoredFold(NamedTuple):
    """A finished fold read back from its checkpoint."""

    metrics: dict[str, Any]
    predictions: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class _FoldCheckpointStore:
    """Per-fold checkpoint files for one fingerprinted walk-forward run.

    Files are written with `joblib`, like model checkpoints, because it round-trips pandas
    dtypes and float64 values exactly. They are only ever read back from a directory this
    module wrote, and only when the stored fingerprint matches the current run.
    """

    directory: Path
    fingerprint: str

    @classmethod
    def create(
        cls, root: Path, df: pd.DataFrame, config: WalkForwardConfig
    ) -> _FoldCheckpointStore:
        """Open (creating if needed) the checkpoint directory for this run's inputs."""
        fingerprint = fold_checkpoint_fingerprint(df, config)
        directory = Path(root) / fingerprint[:20]
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            manifest = {
                "fingerprint": fingerprint,
                "version": FOLD_CHECKPOINT_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "config": config.to_dict(),
                "library_versions": _library_versions(),
            }
            temporary = manifest_path.with_name(f"{manifest_path.name}.tmp")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
            os.replace(temporary, manifest_path)
        log.info("Walk-forward fold checkpoints: %s", directory)
        return cls(directory=directory, fingerprint=fingerprint)

    def _path(self, fold: WalkForwardFold) -> Path:
        """Return the checkpoint file for one fold."""
        return self.directory / f"fold_{int(fold.season)}_w{int(fold.week):02d}.joblib"

    def load(self, fold: WalkForwardFold) -> _RestoredFold | None:
        """Return the saved fold, or None when it is missing, unreadable, or foreign."""
        path = self._path(fold)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            AttributeError,
            ImportError,
            KeyError,
            IndexError,
            UnpicklingError,
        ) as error:
            log.warning("Ignoring unreadable checkpoint %s (%s); training again.", path, error)
            return None
        if not (
            isinstance(payload, dict)
            and payload.get("version") == FOLD_CHECKPOINT_VERSION
            and payload.get("fingerprint") == self.fingerprint
            and payload.get("season") == int(fold.season)
            and payload.get("week") == int(fold.week)
            and isinstance(payload.get("metrics"), dict)
            and isinstance(payload.get("predictions"), pd.DataFrame)
            and isinstance(payload.get("feature_columns"), list)
        ):
            log.warning("Ignoring checkpoint %s: it belongs to another run; training again.", path)
            return None
        return _RestoredFold(
            metrics=payload["metrics"],
            predictions=payload["predictions"],
            feature_columns=list(payload["feature_columns"]),
        )

    def save(
        self,
        fold: WalkForwardFold,
        metrics: dict[str, Any],
        predictions: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:
        """Write one finished fold atomically, so a stop mid-write never leaves a bad file."""
        path = self._path(fold)
        temporary = path.with_name(f"{path.name}.tmp")
        joblib.dump(
            {
                "version": FOLD_CHECKPOINT_VERSION,
                "fingerprint": self.fingerprint,
                "season": int(fold.season),
                "week": int(fold.week),
                "metrics": metrics,
                "predictions": predictions,
                "feature_columns": feature_columns,
            },
            temporary,
        )
        os.replace(temporary, path)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    fold_callback: Callable[[dict[str, Any], WalkForwardFold], None] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run walk-forward training/evaluation and return metrics plus per-game predictions.

    Any feature groups named in `config.disabled_feature_groups` are dropped here, so the
    config alone determines the ablation. The CLI scripts also drop them before building the
    config (to log and record exactly what went away); dropping again is a no-op.

    When `checkpoint_dir` is given, every finished week is saved under a subdirectory named
    for `fold_checkpoint_fingerprint(df, config)`. With `resume` (the default), a week whose
    checkpoint matches is restored instead of trained; with `resume=False` every week is
    trained and its checkpoint overwritten. Weeks are independent and seeded, so a resumed
    run returns exactly what an uninterrupted one would. `fold_callback` fires only for
    weeks that are trained, not for restored ones.
    """
    np.random.seed(config.random_seed)  # noqa: NPY002 (legacy for reproducibility)
    # Fingerprint the caller's frame before any column drops or row filters below.
    store = (
        _FoldCheckpointStore.create(checkpoint_dir, df, config)
        if checkpoint_dir is not None
        else None
    )
    if config.disabled_feature_groups:
        group_columns = resolve_feature_group_columns(
            list(df.columns), config.disabled_feature_groups
        )
        if group_columns:
            log.info(
                "Dropping %d columns for disabled feature groups: %s",
                len(group_columns),
                list(config.disabled_feature_groups),
            )
            df = df.drop(columns=group_columns)
    df = filter_regular_season(df, include_postseason=config.include_postseason)
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
        "market_prob_source": config.market_prob_source,
        "market_prob_blend_method": config.market_prob_blend_method,
        "win_prob_use_uncertainty": config.win_prob_use_uncertainty,
        "include_quantiles": config.include_quantiles,
        "disable_pruning": config.disable_pruning,
        "exclude_incomplete_seasons": config.exclude_incomplete_seasons,
    }

    eval_seasons = resolve_eval_seasons(df, config.eval_seasons, config.eval_last_n_seasons)
    excluded_incomplete: list[int] = []
    if config.exclude_incomplete_seasons:
        eval_seasons, excluded_incomplete = filter_incomplete_eval_seasons(df, eval_seasons)
        if excluded_incomplete:
            log.info("Excluded incomplete seasons from eval window: %s", excluded_incomplete)
        if not eval_seasons:
            raise ValueError("No complete seasons available after filtering eval seasons.")
    resolved_eval_seasons = [int(season) for season in eval_seasons]
    folds = build_walk_forward_folds(
        df,
        eval_seasons,
        config.wf_start_week,
        include_postseason=config.include_postseason,
    )
    if not folds:
        raise ValueError("No walk-forward folds available with the provided settings.")

    if config.win_prob_use_uncertainty and not config.include_quantiles:
        log.info("Uncertainty-aware win probs requested without quantiles; using fallback sigma.")

    params = _resolve_xgb_params(config)

    per_week_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_list: list[str] | None = None
    restored_folds = 0
    computed_folds = 0

    run_start = time.perf_counter()
    for fold_index, fold in enumerate(folds, start=1):
        restored = store.load(fold) if store is not None and resume else None
        if restored is not None:
            if feature_list is None:
                feature_list = restored.feature_columns
            per_week_metrics.append(restored.metrics)
            prediction_frames.append(restored.predictions)
            restored_folds += 1
            log.info(
                "Walk-forward fold %d/%d restored from checkpoint: season %d week %d",
                fold_index,
                len(folds),
                int(fold.season),
                int(fold.week),
            )
            continue

        feature_spec = ml_model._build_feature_spec(
            fold.train_df,
            include_market=include_market,
            max_cardinality_ratio=config.max_cardinality_ratio,
            feature_start=config.feature_start,
            feature_end=config.feature_end,
            market_transform=market_transform,
            disable_pruning=config.disable_pruning,
        )
        fold_features = list(feature_spec.feature_columns)
        if feature_list is None:
            feature_list = fold_features
        preprocessor = ml_model._build_preprocessor(feature_spec, for_tree=True)

        x_train = preprocessor.fit_transform(
            ml_model.apply_feature_spec(fold.train_df, feature_spec)
        )
        y_margin_train, y_total_train, _, _ = ml_model._prepare_margin_total_targets_with_anchor(
            fold.train_df, target_columns, market_anchor
        )
        train_recency = compute_recency_sample_weight(
            fold.train_df,
            half_life_weeks=config.recency_half_life_weeks,
            half_life_seasons=config.recency_half_life_seasons,
        )
        train_weight = combine_sample_weights(train_recency)

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
            sample_weight=train_weight,
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
                sample_weight=train_weight,
            )
            total_quantiles = ml_model._fit_quantile_models(
                x_train,
                y_total_train,
                params,
                quantiles,
                x_eval=x_calibration,
                y_eval=y_total_calibration,
                early_stopping_rounds=config.early_stopping_rounds,
                sample_weight=train_weight,
            )

        calibrator = None
        resolved_calibration = ml_model.resolve_win_prob_calibration_method(
            config.calibration,
            len(calibration_df),
        )
        if config.win_prob_use_uncertainty and resolved_calibration == "elo":
            log.info("Elo calibration ignored for uncertainty-aware probabilities; using 'none'.")
            resolved_calibration = "none"
        calibration_method = resolved_calibration
        if resolved_calibration != "none":
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
                pred_margin_inputs = pred_margin_calibration
                if config.win_prob_use_uncertainty:
                    pred_margin_quantiles_calibration = {
                        q: ml_model._predict_xgb(q_model, x_calibration)
                        for q, q_model in margin_quantiles.items()
                    }
                    if market_anchor and baseline_margin_calibration is not None:
                        for q in list(pred_margin_quantiles_calibration.keys()):
                            pred_margin_quantiles_calibration[q] = (
                                pred_margin_quantiles_calibration[q] + baseline_margin_calibration
                            )
                    sigma_calibration = ml_model._resolve_margin_sigma(
                        pred_margin_calibration,
                        pred_margin_quantiles_calibration,
                        fallback=constants.SCORE_DIFF_STD_DEV,
                    )
                    pred_margin_inputs = pred_margin_calibration / sigma_calibration
                away_col, home_col = target_columns
                actual_home_win = (calibration_df[home_col] > calibration_df[away_col]).astype(int)
                calibration_recency = compute_recency_sample_weight(
                    calibration_df,
                    half_life_weeks=config.recency_half_life_weeks,
                    half_life_seasons=config.recency_half_life_seasons,
                )
                calibrator = _fit_calibrator(
                    pred_margin_inputs,
                    actual_home_win.to_numpy(),
                    resolved_calibration,
                    sample_weight=calibration_recency,
                )
                calibration_method = "none" if calibrator is None else calibrator.method

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
        sigma_eval = None
        if config.win_prob_use_uncertainty:
            sigma_eval = ml_model._resolve_margin_sigma(
                pred_margin,
                pred_margin_quantiles,
                fallback=constants.SCORE_DIFF_STD_DEV,
            )
        home_win_prob = ml_model.predict_home_win_prob(
            pred_margin,
            calibrator,
            sigma=sigma_eval,
            use_uncertainty=config.win_prob_use_uncertainty,
        )
        if config.market_prob_weight or config.market_prob_clamp:
            market_prob_config = ml_model.MarketProbConfig(
                blend_weight=config.market_prob_weight,
                clamp_delta=config.market_prob_clamp,
                prob_source=config.market_prob_source,
                blend_method=config.market_prob_blend_method,
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

        games_count = int(len(fold_predictions))
        metrics = {
            "season": int(fold.season),
            "week": int(fold.week),
            "games": games_count,
            **metrics_utils.margin_total_metrics(
                actual_margin, actual_total, pred_margin, pred_total
            ),
            **metrics_utils.probability_metrics(actual_home_win, home_win_prob),
            **metrics_utils.confidence_pool_summary(confidence_cols),
            "calibration_method": calibration_method,
        }
        metrics["reliability_ece"] = metrics_utils.reliability_ece(
            metrics_utils.reliability_table(
                home_win_prob,
                actual_home_win,
                bins=RELIABILITY_BINS,
            )
        )
        picks_correct = int(metrics["picks_correct"])
        metrics["pick_accuracy"] = picks_correct / games_count if games_count else 0.0

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
        # Saved before the callback, so a callback that stops the run keeps this week.
        if store is not None:
            store.save(fold, metrics, fold_predictions, fold_features)
        computed_folds += 1

        if fold_callback is not None:
            fold_callback(metrics, fold)

        per_week_metrics.append(metrics)

        # Nothing else is logged between the first weeks and the final report, so this
        # line is the only progress signal a long run gives. The remaining-time estimate
        # assumes the average trained fold so far (restored folds cost nothing), which
        # runs low late in a run as training sets grow.
        elapsed = time.perf_counter() - run_start
        log.info(
            "Walk-forward fold %d/%d done: season %d week %d (%d games, Brier %.4f), "
            "%.0fs elapsed, about %.0fs remaining",
            fold_index,
            len(folds),
            int(fold.season),
            int(fold.week),
            games_count,
            float(metrics["brier"]),
            elapsed,
            elapsed / computed_folds * (len(folds) - fold_index),
        )

    if store is not None:
        log.info(
            "Walk-forward checkpoints: %d weeks restored, %d trained (%s)",
            restored_folds,
            computed_folds,
            store.directory,
        )

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
    season_win_totals = _season_win_totals(predictions)
    calibration_drift = _calibration_drift(predictions)

    return {
        "per_week": per_week_metrics,
        "per_season": per_season_metrics,
        "overall": overall_metrics,
        "reliability": reliability,
        "season_win_totals": season_win_totals,
        "calibration_drift": calibration_drift,
        "predictions": predictions,
        "resolved_settings": resolved_settings,
        "resolved_eval_seasons": resolved_eval_seasons,
        "feature_list": feature_list,
        "eval_window": summarize_eval_window(
            df,
            resolved_eval_seasons,
            start_week=config.wf_start_week,
            include_postseason=config.include_postseason,
        ),
        "excluded_incomplete_seasons": excluded_incomplete,
        "checkpoint": (
            None
            if store is None
            else {
                "dir": str(store.directory),
                "restored_folds": restored_folds,
                "computed_folds": computed_folds,
            }
        ),
    }


def build_metrics_report(
    run_id: str,
    created_at: str,
    config_payload: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serializable metrics report payload."""
    fold_summary = _summarize_fold_metrics(results["per_week"])
    summary_table = _build_metrics_summary_table(results["overall"], fold_summary)
    return {
        "run_id": run_id,
        "created_at": created_at,
        "config": config_payload,
        "metric_strategy": metrics_utils.METRIC_STRATEGY,
        "metrics": {
            "per_week": results["per_week"],
            "per_season": results["per_season"],
            "overall": results["overall"],
            "fold_summary": fold_summary,
            "summary_table": summary_table,
        },
        "calibration": {
            "bins": results["reliability"],
            "bin_count": RELIABILITY_BINS,
        },
        "diagnostics": {
            "season_win_totals": results.get("season_win_totals"),
            "calibration_drift": results.get("calibration_drift"),
        },
        "splits": {
            "eval_window": results.get("eval_window"),
            "excluded_incomplete_seasons": results.get("excluded_incomplete_seasons", []),
            "calibration_window": {
                "method": config_payload.get("calibration"),
                "calibration_weeks": config_payload.get("calibration_weeks"),
            },
        },
    }


def _summarize_fold_metrics(per_week: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-week metrics with mean/variance across folds."""
    summary: dict[str, Any] = {"folds": int(len(per_week)), "metrics": {}}
    if not per_week:
        for name in SUMMARY_METRICS:
            summary["metrics"][name] = {"mean": None, "variance": None}
        return summary

    for name in SUMMARY_METRICS:
        values = [row.get(name) for row in per_week if row.get(name) is not None]
        if not values:
            summary["metrics"][name] = {"mean": None, "variance": None}
            continue
        arr = np.asarray(values, dtype=float)
        summary["metrics"][name] = {
            "mean": float(np.mean(arr)),
            "variance": float(np.var(arr)),
        }
    return summary


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
    weeks_count = int(frame["week"].nunique()) if "week" in frame.columns else None
    games_count = int(len(frame))
    expected_points_total = float(frame["expected_points"].sum())
    actual_points_total = float(frame["actual_points"].sum())
    picks_correct = int(frame["pick_correct"].sum())
    metrics = {
        "season": season_value,
        "weeks": weeks_count,
        "games": games_count,
        **metrics_utils.margin_total_metrics(actual_margin, actual_total, pred_margin, pred_total),
        **metrics_utils.probability_metrics(actual_home_win, home_win_prob),
        "expected_points": expected_points_total,
        "actual_points": actual_points_total,
        "picks_correct": picks_correct,
    }
    reliability_bins = metrics_utils.reliability_table(
        home_win_prob,
        actual_home_win,
        bins=RELIABILITY_BINS,
    )
    metrics["reliability_ece"] = metrics_utils.reliability_ece(reliability_bins)
    metrics["pick_accuracy"] = picks_correct / games_count if games_count else 0.0
    if weeks_count:
        metrics["expected_points_avg"] = expected_points_total / weeks_count
        metrics["actual_points_avg"] = actual_points_total / weeks_count

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


def _resolve_team_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Resolve team identifier columns for diagnostics."""
    for candidates in (("away_abbr", "home_abbr"), ("away_name", "home_name")):
        if all(col in frame.columns for col in candidates):
            return candidates
    return None


def _season_win_totals(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute per-team season win totals vs expected wins."""
    if predictions.empty:
        return {"per_team": [], "per_season": [], "overall": None}
    team_cols = _resolve_team_columns(predictions)
    required = {"season", "home_win_prob", "actual_margin"}
    if team_cols is None or not required.issubset(predictions.columns):
        return {"per_team": [], "per_season": [], "overall": None}

    away_col, home_col = team_cols
    home_win_prob = predictions["home_win_prob"].to_numpy(dtype=float)
    actual_margin = predictions["actual_margin"].to_numpy(dtype=float)
    actual_home_win = np.where(
        actual_margin > 0,
        1.0,
        np.where(actual_margin < 0, 0.0, 0.5),
    )

    home_rows = pd.DataFrame(
        {
            "season": predictions["season"].to_numpy(),
            "team": predictions[home_col].to_numpy(),
            "expected_wins": home_win_prob,
            "actual_wins": actual_home_win,
        }
    )
    away_rows = pd.DataFrame(
        {
            "season": predictions["season"].to_numpy(),
            "team": predictions[away_col].to_numpy(),
            "expected_wins": 1.0 - home_win_prob,
            "actual_wins": 1.0 - actual_home_win,
        }
    )
    combined = pd.concat([home_rows, away_rows], ignore_index=True)
    combined = combined.dropna(subset=["season", "team"])
    if combined.empty:
        return {"per_team": [], "per_season": [], "overall": None}

    grouped = combined.groupby(["season", "team"], as_index=False).agg(
        expected_wins=("expected_wins", "sum"),
        actual_wins=("actual_wins", "sum"),
        games=("expected_wins", "size"),
    )
    grouped["error"] = grouped["expected_wins"] - grouped["actual_wins"]
    grouped["abs_error"] = grouped["error"].abs()

    per_team = [
        {
            "season": _scalar_to_int(row["season"]),
            "team": str(row["team"]),
            "expected_wins": float(row["expected_wins"]),
            "actual_wins": float(row["actual_wins"]),
            "games": int(row["games"]),
            "error": float(row["error"]),
            "abs_error": float(row["abs_error"]),
        }
        for _, row in grouped.sort_values(["season", "team"]).iterrows()
    ]

    def _summarize_totals(frame: pd.DataFrame, season: int | None) -> dict[str, Any]:
        errors = frame["error"].to_numpy(dtype=float)
        abs_errors = frame["abs_error"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else None
        return {
            "season": season,
            "teams": int(frame["team"].nunique()),
            "games": int(frame["games"].sum()),
            "mean_abs_error": float(np.mean(abs_errors)) if len(abs_errors) else None,
            "median_abs_error": float(np.median(abs_errors)) if len(abs_errors) else None,
            "max_abs_error": float(np.max(abs_errors)) if len(abs_errors) else None,
            "rmse": rmse,
        }

    per_season = [
        _summarize_totals(season_df, _scalar_to_int(season))
        for season, season_df in grouped.groupby("season")
    ]
    overall = _summarize_totals(grouped, None)

    return {"per_team": per_team, "per_season": per_season, "overall": overall}


def _calibration_drift(predictions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Summarize calibration drift by season and week."""
    required = {"season", "week", "home_win_prob", "actual_home_win"}
    if predictions.empty or not required.issubset(predictions.columns):
        return {"per_week": [], "per_season": []}

    def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
        home_win_prob = frame["home_win_prob"].to_numpy(dtype=float)
        actual_home_win = frame["actual_home_win"].to_numpy(dtype=float)
        prob_metrics = metrics_utils.probability_metrics(actual_home_win, home_win_prob)
        avg_pred = float(np.mean(home_win_prob)) if len(home_win_prob) else None
        avg_actual = float(np.mean(actual_home_win)) if len(actual_home_win) else None
        bias = avg_pred - avg_actual if avg_pred is not None and avg_actual is not None else None
        return {
            "games": int(len(frame)),
            "avg_pred": avg_pred,
            "avg_actual": avg_actual,
            "bias": bias,
            "abs_bias": abs(bias) if bias is not None else None,
            **prob_metrics,
        }

    per_week: list[dict[str, Any]] = []
    for (season, week), frame in predictions.groupby(["season", "week"]):
        row = _summarize(frame)
        row["season"] = _scalar_to_int(season)
        row["week"] = _scalar_to_int(week)
        per_week.append(row)

    per_season: list[dict[str, Any]] = []
    for season, frame in predictions.groupby("season"):
        row = _summarize(frame)
        row["season"] = _scalar_to_int(season)
        per_season.append(row)

    per_week.sort(key=lambda item: (item["season"], item["week"]))
    per_season.sort(key=lambda item: item["season"])

    return {"per_week": per_week, "per_season": per_season}


def _build_metrics_summary_table(
    overall: dict[str, Any],
    fold_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a summary table for first-class metrics."""
    rows: list[dict[str, Any]] = []
    fold_metrics = fold_summary.get("metrics", {}) if fold_summary else {}
    for priority, specs in metrics_utils.METRIC_STRATEGY.items():
        for spec in specs:
            metric = spec["metric"]
            stats = fold_metrics.get(metric, {})
            rows.append(
                {
                    "metric": metric,
                    "priority": priority,
                    "direction": spec.get("direction"),
                    "overall": overall.get(metric),
                    "fold_mean": stats.get("mean"),
                    "fold_variance": stats.get("variance"),
                }
            )
    return rows


def dataset_fingerprint(path: Path) -> str:
    """Compute a SHA-256 fingerprint of the dataset file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_run_id(dataset_hash: str, config: WalkForwardConfig) -> str:
    """Generate a stable-ish run id from timestamp + config hash."""
    created = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    payload = json.dumps(config.to_dict(), sort_keys=True)
    short_hash = hashlib.sha256(f"{dataset_hash}:{payload}".encode()).hexdigest()[:8]
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


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 (git is fixed, not user-controlled)
            cwd=str(constants.ROOT_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
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
        except ImportError:
            versions[module_name] = None
            continue
        versions[module_name] = getattr(module, "__version__", None)
    return versions
