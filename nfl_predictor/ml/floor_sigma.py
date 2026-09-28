"""The spread of the probability floor, estimated from earlier out-of-fold margin errors.

The submitted win probability is the deterministic floor ``Phi(predicted_margin / sigma)``. The
formula treats the actual margin as the predicted margin plus normal noise, so sigma is the
spread of the model's own errors. For a game in ``(season, week)``:

    sigma = sqrt(mean((actual_margin - predicted_margin) ** 2))

over every out-of-fold prediction strictly before ``(season, week)`` (earlier seasons, and the
same season's earlier weeks). It is one value per week, so it never changes a pick or a
confidence rank. Until the pool spans ``constants.FLOOR_SIGMA_MIN_POOL_SEASONS`` complete
earlier seasons, sigma is ``constants.SCORE_DIFF_STD_DEV`` and the record says it fell back.

The root mean square, not the standard deviation around the mean error, is used because the
formula centers the noise on the predicted margin.

The pool is a table of squared errors, one row per game (``ERROR_COLUMNS``). A walk-forward
pools a supplied history with its own earlier folds (``combine``); production pools the
reference runs' fold checkpoints (``load_reference_pool``) with the weekly run's own
current-season folds. Several runs of one configuration (seeds) are averaged per game
(``average_over_runs``), so every game counts once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_predictor import constants

ERROR_COLUMNS = ("game_id", "season", "week", "squared_error")
_WEEK_KEY = ["season", "week"]
_PREDICTION_COLUMNS = ("game_id", "season", "week", "actual_margin", "predicted_margin")


def empty_errors() -> pd.DataFrame:
    """Return an error pool with no games."""
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype=object),
            "season": pd.Series(dtype=int),
            "week": pd.Series(dtype=int),
            "squared_error": pd.Series(dtype=float),
        }
    )


@dataclass(frozen=True, eq=False)
class ErrorPool:
    """Squared out-of-fold margin errors, one row per game, and the runs they came from.

    Pools compare by identity; ``digest`` compares their contents.
    """

    errors: pd.DataFrame = field(default_factory=empty_errors)
    sources: tuple[str, ...] = ()

    def digest(self) -> str:
        """Return a hash of the errors and their sources, for checkpoint and stage keys."""
        frame = self.errors.loc[:, list(ERROR_COLUMNS)].sort_values("game_id")
        digest = hashlib.sha256()
        digest.update(json.dumps(list(self.sources)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class FloorSigma:
    """The sigma used for one week, and what it was estimated from."""

    sigma: float
    fallback: bool
    season: int
    week: int
    pool_games: int
    pool_seasons: tuple[int, ...]
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON record written beside predictions and saved models."""
        return {
            "sigma": float(self.sigma),
            "fallback": bool(self.fallback),
            "season": int(self.season),
            "week": int(self.week),
            "pool_games": int(self.pool_games),
            "pool_seasons": [int(season) for season in self.pool_seasons],
            "sources": list(self.sources),
            "min_pool_seasons": int(constants.FLOOR_SIGMA_MIN_POOL_SEASONS),
            "fallback_sigma": float(constants.SCORE_DIFF_STD_DEV),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FloorSigma:
        """Read a record written by ``to_dict``."""
        return cls(
            sigma=float(payload["sigma"]),
            fallback=bool(payload["fallback"]),
            season=int(payload["season"]),
            week=int(payload["week"]),
            pool_games=int(payload["pool_games"]),
            pool_seasons=tuple(int(season) for season in payload["pool_seasons"]),
            sources=tuple(str(source) for source in payload.get("sources", ())),
        )


def home_win_prob(margin: np.ndarray, sigma: float) -> np.ndarray:
    """Return the floor, ``Phi(margin / sigma)``.

    Raises:
        ValueError: If ``sigma`` is not positive.

    """
    if not sigma > 0:
        raise ValueError(f"The floor's sigma must be positive, got {sigma!r}.")
    return norm.cdf(margin / float(sigma))


def margin_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the squared margin error of every scored game in a prediction frame."""
    frame = predictions.loc[:, list(_PREDICTION_COLUMNS)].copy()
    error = frame["actual_margin"].astype(float) - frame["predicted_margin"].astype(float)
    frame["squared_error"] = error**2
    frame = frame[np.isfinite(frame["squared_error"])]
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    return frame.loc[:, list(ERROR_COLUMNS)].reset_index(drop=True)


def average_over_runs(runs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average several runs' squared errors per game, so each game counts once.

    Over runs that cover the same games this gives the same sigma as pooling every row.
    """
    frames = [run.loc[:, list(ERROR_COLUMNS)] for run in runs if not run.empty]
    if not frames:
        return empty_errors()
    stacked = pd.concat(frames, ignore_index=True)
    averaged = stacked.groupby("game_id", sort=True, as_index=False).agg(
        season=("season", "first"),
        week=("week", "first"),
        squared_error=("squared_error", "mean"),
    )
    return averaged.loc[:, list(ERROR_COLUMNS)]


def combine(history: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    """Pool a history with a run's own errors; the run's weeks replace the history's."""
    if own.empty:
        return history.loc[:, list(ERROR_COLUMNS)].reset_index(drop=True)
    own_weeks = pd.MultiIndex.from_frame(own[_WEEK_KEY].drop_duplicates())
    covered = pd.MultiIndex.from_frame(history[_WEEK_KEY]).isin(own_weeks)
    kept = history.loc[~covered, list(ERROR_COLUMNS)]
    return pd.concat([kept, own.loc[:, list(ERROR_COLUMNS)]], ignore_index=True)


def estimate(
    errors: pd.DataFrame, season: int, week: int, *, sources: Sequence[str] = ()
) -> FloorSigma:
    """Return the sigma for games in ``(season, week)`` from errors strictly before it."""
    seasons = errors["season"].to_numpy(dtype=int)
    weeks = errors["week"].to_numpy(dtype=int)
    before = (seasons < season) | ((seasons == season) & (weeks < week))
    pool = errors.loc[before].sort_values("game_id", kind="mergesort")
    pool_seasons = tuple(sorted(int(value) for value in pool["season"].unique()))
    earlier_seasons = sum(1 for value in pool_seasons if value < season)
    fallback = earlier_seasons < constants.FLOOR_SIGMA_MIN_POOL_SEASONS
    sigma = (
        float(constants.SCORE_DIFF_STD_DEV)
        if fallback
        else float(np.sqrt(pool["squared_error"].to_numpy(dtype=float).mean()))
    )
    return FloorSigma(
        sigma=sigma,
        fallback=fallback,
        season=int(season),
        week=int(week),
        pool_games=len(pool),
        pool_seasons=pool_seasons,
        sources=tuple(sources),
    )


def _resolve(path: Path) -> Path:
    """Return ``path``, relative paths taken from the repository root."""
    path = Path(path)
    return path if path.is_absolute() else Path(constants.ROOT_DIR) / path


def _checkpoint_dir(run_path: Path) -> Path:
    """Return the fold checkpoint directory of a run directory or checkpoint directory.

    Raises:
        FileNotFoundError: If the path is missing or names no fold checkpoints.

    """
    if not run_path.exists():
        raise FileNotFoundError(
            f"Floor sigma reference run {run_path} does not exist. Name existing walk-forward "
            "runs, or give an empty list to use only the run's own folds."
        )
    metadata_path = run_path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = (metadata.get("config") or {}).get("checkpoint") or {}
        if not checkpoint.get("dir"):
            raise FileNotFoundError(f"{metadata_path} names no walk-forward checkpoint directory")
        return Path(checkpoint["dir"])
    return run_path


def _read_run(run_path: Path) -> pd.DataFrame:
    """Return one reference run's squared errors from its fold checkpoints.

    Raises:
        FileNotFoundError: If the run has no fold checkpoints.
        ValueError: If the checkpoints lack a needed column or repeat a game.

    """
    checkpoint_dir = _checkpoint_dir(run_path)
    files = sorted(checkpoint_dir.glob("fold_*.joblib"))
    if not files:
        raise FileNotFoundError(f"No fold checkpoints in {checkpoint_dir} (run {run_path}).")
    predictions = pd.concat([joblib.load(path)["predictions"] for path in files], ignore_index=True)
    missing = [column for column in _PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"Fold checkpoints of {run_path} lack {missing}.")
    if predictions["game_id"].duplicated().any():
        raise ValueError(f"Fold checkpoints of {run_path} repeat a game.")
    return margin_errors(predictions)


def load_reference_pool(run_paths: Sequence[Path]) -> ErrorPool:
    """Read the reference runs' fold checkpoints (read-only) into one per-game error pool."""
    resolved = [_resolve(path) for path in run_paths]
    runs = [_read_run(path) for path in resolved]
    return ErrorPool(average_over_runs(runs), tuple(str(path) for path in resolved))


def weekly_home_win_prob(
    margin: np.ndarray, seasons: np.ndarray, weeks: np.ndarray, errors: pd.DataFrame
) -> np.ndarray:
    """Return the floor for games in several weeks, each week through its own sigma."""
    margin = np.asarray(margin)
    seasons = np.asarray(seasons, dtype=int)
    weeks = np.asarray(weeks, dtype=int)
    probs = np.empty(len(margin), dtype=float)
    for season, week in sorted(set(zip(seasons.tolist(), weeks.tolist(), strict=True))):
        games = (seasons == season) & (weeks == week)
        probs[games] = home_win_prob(margin[games], estimate(errors, season, week).sigma)
    return probs


def model_record(model: Any) -> dict[str, Any] | None:
    """Return a model's recorded floor sigma for its metadata, or None when it has none."""
    record = getattr(model, "floor_sigma", None)
    return None if record is None else record.to_dict()


def week_after(games: pd.DataFrame) -> tuple[int, int]:
    """Return the week after the newest game: its season, and its week plus one."""
    season = int(games["season"].max())
    return season, int(games.loc[games["season"] == season, "week"].max()) + 1
