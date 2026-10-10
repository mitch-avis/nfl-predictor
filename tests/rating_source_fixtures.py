"""Walk-forward fold checkpoints for the rating-source tests, in the backtest's file format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import pandas as pd

from nfl_predictor.ml.walk_forward import FOLD_CHECKPOINT_VERSION

if TYPE_CHECKING:
    from pathlib import Path


def fold_frame(season: int, week: int, games: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Return a fold's prediction frame as the walk-forward stores it (extra columns included).

    Each game is ``(away, home, predicted margin)``.
    """
    return pd.DataFrame(
        {
            "game_id": [f"{season}_{week:02d}_{away}_{home}" for away, home, _ in games],
            "season": [season] * len(games),
            "week": [week] * len(games),
            "game_type": ["REG"] * len(games),
            "away_abbr": [away for away, _, _ in games],
            "home_abbr": [home for _, home, _ in games],
            "neutral": [0] * len(games),
            "home_rest": [7] * len(games),
            "predicted_margin": [margin for _, _, margin in games],
            "deterministic_home_win_prob": [0.5] * len(games),
        }
    )


def write_fold_checkpoints(directory: Path, frames: list[pd.DataFrame]) -> Path:
    """Write one ``fold_<season>_w<week>.joblib`` file per frame, as a walk-forward does."""
    directory.mkdir(parents=True)
    for frame in frames:
        season, week = int(frame["season"].iloc[0]), int(frame["week"].iloc[0])
        payload = {
            "version": FOLD_CHECKPOINT_VERSION,
            "fingerprint": "f" * 64,
            "season": season,
            "week": week,
            "metrics": {"margin_model.best_iteration": 199},
            "predictions": frame,
            "feature_columns": ["home_rest", "neutral"],
        }
        joblib.dump(payload, directory / f"fold_{season}_w{week:02d}.joblib")
    return directory
