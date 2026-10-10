"""Final scores of played games, read from the ETL's full dataset for pool scoring.

A predictions file is written before its games, so its score columns stay empty; the ETL's
``all_data.csv`` gains each final score at the next refresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.api.readers.cache import cached
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from pathlib import Path

RESULTS_FILE = "all_data.csv"
SCORE_COLUMNS = ["game_id", "away_score", "home_score"]


def _read_scores(path: Path) -> dict[str, tuple[int, int]]:
    """Read ``game_id -> (away_score, home_score)`` for every game with both scores."""
    try:
        frame = pl.read_csv(path, columns=SCORE_COLUMNS, infer_schema_length=10000)
    except pl.exceptions.PolarsError as exc:
        log.warning("No final scores from %s: %s", path, exc)
        return {}
    played = frame.drop_nulls(["away_score", "home_score"])
    return {
        str(game_id): (int(away), int(home))
        for game_id, away, home in played.select(SCORE_COLUMNS).iter_rows()
    }


def final_scores(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Return the final score of every played game in ``data_dir``'s full dataset.

    Returns an empty mapping when the dataset is missing or has no score columns.
    """
    path = data_dir / RESULTS_FILE
    if not path.is_file():
        return {}
    scores: dict[str, tuple[int, int]] = cached(path, _read_scores, "final_scores")
    return scores
