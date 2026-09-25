"""Weekly-run inputs and outputs: which prediction file to use and where results go."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

_WEEK_FILE_RE = re.compile(r"week_(\d+)_games_to_predict", re.IGNORECASE)


def _default_power_rankings_through_week(season: int | None, week: int | None) -> int | None:
    """Derive the default power-rankings through-week for a prediction week.

    The rankings read a strength snapshot for ``through_week + 1``, and the ETL writes
    snapshots only through the week after the regular season, so a postseason prediction
    week is clamped back to the last regular-season week.

    Args:
        season: Season of the prediction week, when known.
        week: Prediction week, when known.

    Returns:
        The through-week to rank on, or ``None`` when no week is known.

    """
    if week is None:
        return None
    derived = max(week - 1, 0)
    if season is None:
        return derived
    last_regular_week = constants.get_regular_season_weeks(season)
    if derived > last_regular_week:
        log.info(
            "Prediction week %s is postseason; power rankings through the regular season week %s.",
            week,
            last_regular_week,
        )
        return last_regular_week
    return derived


def _extract_week(path: Path) -> int | None:
    match = _WEEK_FILE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _parse_prediction_file_int(value: object) -> int | None:
    """Parse an integer-like CSV field from a prediction file row."""
    if not isinstance(value, str) or value == "":
        return None
    return int(float(value))


def _predict_file_sort_key(path: Path) -> tuple[int, int, str]:
    """Return a sortable `(season, week, name)` key for a prediction input file."""
    week = _extract_week(path)
    season = -1
    resolved_week = week if week is not None else -1
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle), None)
    except OSError:
        row = None

    if row is not None:
        parsed_season = _parse_prediction_file_int(row.get("season"))
        parsed_week = _parse_prediction_file_int(row.get("week"))
        if parsed_season is not None:
            season = parsed_season
        if parsed_week is not None:
            resolved_week = parsed_week

    return season, resolved_week, path.name


def _resolve_predict_path(predict_path: Path | None, data_dir: Path) -> Path:
    """Resolve the default prediction file path when not provided."""
    if predict_path is not None:
        if not predict_path.exists():
            raise FileNotFoundError(f"Missing predict dataset: {predict_path}")
        return predict_path

    predict_dir = data_dir / "predict"
    if not predict_dir.exists():
        raise FileNotFoundError(f"Missing predict directory: {predict_dir}")

    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for candidate in predict_dir.glob("week_*_games_to_predict.csv"):
        week = _extract_week(candidate)
        if week is not None:
            candidates.append((_predict_file_sort_key(candidate), candidate))
    if not candidates:
        raise FileNotFoundError(f"No week_XX_games_to_predict.csv files found in {predict_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _infer_season_week(df: pd.DataFrame) -> tuple[int | None, int | None]:
    """Infer a single season/week from a prediction frame."""
    season = None
    week = None
    if "season" in df.columns:
        seasons = pd.Series(df["season"]).dropna().unique()
        if len(seasons) == 1:
            season = int(seasons[0])
    if "week" in df.columns:
        weeks = pd.Series(df["week"]).dropna().unique()
        if len(weeks) == 1:
            week = int(weeks[0])
    return season, week


def _resolve_output_paths(
    output_dir: Path,
    season: int | None,
    week: int | None,
) -> dict[str, Path]:
    """Resolve output paths for predictions and reports."""
    if season is not None and week is not None:
        suffix = f"season_{season}_week_{week:02d}"
    else:
        suffix = "weekly"

    return {
        "predictions": output_dir / f"{suffix}_predictions.csv",
        "confidence_picks": output_dir / f"{suffix}_confidence_picks.csv",
        "betting_report": output_dir / f"{suffix}_betting_report.csv",
    }
