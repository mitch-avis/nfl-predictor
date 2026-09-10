"""Dataset and ETL status routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.readers import data_status as reader
from nfl_predictor.api.schemas.data import DataFileOut, DataStatusOut

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status", response_model=DataStatusOut)
def status(_user: CurrentUser, db: DbDep, settings: SettingsDep, request: Request) -> DataStatusOut:
    """Describe the dataset files, their freshness, cache coverage, and the latest leakage audit."""
    fingerprints: reader.FingerprintCache = request.app.state.fingerprints
    season, week = reader.current_season_week()
    files = [
        DataFileOut(**reader.file_status(settings.data_path, name, description).__dict__)
        for name, description in reader.DATASET_FILES
    ]
    return DataStatusOut(
        current_season=season,
        current_week=week,
        files=files,
        predict_files=reader.unattached_files(settings.data_path, settings.reports_path),
        fingerprint=fingerprints.get(settings.data_path / "completed_games_ml.csv"),
        cache=reader.cache_coverage(settings.data_path / "cache" / "nflreadpy"),
        leakage_audit=reader.latest_leakage_audit(settings.models_path, settings.reports_path),
    )


@router.get("/unattached", response_model=list[dict[str, Any]])
def unattached(_user: CurrentUser, settings: SettingsDep) -> list[dict[str, Any]]:
    """List ad-hoc prediction files and workbooks not owned by a run."""
    return reader.unattached_files(settings.data_path, settings.reports_path)
