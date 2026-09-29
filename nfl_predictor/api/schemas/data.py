"""Payloads for predictions, betting, power rankings, model, and data status routes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from nfl_predictor.api.schemas.common import TablePayload


class WeekQuery(BaseModel):
    """Which week's predictions a request asks for: a run, or a season and week."""

    run: str | None = None
    season: int | None = None
    week: int | None = None


class PredictionQuery(WeekQuery):
    """A week query that may also insist on an unattached ``data/predict`` file."""

    source: str | None = None


class WeekRef(BaseModel):
    """A week for which predictions exist, and where they live."""

    season: int | None
    week: int | None
    source: str
    run_id: str | None
    label: str


class PredictionsOut(BaseModel):
    """A week of predictions."""

    run_id: str | None
    source: str
    season: int | None
    week: int | None
    generated_at: str | None
    table: TablePayload
    summary: dict[str, Any]
    weeks: list[WeekRef]


class PicksOut(BaseModel):
    """Confidence picks for a week."""

    run_id: str | None
    season: int | None
    week: int | None
    table: TablePayload


class LadderStep(BaseModel):
    """One rung of the action ladder."""

    action: str
    min_edge: float


class BettingOut(BaseModel):
    """The betting report for a week."""

    run_id: str | None
    season: int | None
    week: int | None
    generated_at: str | None
    table: TablePayload
    ladder: list[LadderStep]
    notes: list[str]


class PowerOut(BaseModel):
    """Power rankings and projected standings."""

    run_id: str
    season: int | None
    through_week: int | None
    previous_run_id: str | None
    rankings: TablePayload
    standings: TablePayload | None
    division_standings: TablePayload | None


class FeatureImportanceOut(BaseModel):
    """Top base features and the importance measure that ranks them.

    ``mean_abs_shap`` is the mean absolute SHAP value in points over the final model's training
    rows, the headline measure; ``total_gain`` is XGBoost's loss reduction summed over every
    split, for runs written before SHAP was recorded; ``summed_average_gain`` is the average
    gain per split summed over encoded columns and heads, the only measure the oldest runs
    recorded. ``None`` means the run recorded no usable importance.
    """

    measure: Literal["mean_abs_shap", "total_gain", "summed_average_gain"] | None
    rows: list[dict[str, Any]]


class ModelOut(BaseModel):
    """Everything the Model page shows."""

    run_id: str
    kind: str
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    feature_importance: FeatureImportanceOut
    calibration: dict[str, Any] | None
    wf_compare: TablePayload | None
    wf_best: dict[str, Any] | None
    shap_available: bool


class DataFileOut(BaseModel):
    """One dataset file."""

    name: str
    description: str
    exists: bool
    size: int | None
    modified_at: str | None
    rows: int | None
    seasons: tuple[int, int] | None


class DataStatusOut(BaseModel):
    """ETL and dataset health."""

    current_season: int
    current_week: int
    files: list[DataFileOut]
    predict_files: list[dict[str, Any]]
    fingerprint: dict[str, Any] | None
    cache: dict[str, list[int]]
    leakage_audit: dict[str, Any] | None
