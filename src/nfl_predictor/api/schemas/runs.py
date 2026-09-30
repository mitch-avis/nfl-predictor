"""Run payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class RunOut(BaseModel):
    """One run directory."""

    run_id: str
    created_at: str
    kind: Literal["weekly", "training", "walk_forward"]
    season: int | None
    week: int | None
    stages: dict[str, bool]
    complete: bool
    git_commit_hash: str | None
    dataset_hash: str | None
    model_kind: str | None
    holdout: dict[str, float] | None
    files: dict[str, bool]
    is_active: bool


class RunListOut(BaseModel):
    """The run list plus the active pointer."""

    active_run_id: str | None
    pinned_run_id: str | None
    runs: list[RunOut]


class RunDetailOut(RunOut):
    """A run plus its training configuration."""

    config: dict[str, Any] | None
    splits: dict[str, Any] | None
    library_versions: dict[str, Any] | None
    feature_count: int | None
