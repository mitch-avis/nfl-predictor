"""Request and response shapes for the job routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParamSpecOut(BaseModel):
    """One parameter of a job template, as the frontend form renders it."""

    name: str
    label: str
    kind: str
    description: str
    required: bool
    default: Any = None
    choices: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None


class JobTemplateOut(BaseModel):
    """A launchable job template."""

    id: str
    label: str
    description: str
    category: str
    exclusive_group: str | None
    chain_template_id: str | None
    writes_datasets: bool
    needs_active_run: bool
    params: list[ParamSpecOut]


class JobCatalogOut(BaseModel):
    """Every template plus the exclusive groups that are currently busy."""

    templates: list[JobTemplateOut]
    busy_groups: list[str]


class JobCreateIn(BaseModel):
    """A job submission."""

    template_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    """A job record."""

    id: str
    template_id: str
    template_label: str
    params: dict[str, Any]
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    created_by: str | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None
    parent_job_id: str | None = None


class JobListOut(BaseModel):
    """Recent jobs, newest first."""

    jobs: list[JobOut]
    busy_groups: list[str]


class JobLogLineOut(BaseModel):
    """One stored output line."""

    seq: int
    ts: str
    level: str
    line: str


class JobLogsOut(BaseModel):
    """A page of a job's logs plus the cursor to continue from."""

    job_id: str
    status: str
    next_seq: int
    lines: list[JobLogLineOut]
