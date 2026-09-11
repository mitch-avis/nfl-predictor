"""Routes for launching, watching, and cancelling jobs."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from nfl_predictor.api.deps import AdminUser, CurrentUser
from nfl_predictor.api.jobs import catalog, stream
from nfl_predictor.api.jobs.runner import JobRunner, Submission
from nfl_predictor.api.jobs.store import JobRecord
from nfl_predictor.api.schemas.jobs import (
    JobCatalogOut,
    JobCreateIn,
    JobListOut,
    JobLogLineOut,
    JobLogsOut,
    JobOut,
    JobTemplateOut,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])
MAX_JOBS = 200


def get_runner(request: Request) -> JobRunner:
    """Return the application's job runner."""
    runner: JobRunner = request.app.state.job_runner
    return runner


def to_job_out(record: JobRecord) -> JobOut:
    """Convert a job record into its API shape."""
    template = catalog.TEMPLATES_BY_ID.get(record.template_id)
    return JobOut(
        id=record.id,
        template_id=record.template_id,
        template_label=template.label if template else record.template_id,
        params=record.params,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        exit_code=record.exit_code,
        created_by=record.created_by,
        progress=record.progress,
        error=record.error,
        parent_job_id=record.parent_job_id,
    )


@router.get("/catalog", response_model=JobCatalogOut)
def job_catalog(_user: CurrentUser, request: Request) -> JobCatalogOut:
    """List every job template and which exclusive groups are busy."""
    runner = get_runner(request)
    return JobCatalogOut(
        templates=[
            JobTemplateOut(**catalog.describe(template).__dict__) for template in catalog.TEMPLATES
        ],
        busy_groups=sorted(runner.busy_groups()),
    )


@router.post("", response_model=JobOut, status_code=201)
def create_job(payload: JobCreateIn, admin: AdminUser, request: Request) -> JobOut:
    """Validate a submission and queue it."""
    template = catalog.get_template(payload.template_id)
    params = catalog.validate_params(template, payload.params)
    record = get_runner(request).submit(
        Submission(template=template, params=params, created_by=admin.username)
    )
    return to_job_out(record)


@router.get("", response_model=JobListOut)
def list_jobs(
    _user: CurrentUser,
    request: Request,
    limit: int = Query(default=50, ge=1, le=MAX_JOBS),
    template: str | None = None,
) -> JobListOut:
    """List recent jobs, newest first."""
    runner = get_runner(request)
    records = runner.store.recent(limit=limit, template_id=template)
    return JobListOut(
        jobs=[to_job_out(record) for record in records],
        busy_groups=sorted(runner.busy_groups()),
    )


@router.get("/{job_id}", response_model=JobOut)
def job_detail(job_id: str, _user: CurrentUser, request: Request) -> JobOut:
    """Return one job."""
    return to_job_out(get_runner(request).store.require(job_id))


@router.get("/{job_id}/logs", response_model=JobLogsOut)
def job_logs(
    job_id: str,
    _user: CurrentUser,
    request: Request,
    after: int = Query(default=0, ge=0),
) -> JobLogsOut:
    """Return the job's log lines after ``after``."""
    store = get_runner(request).store
    record = store.require(job_id)
    lines = store.logs(job_id, after=after)
    return JobLogsOut(
        job_id=job_id,
        status=record.status,
        next_seq=lines[-1]["seq"] if lines else after,
        lines=[JobLogLineOut(**row) for row in lines],
    )


@router.get("/{job_id}/stream")
def job_stream(
    job_id: str,
    _user: CurrentUser,
    request: Request,
    after: int = Query(default=0, ge=0),
) -> EventSourceResponse:
    """Stream the job's logs and status until it reaches a terminal state."""
    store = get_runner(request).store
    store.require(job_id)
    return EventSourceResponse(stream.job_events(store, job_id, after=after))


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, _admin: AdminUser, request: Request) -> JobOut:
    """Cancel a queued or running job."""
    return to_job_out(get_runner(request).cancel(job_id))
