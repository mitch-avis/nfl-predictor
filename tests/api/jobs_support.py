"""Shared helpers for the job runner and job route tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from nfl_predictor.api.jobs.catalog import JobContext, JobTemplate
from nfl_predictor.api.jobs.store import JobStore

FAKE_SCRIPT = Path(__file__).parent / "fake_script.py"
TIMEOUT_SECONDS = 20.0


def fake_template(
    template_id: str = "fake",
    *,
    args: tuple[str, ...] = (),
    group: str | None = None,
    chain: str | None = None,
) -> JobTemplate:
    """Return a template that runs ``fake_script.py`` with ``args``."""

    def build(ctx: JobContext) -> list[str]:
        """Build the fake script's command line."""
        return [ctx.python, str(FAKE_SCRIPT), *args]

    return JobTemplate(
        id=template_id,
        label=template_id,
        description="A fake job.",
        category="Test",
        build=build,
        exclusive_group=group,
        chain_template_id=chain,
    )


def wait_for(predicate: Callable[[], bool], *, timeout: float = TIMEOUT_SECONDS) -> None:
    """Block until ``predicate`` is true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for the job runner")


def wait_for_status(store: JobStore, job_id: str, status: str) -> None:
    """Block until a job reaches ``status``."""
    wait_for(lambda: (store.require(job_id)).status == status)


def wait_for_log(store: JobStore, job_id: str, needle: str) -> None:
    """Block until a job has logged a line containing ``needle``.

    Waiting on the status alone is not enough before cancelling: a job counts as running the
    moment it is launched, well before the interpreter has started and installed its handlers.
    """
    wait_for(lambda: any(needle in row["line"] for row in store.logs(job_id)))
