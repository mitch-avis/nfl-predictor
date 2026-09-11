"""Server-sent events for one job's logs and status.

The runner writes to SQLite from worker threads, so the stream is a poller rather than a
subscription: it replays everything after the client's cursor, then re-checks a few times a
second until the job reaches a terminal state.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from nfl_predictor.api.jobs.store import JobRecord, JobStore

POLL_SECONDS = 0.3
LOG_PAGE = 500


def status_payload(record: JobRecord) -> dict[str, Any]:
    """Return the status fields a client needs to update its header."""
    return {
        "id": record.id,
        "status": record.status,
        "progress": record.progress,
        "exit_code": record.exit_code,
        "error": record.error,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


async def job_events(
    store: JobStore, job_id: str, *, after: int = 0, poll_seconds: float = POLL_SECONDS
) -> AsyncIterator[dict[str, str]]:
    """Yield ``log``, ``status``, and a final ``end`` event for ``job_id``.

    Args:
        store: Store holding the job and its logs.
        job_id: Job to follow.
        after: Sequence number the client already has; replay starts after it.
        poll_seconds: How often to re-read the store.

    Yields:
        ``sse-starlette`` event dictionaries.

    """
    cursor = after
    last_status: dict[str, Any] | None = None
    while True:
        record = store.require(job_id)
        while True:
            rows = store.logs(job_id, after=cursor, limit=LOG_PAGE)
            if not rows:
                break
            for row in rows:
                cursor = int(row["seq"])
                yield {"event": "log", "data": json.dumps(row)}
        payload = status_payload(record)
        if payload != last_status:
            last_status = payload
            yield {"event": "status", "data": json.dumps(payload)}
        if record.is_terminal and not store.logs(job_id, after=cursor, limit=1):
            yield {"event": "end", "data": json.dumps({"status": record.status})}
            return
        await asyncio.sleep(poll_seconds)
