"""Persistence for job records and their logs.

Jobs outlive the request that submitted them and their logs outlive the browser tab that watched
them, so both live in SQLite. Log lines carry a monotonic per-job ``seq`` the stream endpoint
replays from.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import NotFoundError

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "canceled"})
ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})
RESTART_ERROR = "The server restarted while this job was running."


def _now() -> str:
    """Return the current UTC time in ISO-8601."""
    return datetime.now(UTC).isoformat()


def _loads(payload: str | None) -> dict[str, Any]:
    """Return a JSON object column as a dict, tolerating nulls and bad rows."""
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class LogLine:
    """One parsed line of a job's output."""

    level: str
    message: str
    ts: str = ""


@dataclass(frozen=True)
class JobRecord:
    """A submitted job and everything known about its progress."""

    id: str
    template_id: str
    params: dict[str, Any]
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    created_by: str | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None
    parent_job_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has finished, one way or another."""
        return self.status in TERMINAL_STATUSES


def _to_record(row: Any) -> JobRecord:
    """Build a :class:`JobRecord` from a database row."""
    progress = _loads(row["progress_json"])
    return JobRecord(
        id=str(row["id"]),
        template_id=str(row["template_id"]),
        params=_loads(row["params_json"]),
        status=cast(JobStatus, str(row["status"])),
        created_at=str(row["created_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        created_by=row["created_by"],
        progress=progress or None,
        error=row["error"],
        parent_job_id=row["parent_job_id"],
    )


class JobStore:
    """Reads and writes the ``jobs`` and ``job_logs`` tables."""

    def __init__(self, db: Database) -> None:
        """Remember the database the store writes to."""
        self.db = db

    def create(
        self,
        template_id: str,
        params: dict[str, Any],
        *,
        created_by: str | None = None,
        parent_job_id: str | None = None,
    ) -> JobRecord:
        """Insert a queued job and return it."""
        record = JobRecord(
            id=uuid.uuid4().hex[:16],
            template_id=template_id,
            params=params,
            status="queued",
            created_at=_now(),
            created_by=created_by,
            parent_job_id=parent_job_id,
        )
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, template_id, params_json, status, created_at, created_by, "
                "parent_job_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.template_id,
                    json.dumps(record.params),
                    record.status,
                    record.created_at,
                    record.created_by,
                    record.parent_job_id,
                ),
            )
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """Return one job or ``None``."""
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else _to_record(row)

    def require(self, job_id: str) -> JobRecord:
        """Return one job or raise 404."""
        record = self.get(job_id)
        if record is None:
            raise NotFoundError(f"Job {job_id!r} not found", code="job_not_found")
        return record

    def recent(self, *, limit: int = 50, template_id: str | None = None) -> list[JobRecord]:
        """Return recent jobs, newest first."""
        query = "SELECT * FROM jobs"
        args: list[Any] = []
        if template_id is not None:
            query += " WHERE template_id = ?"
            args.append(template_id)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self.db.connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [_to_record(row) for row in rows]

    def active(self) -> list[JobRecord]:
        """Return every queued or running job, oldest first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def mark_running(self, job_id: str) -> None:
        """Move a job into the running state."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (_now(), job_id),
            )

    def finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> None:
        """Move a job into a terminal state."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, exit_code = ?, error = ? "
                "WHERE id = ?",
                (status, _now(), exit_code, error, job_id),
            )

    def set_progress(self, job_id: str, progress: dict[str, Any] | None) -> None:
        """Store the job's latest progress payload."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress_json = ? WHERE id = ?",
                (None if progress is None else json.dumps(progress), job_id),
            )

    def max_seq(self, job_id: str) -> int:
        """Return the highest log sequence written for a job, or 0."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM job_logs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return int(row["seq"])

    def append_logs(self, job_id: str, lines: Sequence[LogLine]) -> int:
        """Append log lines and return the new highest sequence number."""
        if not lines:
            return self.max_seq(job_id)
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM job_logs WHERE job_id = ?", (job_id,)
            ).fetchone()
            seq = int(row["seq"])
            payload = []
            for line in lines:
                seq += 1
                payload.append((job_id, seq, line.ts or _now(), line.level, line.message))
            conn.executemany(
                "INSERT INTO job_logs (job_id, seq, ts, level, line) VALUES (?, ?, ?, ?, ?)",
                payload,
            )
        return seq

    def logs(self, job_id: str, *, after: int = 0, limit: int = 5000) -> list[dict[str, Any]]:
        """Return log rows with ``seq`` greater than ``after``, oldest first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT seq, ts, level, line FROM job_logs WHERE job_id = ? AND seq > ? "
                "ORDER BY seq ASC LIMIT ?",
                (job_id, after, limit),
            ).fetchall()
        return [
            {"seq": int(row["seq"]), "ts": row["ts"], "level": row["level"], "line": row["line"]}
            for row in rows
        ]

    def recover_orphans(self) -> list[str]:
        """Fail jobs left running or queued by a previous process and return their ids.

        The runner owns its subprocesses, so nothing survives a restart; leaving rows in a
        non-terminal state would block their exclusive group forever.
        """
        orphans = [record.id for record in self.active()]
        if not orphans:
            return []
        with self.db.connect() as conn:
            conn.executemany(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
                [(_now(), RESTART_ERROR, job_id) for job_id in orphans],
            )
        return orphans
