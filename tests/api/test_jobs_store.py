"""Tests for job persistence and the event stream's polling loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.jobs.store import JobStore, LogLine
from nfl_predictor.api.jobs.stream import job_events


@pytest.fixture
def store(db: Database) -> JobStore:
    """Return a store over the test database."""
    return JobStore(db)


def test_create_get_and_require(store: JobStore) -> None:
    """A created job round-trips, and an unknown id raises 404."""
    record = store.create("predict", {"season": 2026}, created_by="admin")

    fetched = store.require(record.id)
    assert fetched.params == {"season": 2026}
    assert fetched.status == "queued"
    assert fetched.created_by == "admin"
    assert fetched.is_terminal is False
    with pytest.raises(NotFoundError):
        store.require("missing")


def test_list_filters_by_template_and_orders_newest_first(store: JobStore) -> None:
    """The history can be narrowed to one template."""
    first = store.create("predict", {})
    second = store.create("etl_full", {})

    assert [job.id for job in store.recent()][0] == second.id
    assert [job.id for job in store.recent(template_id="predict")] == [first.id]
    assert store.recent(limit=1) != []


def test_logs_are_sequenced_and_paged(store: JobStore) -> None:
    """Sequence numbers continue across batches and drive the read cursor."""
    record = store.create("predict", {})

    assert store.max_seq(record.id) == 0
    assert store.append_logs(record.id, []) == 0
    assert store.append_logs(record.id, [LogLine("INFO", "one")]) == 1
    assert store.append_logs(record.id, [LogLine("INFO", "two"), LogLine("ERROR", "three")]) == 3
    assert [row["line"] for row in store.logs(record.id, after=1)] == ["two", "three"]
    assert store.logs(record.id, after=3) == []


def test_progress_and_finish_are_persisted(store: JobStore) -> None:
    """Progress and the terminal state survive a re-read."""
    record = store.create("weekly_run", {})
    store.mark_running(record.id)
    store.set_progress(record.id, {"current": 2, "total": 5, "label": "WF candidate 2/5"})
    store.finish(record.id, "failed", exit_code=1, error="boom")

    finished = store.require(record.id)
    assert finished.status == "failed"
    assert finished.exit_code == 1
    assert finished.error == "boom"
    assert finished.progress == {"current": 2, "total": 5, "label": "WF candidate 2/5"}
    assert finished.started_at is not None
    assert finished.is_terminal is True
    store.set_progress(record.id, None)
    assert store.require(record.id).progress is None


def test_a_corrupt_json_column_reads_as_empty(store: JobStore, db: Database) -> None:
    """A hand-edited or truncated JSON column never breaks the job list."""
    record = store.create("predict", {"season": 2026})
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET params_json = ? WHERE id = ?", ("not json", record.id))

    assert store.require(record.id).params == {}


def test_recover_orphans_fails_only_unfinished_jobs(store: JobStore) -> None:
    """Restart recovery frees queued and running rows and leaves finished ones alone."""
    running = store.create("weekly_run", {})
    store.mark_running(running.id)
    queued = store.create("predict", {})
    done = store.create("predict", {})
    store.finish(done.id, "succeeded", exit_code=0)

    recovered = store.recover_orphans()

    assert set(recovered) == {running.id, queued.id}
    assert store.require(running.id).status == "failed"
    assert store.require(queued.id).status == "failed"
    assert store.require(done.id).status == "succeeded"
    assert store.recover_orphans() == []


async def _collect(events: AsyncIterator[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Return at most ``limit`` events from an event stream."""
    collected: list[dict[str, str]] = []
    async for event in events:
        collected.append(event)
        if len(collected) >= limit:
            break
    return collected


def test_the_stream_keeps_polling_a_running_job(store: JobStore) -> None:
    """A running job's stream stays open and picks up lines written after it started."""
    record = store.create("predict", {})
    store.mark_running(record.id)
    store.append_logs(record.id, [LogLine("INFO", "first")])

    async def scenario() -> list[dict[str, str]]:
        events = job_events(store, record.id, poll_seconds=0.01)
        first = await _collect(events, 2)
        store.append_logs(record.id, [LogLine("INFO", "second")])
        store.finish(record.id, "succeeded", exit_code=0)
        rest = [event async for event in events]
        return first + rest

    collected = asyncio.run(scenario())

    kinds = [event["event"] for event in collected]
    assert kinds[0] == "log"
    assert kinds[-1] == "end"
    lines = [json.loads(e["data"])["line"] for e in collected if e["event"] == "log"]
    assert lines == ["first", "second"]
    assert json.loads(collected[-1]["data"])["status"] == "succeeded"
