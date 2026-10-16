"""Tests for job persistence and the event stream's polling loop."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.errors import NotFoundError
from nfl_predictor.api.jobs.store import JobStore, LogLine
from nfl_predictor.api.jobs.stream import job_events

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nfl_predictor.api.db import Database


@pytest.fixture
def store(db: Database) -> JobStore:
    """Return a store over the test database."""
    return JobStore(db)


def test_create_get_and_require(store: JobStore) -> None:
    """A created job round-trips."""
    # Arrange
    record = store.create("predict", {"season": 2026}, created_by="admin")

    # Act
    fetched = store.require(record.id)

    # Assert
    assert fetched.params == {"season": 2026}
    assert fetched.status == "queued"
    assert fetched.created_by == "admin"
    assert fetched.is_terminal is False


def test_require_unknown_id_raises_not_found(store: JobStore) -> None:
    """An unknown id raises 404."""
    # Arrange
    store.create("predict", {"season": 2026}, created_by="admin")

    # Act & Assert
    with pytest.raises(NotFoundError):
        store.require("missing")


def test_list_orders_newest_first(store: JobStore) -> None:
    """The history lists the newest job first."""
    # Arrange
    store.create("predict", {})
    second = store.create("etl_full", {})

    # Act
    recent = store.recent()

    # Assert
    assert next(job.id for job in recent) == second.id


def test_list_filters_by_template(store: JobStore) -> None:
    """The history can be narrowed to one template."""
    # Arrange
    first = store.create("predict", {})
    store.create("etl_full", {})

    # Act
    recent = store.recent(template_id="predict")

    # Assert
    assert [job.id for job in recent] == [first.id]


def test_list_honours_the_limit(store: JobStore) -> None:
    """A limited history still returns jobs."""
    # Arrange
    store.create("predict", {})
    store.create("etl_full", {})

    # Act
    recent = store.recent(limit=1)

    # Assert
    assert recent != []


def test_max_seq_of_a_job_without_logs_is_zero(store: JobStore) -> None:
    """A job with no log lines starts its sequence at zero."""
    # Arrange
    record = store.create("predict", {})

    # Act
    max_seq = store.max_seq(record.id)

    # Assert
    assert max_seq == 0


def test_append_no_logs_returns_zero(store: JobStore) -> None:
    """Appending an empty batch leaves the sequence at zero."""
    # Arrange
    record = store.create("predict", {})

    # Act
    seq = store.append_logs(record.id, [])

    # Assert
    assert seq == 0


def test_append_logs_returns_the_last_sequence_number(store: JobStore) -> None:
    """The first appended line gets sequence number one."""
    # Arrange
    record = store.create("predict", {})
    store.append_logs(record.id, [])

    # Act
    seq = store.append_logs(record.id, [LogLine("INFO", "one")])

    # Assert
    assert seq == 1


def test_logs_are_sequenced_across_batches(store: JobStore) -> None:
    """Sequence numbers continue across batches."""
    # Arrange
    record = store.create("predict", {})
    store.append_logs(record.id, [LogLine("INFO", "one")])

    # Act
    seq = store.append_logs(record.id, [LogLine("INFO", "two"), LogLine("ERROR", "three")])

    # Assert
    assert seq == 3


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        (1, ["two", "three"]),
        (3, []),
    ],
)
def test_logs_are_paged(store: JobStore, after: int, expected: list[str]) -> None:
    """Sequence numbers drive the read cursor."""
    # Arrange
    record = store.create("predict", {})
    store.append_logs(record.id, [LogLine("INFO", "one")])
    store.append_logs(record.id, [LogLine("INFO", "two"), LogLine("ERROR", "three")])

    # Act
    rows = store.logs(record.id, after=after)

    # Assert
    assert [row["line"] for row in rows] == expected


def test_progress_and_finish_are_persisted(store: JobStore) -> None:
    """Progress and the terminal state survive a re-read."""
    # Arrange
    record = store.create("weekly_run", {})
    store.mark_running(record.id)
    store.set_progress(record.id, {"current": 2, "total": 5, "label": "WF candidate 2/5"})
    store.finish(record.id, "failed", exit_code=1, error="boom")

    # Act
    finished = store.require(record.id)

    # Assert
    assert finished.status == "failed"
    assert finished.exit_code == 1
    assert finished.error == "boom"
    assert finished.progress == {"current": 2, "total": 5, "label": "WF candidate 2/5"}
    assert finished.started_at is not None
    assert finished.is_terminal is True


def test_progress_can_be_cleared(store: JobStore) -> None:
    """Setting progress to ``None`` clears it."""
    # Arrange
    record = store.create("weekly_run", {})
    store.mark_running(record.id)
    store.set_progress(record.id, {"current": 2, "total": 5, "label": "WF candidate 2/5"})
    store.finish(record.id, "failed", exit_code=1, error="boom")

    # Act
    store.set_progress(record.id, None)

    # Assert
    assert store.require(record.id).progress is None


def test_a_corrupt_json_column_reads_as_empty(store: JobStore, db: Database) -> None:
    """A hand-edited or truncated JSON column never breaks the job list."""
    # Arrange
    record = store.create("predict", {"season": 2026})
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET params_json = ? WHERE id = ?", ("not json", record.id))

    # Act
    params = store.require(record.id).params

    # Assert
    assert params == {}


def test_recover_orphans_fails_only_unfinished_jobs(store: JobStore) -> None:
    """Restart recovery frees queued and running rows and leaves finished ones alone."""
    # Arrange
    running = store.create("weekly_run", {})
    store.mark_running(running.id)
    queued = store.create("predict", {})
    done = store.create("predict", {})
    store.finish(done.id, "succeeded", exit_code=0)

    # Act
    recovered = store.recover_orphans()

    # Assert
    assert set(recovered) == {running.id, queued.id}
    assert store.require(running.id).status == "failed"
    assert store.require(queued.id).status == "failed"
    assert store.require(done.id).status == "succeeded"


def test_recover_orphans_twice_recovers_nothing_more(store: JobStore) -> None:
    """A second recovery pass finds no unfinished jobs."""
    # Arrange
    running = store.create("weekly_run", {})
    store.mark_running(running.id)
    store.create("predict", {})
    done = store.create("predict", {})
    store.finish(done.id, "succeeded", exit_code=0)
    store.recover_orphans()

    # Act
    recovered = store.recover_orphans()

    # Assert
    assert recovered == []


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
    # Arrange
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

    # Act
    collected = asyncio.run(scenario())

    # Assert
    kinds = [event["event"] for event in collected]
    assert kinds[0] == "log"
    assert kinds[-1] == "end"
    lines = [json.loads(e["data"])["line"] for e in collected if e["event"] == "log"]
    assert lines == ["first", "second"]
    assert json.loads(collected[-1]["data"])["status"] == "succeeded"
