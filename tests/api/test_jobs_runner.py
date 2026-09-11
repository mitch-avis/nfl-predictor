"""Tests for the subprocess job runner: logs, progress, scheduling, and cancellation."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import ConflictError
from nfl_predictor.api.jobs import catalog
from nfl_predictor.api.jobs import runner as runner_module
from nfl_predictor.api.jobs.catalog import JobContext, JobTemplate
from nfl_predictor.api.jobs.runner import JobRunner, Submission
from nfl_predictor.api.settings import Settings
from tests.api.jobs_support import fake_template, wait_for, wait_for_log, wait_for_status


@pytest.fixture
def runner(settings: Settings, db: Database) -> Iterator[JobRunner]:
    """Return a started runner that is stopped at the end of the test."""
    started = JobRunner(db, settings, sigkill_delay=1.0)
    yield started
    started.stop(timeout=5.0)


def test_strip_ansi_removes_color_codes() -> None:
    """The color codes the project logger writes never reach the store."""
    assert runner_module.strip_ansi("\x1b[32mhello\x1b[0m") == "hello"


def test_parse_log_line_splits_the_project_prefix() -> None:
    """A project log line yields its level, timestamp, and bare message."""
    line = runner_module.parse_log_line(
        "\x1b[32m[2026-09-10 12:00:00.123][WARNING][mod:func:12] be careful\x1b[0m"
    )
    assert line.level == "WARNING"
    assert line.ts == "2026-09-10 12:00:00.123"
    assert line.message == "be careful"


def test_parse_log_line_keeps_unprefixed_output() -> None:
    """Output from a library that does not use the project logger is kept verbatim."""
    line = runner_module.parse_log_line("Traceback (most recent call last):")
    assert line.level == "INFO"
    assert line.message == "Traceback (most recent call last):"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("WF candidate 3/12 done in 1m2.0s", {"current": 3, "total": 12}),
        ("Walk-forward fold 2/5 done", {"current": 2, "total": 5}),
        ("nothing to report", None),
        ("candidate 1/0 done", None),
    ],
)
def test_parse_progress(message: str, expected: dict[str, int] | None) -> None:
    """Progress is read from the fold and candidate counters, and only from those."""
    progress = runner_module.parse_progress(message)
    if expected is None:
        assert progress is None
    else:
        assert progress is not None
        assert progress["current"] == expected["current"]
        assert progress["total"] == expected["total"]


def test_runner_records_logs_and_success(runner: JobRunner, register: Callable[..., None]) -> None:
    """A job's output is stored line by line, stripped and parsed, and the job succeeds."""
    register(fake_template(args=("--lines", "3", "--stderr", "--plain")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "succeeded")

    finished = runner.store.require(record.id)
    assert finished.exit_code == 0
    assert finished.error is None
    lines = runner.store.logs(record.id)
    assert lines[0]["line"].startswith("$ ")
    messages = [row["line"] for row in lines]
    assert "line 1" in messages
    assert "from stderr" in messages
    assert "a line with no log prefix" in messages
    assert all("\x1b" not in row["line"] for row in lines)
    assert {row["level"] for row in lines} >= {"INFO", "ERROR"}
    assert [row["seq"] for row in lines] == sorted(row["seq"] for row in lines)


def test_runner_tracks_progress(runner: JobRunner, register: Callable[..., None]) -> None:
    """The candidate counter drives the job's progress payload."""
    register(fake_template(args=("--lines", "1", "--progress", "4")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "succeeded")

    progress = runner.store.require(record.id).progress
    assert progress == {"current": 4, "total": 4, "label": "WF candidate 4/4 done in 0m1.0s"}


def test_runner_marks_a_nonzero_exit_as_failed(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """A job that exits non-zero fails and keeps its exit code."""
    register(fake_template(args=("--exit-code", "3")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "failed")

    finished = runner.store.require(record.id)
    assert finished.exit_code == 3
    assert finished.error is not None and "3" in finished.error


def test_runner_reports_a_command_that_cannot_start(
    runner: JobRunner, register: Callable[..., None], settings: Settings
) -> None:
    """A missing interpreter fails the job instead of killing the worker."""

    def build(_ctx: JobContext) -> list[str]:
        """Return a command that does not exist."""
        return [str(settings.root_dir / "no-such-binary")]

    register(
        JobTemplate(
            id="fake",
            label="fake",
            description="A job that cannot start.",
            category="Test",
            build=build,
        )
    )
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "failed")

    assert "Could not start" in (runner.store.require(record.id).error or "")


def test_runner_fails_a_job_whose_command_cannot_be_built(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """A template needing an active run fails cleanly when none is pinned."""

    def build(ctx: JobContext) -> list[str]:
        """Require an active run that the empty models directory cannot provide."""
        run = ctx.require_run()
        return [ctx.python, str(run.run_files.model)]

    register(
        JobTemplate(
            id="fake",
            label="fake",
            description="A job that needs a run.",
            category="Test",
            build=build,
            needs_active_run=True,
        )
    )
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "failed")

    assert "active run" in (runner.store.require(record.id).error or "")
    assert runner.store.logs(record.id)[0]["level"] == "ERROR"


def test_runner_refuses_a_second_job_in_a_busy_exclusive_group(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """The API-facing guard rejects a walk-forward job while one is already active."""
    register(fake_template(args=("--sleep", "5"), group="walk_forward"))
    runner.start()
    template = catalog.TEMPLATES[0]

    first = runner.submit(Submission(template=template, params={}))
    with pytest.raises(ConflictError):
        runner.submit(Submission(template=template, params={}))

    runner.cancel(first.id)


def test_exclusive_group_jobs_never_overlap(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """Two jobs queued on an exclusive group run one after the other, not together."""
    register(fake_template(args=("--sleep", "0.5"), group="walk_forward"))
    runner.start()
    template = catalog.TEMPLATES[0]
    first = runner.store.create(template.id, {})
    second = runner.store.create(template.id, {})

    runner._enqueue(template, first.id)
    runner._enqueue(template, second.id)
    wait_for(lambda: runner.store.require(first.id).status == "running")
    assert runner.store.require(second.id).status == "queued"

    wait_for_status(runner.store, second.id, "succeeded")
    first_finished = runner.store.require(first.id).finished_at
    second_started = runner.store.require(second.id).started_at
    assert first_finished is not None and second_started is not None
    assert first_finished <= second_started


def test_cancel_terminates_a_running_job(runner: JobRunner, register: Callable[..., None]) -> None:
    """A running job is signalled and ends up canceled."""
    register(fake_template(args=("--sleep", "30")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_log(runner.store, record.id, "line 1")
    runner.cancel(record.id)
    wait_for_status(runner.store, record.id, "canceled")

    assert runner.store.require(record.id).error == "Canceled by an administrator."


def test_cancel_escalates_to_sigkill(runner: JobRunner, register: Callable[..., None]) -> None:
    """A job that ignores SIGTERM is killed after the grace period."""
    register(fake_template(args=("--sleep", "30", "--ignore-sigterm")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_log(runner.store, record.id, "line 1")
    runner.cancel(record.id)
    wait_for_status(runner.store, record.id, "canceled")

    messages = [row["line"] for row in runner.store.logs(record.id)]
    assert "Still running; sending SIGKILL." in messages


def test_cancel_a_queued_job_never_starts_it(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """Cancelling before a worker picks the job up marks it canceled without running it."""
    register(fake_template(args=("--sleep", "5"), group="walk_forward"))
    runner.start()
    template = catalog.TEMPLATES[0]
    first = runner.store.create(template.id, {})
    queued = runner.store.create(template.id, {})
    runner._enqueue(template, first.id)
    wait_for(lambda: runner.store.require(first.id).status == "running")
    runner._enqueue(template, queued.id)

    canceled = runner.cancel(queued.id)

    assert canceled.status == "canceled"
    assert canceled.started_at is None
    runner.cancel(first.id)


def test_cancel_rejects_a_finished_job(runner: JobRunner, register: Callable[..., None]) -> None:
    """Cancelling a job that already finished is a conflict."""
    register(fake_template(args=("--lines", "1")))
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "succeeded")

    with pytest.raises(ConflictError):
        runner.cancel(record.id)


def test_a_successful_job_chains_its_follow_up(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """A template with ``chain_template_id`` queues the follow-up once it succeeds."""
    register(
        fake_template("first", args=("--lines", "1"), chain="second"),
        fake_template("second", args=("--lines", "1")),
    )
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES_BY_ID["first"], params={}))
    wait_for_status(runner.store, record.id, "succeeded")
    wait_for(lambda: any(job.template_id == "second" for job in runner.store.recent()))

    chained = next(job for job in runner.store.recent() if job.template_id == "second")
    assert chained.parent_job_id == record.id
    wait_for_status(runner.store, chained.id, "succeeded")


def test_a_failed_job_does_not_chain(runner: JobRunner, register: Callable[..., None]) -> None:
    """The follow-up only runs when the first job succeeded."""
    register(
        fake_template("first", args=("--exit-code", "1"), chain="second"),
        fake_template("second", args=("--lines", "1")),
    )
    runner.start()

    record = runner.submit(Submission(template=catalog.TEMPLATES_BY_ID["first"], params={}))
    wait_for_status(runner.store, record.id, "failed")

    assert all(job.template_id != "second" for job in runner.store.recent())


def test_start_fails_jobs_left_behind_by_a_restart(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """Rows still marked running when the process starts are failed, freeing their group."""
    register(fake_template(group="walk_forward"))
    orphan = runner.store.create("fake", {})
    runner.store.mark_running(orphan.id)

    runner.start()

    recovered = runner.store.require(orphan.id)
    assert recovered.status == "failed"
    assert recovered.error is not None and "restarted" in recovered.error
    assert runner.busy_groups() == set()


def test_start_and_stop_are_idempotent(runner: JobRunner, register: Callable[..., None]) -> None:
    """Starting twice adds no workers and stopping an unstarted runner does nothing."""
    register(fake_template())
    runner.stop()
    runner.start()
    workers = len(runner._workers)
    runner.start()

    assert len(runner._workers) == workers
    runner.stop(timeout=5.0)
    runner.stop(timeout=5.0)


def test_a_runner_without_workers_executes_inline(
    settings: Settings, db: Database, register: Callable[..., None]
) -> None:
    """Before :meth:`start`, submitting runs the job on the calling thread.

    This keeps a job submitted during startup from being silently dropped.
    """
    register(fake_template(args=("--lines", "1")))
    inline = JobRunner(db, settings)

    record = inline.submit(Submission(template=catalog.TEMPLATES[0], params={}))

    assert inline.store.require(record.id).status == "succeeded"


def test_signalling_a_process_that_already_exited_is_harmless(
    runner: JobRunner, register: Callable[..., None]
) -> None:
    """Racing a cancel against a job that just finished does not raise."""
    register(fake_template(args=("--lines", "1")))
    runner.start()
    record = runner.submit(Submission(template=catalog.TEMPLATES[0], params={}))
    wait_for_status(runner.store, record.id, "succeeded")

    runner._force_kill(record.id)  # no process is registered any more

    assert runner.store.require(record.id).status == "succeeded"
