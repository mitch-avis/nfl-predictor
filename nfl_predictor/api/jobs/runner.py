"""Run job templates as subprocesses and stream their output into the store.

Project entrypoints configure logging globally and read ``sys.argv``, so every job is a subprocess
of the venv interpreter started in its own process group. Each job has a dedicated worker thread
that reads the merged stdout/stderr stream, strips the terminal colors the project logger emits,
parses the log prefix, batches lines into SQLite, and updates the job's progress from the
walk-forward candidate counter.

Scheduling is a small fixed set of queues: one worker per exclusive group (so two walk-forward
jobs never overlap) and a two-slot pool for everything else.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import IO, Any

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import ApiError, ConflictError
from nfl_predictor.api.jobs import catalog
from nfl_predictor.api.jobs.catalog import JobContext, JobTemplate
from nfl_predictor.api.jobs.store import JobRecord, JobStore, LogLine
from nfl_predictor.api.runs import active as active_run
from nfl_predictor.api.runs.indexer import RunIndex
from nfl_predictor.api.settings import Settings
from nfl_predictor.utils.logger import log

DEFAULT_POOL = "default"
DEFAULT_POOL_WORKERS = 2
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LOG_PREFIX_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:.]+)\]\[(?P<level>[A-Z]+)\s*\]\[(?P<origin>[^\]]*)\]\s?(?P<message>.*)$"
)
PROGRESS_RE = re.compile(r"\b(?:fold|candidate)\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
LOG_BATCH_SIZE = 50
LOG_FLUSH_SECONDS = 0.25
SIGKILL_DELAY_SECONDS = 10.0
STOP = None


def strip_ansi(text: str) -> str:
    """Return ``text`` without the SGR escape sequences ``coloredlogs`` emits."""
    return ANSI_RE.sub("", text)


def parse_log_line(raw: str) -> LogLine:
    """Split one output line into level, timestamp, and message.

    Lines the project logger wrote carry ``[ts][LEVEL][file:func:line] message``; anything else
    (a traceback, a library writing to stdout) is kept verbatim at ``INFO``.
    """
    text = strip_ansi(raw).rstrip()
    match = LOG_PREFIX_RE.match(text)
    if match is None:
        return LogLine(level="INFO", message=text)
    return LogLine(level=match.group("level"), message=match.group("message"), ts=match.group("ts"))


def parse_progress(message: str) -> dict[str, Any] | None:
    """Return ``{"current", "total", "label"}`` when a message reports fold/candidate progress."""
    match = PROGRESS_RE.search(message)
    if match is None:
        return None
    current, total = int(match.group(1)), int(match.group(2))
    if total <= 0:
        return None
    return {"current": current, "total": total, "label": message}


@dataclass(frozen=True)
class Submission:
    """A validated job about to be queued."""

    template: JobTemplate
    params: dict[str, Any]
    created_by: str | None = None
    parent_job_id: str | None = None


def pool_name(template: JobTemplate) -> str:
    """Return the queue a template runs on."""
    return template.exclusive_group or DEFAULT_POOL


class JobRunner:
    """Owns the worker threads, the running subprocesses, and the job lifecycle."""

    def __init__(
        self,
        db: Database,
        settings: Settings,
        index: RunIndex | None = None,
        *,
        sigkill_delay: float = SIGKILL_DELAY_SECONDS,
    ) -> None:
        """Build a runner; call :meth:`start` to spin up its workers."""
        self.db = db
        self.settings = settings
        self.index = index
        self.store = JobStore(db)
        self.sigkill_delay = sigkill_delay
        self._queues: dict[str, queue.Queue[str | None]] = {}
        self._workers: list[threading.Thread] = []
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._canceled: set[str] = set()
        self._lock = threading.RLock()
        self._started = False

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        """Recover orphaned rows and start one thread per pool slot."""
        if self._started:
            return
        self._started = True
        orphans = self.store.recover_orphans()
        if orphans:
            log.warning("Marked %d job(s) failed after a server restart.", len(orphans))
        pools = {DEFAULT_POOL: DEFAULT_POOL_WORKERS} | {
            template.exclusive_group: 1
            for template in catalog.TEMPLATES
            if template.exclusive_group
        }
        for name, slots in pools.items():
            self._queues[name] = queue.Queue()
            for slot in range(slots):
                worker = threading.Thread(
                    target=self._work, args=(name,), name=f"job-{name}-{slot}", daemon=True
                )
                worker.start()
                self._workers.append(worker)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask every worker to finish its current job and exit."""
        if not self._started:
            return
        for pool in self._queues.values():
            for _ in range(DEFAULT_POOL_WORKERS):
                pool.put(STOP)
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()
        self._queues.clear()
        self._started = False

    # -- submission -----------------------------------------------------------------

    def busy_groups(self) -> set[str]:
        """Return the exclusive groups that already hold a queued or running job."""
        groups: set[str] = set()
        for record in self.store.active():
            template = catalog.TEMPLATES_BY_ID.get(record.template_id)
            if template is not None and template.exclusive_group:
                groups.add(template.exclusive_group)
        return groups

    def submit(self, submission: Submission) -> JobRecord:
        """Queue a job, refusing when its exclusive group is already busy."""
        group = submission.template.exclusive_group
        if group and group in self.busy_groups():
            raise ConflictError(
                f"Another {group.replace('_', ' ')} job is already queued or running.",
                code="group_busy",
            )
        record = self.store.create(
            submission.template.id,
            submission.params,
            created_by=submission.created_by,
            parent_job_id=submission.parent_job_id,
        )
        self._enqueue(submission.template, record.id)
        return record

    def _enqueue(self, template: JobTemplate, job_id: str) -> None:
        """Put a job on its pool's queue, running it inline when no workers exist."""
        pool = self._queues.get(pool_name(template))
        if pool is None:
            self._execute(job_id)
            return
        pool.put(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel a queued or running job and return its record."""
        record = self.store.require(job_id)
        if record.is_terminal:
            raise ConflictError(
                f"Job {job_id} already finished as {record.status}.", code="job_finished"
            )
        with self._lock:
            self._canceled.add(job_id)
            process = self._processes.get(job_id)
        if process is None:
            self.store.append_logs(job_id, [LogLine("WARNING", "Canceled before it started.")])
            self.store.finish(job_id, "canceled", error="Canceled before it started.")
            return self.store.require(job_id)
        self.store.append_logs(job_id, [LogLine("WARNING", "Cancel requested; sending SIGTERM.")])
        self._signal_group(process, signal.SIGTERM)
        timer = threading.Timer(self.sigkill_delay, self._force_kill, args=(job_id,))
        timer.daemon = True
        timer.start()
        return self.store.require(job_id)

    # -- execution ------------------------------------------------------------------

    def _work(self, pool: str) -> None:
        """Worker loop for one pool slot."""
        pending = self._queues[pool]
        while True:
            job_id = pending.get()
            if job_id is None:
                return
            try:
                self._execute(job_id)
            except Exception:
                log.exception("Job %s crashed the runner", job_id)
                self.store.finish(job_id, "failed", error="The runner failed to execute this job.")

    def _build_context(self, record: JobRecord) -> JobContext:
        """Build the context a template's build function reads."""
        index = self.index or RunIndex(self.settings.models_path)
        run = active_run.resolve_active_run(self.db, index)
        return JobContext(settings=self.settings, job_id=record.id, params=record.params, run=run)

    def _execute(self, job_id: str) -> None:
        """Run one job start to finish."""
        record = self.store.get(job_id)
        if record is None or record.status != "queued":
            return
        if job_id in self._canceled:
            self.store.finish(job_id, "canceled", error="Canceled before it started.")
            return
        template = catalog.TEMPLATES_BY_ID[record.template_id]
        try:
            argv = template.build(self._build_context(record))
        except ApiError as exc:
            self.store.append_logs(job_id, [LogLine("ERROR", exc.message)])
            self.store.finish(job_id, "failed", error=exc.message)
            return
        self.store.mark_running(job_id)
        self.store.append_logs(job_id, [LogLine("INFO", f"$ {shlex.join(argv)}")])
        exit_code, error = self._run_process(job_id, argv)
        status = "succeeded" if exit_code == 0 else "failed"
        if job_id in self._canceled:
            status, error = "canceled", "Canceled by an administrator."
        self.store.finish(job_id, status, exit_code=exit_code, error=error)
        if self.index is not None:
            self.index.invalidate()
        if status == "succeeded":
            self._chain(template, record)

    def _run_process(self, job_id: str, argv: list[str]) -> tuple[int, str | None]:
        """Launch ``argv``, drain its output into the store, and return its exit code."""
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built from the vetted catalog
                argv,
                cwd=self.settings.root_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            message = f"Could not start {argv[0]}: {exc}"
            self.store.append_logs(job_id, [LogLine("ERROR", message)])
            return -1, message
        with self._lock:
            self._processes[job_id] = process
        try:
            if process.stdout is not None:
                self._drain(job_id, process.stdout)
            exit_code = process.wait()
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        error = None if exit_code == 0 else f"Exited with code {exit_code}."
        return exit_code, error

    def _drain(self, job_id: str, stream: IO[str]) -> None:
        """Read the process output and flush it to the store on a fixed cadence.

        A dedicated thread does the blocking reads while this one flushes every
        ``LOG_FLUSH_SECONDS``, so a job that logs one line and then works for a minute still shows
        that line immediately instead of when its next line arrives.
        """
        buffer: list[LogLine] = []
        buffer_lock = threading.Lock()
        finished = threading.Event()

        def read() -> None:
            """Parse each line into the shared buffer until the stream closes."""
            for raw in stream:
                with buffer_lock:
                    buffer.append(parse_log_line(raw))
            finished.set()

        def flush() -> None:
            """Write whatever the reader has buffered, updating progress as it goes."""
            with buffer_lock:
                batch, buffer[:] = list(buffer), []
            if not batch:
                return
            for line in batch:
                progress = parse_progress(line.message)
                if progress is not None:
                    self.store.set_progress(job_id, progress)
            for start in range(0, len(batch), LOG_BATCH_SIZE):
                self.store.append_logs(job_id, batch[start : start + LOG_BATCH_SIZE])

        reader = threading.Thread(target=read, name=f"job-log-{job_id}", daemon=True)
        reader.start()
        while not finished.wait(LOG_FLUSH_SECONDS):
            flush()
        reader.join(timeout=LOG_FLUSH_SECONDS)
        flush()

    def _chain(self, template: JobTemplate, record: JobRecord) -> None:
        """Queue the follow-up job a template declares, if any."""
        if template.chain_template_id is None:
            return
        chained = catalog.get_template(template.chain_template_id)
        params = catalog.chained_params(template, record.params)
        try:
            self.submit(
                Submission(
                    template=chained,
                    params=params,
                    created_by=record.created_by,
                    parent_job_id=record.id,
                )
            )
        except ApiError as exc:
            self.store.append_logs(
                record.id, [LogLine("WARNING", f"Could not chain {chained.id}: {exc.message}")]
            )

    # -- signals --------------------------------------------------------------------

    def _signal_group(self, process: subprocess.Popen[str], sig: int) -> None:
        """Send ``sig`` to the process group, tolerating a process that already exited."""
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError, PermissionError:
            log.debug("Job process %d already gone", process.pid)

    def _force_kill(self, job_id: str) -> None:
        """SIGKILL a job that ignored the earlier SIGTERM."""
        with self._lock:
            process = self._processes.get(job_id)
        if process is None or process.poll() is not None:
            return
        self.store.append_logs(job_id, [LogLine("WARNING", "Still running; sending SIGKILL.")])
        self._signal_group(process, signal.SIGKILL)
