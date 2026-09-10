"""SQLite storage for users, the active-run pointer, and job records.

The database is small and single-process, so plain ``sqlite3`` behind a lock is enough. Every
access goes through :meth:`Database.connect` so the schema is created lazily and connections are
always closed.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'admin')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    exit_code INTEGER,
    created_by TEXT,
    progress_json TEXT,
    result_json TEXT,
    error TEXT,
    parent_job_id TEXT
);
CREATE TABLE IF NOT EXISTS job_logs (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    line TEXT NOT NULL,
    PRIMARY KEY (job_id, seq)
);
"""


class Database:
    """A tiny wrapper around a SQLite file with lazy schema creation."""

    def __init__(self, path: Path) -> None:
        """Remember the database path; nothing is opened until first use."""
        self.path = path
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> None:
        """Create the parent directory and the schema if they do not exist."""
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as conn:
                conn.executescript(SCHEMA)
            self._initialized = True

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with ``Row`` access; commits on success, rolls back on error."""
        self.initialize()
        with self._lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_value(self, key: str) -> str | None:
        """Return the ``kv`` value for ``key`` or ``None``."""
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_value(self, key: str, value: str | None) -> None:
        """Upsert ``key`` in the ``kv`` table, deleting it when ``value`` is ``None``."""
        with self.connect() as conn:
            if value is None:
                conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            else:
                conn.execute(
                    "INSERT INTO kv (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
