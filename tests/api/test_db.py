"""Tests for the SQLite wrapper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nfl_predictor.api.db import Database


def test_schema_created_lazily(tmp_path: Path) -> None:
    """The database file and tables appear on first use."""
    db = Database(tmp_path / "nested" / "app.db")
    assert not db.path.exists()
    with db.connect() as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert {"users", "kv", "jobs", "job_logs"} <= names


def test_kv_roundtrip(tmp_path: Path) -> None:
    """Values can be set, replaced, and deleted."""
    db = Database(tmp_path / "app.db")
    assert db.get_value("active") is None
    db.set_value("active", "run_a")
    db.set_value("active", "run_b")
    assert db.get_value("active") == "run_b"
    db.set_value("active", None)
    assert db.get_value("active") is None


def test_rollback_on_error(tmp_path: Path) -> None:
    """A failing block rolls back its writes."""
    db = Database(tmp_path / "app.db")
    with pytest.raises(sqlite3.OperationalError), db.connect() as conn:
        conn.execute("INSERT INTO kv (key, value) VALUES ('k', 'v')")
        conn.execute("SELECT * FROM missing_table")
    assert db.get_value("k") is None
