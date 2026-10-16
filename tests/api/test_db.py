"""Tests for the SQLite wrapper."""

from __future__ import annotations

import gc
import sqlite3
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.db import Database

if TYPE_CHECKING:
    from pathlib import Path


def test_database_file_not_created_on_construction(tmp_path: Path) -> None:
    """Constructing the wrapper does not create the database file."""
    # Act
    db = Database(tmp_path / "nested" / "app.db")

    # Assert
    assert not db.path.exists()


def test_schema_created_lazily(tmp_path: Path) -> None:
    """The database file and tables appear on first use."""
    # Arrange
    db = Database(tmp_path / "nested" / "app.db")

    # Act
    with db.connect() as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}

    # Assert
    assert {"users", "kv", "jobs", "job_logs"} <= names


def test_kv_get_missing_key_returns_none(tmp_path: Path) -> None:
    """An unset key reads as ``None``."""
    # Arrange
    db = Database(tmp_path / "app.db")

    # Act
    value = db.get_value("active")

    # Assert
    assert value is None


def test_kv_roundtrip(tmp_path: Path) -> None:
    """Values can be set and replaced."""
    # Arrange
    db = Database(tmp_path / "app.db")
    db.set_value("active", "run_a")

    # Act
    db.set_value("active", "run_b")

    # Assert
    assert db.get_value("active") == "run_b"


def test_kv_set_none_deletes_value(tmp_path: Path) -> None:
    """Setting ``None`` deletes the stored value."""
    # Arrange
    db = Database(tmp_path / "app.db")
    db.set_value("active", "run_b")

    # Act
    db.set_value("active", None)

    # Assert
    assert db.get_value("active") is None


def test_rollback_on_error(tmp_path: Path) -> None:
    """A failing block rolls back its writes."""
    # Arrange
    db = Database(tmp_path / "app.db")

    # Act
    with pytest.raises(sqlite3.OperationalError):
        _write_then_fail(db)

    # Assert
    assert db.get_value("k") is None


def _write_then_fail(db: Database) -> None:
    """Write a row, then fail inside the same transaction."""
    with db.connect() as conn:
        conn.execute("INSERT INTO kv (key, value) VALUES ('k', 'v')")
        conn.execute("SELECT * FROM missing_table")


@pytest.mark.filterwarnings(
    "error::ResourceWarning", "error::pytest.PytestUnraisableExceptionWarning"
)
def test_initialize_closes_its_connection(tmp_path: Path) -> None:
    """Creating the schema leaves no SQLite connection open for the garbage collector."""
    # Arrange
    db = Database(tmp_path / "app.db")

    # Act
    db.initialize()

    # Assert
    gc.collect()
