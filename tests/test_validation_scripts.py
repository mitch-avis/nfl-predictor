"""Tests for validation script entrypoints and exit propagation."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest


class _FakeLogger:
    """Collect logger calls for assertion in script tests."""

    def __init__(self) -> None:
        """Initialize the in-memory log record store."""
        self.records: list[tuple[str, str]] = []

    def info(self, message: str, *args: object) -> None:
        """Record an info-level message."""
        self.records.append(("info", message % args if args else message))

    def warning(self, message: str, *args: object) -> None:
        """Record a warning-level message."""
        self.records.append(("warning", message % args if args else message))

    def error(self, message: str, *args: object) -> None:
        """Record an error-level message."""
        self.records.append(("error", message % args if args else message))


def _script_path(script_name: str) -> Path:
    """Return the path to a script under the repository's `scripts/` folder."""
    return Path(__file__).resolve().parents[1] / "scripts" / f"{script_name}.py"


def _load_script_module(script_name: str) -> ModuleType:
    """Import a script file as a regular module for direct `main()` testing."""
    script_path = _script_path(script_name)
    spec = importlib.util.spec_from_file_location(f"test_{script_name}", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dummy_all_data(tmp_path: Path) -> Path:
    """Create a minimal all-data CSV path that satisfies the existence check."""
    csv_path = tmp_path / "all_data.csv"
    csv_path.write_text("season,week\n2024,1\n", encoding="utf-8")
    return csv_path


def test_validate_offline_main_missing_file_logs_error(tmp_path: Path, monkeypatch) -> None:
    """Offline validation returns code 2 and logs an error when the CSV is missing."""
    module = _load_script_module("validate_offline")
    logger = _FakeLogger()

    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))

    assert module.main() == 2
    assert logger.records == [("error", f"Missing data file: {tmp_path / 'all_data.csv'}")]


def test_validate_offline_main_logs_errors_and_warnings(tmp_path: Path, monkeypatch) -> None:
    """Offline validation emits logger output for both errors and warnings."""
    module = _load_script_module("validate_offline")
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(module.pl, "read_csv", lambda _path: pl.DataFrame({"season": [2024]}))
    monkeypatch.setattr(
        module.validation_utils,
        "validate_dataframe",
        lambda _df: module.validation_utils.ValidationResult(
            errors=["bad schema"],
            warnings=["missing optional field"],
        ),
    )

    assert module.main() == 1
    assert ("error", "Errors:") in logger.records
    assert ("error", "- bad schema") in logger.records
    assert ("warning", "Warnings:") in logger.records
    assert ("warning", "- missing optional field") in logger.records


def test_validate_offline_main_logs_success(tmp_path: Path, monkeypatch) -> None:
    """Offline validation logs success and returns zero when no issues are found."""
    module = _load_script_module("validate_offline")
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(module.pl, "read_csv", lambda _path: pl.DataFrame({"season": [2024]}))
    monkeypatch.setattr(
        module.validation_utils,
        "validate_dataframe",
        lambda _df: module.validation_utils.ValidationResult(errors=[], warnings=[]),
    )

    assert module.main() == 0
    assert logger.records == [("info", "Validation OK")]


def test_validate_live_main_missing_file_logs_error(tmp_path: Path, monkeypatch) -> None:
    """Live validation returns code 2 and logs an error when the CSV is missing."""
    module = _load_script_module("validate_live")
    logger = _FakeLogger()

    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))

    assert module.main() == 2
    assert logger.records == [("error", f"Missing data file: {tmp_path / 'all_data.csv'}")]


def test_validate_live_main_logs_success_without_mismatches(tmp_path: Path, monkeypatch) -> None:
    """Live validation logs success and returns zero when no score mismatches exist."""
    module = _load_script_module("validate_live")
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(module.pl, "read_csv", lambda _path: pl.DataFrame({"season": [2024]}))
    monkeypatch.setattr(
        module.validation_utils,
        "compare_latest_week_scores",
        lambda _df: pl.DataFrame(),
    )

    assert module.main() == 0
    assert logger.records == [("info", "No mismatches found")]


def test_validate_live_main_logs_mismatches(tmp_path: Path, monkeypatch) -> None:
    """Live validation logs mismatch details and returns one when scores differ."""
    module = _load_script_module("validate_live")
    logger = _FakeLogger()

    mismatches = pl.DataFrame({"away_abbr": ["BUF"], "home_abbr": ["KC"]})
    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(module.pl, "read_csv", lambda _path: pl.DataFrame({"season": [2024]}))
    monkeypatch.setattr(
        module.validation_utils,
        "compare_latest_week_scores",
        lambda _df: mismatches,
    )

    assert module.main() == 1
    assert ("error", "Score mismatches detected:") in logger.records
    assert any(
        level == "error" and "BUF" in message and "KC" in message
        for level, message in logger.records
    )


@pytest.mark.parametrize("script_name", ["validate_offline", "validate_live"])
def test_validation_scripts_propagate_exit_codes_from_main(
    script_name: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Running either validation script as `__main__` exits with the `main()` return code."""
    from nfl_predictor import constants

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_script_path(script_name)), run_name="__main__")

    assert exc_info.value.code == 2
