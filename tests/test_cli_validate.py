"""Tests for the ``validate`` command (offline checks, or ``--live``)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl_predictor.cli import main as front_door
from nfl_predictor.cli import validate as module

LIVE = ["--live"]


class _FakeLogger:
    """Collect logger calls for assertion in command tests."""

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


def _write_dummy_all_data(tmp_path: Path) -> Path:
    """Create a minimal all-data CSV path that satisfies the existence check."""
    csv_path = tmp_path / "all_data.csv"
    csv_path.write_text("season,week\n2024,1\n", encoding="utf-8")
    return csv_path


def test_validate_offline_main_missing_file_logs_error(tmp_path: Path, monkeypatch) -> None:
    """Offline validation returns code 2 and logs an error when the CSV is missing."""
    logger = _FakeLogger()

    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))

    assert module.main([]) == 2
    assert logger.records == [("error", f"Missing data file: {tmp_path / 'all_data.csv'}")]


def test_validate_offline_main_logs_errors_and_warnings(tmp_path: Path, monkeypatch) -> None:
    """Offline validation emits logger output for both errors and warnings."""
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        module.pl, "read_csv", lambda _path, **_kwargs: pl.DataFrame({"season": [2024]})
    )
    monkeypatch.setattr(
        module.validation_utils,
        "validate_dataframe",
        lambda _df: module.validation_utils.ValidationResult(
            errors=["bad schema"],
            warnings=["missing optional field"],
        ),
    )

    assert module.main([]) == 1
    assert ("error", "Errors:") in logger.records
    assert ("error", "- bad schema") in logger.records
    assert ("warning", "Warnings:") in logger.records
    assert ("warning", "- missing optional field") in logger.records


def test_validate_offline_main_logs_success(tmp_path: Path, monkeypatch) -> None:
    """Offline validation logs success and returns zero when no issues are found."""
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        module.pl, "read_csv", lambda _path, **_kwargs: pl.DataFrame({"season": [2024]})
    )
    monkeypatch.setattr(
        module.validation_utils,
        "validate_dataframe",
        lambda _df: module.validation_utils.ValidationResult(errors=[], warnings=[]),
    )

    assert module.main([]) == 0
    assert logger.records == [("info", "Validation OK")]


def test_validate_live_main_missing_file_logs_error(tmp_path: Path, monkeypatch) -> None:
    """Live validation returns code 2 and logs an error when the CSV is missing."""
    logger = _FakeLogger()

    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))

    assert module.main(LIVE) == 2
    assert logger.records == [("error", f"Missing data file: {tmp_path / 'all_data.csv'}")]


def test_validate_live_main_logs_success_without_mismatches(tmp_path: Path, monkeypatch) -> None:
    """Live validation logs success and returns zero when no score mismatches exist."""
    logger = _FakeLogger()

    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        module.pl, "read_csv", lambda _path, **_kwargs: pl.DataFrame({"season": [2024]})
    )
    monkeypatch.setattr(
        module.validation_utils,
        "compare_latest_week_scores",
        lambda _df: pl.DataFrame(),
    )

    assert module.main(LIVE) == 0
    assert logger.records == [("info", "No mismatches found")]


def test_validate_live_main_logs_mismatches(tmp_path: Path, monkeypatch) -> None:
    """Live validation logs mismatch details and returns one when scores differ."""
    logger = _FakeLogger()

    mismatches = pl.DataFrame({"away_abbr": ["BUF"], "home_abbr": ["KC"]})
    _write_dummy_all_data(tmp_path)
    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        module.pl, "read_csv", lambda _path, **_kwargs: pl.DataFrame({"season": [2024]})
    )
    monkeypatch.setattr(
        module.validation_utils,
        "compare_latest_week_scores",
        lambda _df: mismatches,
    )

    assert module.main(LIVE) == 1
    assert ("error", "Score mismatches detected:") in logger.records
    assert any(
        level == "error" and "BUF" in message and "KC" in message
        for level, message in logger.records
    )


@pytest.mark.parametrize("mode", [[], LIVE])
def test_validate_reads_the_data_dir_option(
    mode: list[str],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An explicit data directory is read instead of the packaged one, in either mode."""
    logger = _FakeLogger()
    chosen = tmp_path / "configured"
    chosen.mkdir()

    monkeypatch.setattr(module, "log", logger, raising=False)
    monkeypatch.setattr(module.constants, "DATA_PATH", str(tmp_path / "packaged"))

    assert module.main([*mode, "--data-dir", str(chosen)]) == 2
    assert logger.records == [("error", f"Missing data file: {chosen / 'all_data.csv'}")]


@pytest.mark.parametrize("extra", [[], LIVE])
def test_the_front_door_returns_the_validation_exit_code(
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nfl-predictor validate`` (and ``--live``) exits with the command's own return code."""
    monkeypatch.setattr(module.log, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.log, "error", lambda *_args, **_kwargs: None)

    assert front_door.main(["validate", *extra, "--data-dir", str(tmp_path)]) == 2


_SCORED_MATCHUPS = (("BUF", "KC"), ("DAL", "PHI"), ("SF", "SEA"))


def _write_newest_first_all_data(tmp_path: Path, unplayed_rows: int = 150) -> Path:
    """Write an all-data CSV whose first rows are unplayed games with empty scores."""
    columns = [
        *module.constants.METADATA_COLUMNS,
        *module.constants.LINES_COLUMNS,
        *module.constants.RESULT_COLUMNS,
    ]
    rows: list[dict[str, object]] = [
        {
            "game_id": f"2026_{index}",
            "season": 2026,
            "week": 1 + index % 18,
            "away_abbr": "BUF",
            "home_abbr": "KC",
        }
        for index in range(unplayed_rows)
    ]
    rows.extend(
        {
            "game_id": f"2025_{index}",
            "season": 2025,
            "week": 18,
            "away_abbr": away,
            "home_abbr": home,
            "away_score": 17 + index,
            "home_score": 20 + index,
        }
        for index, (away, home) in enumerate(_SCORED_MATCHUPS)
    )
    lines = [",".join(columns)]
    lines.extend(",".join(str(row.get(col, "")) for col in columns) for row in rows)
    csv_path = tmp_path / "all_data.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def test_validate_offline_reads_scores_after_many_unplayed_rows(tmp_path: Path) -> None:
    """Scores that first appear after 100+ empty rows are still read as numbers."""
    _write_newest_first_all_data(tmp_path)

    assert module.main(["--data-dir", str(tmp_path)]) == 0


def test_validate_live_reads_scores_after_many_unplayed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live comparison matches numeric schedule scores after 100+ empty rows."""
    _write_newest_first_all_data(tmp_path)
    schedule = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [18, 18, 18],
            "away_abbr": [away for away, _ in _SCORED_MATCHUPS],
            "home_abbr": [home for _, home in _SCORED_MATCHUPS],
            "away_score": [17, 18, 19],
            "home_score": [20, 21, 22],
        }
    )
    monkeypatch.setattr(
        module.validation_utils.polars_utils, "load_schedule", lambda _seasons: schedule
    )

    assert module.main([*LIVE, "--data-dir", str(tmp_path)]) == 0
