"""Tests for run discovery, classification, and the active-run pointer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.errors import ConflictError, NotFoundError
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.files import resolve_run_files
from nfl_predictor.api.runs.indexer import RunIndex, scan_runs, summarize_run
from tests.api.factories import RunSpec, make_run_dir

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.api.db import Database


def _make_bare_training_run(project_root: Path) -> tuple[Path, dict[str, object]]:
    """Write a training run whose metadata config is not a dict; return it and its metadata."""
    run_dir = make_run_dir(project_root / "models", "bare", RunSpec(kind="training"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    meta["config"] = "not a dict"
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir, meta


def _make_active_candidates(models: Path) -> None:
    """Write an incomplete, a complete weekly, a training and a walk-forward run."""
    make_run_dir(
        models, "incomplete", RunSpec(complete=False, created_at="2026-09-12T00:00:00+00:00")
    )
    make_run_dir(models, "done", RunSpec(created_at="2026-09-10T00:00:00+00:00"))
    make_run_dir(models, "train", RunSpec(kind="training", created_at="2026-09-13T00:00:00+00:00"))
    make_run_dir(models, "wf", RunSpec(kind="walk_forward", created_at="2026-09-14T00:00:00+00:00"))


def test_scan_classifies_and_orders_runs(project_root: Path) -> None:
    """Weekly, training, and walk-forward runs are recognized; junk is skipped."""
    # Arrange
    models = project_root / "models"
    make_run_dir(
        models, "weekly_old", RunSpec(created_at="2026-01-01T00:00:00+00:00", season=2025, week=22)
    )
    make_run_dir(models, "weekly_new", RunSpec(created_at="2026-09-09T00:00:00+00:00"))
    make_run_dir(
        models, "train_only", RunSpec(kind="training", created_at="2026-05-01T00:00:00+00:00")
    )
    make_run_dir(
        models, "wf_report", RunSpec(kind="walk_forward", created_at="2026-06-01T00:00:00+00:00")
    )
    (models / "empty_dir").mkdir()
    (models / "stray.json").write_text("{}", encoding="utf-8")
    (models / "broken").mkdir()
    (models / "broken" / "metadata.json").write_text("not json", encoding="utf-8")
    (models / "listy").mkdir()
    (models / "listy" / "metadata.json").write_text("[1]", encoding="utf-8")

    # Act
    runs = scan_runs(models)

    # Assert
    assert [r.run_id for r in runs] == ["weekly_new", "wf_report", "train_only", "weekly_old"]
    by_id = {r.run_id: r for r in runs}
    assert by_id["weekly_new"].kind == "weekly"
    assert by_id["weekly_new"].complete
    assert (by_id["weekly_new"].season, by_id["weekly_new"].week) == (2026, 1)
    assert by_id["weekly_new"].run_files.power_week == 0
    assert by_id["train_only"].kind == "training"
    assert (by_id["train_only"].season, by_id["train_only"].week) == (2026, 1)
    assert by_id["train_only"].holdout == pytest.approx(
        {"winner_accuracy": 0.6471, "brier": 0.2192, "margin_mae": 9.847, "total_mae": 10.997}
    )
    assert by_id["wf_report"].kind == "walk_forward"
    assert not by_id["wf_report"].has_model
    assert by_id["wf_report"].holdout is not None
    assert by_id["wf_report"].holdout["brier"] == pytest.approx(0.2277)
    assert "betting_xlsx" not in by_id["weekly_old"].files


def test_scan_missing_dir_and_partial_runs(project_root: Path) -> None:
    """A missing models dir yields nothing."""
    # Act
    runs = scan_runs(project_root / "nope")

    # Assert
    assert runs == []


def test_summarize_partial_run_reports_stages(project_root: Path) -> None:
    """Incomplete runs report their stages."""
    # Arrange
    run_dir = make_run_dir(
        project_root / "models", "partial", RunSpec(complete=False, with_model=False)
    )
    (run_dir / "metrics_report.json").write_text("{}", encoding="utf-8")

    # Act
    summary = summarize_run(run_dir)

    # Assert
    assert summary is not None
    assert summary.stages == {
        "wf_compare": True,
        "train": True,
        "predictions": True,
        "reports": False,
    }
    assert not summary.complete
    assert summary.holdout is None
    assert not summary.has_model


def test_summarize_missing_run_is_none(project_root: Path) -> None:
    """A run directory that does not exist has no summary."""
    # Act
    summary = summarize_run(project_root / "models" / "missing")

    # Assert
    assert summary is None


def test_season_week_from_config_without_files(project_root: Path) -> None:
    """A training run with no predict path has no season/week; odd configs are tolerated."""
    # Arrange
    run_dir, _ = _make_bare_training_run(project_root)

    # Act
    summary = summarize_run(run_dir)

    # Assert
    assert summary is not None
    assert (summary.season, summary.week) == (None, None)
    assert summary.model_kind is None


def test_week_from_predict_path_with_bogus_created_at(project_root: Path) -> None:
    """The week comes from the predict path even when ``created_at`` is not a date."""
    # Arrange
    run_dir, meta = _make_bare_training_run(project_root)
    meta["config"] = {"predict_path": "data/predict/week_05_games_to_predict.csv"}
    meta["created_at"] = "bogus"
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Act
    summary = summarize_run(run_dir)

    # Assert
    assert summary is not None
    assert (summary.season, summary.week) == (None, 5)


def test_run_files_prefers_newest_week(project_root: Path) -> None:
    """When a run holds several weeks the newest is exposed."""
    # Arrange
    run_dir = make_run_dir(project_root / "models", "multi", RunSpec(season=2026, week=1))
    (run_dir / "season_2026_week_03_predictions.csv").write_text("game_id\n", encoding="utf-8")
    (run_dir / "season_bad_predictions.csv").write_text("game_id\n", encoding="utf-8")

    # Act
    files = resolve_run_files(run_dir)

    # Assert
    assert files.predictions is not None
    assert files.predictions.name == "season_2026_week_03_predictions.csv"
    assert (files.season, files.week) == (2026, 3)
    assert files.picks is not None
    assert files.picks.name.endswith("week_01_confidence_picks.csv")


def test_index_caches_and_invalidates(project_root: Path) -> None:
    """The index scans on first use."""
    # Arrange
    models = project_root / "models"
    make_run_dir(models, "first")
    index = RunIndex(models, ttl_seconds=3600)

    # Act
    runs = index.runs()

    # Assert
    assert [r.run_id for r in runs] == ["first"]


def test_index_reuses_its_scan_until_invalidated(project_root: Path) -> None:
    """A run added after the scan is not seen before invalidation."""
    # Arrange
    models = project_root / "models"
    make_run_dir(models, "first")
    index = RunIndex(models, ttl_seconds=3600)
    index.runs()
    make_run_dir(models, "second", RunSpec(created_at="2026-09-10T00:00:00+00:00"))

    # Act
    runs = index.runs()

    # Assert
    assert [r.run_id for r in runs] == ["first"]


def test_index_invalidate_rescans(project_root: Path) -> None:
    """After invalidation the index sees new runs and looks runs up by id."""
    # Arrange
    models = project_root / "models"
    make_run_dir(models, "first")
    index = RunIndex(models, ttl_seconds=3600)
    index.runs()
    make_run_dir(models, "second", RunSpec(created_at="2026-09-10T00:00:00+00:00"))
    index.runs()

    # Act
    index.invalidate()

    # Assert
    assert [r.run_id for r in index.runs()] == ["second", "first"]
    assert index.get("first") is not None
    assert index.get("missing") is None


def test_index_rescans_once_expired(project_root: Path) -> None:
    """An expired index rescans and finds runs added after its last scan."""
    # Arrange
    models = project_root / "models"
    make_run_dir(models, "first")
    expired = RunIndex(models, ttl_seconds=0)
    expired.runs()
    make_run_dir(models, "third", RunSpec(created_at="2026-09-11T00:00:00+00:00"))

    # Act
    found = expired.get("third")

    # Assert
    assert found is not None


def test_active_run_resolution(project_root: Path, db: Database) -> None:
    """With no runs there is no active run."""
    # Arrange
    index = RunIndex(project_root / "models", ttl_seconds=0)

    # Act
    resolved = active.resolve_active_run(db, index)

    # Assert
    assert resolved is None


def test_active_run_is_newest_complete_weekly_run(project_root: Path, db: Database) -> None:
    """Without a pin the newest complete weekly run with a model is active."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)

    # Act
    resolved = active.resolve_active_run(db, index)

    # Assert
    assert resolved is not None
    assert resolved.run_id == "done"


def test_pinned_run_wins(project_root: Path, db: Database) -> None:
    """A pinned run is the active run."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)
    active.set_active_run(db, index, "train")

    # Act
    resolved = active.resolve_active_run(db, index)

    # Assert
    assert resolved is not None
    assert resolved.run_id == "train"


@pytest.mark.parametrize(("run_id", "error"), [("ghost", NotFoundError), ("wf", ConflictError)])
def test_set_active_run_rejects_unknown_or_walk_forward_run(
    project_root: Path, db: Database, run_id: str, error: type[Exception]
) -> None:
    """Unknown runs and walk-forward runs cannot be pinned."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)
    active.set_active_run(db, index, "train")

    # Act & Assert
    with pytest.raises(error):
        active.set_active_run(db, index, run_id)


def test_pin_to_deleted_run_falls_back(project_root: Path, db: Database) -> None:
    """A pin to a run that no longer exists falls back to the newest weekly run."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)
    active.set_active_run(db, index, "train")
    db.set_value(active.ACTIVE_RUN_KEY, "deleted")

    # Act
    resolved = active.resolve_active_run(db, index)

    # Assert
    assert resolved is not None
    assert resolved.run_id == "done"


def test_clear_active_run_removes_pin(project_root: Path, db: Database) -> None:
    """Clearing the active run removes the pin."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)
    active.set_active_run(db, index, "train")

    # Act
    active.clear_active_run(db)

    # Assert
    assert active.pinned_run_id(db) is None


@pytest.mark.parametrize(("requested", "expected"), [(None, "done"), ("train", "train")])
def test_require_run_resolves_default_or_named_run(
    project_root: Path, db: Database, requested: str | None, expected: str
) -> None:
    """``require_run`` returns the active run by default and a named run when asked."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)
    active.set_active_run(db, index, "train")
    active.clear_active_run(db)

    # Act
    run = active.require_run(db, index, requested)

    # Assert
    assert run.run_id == expected


def test_require_run_unknown_run_not_found(project_root: Path, db: Database) -> None:
    """``require_run`` raises not found for an unknown run id."""
    # Arrange
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    _make_active_candidates(models)

    # Act & Assert
    with pytest.raises(NotFoundError):
        active.require_run(db, index, "ghost")


def test_require_run_without_any_run(project_root: Path, db: Database) -> None:
    """With nothing on disk the active-run lookup is a 404."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        active.require_run(db, RunIndex(project_root / "models", ttl_seconds=0), None)
