"""Tests for run discovery, classification, and the active-run pointer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import ConflictError, NotFoundError
from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.files import resolve_run_files
from nfl_predictor.api.runs.indexer import RunIndex, scan_runs, summarize_run
from tests.api.factories import make_run_dir


def test_scan_classifies_and_orders_runs(project_root: Path) -> None:
    """Weekly, training, and walk-forward runs are recognized; junk is skipped."""
    models = project_root / "models"
    make_run_dir(models, "weekly_old", created_at="2026-01-01T00:00:00+00:00", season=2025, week=22)
    make_run_dir(models, "weekly_new", created_at="2026-09-09T00:00:00+00:00")
    make_run_dir(models, "train_only", kind="training", created_at="2026-05-01T00:00:00+00:00")
    make_run_dir(models, "wf_report", kind="walk_forward", created_at="2026-06-01T00:00:00+00:00")
    (models / "empty_dir").mkdir()
    (models / "stray.json").write_text("{}", encoding="utf-8")
    (models / "broken").mkdir()
    (models / "broken" / "metadata.json").write_text("not json", encoding="utf-8")
    (models / "listy").mkdir()
    (models / "listy" / "metadata.json").write_text("[1]", encoding="utf-8")

    runs = scan_runs(models)
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
    assert by_id["weekly_old"].files["betting_xlsx"] is False


def test_scan_missing_dir_and_partial_runs(project_root: Path) -> None:
    """A missing models dir yields nothing; incomplete runs report their stages."""
    assert scan_runs(project_root / "nope") == []
    models = project_root / "models"
    run_dir = make_run_dir(models, "partial", complete=False, with_model=False)
    (run_dir / "metrics_report.json").write_text("{}", encoding="utf-8")
    summary = summarize_run(run_dir)
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
    assert summarize_run(models / "missing") is None


def test_season_week_from_config_without_files(project_root: Path) -> None:
    """A training run with no predict path has no season/week; odd configs are tolerated."""
    models = project_root / "models"
    run_dir = make_run_dir(models, "bare", kind="training")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    meta["config"] = "not a dict"
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    summary = summarize_run(run_dir)
    assert summary is not None
    assert (summary.season, summary.week) == (None, None)
    assert summary.model_kind is None
    meta["config"] = {"predict_path": "data/predict/week_05_games_to_predict.csv"}
    meta["created_at"] = "bogus"
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    summary = summarize_run(run_dir)
    assert summary is not None
    assert (summary.season, summary.week) == (None, 5)


def test_run_files_prefers_newest_week(project_root: Path) -> None:
    """When a run holds several weeks the newest is exposed."""
    run_dir = make_run_dir(project_root / "models", "multi", season=2026, week=1)
    (run_dir / "season_2026_week_03_predictions.csv").write_text("game_id\n", encoding="utf-8")
    (run_dir / "season_bad_predictions.csv").write_text("game_id\n", encoding="utf-8")
    files = resolve_run_files(run_dir)
    assert files.predictions is not None
    assert files.predictions.name == "season_2026_week_03_predictions.csv"
    assert (files.season, files.week) == (2026, 3)
    assert files.picks is not None and files.picks.name.endswith("week_01_confidence_picks.csv")


def test_index_caches_and_invalidates(project_root: Path) -> None:
    """The index reuses its scan until invalidated or expired."""
    models = project_root / "models"
    make_run_dir(models, "first")
    index = RunIndex(models, ttl_seconds=3600)
    assert [r.run_id for r in index.runs()] == ["first"]
    make_run_dir(models, "second", created_at="2026-09-10T00:00:00+00:00")
    assert [r.run_id for r in index.runs()] == ["first"]
    index.invalidate()
    assert [r.run_id for r in index.runs()] == ["second", "first"]
    assert index.get("first") is not None
    assert index.get("missing") is None
    expired = RunIndex(models, ttl_seconds=0)
    expired.runs()
    make_run_dir(models, "third", created_at="2026-09-11T00:00:00+00:00")
    assert expired.get("third") is not None


def test_active_run_resolution(project_root: Path, db: Database) -> None:
    """Pinned wins; else newest complete weekly run with a model; else none."""
    models = project_root / "models"
    index = RunIndex(models, ttl_seconds=0)
    assert active.resolve_active_run(db, index) is None
    make_run_dir(models, "incomplete", complete=False, created_at="2026-09-12T00:00:00+00:00")
    make_run_dir(models, "done", created_at="2026-09-10T00:00:00+00:00")
    make_run_dir(models, "train", kind="training", created_at="2026-09-13T00:00:00+00:00")
    make_run_dir(models, "wf", kind="walk_forward", created_at="2026-09-14T00:00:00+00:00")
    resolved = active.resolve_active_run(db, index)
    assert resolved is not None and resolved.run_id == "done"
    active.set_active_run(db, index, "train")
    resolved = active.resolve_active_run(db, index)
    assert resolved is not None and resolved.run_id == "train"
    with pytest.raises(NotFoundError):
        active.set_active_run(db, index, "ghost")
    with pytest.raises(ConflictError):
        active.set_active_run(db, index, "wf")
    db.set_value(active.ACTIVE_RUN_KEY, "deleted")
    resolved = active.resolve_active_run(db, index)
    assert resolved is not None and resolved.run_id == "done"
    active.clear_active_run(db)
    assert active.pinned_run_id(db) is None
    assert active.require_run(db, index, None).run_id == "done"
    assert active.require_run(db, index, "train").run_id == "train"
    with pytest.raises(NotFoundError):
        active.require_run(db, index, "ghost")


def test_require_run_without_any_run(project_root: Path, db: Database) -> None:
    """With nothing on disk the active-run lookup is a 404."""
    with pytest.raises(NotFoundError):
        active.require_run(db, RunIndex(project_root / "models", ttl_seconds=0), None)
