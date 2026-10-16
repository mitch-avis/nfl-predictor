"""A run's own week stays visible after later weeks are predicted into its directory.

The ``predict`` job writes ``season_S_week_WW_predictions.csv`` into the active run, so a weekly
run can hold its own week and any number of later weeks. These tests pin which file each route
reads.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.runs import active
from nfl_predictor.api.runs.files import resolve_run_files
from tests.api import factories

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nfl_predictor.api.db import Database

OWN_WEEK = 5
LATER_WEEK = 6


def _weekly_run_with_a_later_week(models: Path, *, marker: bool = True) -> Path:
    """Create a week-5 weekly run that also holds predictions for week 6.

    With ``marker`` the run's ``predictions_state.json`` names its own predictions file by an
    absolute path from another machine, the way a copied run directory carries it.
    """
    run_dir = factories.make_run_dir(
        models, "weekly_w5", factories.RunSpec(season=2026, week=OWN_WEEK)
    )
    if marker:
        own = f"/elsewhere/models/weekly_w5/season_2026_week_{OWN_WEEK:02d}_predictions.csv"
        (run_dir / "predictions_state.json").write_text(
            json.dumps({"stage": "predictions", "outputs": [own]}), encoding="utf-8"
        )
    factories.write_predictions_csv(
        run_dir / f"season_2026_week_{LATER_WEEK:02d}_predictions.csv", 2026, LATER_WEEK
    )
    return run_dir


def _newer_weekly_run(models: Path, *, marker: bool = True) -> Path:
    """Create the week-6 weekly run, written after the week-5 run.

    With ``marker`` its predictions stage names its week-6 file; without, week 6 is only its
    newest predictions file.
    """
    run_dir = factories.make_run_dir(
        models,
        "weekly_w6",
        factories.RunSpec(season=2026, week=LATER_WEEK, created_at="2026-10-14T00:00:00+00:00"),
    )
    if marker:
        own = f"/elsewhere/models/weekly_w6/season_2026_week_{LATER_WEEK:02d}_predictions.csv"
        (run_dir / "predictions_state.json").write_text(
            json.dumps({"stage": "predictions", "outputs": [own]}), encoding="utf-8"
        )
    return run_dir


def _pin_the_older_run(models: Path, app: FastAPI, db: Database, *, marker: bool = True) -> None:
    """Create the week-5 run holding a week-6 file and the newer week-6 run; pin the older one."""
    _weekly_run_with_a_later_week(models)
    _newer_weekly_run(models, marker=marker)
    app.state.run_index.invalidate()
    active.set_active_run(db, app.state.run_index, "weekly_w5")


def test_a_newer_runs_own_week_outranks_a_pinned_runs_file_for_it(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """A week a newer run predicted as its own wins over the pinned run's ahead-of-time file."""
    # Arrange
    _pin_the_older_run(project_root / "models", app, db)

    # Act
    payload = viewer_client.get("/api/predictions", params={"season": 2026, "week": 6}).json()

    # Assert
    assert (payload["source"], payload["run_id"], payload["week"]) == ("run", "weekly_w6", 6)


def test_weeks_list_a_newer_runs_own_week_instead_of_a_pinned_runs_file(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """The selector offers the newer run's week 6, not the pinned run's ahead-of-time copy."""
    # Arrange
    _pin_the_older_run(project_root / "models", app, db)

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    assert [(w["week"], w["source"], w["run_id"]) for w in weeks] == [
        (OWN_WEEK, "active", "weekly_w5"),
        (LATER_WEEK, "run", "weekly_w6"),
    ]


def test_an_unmarked_runs_newest_file_does_not_outrank_the_pinned_runs_file(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """Only a run whose predictions stage wrote the week outranks the pinned run's file."""
    # Arrange
    _pin_the_older_run(project_root / "models", app, db, marker=False)

    # Act
    payload = viewer_client.get("/api/predictions", params={"season": 2026, "week": 6}).json()

    # Assert
    assert (payload["source"], payload["run_id"], payload["week"]) == ("active", "weekly_w5", 6)


def _marked_run(models: Path, run_id: str, week: int, created_at: str, *, marker: bool) -> None:
    """Create a weekly run for ``week``, with or without its predictions stage marker."""
    run_dir = factories.make_run_dir(
        models, run_id, factories.RunSpec(season=2026, week=week, created_at=created_at)
    )
    if marker:
        own = f"/elsewhere/models/{run_id}/season_2026_week_{week:02d}_predictions.csv"
        (run_dir / "predictions_state.json").write_text(
            json.dumps({"stage": "predictions", "outputs": [own]}), encoding="utf-8"
        )


def _marked_and_unmarked_week_six(models: Path, app: FastAPI, db: Database) -> None:
    """Pin a week-5 run beside an older marked week-6 run and a newer unmarked one."""
    _marked_run(models, "weekly_w5", OWN_WEEK, "2026-10-07T00:00:00+00:00", marker=True)
    _marked_run(models, "weekly_w6", LATER_WEEK, "2026-10-14T00:00:00+00:00", marker=True)
    _marked_run(models, "copy_w6", LATER_WEEK, "2026-10-15T00:00:00+00:00", marker=False)
    app.state.run_index.invalidate()
    active.set_active_run(db, app.state.run_index, "weekly_w5")


def test_weeks_list_the_marked_runs_week_over_a_newer_unmarked_copy(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """The selector offers the week the routes resolve to: the run whose stage wrote it."""
    # Arrange
    _marked_and_unmarked_week_six(project_root / "models", app, db)

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    assert [(w["week"], w["source"], w["run_id"]) for w in weeks] == [
        (OWN_WEEK, "active", "weekly_w5"),
        (LATER_WEEK, "run", "weekly_w6"),
    ]


def test_predictions_resolve_the_marked_runs_week_over_a_newer_unmarked_copy(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """A season-and-week query takes the run whose predictions stage wrote the week."""
    # Arrange
    _marked_and_unmarked_week_six(project_root / "models", app, db)

    # Act
    payload = viewer_client.get("/api/predictions", params={"season": 2026, "week": 6}).json()

    # Assert
    assert (payload["source"], payload["run_id"]) == ("run", "weekly_w6")


def test_run_files_keep_the_week_the_run_predicted(tmp_path: Path) -> None:
    """The predictions stage marker names the run's own week, whatever was predicted later."""
    # Arrange
    run_dir = _weekly_run_with_a_later_week(tmp_path)

    # Act
    files = resolve_run_files(run_dir)

    # Assert
    assert files.predictions == run_dir / "season_2026_week_05_predictions.csv"
    assert (files.season, files.week) == (2026, OWN_WEEK)
    assert files.week_predictions == {
        (2026, OWN_WEEK): run_dir / "season_2026_week_05_predictions.csv",
        (2026, LATER_WEEK): run_dir / "season_2026_week_06_predictions.csv",
    }


def test_run_files_without_a_marker_use_the_newest_week(tmp_path: Path) -> None:
    """A run whose marker names no predictions file falls back to its newest week."""
    # Arrange
    run_dir = _weekly_run_with_a_later_week(tmp_path, marker=False)

    # Act
    files = resolve_run_files(run_dir)

    # Assert
    assert files.predictions == run_dir / "season_2026_week_06_predictions.csv"
    assert files.week == LATER_WEEK


@pytest.mark.parametrize(
    "marker",
    [
        "not json",
        '["season_2026_week_05_predictions.csv"]',
        '{"outputs": "season_2026_week_05_predictions.csv"}',
        '{"outputs": [7, "season_2026_week_09_predictions.csv"]}',
    ],
    ids=["not-json", "not-an-object", "outputs-not-a-list", "no-output-matches"],
)
def test_run_files_with_an_unusable_marker_use_the_newest_week(tmp_path: Path, marker: str) -> None:
    """A marker that names no predictions file in the directory falls back to the newest week."""
    # Arrange
    run_dir = _weekly_run_with_a_later_week(tmp_path)
    (run_dir / "predictions_state.json").write_text(marker, encoding="utf-8")

    # Act
    files = resolve_run_files(run_dir)

    # Assert
    assert files.week == LATER_WEEK


def test_predictions_skip_a_pinned_run_without_predictions(
    project_root: Path, viewer_client: TestClient, app: FastAPI, db: Database
) -> None:
    """A pinned training run predicts nothing, so a week resolves to the run that predicted it."""
    # Arrange
    models = project_root / "models"
    _weekly_run_with_a_later_week(models)
    factories.make_run_dir(models, "train", factories.RunSpec(kind="training"))
    app.state.run_index.invalidate()
    active.set_active_run(db, app.state.run_index, "train")

    # Act
    payload = viewer_client.get("/api/predictions", params={"season": 2026, "week": 5}).json()

    # Assert
    assert (payload["source"], payload["run_id"], payload["week"]) == ("run", "weekly_w5", 5)


def test_predictions_default_to_the_active_runs_own_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """With no week asked for, the page shows the week the active run predicted."""
    # Arrange
    _weekly_run_with_a_later_week(project_root / "models")
    app.state.run_index.invalidate()

    # Act
    payload = viewer_client.get("/api/predictions").json()

    # Assert
    assert (payload["source"], payload["season"], payload["week"]) == ("active", 2026, OWN_WEEK)


def test_predictions_read_a_later_week_from_the_active_run(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A week predicted into the active run after its own week resolves to that week's file."""
    # Arrange
    _weekly_run_with_a_later_week(project_root / "models")
    app.state.run_index.invalidate()

    # Act
    payload = viewer_client.get("/api/predictions", params={"season": 2026, "week": 6}).json()

    # Assert
    assert (payload["source"], payload["run_id"], payload["week"]) == ("active", "weekly_w5", 6)
    assert {row["game_id"][:7] for row in payload["table"]["rows"]} == {"2026_06"}


def test_weeks_list_every_week_the_active_run_predicted(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The week selector offers the active run's own week first, then its later weeks."""
    # Arrange
    _weekly_run_with_a_later_week(project_root / "models")
    app.state.run_index.invalidate()

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    assert [(w["week"], w["source"], w["label"]) for w in weeks] == [
        (OWN_WEEK, "active", "Week 5 (active run)"),
        (LATER_WEEK, "active", "Week 6 (active run)"),
    ]


def test_picks_for_a_later_week_come_from_its_predictions(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The run's picks file belongs to its own week, so a later week derives its picks."""
    # Arrange
    _weekly_run_with_a_later_week(project_root / "models")
    app.state.run_index.invalidate()

    # Act
    payload = viewer_client.get("/api/predictions/picks", params={"season": 2026, "week": 6}).json()

    # Assert
    assert payload["week"] == LATER_WEEK
    assert {row["game_id"][:7] for row in payload["table"]["rows"]} == {"2026_06"}


def test_picks_follow_predictions_written_after_the_picks_file(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Predicting the run's own week again before its lock supersedes its picks file.

    The fixture week's first kickoff is Sept. 13; the picks are written on Sept. 10 and the week
    is predicted again (after a lines refresh) on Sept. 12.
    """
    # Arrange
    run_dir = _weekly_run_with_a_later_week(project_root / "models")
    picks = run_dir / "season_2026_week_05_confidence_picks.csv"
    factories.write_picks_csv(picks, 2026, OWN_WEEK)
    picks.write_text(picks.read_text(encoding="utf-8").replace("2026_05_", "stale_"), "utf-8")
    picks_day = datetime(2026, 9, 10, tzinfo=UTC).timestamp()
    repredicted = datetime(2026, 9, 12, tzinfo=UTC).timestamp()
    os.utime(picks, (picks_day, picks_day))
    os.utime(run_dir / "season_2026_week_05_predictions.csv", (repredicted, repredicted))
    app.state.run_index.invalidate()

    # Act
    payload = viewer_client.get("/api/predictions/picks").json()

    # Assert
    assert {row["game_id"][:7] for row in payload["table"]["rows"]} == {"2026_05"}


def test_picks_read_the_picks_file_of_the_runs_own_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A picks file at least as new as its week's predictions is the one served."""
    # Arrange
    run_dir = _weekly_run_with_a_later_week(project_root / "models")
    picks = run_dir / "season_2026_week_05_confidence_picks.csv"
    picks.write_text(picks.read_text(encoding="utf-8").replace("2026_05_", "files_"), "utf-8")
    predictions = run_dir / "season_2026_week_05_predictions.csv"
    os.utime(predictions, (1_000_000_000, 1_000_000_000))
    app.state.run_index.invalidate()

    # Act
    payload = viewer_client.get("/api/predictions/picks").json()

    # Assert
    assert {row["game_id"][:6] for row in payload["table"]["rows"]} == {"files_"}
