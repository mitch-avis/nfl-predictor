"""A run's own week stays visible after later weeks are predicted into its directory.

The ``predict`` job writes ``season_S_week_WW_predictions.csv`` into the active run, so a weekly
run can hold its own week and any number of later weeks. These tests pin which file each route
reads.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from tests.api import factories

from nfl_predictor.api.runs.files import resolve_run_files

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

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
    """Predicting the run's own week again (after a lines refresh) supersedes its picks file."""
    # Arrange
    run_dir = _weekly_run_with_a_later_week(project_root / "models")
    picks = run_dir / "season_2026_week_05_confidence_picks.csv"
    factories.write_picks_csv(picks, 2026, OWN_WEEK)
    picks.write_text(picks.read_text(encoding="utf-8").replace("2026_05_", "stale_"), "utf-8")
    os.utime(picks, (1_000_000_000, 1_000_000_000))
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
