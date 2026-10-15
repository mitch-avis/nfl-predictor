"""Tests for the runs routes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.api.factories import RunSpec, make_run_dir

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


def _make_two_runs(project_root: Path, app) -> None:
    """Write a weekly and a training run and refresh the run index."""
    models = project_root / "models"
    make_run_dir(models, "weekly_a", RunSpec(created_at="2026-09-01T00:00:00+00:00"))
    make_run_dir(
        models, "train_b", RunSpec(kind="training", created_at="2026-09-05T00:00:00+00:00")
    )
    app.state.run_index.invalidate()


def test_list_detail_activate_and_download(
    project_root: Path,
    admin_client: TestClient,
    app,
) -> None:
    """The run list marks the active run."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    listed = admin_client.get("/api/runs").json()

    # Assert
    assert listed["active_run_id"] == "weekly_a"
    assert listed["pinned_run_id"] is None
    assert [r["run_id"] for r in listed["runs"]] == ["train_b", "weekly_a"]
    assert listed["runs"][1]["is_active"] is True


def test_list_runs_filters_by_kind(project_root: Path, admin_client: TestClient, app) -> None:
    """The run list filters on the run kind."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    only_training = admin_client.get("/api/runs", params={"kind": "training"}).json()

    # Assert
    assert [r["run_id"] for r in only_training["runs"]] == ["train_b"]


def test_run_detail_exposes_config(project_root: Path, admin_client: TestClient, app) -> None:
    """Run details expose the config and the feature count."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    detail = admin_client.get("/api/runs/weekly_a").json()

    # Assert
    assert detail["config"]["model_kind"] == "margin_total"
    assert detail["feature_count"] == 3


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("get", "/api/runs/nope"),
        ("post", "/api/runs/nope/activate"),
        ("get", "/api/runs/weekly_a/files/betting_xlsx"),
        ("get", "/api/runs/weekly_a/files/model"),
        ("get", "/api/runs/nope/files/metadata"),
    ],
)
def test_unknown_run_or_file_is_404(
    project_root: Path, admin_client: TestClient, app, method: str, url: str
) -> None:
    """Unknown runs and missing run files return 404."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    response = getattr(admin_client, method)(url)

    # Assert
    assert response.status_code == 404


def test_activate_run_returns_active_run(project_root: Path, admin_client: TestClient, app) -> None:
    """Activating a run marks it active in the response."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    activated = admin_client.post("/api/runs/train_b/activate").json()

    # Assert
    assert activated["is_active"] is True


def test_activate_run_pins_it_in_the_list(
    project_root: Path, admin_client: TestClient, app
) -> None:
    """After activation the run list reports the run as active and pinned."""
    # Arrange
    _make_two_runs(project_root, app)
    admin_client.post("/api/runs/train_b/activate")

    # Act
    listed = admin_client.get("/api/runs").json()

    # Assert
    assert listed["active_run_id"] == "train_b"
    assert listed["pinned_run_id"] == "train_b"


def test_delete_active_unpins_run(project_root: Path, admin_client: TestClient, app) -> None:
    """Deleting the active pin returns 204 and clears the pinned run."""
    # Arrange
    _make_two_runs(project_root, app)
    admin_client.post("/api/runs/train_b/activate")

    # Act
    response = admin_client.delete("/api/runs/active")

    # Assert
    assert response.status_code == 204
    assert admin_client.get("/api/runs").json()["pinned_run_id"] is None


def test_download_run_metadata(project_root: Path, admin_client: TestClient, app) -> None:
    """The metadata file downloads as an attachment named after the run."""
    # Arrange
    _make_two_runs(project_root, app)

    # Act
    metadata = admin_client.get("/api/runs/weekly_a/files/metadata")

    # Assert
    assert metadata.status_code == 200
    assert json.loads(metadata.content)["run_id"] == "weekly_a"
    assert "weekly_a_metadata.json" in metadata.headers["content-disposition"]


@pytest.mark.parametrize(
    "request_and_status",
    [
        ("get", "/api/runs", 200),
        ("post", "/api/runs/weekly_a/activate", 403),
        ("delete", "/api/runs/active", 403),
    ],
)
def test_viewer_cannot_activate(
    project_root: Path,
    viewer_client: TestClient,
    app,
    request_and_status: tuple[str, str, int],
) -> None:
    """Activation is admin-only, but viewers can read runs."""
    # Arrange
    method, url, expected_status = request_and_status
    make_run_dir(project_root / "models", "weekly_a")
    app.state.run_index.invalidate()

    # Act
    response = getattr(viewer_client, method)(url)

    # Assert
    assert response.status_code == expected_status
