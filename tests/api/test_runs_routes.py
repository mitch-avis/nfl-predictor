"""Tests for the runs routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.factories import make_run_dir


def test_list_detail_activate_and_download(
    project_root: Path,
    admin_client: TestClient,
    app,  # noqa: ANN001
) -> None:
    """The run list marks the active run, details expose config, activation pins."""
    models = project_root / "models"
    make_run_dir(models, "weekly_a", created_at="2026-09-01T00:00:00+00:00", with_xlsx=True)
    make_run_dir(models, "train_b", kind="training", created_at="2026-09-05T00:00:00+00:00")
    app.state.run_index.invalidate()

    listed = admin_client.get("/api/runs").json()
    assert listed["active_run_id"] == "weekly_a"
    assert listed["pinned_run_id"] is None
    assert [r["run_id"] for r in listed["runs"]] == ["train_b", "weekly_a"]
    assert listed["runs"][1]["is_active"] is True
    only_training = admin_client.get("/api/runs", params={"kind": "training"}).json()
    assert [r["run_id"] for r in only_training["runs"]] == ["train_b"]

    detail = admin_client.get("/api/runs/weekly_a").json()
    assert detail["config"]["model_kind"] == "margin_total"
    assert detail["feature_count"] == 3
    assert admin_client.get("/api/runs/nope").status_code == 404

    activated = admin_client.post("/api/runs/train_b/activate").json()
    assert activated["is_active"] is True
    listed = admin_client.get("/api/runs").json()
    assert listed["active_run_id"] == "train_b"
    assert listed["pinned_run_id"] == "train_b"
    assert admin_client.post("/api/runs/nope/activate").status_code == 404

    assert admin_client.delete("/api/runs/active").status_code == 204
    assert admin_client.get("/api/runs").json()["pinned_run_id"] is None

    xlsx = admin_client.get("/api/runs/weekly_a/files/betting_xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content == b"PK-fake-xlsx"
    assert "weekly_a_betting_report.xlsx" in xlsx.headers["content-disposition"]
    assert admin_client.get("/api/runs/train_b/files/betting_xlsx").status_code == 404
    assert admin_client.get("/api/runs/weekly_a/files/model").status_code == 404
    assert admin_client.get("/api/runs/nope/files/metadata").status_code == 404


def test_viewer_cannot_activate(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """Activation is admin-only, but viewers can read runs."""
    make_run_dir(project_root / "models", "weekly_a")
    app.state.run_index.invalidate()
    assert viewer_client.get("/api/runs").status_code == 200
    assert viewer_client.post("/api/runs/weekly_a/activate").status_code == 403
    assert viewer_client.delete("/api/runs/active").status_code == 403
