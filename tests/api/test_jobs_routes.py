"""Tests for the job routes: catalog, submission, listing, logs, streaming, and cancel."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api.jobs import catalog
from nfl_predictor.api.jobs.runner import JobRunner
from nfl_predictor.api.jobs.store import JobStore, LogLine
from tests.api.jobs_support import fake_template, wait_for, wait_for_status


@pytest.fixture
def runner(app) -> JobRunner:  # noqa: ANN001 - the app fixture has no public type alias
    """Return the runner the application built."""
    return app.state.job_runner


@pytest.fixture
def store(runner: JobRunner) -> JobStore:
    """Return the runner's job store."""
    return runner.store


def test_catalog_lists_every_template(admin_client: TestClient) -> None:
    """The catalog exposes each template with its parameter schema."""
    response = admin_client.get("/api/jobs/catalog")

    assert response.status_code == 200
    payload = response.json()
    ids = {template["id"] for template in payload["templates"]}
    assert {"etl_full", "lines_refresh", "weekly_run", "predict"} <= ids
    lines = next(t for t in payload["templates"] if t["id"] == "lines_refresh")
    assert lines["chain_template_id"] == "predict"
    assert [param["name"] for param in lines["params"]] == ["season", "week"]
    assert payload["busy_groups"] == []


def test_catalog_requires_a_session(client: TestClient) -> None:
    """Anonymous callers cannot read the catalog."""
    assert client.get("/api/jobs/catalog").status_code == 401


def test_creating_a_job_requires_an_admin(viewer_client: TestClient) -> None:
    """Viewers can watch jobs but not launch them."""
    response = viewer_client.post(
        "/api/jobs", json={"template_id": "validate_offline", "params": {}}
    )
    assert response.status_code == 403


def test_creating_a_job_rejects_bad_parameters(admin_client: TestClient) -> None:
    """Parameter problems come back as 422 with a code the form can show."""
    response = admin_client.post(
        "/api/jobs", json={"template_id": "predict", "params": {"season": 2026}}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_param"

    unknown = admin_client.post("/api/jobs", json={"template_id": "nope", "params": {}})
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "unknown_job"


def test_submitting_running_and_listing_a_job(
    admin_client: TestClient,
    runner: JobRunner,
    register: Callable[..., None],
) -> None:
    """A submitted job runs, its logs are readable, and it appears in the list."""
    register(fake_template("validate_offline", args=("--lines", "2")))

    response = admin_client.post(
        "/api/jobs", json={"template_id": "validate_offline", "params": {}}
    )

    assert response.status_code == 201
    job = response.json()
    assert job["status"] == "queued"
    assert job["created_by"] == "admin"
    wait_for_status(runner.store, job["id"], "succeeded")

    detail = admin_client.get(f"/api/jobs/{job['id']}").json()
    assert detail["status"] == "succeeded"
    assert detail["exit_code"] == 0
    logs = admin_client.get(f"/api/jobs/{job['id']}/logs").json()
    assert [line["line"] for line in logs["lines"]][1:] == ["line 1", "line 2"]
    assert logs["next_seq"] == logs["lines"][-1]["seq"]
    listing = admin_client.get("/api/jobs").json()
    assert [item["id"] for item in listing["jobs"]] == [job["id"]]
    assert listing["jobs"][0]["template_label"] == "validate_offline"


def test_logs_can_be_read_from_a_cursor(admin_client: TestClient, store: JobStore) -> None:
    """``after`` returns only the lines the client has not seen."""
    record = store.create("validate_offline", {})
    store.append_logs(record.id, [LogLine("INFO", "one"), LogLine("INFO", "two")])

    payload = admin_client.get(f"/api/jobs/{record.id}/logs?after=1").json()

    assert [line["line"] for line in payload["lines"]] == ["two"]
    assert payload["next_seq"] == 2


def test_logs_of_an_unknown_job_are_a_404(admin_client: TestClient) -> None:
    """An unknown job id is reported as such."""
    response = admin_client.get("/api/jobs/nope/logs")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_a_busy_exclusive_group_is_a_conflict(
    admin_client: TestClient, runner: JobRunner, register: Callable[..., None]
) -> None:
    """A second walk-forward submission is refused while one is active."""
    register(fake_template("weekly_run", args=("--sleep", "10"), group="walk_forward"))

    first = admin_client.post("/api/jobs", json={"template_id": "weekly_run", "params": {}})
    assert first.status_code == 201
    second = admin_client.post("/api/jobs", json={"template_id": "weekly_run", "params": {}})

    assert second.status_code == 409
    assert second.json()["error"]["code"] == "group_busy"
    assert admin_client.get("/api/jobs/catalog").json()["busy_groups"] == ["walk_forward"]

    admin_client.post(f"/api/jobs/{first.json()['id']}/cancel")


def test_cancelling_a_running_job(
    admin_client: TestClient, runner: JobRunner, register: Callable[..., None]
) -> None:
    """An admin can cancel a running job; cancelling a finished one is a conflict."""
    register(fake_template("validate_offline", args=("--sleep", "20")))
    job = admin_client.post(
        "/api/jobs", json={"template_id": "validate_offline", "params": {}}
    ).json()
    wait_for(lambda: any("line 1" in row["line"] for row in runner.store.logs(job["id"])))

    response = admin_client.post(f"/api/jobs/{job['id']}/cancel")

    assert response.status_code == 200
    wait_for_status(runner.store, job["id"], "canceled")
    again = admin_client.post(f"/api/jobs/{job['id']}/cancel")
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "job_finished"


def test_cancelling_requires_an_admin(viewer_client: TestClient, store: JobStore) -> None:
    """Viewers cannot cancel other people's jobs."""
    record = store.create("validate_offline", {})
    assert viewer_client.post(f"/api/jobs/{record.id}/cancel").status_code == 403


def test_stream_replays_logs_and_ends_with_the_final_status(
    admin_client: TestClient, store: JobStore
) -> None:
    """The SSE stream replays stored lines, reports the status, and closes when terminal."""
    record = store.create("validate_offline", {})
    store.append_logs(record.id, [LogLine("INFO", "one"), LogLine("WARNING", "two")])
    store.finish(record.id, "succeeded", exit_code=0)

    with admin_client.stream("GET", f"/api/jobs/{record.id}/stream") as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [line for line in body.splitlines() if line.startswith(("event:", "data:"))]
    assert "event: log" in events
    assert "event: end" in events
    logged = [json.loads(line[6:]) for line in events if line.startswith("data:")]
    assert logged[0]["line"] == "one"
    assert logged[-1]["status"] == "succeeded"


def test_stream_resumes_from_a_cursor(admin_client: TestClient, store: JobStore) -> None:
    """A reconnecting client passes ``after`` and does not receive lines twice."""
    record = store.create("validate_offline", {})
    store.append_logs(record.id, [LogLine("INFO", "one"), LogLine("INFO", "two")])
    store.finish(record.id, "failed", exit_code=1, error="boom")

    with admin_client.stream("GET", f"/api/jobs/{record.id}/stream?after=1") as response:
        body = "".join(response.iter_text())

    assert "one" not in body
    assert "two" in body
    assert "boom" in body


def test_stream_rejects_an_unknown_job(admin_client: TestClient) -> None:
    """Streaming a job that does not exist fails before the stream opens."""
    assert admin_client.get("/api/jobs/nope/stream").status_code == 404


def test_the_catalog_the_app_serves_is_the_real_one(admin_client: TestClient) -> None:
    """Every real template is exposed, so the UI can launch all of them."""
    payload = admin_client.get("/api/jobs/catalog").json()
    assert {template["id"] for template in payload["templates"]} == {
        template.id for template in catalog.TEMPLATES
    }
