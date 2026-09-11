"""Route tests for predictions, betting, power, model, data status, and the registry."""

from __future__ import annotations

from pathlib import Path

import polars
from fastapi.testclient import TestClient

from nfl_predictor.api.routers import resolve
from tests.api import factories


def _seed(project_root: Path, app) -> None:  # noqa: ANN001
    """Create a complete weekly run, a prior-week run, and an unattached prediction file."""
    models = project_root / "models"
    factories.make_run_dir(
        models,
        "weekly_w1",
        season=2026,
        week=1,
        created_at="2026-09-09T00:00:00+00:00",
        with_xlsx=True,
    )
    factories.make_run_dir(
        models,
        "weekly_w0",
        season=2026,
        week=0,
        created_at="2026-09-01T00:00:00+00:00",
        power_order=["DEN", "KC", "GB", "BUF", "MIN"],
    )
    factories.write_predictions_csv(
        project_root / "data" / "predict" / "week_03_predictions.csv", 2026, 3
    )
    app.state.run_index.invalidate()


def test_registry_route(viewer_client: TestClient) -> None:
    """The registry is readable by any signed-in user."""
    payload = viewer_client.get("/api/registry").json()
    assert "home_win_prob" in payload["columns"]
    assert "Model" in payload["groups"]


def test_predictions_routes(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """The active run is the default; weeks list every source; explicit selections resolve."""
    _seed(project_root, app)
    payload = viewer_client.get("/api/predictions").json()
    assert payload["run_id"] == "weekly_w1" and payload["source"] == "active"
    assert payload["season"] == 2026 and payload["week"] == 1
    assert payload["summary"]["games"] == 4
    assert len(payload["table"]["rows"]) == 4
    labels = [w["label"] for w in payload["weeks"]]
    assert labels[0] == "Week 1 (active run)"
    assert "Week 0 (weekly_w0)" in labels
    assert "Week 3 (data/predict)" in labels

    other = viewer_client.get("/api/predictions", params={"week": 0}).json()
    assert other["run_id"] == "weekly_w0" and other["source"] == "run"
    unattached = viewer_client.get("/api/predictions", params={"week": 3}).json()
    assert unattached["run_id"] is None and unattached["source"] == "unattached"
    forced = viewer_client.get("/api/predictions", params={"source": "unattached"}).json()
    assert forced["week"] == 3
    explicit = viewer_client.get("/api/predictions", params={"run": "weekly_w0"}).json()
    assert explicit["run_id"] == "weekly_w0"
    assert viewer_client.get("/api/predictions", params={"week": 9}).status_code == 404
    assert viewer_client.get("/api/predictions", params={"run": "nope"}).status_code == 404

    weeks = viewer_client.get("/api/predictions/weeks").json()
    assert len(weeks) == 3
    picks = viewer_client.get("/api/predictions/picks").json()
    assert picks["table"]["rows"][0]["confidence_rank"] == 4
    derived = viewer_client.get("/api/predictions/picks", params={"week": 3}).json()
    assert derived["run_id"] is None and len(derived["table"]["rows"]) == 4


def test_predictions_without_any_source(viewer_client: TestClient) -> None:
    """With nothing on disk the predictions route is a 404 with a helpful code."""
    response = viewer_client.get("/api/predictions")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_training_run_without_predictions(
    project_root: Path, viewer_client: TestClient, app
) -> None:  # noqa: ANN001
    """An explicit training run has no predictions file."""
    factories.make_run_dir(project_root / "models", "train", kind="training")
    app.state.run_index.invalidate()
    response = viewer_client.get("/api/predictions", params={"run": "train"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_betting_routes(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """The betting table, ladder, and workbook download work for the active run."""
    _seed(project_root, app)
    payload = viewer_client.get("/api/betting").json()
    assert payload["run_id"] == "weekly_w1"
    assert payload["xlsx_available"] is True
    assert [step["action"] for step in payload["ladder"]] == [
        "PASS",
        "LEAN",
        "SMALL",
        "MEDIUM",
        "STRONG",
    ]
    assert payload["table"]["column_metadata"]["total_action"]["actionable"] is False
    xlsx = viewer_client.get("/api/betting/xlsx")
    assert xlsx.status_code == 200 and xlsx.content == b"PK-fake-xlsx"
    assert viewer_client.get("/api/betting/xlsx", params={"run": "weekly_w0"}).status_code == 404
    unattached = viewer_client.get("/api/betting", params={"week": 3}).json()
    assert unattached["xlsx_available"] is False


def test_power_routes(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """Rankings carry movement against the prior-week run and standings tabs are present."""
    _seed(project_root, app)
    payload = viewer_client.get("/api/power").json()
    assert payload["run_id"] == "weekly_w1"
    assert payload["through_week"] == 0
    assert payload["previous_run_id"] is None  # week 0 has no week -1 predecessor
    rows = {row["team_abbr"]: row for row in payload["rankings"]["rows"]}
    assert rows["KC"]["rank"] == 1
    assert payload["standings"] is not None and payload["division_standings"] is not None

    factories.make_run_dir(
        project_root / "models",
        "weekly_w2",
        season=2026,
        week=2,
        created_at="2026-09-16T00:00:00+00:00",
        power_order=["DEN", "KC", "BUF", "GB", "MIN"],
    )
    app.state.run_index.invalidate()
    payload = viewer_client.get("/api/power", params={"run": "weekly_w2"}).json()
    assert payload["previous_run_id"] == "weekly_w1"
    rows = {row["team_abbr"]: row for row in payload["rankings"]["rows"]}
    assert rows["DEN"]["rank_change"] == 1 and rows["KC"]["rank_change"] == -1

    factories.make_run_dir(project_root / "models", "train", kind="training")
    app.state.run_index.invalidate()
    assert viewer_client.get("/api/power", params={"run": "train"}).status_code == 404


def test_model_routes(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """The model payload is available for the active run and by run id."""
    _seed(project_root, app)
    payload = viewer_client.get("/api/model").json()
    assert payload["run_id"] == "weekly_w1" and payload["kind"] == "weekly"
    assert payload["metrics"]["holdout"]["brier"] == 0.2192
    assert payload["wf_compare"]["rows"][0]["label"] == "base"
    by_id = viewer_client.get("/api/runs/weekly_w0/model").json()
    assert by_id["run_id"] == "weekly_w0"
    assert viewer_client.get("/api/runs/nope/model").status_code == 404


def test_data_routes(project_root: Path, viewer_client: TestClient, app) -> None:  # noqa: ANN001
    """Data status reports files, cache coverage, and unattached outputs."""
    _seed(project_root, app)
    (project_root / "data" / "completed_games_ml.csv").write_text(
        "season,week\n2025,1\n", encoding="utf-8"
    )
    payload = viewer_client.get("/api/data/status").json()
    assert payload["current_season"] >= 2026
    names = {f["name"]: f for f in payload["files"]}
    assert names["completed_games_ml.csv"]["exists"] is True
    assert names["completed_games_ml.csv"]["rows"] == 1
    assert names["all_data_ml.csv"]["exists"] is False
    assert payload["cache"] == {"schedule": [], "pbp": []}
    assert payload["leakage_audit"] is None
    assert [f["name"] for f in payload["predict_files"]] == ["week_03_predictions.csv"]
    unattached = viewer_client.get("/api/data/unattached").json()
    assert len(unattached) == 1


def test_weeks_offer_unpredicted_weeks_of_the_current_season(
    project_root: Path,
    viewer_client: TestClient,
    app,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
) -> None:
    """Unplayed weeks with no predictions are listed so the UI can offer to generate them."""
    _seed(project_root, app)
    monkeypatch.setattr(resolve, "current_season_week", lambda: (2026, 2))
    polars.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 2, 4],
            "away_score": [17, None, None],
            "home_score": [24, None, None],
        }
    ).write_csv(project_root / "data" / "all_data_ml.csv")

    weeks = viewer_client.get("/api/predictions/weeks").json()

    available = [week for week in weeks if week["source"] == "available"]
    assert [week["week"] for week in available] == [2, 4]
    assert available[0]["label"] == "Week 2 (not predicted yet)"
    assert all(week["run_id"] is None for week in available)


def test_weeks_offer_nothing_when_the_dataset_is_missing(
    project_root: Path,
    viewer_client: TestClient,
    app,  # noqa: ANN001
) -> None:
    """Without an ML dataset there is nothing to generate from."""
    _seed(project_root, app)
    weeks = viewer_client.get("/api/predictions/weeks").json()
    assert [week for week in weeks if week["source"] == "available"] == []
