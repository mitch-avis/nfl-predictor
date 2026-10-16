"""Route tests for predictions, betting, power, model, data status, and the registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.api.routers import resolve
from tests.api import factories

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient


def _seed(project_root: Path, app: FastAPI) -> None:
    """Create a complete weekly run, a prior-week run, and an unattached prediction file."""
    models = project_root / "models"
    factories.make_run_dir(
        models,
        "weekly_w1",
        factories.RunSpec(season=2026, week=1, created_at="2026-09-09T00:00:00+00:00"),
    )
    factories.make_run_dir(
        models,
        "weekly_w0",
        factories.RunSpec(
            season=2026,
            week=0,
            created_at="2026-09-01T00:00:00+00:00",
            power_order=["DEN", "KC", "GB", "BUF", "MIN"],
        ),
    )
    factories.write_predictions_csv(
        project_root / "data" / "predict" / "week_03_predictions.csv", 2026, 3
    )
    app.state.run_index.invalidate()


def _add_weekly_w2(project_root: Path, app: FastAPI) -> None:
    """Add a week-2 run whose power order swaps DEN and KC against week 1."""
    factories.make_run_dir(
        project_root / "models",
        "weekly_w2",
        factories.RunSpec(
            season=2026,
            week=2,
            created_at="2026-09-16T00:00:00+00:00",
            power_order=["DEN", "KC", "BUF", "GB", "MIN"],
        ),
    )
    app.state.run_index.invalidate()


def test_registry_route(viewer_client: TestClient) -> None:
    """The registry is readable by any signed-in user."""
    # Act
    payload = viewer_client.get("/api/registry").json()

    # Assert
    assert "home_win_prob" in payload["columns"]
    assert "Model" in payload["groups"]


def test_predictions_routes(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """The active run is the default and weeks list every source."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/predictions").json()

    # Assert
    assert payload["run_id"] == "weekly_w1"
    assert payload["source"] == "active"
    assert payload["season"] == 2026
    assert payload["week"] == 1
    assert payload["summary"]["games"] == 4
    assert len(payload["table"]["rows"]) == 4
    labels = [w["label"] for w in payload["weeks"]]
    assert labels[0] == "Week 1 (active run)"
    assert "Week 0 (weekly_w0)" in labels
    assert "Week 3 (data/predict)" in labels


def test_predictions_week_selects_its_run(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A week with a run resolves to that run."""
    # Arrange
    _seed(project_root, app)

    # Act
    other = viewer_client.get("/api/predictions", params={"week": 0}).json()

    # Assert
    assert other["run_id"] == "weekly_w0"
    assert other["source"] == "run"


def test_predictions_week_without_run_is_unattached(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A week with only a prediction file resolves to the unattached source."""
    # Arrange
    _seed(project_root, app)

    # Act
    unattached = viewer_client.get("/api/predictions", params={"week": 3}).json()

    # Assert
    assert unattached["run_id"] is None
    assert unattached["source"] == "unattached"


def test_predictions_forced_unattached_source(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Forcing the unattached source picks the unattached prediction file."""
    # Arrange
    _seed(project_root, app)

    # Act
    forced = viewer_client.get("/api/predictions", params={"source": "unattached"}).json()

    # Assert
    assert forced["week"] == 3


def test_predictions_explicit_run(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """An explicit run id resolves to that run."""
    # Arrange
    _seed(project_root, app)

    # Act
    explicit = viewer_client.get("/api/predictions", params={"run": "weekly_w0"}).json()

    # Assert
    assert explicit["run_id"] == "weekly_w0"


@pytest.mark.parametrize("params", [{"week": 9}, {"run": "nope"}])
def test_predictions_unknown_selection_is_404(
    project_root: Path, viewer_client: TestClient, app: FastAPI, params: dict[str, str | int]
) -> None:
    """An unknown week or run is a 404."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = viewer_client.get("/api/predictions", params=params)

    # Assert
    assert response.status_code == 404


def test_predictions_weeks_lists_every_source(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The weeks route lists every prediction source."""
    # Arrange
    _seed(project_root, app)

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    assert len(weeks) == 3


def test_predictions_picks_for_active_run(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The picks route ranks the active run's games."""
    # Arrange
    _seed(project_root, app)

    # Act
    picks = viewer_client.get("/api/predictions/picks").json()

    # Assert
    assert picks["table"]["rows"][0]["confidence_rank"] == 4


def test_predictions_picks_derived_from_unattached_file(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Picks are derived for a week that only has a prediction file."""
    # Arrange
    _seed(project_root, app)

    # Act
    derived = viewer_client.get("/api/predictions/picks", params={"week": 3}).json()

    # Assert
    assert derived["run_id"] is None
    assert len(derived["table"]["rows"]) == 4


def test_predictions_without_any_source(viewer_client: TestClient) -> None:
    """With nothing on disk the predictions route is a 404 with a helpful code."""
    # Act
    response = viewer_client.get("/api/predictions")

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_training_run_without_predictions(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """An explicit training run has no predictions file."""
    # Arrange
    factories.make_run_dir(project_root / "models", "train", factories.RunSpec(kind="training"))
    app.state.run_index.invalidate()

    # Act
    response = viewer_client.get("/api/predictions", params={"run": "train"})

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_betting_routes(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """The betting table and ladder work for the active run."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/betting").json()

    # Assert
    assert payload["run_id"] == "weekly_w1"
    assert "xlsx_available" not in payload
    assert [step["action"] for step in payload["ladder"]] == [
        "PASS",
        "LEAN",
        "SMALL",
        "MEDIUM",
        "STRONG",
    ]
    assert payload["table"]["column_metadata"]["total_action"]["actionable"] is False


def test_betting_workbook_route_is_not_registered(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The betting workbook route is gone from the app."""
    # Arrange
    _seed(project_root, app)

    # Act
    paths = {getattr(route, "path", None) for route in app.routes}

    # Assert
    assert "/api/betting/xlsx" not in paths


def test_betting_workbook_request_is_404(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Requesting the betting workbook is a 404."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = viewer_client.get("/api/betting/xlsx")

    # Assert
    assert response.status_code == 404


def test_betting_unattached_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The betting table works for a week with only a prediction file."""
    # Arrange
    _seed(project_root, app)

    # Act
    unattached = viewer_client.get("/api/betting", params={"week": 3}).json()

    # Assert
    assert unattached["run_id"] is None


def test_power_routes(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """Rankings for the active run carry standings tabs and no predecessor for week 0."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/power").json()

    # Assert
    assert payload["run_id"] == "weekly_w1"
    assert payload["through_week"] == 0
    assert payload["previous_run_id"] is None  # week 0 has no week -1 predecessor
    rows = {row["team_abbr"]: row for row in payload["rankings"]["rows"]}
    assert rows["KC"]["rank"] == 1
    assert payload["standings"] is not None
    assert payload["division_standings"] is not None


def test_power_rankings_carry_movement_against_prior_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Rankings carry movement against the prior-week run."""
    # Arrange
    _seed(project_root, app)
    _add_weekly_w2(project_root, app)

    # Act
    payload = viewer_client.get("/api/power", params={"run": "weekly_w2"}).json()

    # Assert
    assert payload["previous_run_id"] == "weekly_w1"
    rows = {row["team_abbr"]: row for row in payload["rankings"]["rows"]}
    assert rows["DEN"]["rank_change"] == 1
    assert rows["KC"]["rank_change"] == -1


def test_power_for_training_run_is_404(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A training run has no power rankings."""
    # Arrange
    _seed(project_root, app)
    _add_weekly_w2(project_root, app)
    factories.make_run_dir(project_root / "models", "train", factories.RunSpec(kind="training"))
    app.state.run_index.invalidate()

    # Act
    response = viewer_client.get("/api/power", params={"run": "train"})

    # Assert
    assert response.status_code == 404


def test_model_routes(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """The model payload is available for the active run."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/model").json()

    # Assert
    assert payload["run_id"] == "weekly_w1"
    assert payload["kind"] == "weekly"
    assert payload["metrics"]["holdout"]["brier"] == 0.2192
    assert payload["wf_compare"]["rows"][0]["label"] == "base"
    assert payload["feature_importance"]["measure"] == "mean_abs_shap"
    assert payload["feature_importance"]["rows"][0]["feature"] == "away_rest"


def test_model_route_by_run_id(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """The model payload is available by run id."""
    # Arrange
    _seed(project_root, app)

    # Act
    by_id = viewer_client.get("/api/runs/weekly_w0/model").json()

    # Assert
    assert by_id["run_id"] == "weekly_w0"


def test_model_route_unknown_run_is_404(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """An unknown run has no model payload."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = viewer_client.get("/api/runs/nope/model")

    # Assert
    assert response.status_code == 404


def test_data_routes(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """Data status reports files, cache coverage, and unattached outputs."""
    # Arrange
    _seed(project_root, app)
    (project_root / "data" / "completed_games_ml.csv").write_text(
        "season,week\n2025,1\n", encoding="utf-8"
    )

    # Act
    payload = viewer_client.get("/api/data/status").json()

    # Assert
    assert payload["current_season"] >= 2026
    names = {f["name"]: f for f in payload["files"]}
    assert names["completed_games_ml.csv"]["exists"] is True
    assert names["completed_games_ml.csv"]["rows"] == 1
    assert names["all_data_ml.csv"]["exists"] is False
    assert names["ratings/"]["file_count"] is None
    assert payload["cache"] == {"schedule": [], "pbp": []}
    assert payload["cache_files"] == []
    assert payload["other_seasons"] == {"team_rankings": [], "etl_seasons": []}
    assert payload["leakage_audit"] is None
    assert [f["name"] for f in payload["predict_files"]] == ["week_03_predictions.csv"]


def test_data_status_reports_the_caches_on_disk(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Data status reads the nflreadpy and nfelo caches under the settings' data directory."""
    # Arrange
    _seed(project_root, app)
    cache_root = project_root / "data" / "cache"
    (cache_root / "nflreadpy").mkdir(parents=True)
    (cache_root / "nflreadpy" / "snap_counts_2025_reg.parquet").write_bytes(b"x")
    (cache_root / "nflreadpy" / "players.parquet").write_bytes(b"x")
    (cache_root / "nfelo").mkdir()
    (cache_root / "nfelo" / "lines.csv").write_bytes(b"x")

    # Act
    payload = viewer_client.get("/api/data/status").json()

    # Assert
    assert payload["cache"]["snap_counts"] == [2025]
    assert [(f["cache"], f["name"]) for f in payload["cache_files"]] == [
        ("nflreadpy", "players.parquet"),
        ("nfelo", "lines.csv"),
    ]


def test_data_unattached_lists_prediction_files(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The unattached route lists prediction files that belong to no run."""
    # Arrange
    _seed(project_root, app)
    (project_root / "data" / "completed_games_ml.csv").write_text(
        "season,week\n2025,1\n", encoding="utf-8"
    )

    # Act
    unattached = viewer_client.get("/api/data/unattached").json()

    # Assert
    assert len(unattached) == 1


def test_weeks_offer_unpredicted_weeks_of_the_current_season(
    project_root: Path,
    viewer_client: TestClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unplayed weeks with no predictions are listed so the UI can offer to generate them."""
    # Arrange
    _seed(project_root, app)
    monkeypatch.setattr(resolve, "current_season_week", lambda: (2026, 2))
    pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 2, 4],
            "away_score": [17, None, None],
            "home_score": [24, None, None],
        }
    ).write_csv(project_root / "data" / "all_data_ml.csv")

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    available = [week for week in weeks if week["source"] == "available"]
    assert [week["week"] for week in available] == [2, 4]
    assert available[0]["label"] == "Week 2 (not predicted yet)"
    assert all(week["run_id"] is None for week in available)


def test_weeks_offer_nothing_when_the_dataset_is_missing(
    project_root: Path,
    viewer_client: TestClient,
    app: FastAPI,
) -> None:
    """Without an ML dataset there is nothing to generate from."""
    # Arrange
    _seed(project_root, app)

    # Act
    weeks = viewer_client.get("/api/predictions/weeks").json()

    # Assert
    assert [week for week in weeks if week["source"] == "available"] == []
