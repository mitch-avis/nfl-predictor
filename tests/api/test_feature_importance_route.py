"""Route test for the Model page's feature importance and its margin-only ranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.api import factories

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient


def test_model_route_returns_the_margin_head_ranking(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The model payload carries the margin head's own ranking beside the combined one."""
    # Arrange
    factories.make_run_dir(project_root / "models", "weekly_w1")
    app.state.run_index.invalidate()

    # Act
    response = viewer_client.get("/api/model")

    # Assert
    assert response.status_code == 200
    importance = response.json()["feature_importance"]
    assert importance["measure"] == "mean_abs_shap"
    assert [row["feature"] for row in importance["margin_rows"]] == [
        "away_rest",
        "away_elo_pre",
        "home_rest",
    ]
    assert importance["margin_rows"][0]["value"] == 2.5
