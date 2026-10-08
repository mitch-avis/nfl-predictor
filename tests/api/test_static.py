"""Tests for the SPA mount and fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import create_app
from nfl_predictor.api.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/predictions?week=3", "<html>spa</html>"),
        ("/", "<html>spa</html>"),
        ("/favicon.svg", "<svg/>"),
        ("/assets/app.js", "console.log(1)"),
        ("/../outside.txt", "<html>spa</html>"),
    ],
)
def test_spa_fallback_and_files(
    path: str, expected: str, project_root: Path, settings: Settings
) -> None:
    """Deep links get ``index.html``; real files and assets are served; traversal is blocked."""
    # Arrange
    dist = project_root / "web" / "dist"
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (project_root / "outside.txt").write_text("secret", encoding="utf-8")

    # Act
    with TestClient(create_app(settings)) as client:
        response = client.get(path)

    # Assert
    assert response.text == expected


def test_missing_build_returns_503(tmp_path: Path) -> None:
    """Without a built frontend the fallback explains how to build it."""
    # Arrange
    settings = Settings(root_dir=tmp_path, jwt_secret="s" * 32)

    # Act
    with TestClient(create_app(settings)) as client:
        response = client.get("/anything")

    # Assert
    assert response.status_code == 503
    assert "pnpm run build" in response.text


def test_frontend_can_be_disabled(settings: Settings) -> None:
    """``serve_frontend=False`` leaves non-API paths unrouted."""
    # Act
    with TestClient(create_app(settings, serve_frontend=False)) as client:
        response = client.get("/anything")

    # Assert
    assert response.status_code == 404


@pytest.mark.parametrize(
    "path", ["/api/no-such-route", "/api/runs/x/no-such-child", "/api", "/api/"]
)
def test_unknown_api_paths_get_a_json_404_not_the_frontend(path: str, settings: Settings) -> None:
    """The fallback serves only paths outside ``/api``; an unknown API route is a JSON 404."""
    # Act
    with TestClient(create_app(settings)) as client:
        response = client.get(path)

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "not_found"


def test_a_path_that_only_starts_like_api_gets_the_frontend(settings: Settings) -> None:
    """A path such as ``/apiary`` is not under ``/api``, so the fallback serves the app."""
    # Act
    with TestClient(create_app(settings)) as client:
        response = client.get("/apiary")

    # Assert
    assert response.text == "<html>spa</html>"
