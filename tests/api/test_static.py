"""Tests for the SPA mount and fallback."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nfl_predictor.api import create_app
from nfl_predictor.api.settings import Settings


def test_spa_fallback_and_files(project_root: Path, settings: Settings) -> None:
    """Deep links get ``index.html``; real files and assets are served; traversal is blocked."""
    dist = project_root / "web" / "dist"
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (project_root / "outside.txt").write_text("secret", encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        assert client.get("/predictions?week=3").text == "<html>spa</html>"
        assert client.get("/").text == "<html>spa</html>"
        assert client.get("/favicon.svg").text == "<svg/>"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/../outside.txt").text == "<html>spa</html>"


def test_missing_build_returns_503(tmp_path: Path) -> None:
    """Without a built frontend the fallback explains how to build it."""
    settings = Settings(root_dir=tmp_path, jwt_secret="s" * 32)
    with TestClient(create_app(settings)) as client:
        response = client.get("/anything")
    assert response.status_code == 503
    assert "npm run build" in response.text


def test_frontend_can_be_disabled(settings: Settings) -> None:
    """``serve_frontend=False`` leaves non-API paths unrouted."""
    with TestClient(create_app(settings, serve_frontend=False)) as client:
        assert client.get("/anything").status_code == 404
