"""Shared fixtures: an isolated project root, an app, and logged-in clients."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import create_app
from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.db import Database
from nfl_predictor.api.settings import CSRF_HEADER_VALUE, Settings

TEST_SECRET = "test-secret-" * 4
CSRF_HEADERS = {"X-Requested-With": CSRF_HEADER_VALUE}
ADMIN_PASSWORD = "admin-pass-123"
VIEWER_PASSWORD = "viewer-pass-123"


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return an empty fake repository root with the expected top-level directories."""
    for name in ("data", "data/predict", "models", "reports", "web/dist"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "web" / "dist" / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> Settings:
    """Return settings rooted at the fake repository."""
    return Settings(
        root_dir=project_root,
        jwt_secret=TEST_SECRET,
        python_executable=Path(sys.executable),
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    """Return the database for ``settings``."""
    return Database(settings.database_path)


@pytest.fixture
def app(settings: Settings):  # noqa: ANN201 - FastAPI has no stable public type alias here
    """Return the application for ``settings``."""
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:  # noqa: ANN001
    """Return an anonymous client that sends the CSRF header."""
    with TestClient(app, base_url="http://testserver") as test_client:
        test_client.headers.update(CSRF_HEADERS)
        yield test_client


def login(test_client: TestClient, username: str, password: str) -> None:
    """Log ``test_client`` in and keep the session cookie."""
    response = test_client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def admin_client(app, db: Database) -> Iterator[TestClient]:  # noqa: ANN001
    """Return a client logged in as an admin user named ``admin``."""
    user_store.create_user(db, "admin", ADMIN_PASSWORD, "admin")
    with TestClient(app, base_url="http://testserver") as test_client:
        test_client.headers.update(CSRF_HEADERS)
        login(test_client, "admin", ADMIN_PASSWORD)
        yield test_client


@pytest.fixture
def viewer_client(app, db: Database) -> Iterator[TestClient]:  # noqa: ANN001
    """Return a client logged in as a viewer named ``viewer``."""
    user_store.create_user(db, "viewer", VIEWER_PASSWORD, "viewer")
    with TestClient(app, base_url="http://testserver") as test_client:
        test_client.headers.update(CSRF_HEADERS)
        login(test_client, "viewer", VIEWER_PASSWORD)
        yield test_client
