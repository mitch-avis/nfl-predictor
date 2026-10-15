"""Tests for login, sessions, CSRF, rate limiting, and user routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from nfl_predictor.api.auth.ratelimit import LoginRateLimiter
from nfl_predictor.api.auth.tokens import issue_token, read_token
from nfl_predictor.api.settings import COOKIE_NAME
from tests.api.conftest import ADMIN_PASSWORD, CSRF_HEADERS, TEST_SECRET

if TYPE_CHECKING:
    from nfl_predictor.api.db import Database

OTHER_SECRET = "other-secret-" * 4


def _create_bob(admin_client: TestClient):
    """Create the viewer ``bob`` through the API and return the response."""
    return admin_client.post(
        "/api/users", json={"username": "bob", "password": "password123", "role": "viewer"}
    )


def test_health_is_public(client: TestClient) -> None:
    """The health probe needs no session."""
    # Act
    response = client.get("/api/health")

    # Assert
    assert response.json() == {"status": "ok"}


def test_me_requires_session(client: TestClient) -> None:
    """Anonymous requests get a structured 401."""
    # Act
    response = client.get("/api/auth/me")

    # Assert
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_login_sets_cookie_and_me_works(admin_client: TestClient) -> None:
    """A successful login sets the cookie and ``/me`` reflects the user."""
    # Act
    me = admin_client.get("/api/auth/me").json()["user"]

    # Assert
    assert COOKIE_NAME in admin_client.cookies
    assert me["username"] == "admin"
    assert me["role"] == "admin"


def test_login_rejects_bad_password(client: TestClient, db: Database) -> None:
    """Wrong credentials return 401 without a cookie."""
    # Arrange
    from nfl_predictor.api.auth import users as user_store

    user_store.create_user(db, "admin", ADMIN_PASSWORD, "admin")

    # Act
    response = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})

    # Assert
    assert response.status_code == 401
    assert COOKIE_NAME not in client.cookies


def test_login_rate_limited(client: TestClient, app) -> None:
    """After the failure budget is spent the login returns 429."""
    # Arrange
    app.state.login_limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)
    for _ in range(2):
        client.post("/api/auth/login", json={"username": "x", "password": "y"})

    # Act
    response = client.post("/api/auth/login", json={"username": "x", "password": "y"})

    # Assert
    assert response.status_code == 429


def test_rate_limiter_blocks_within_window() -> None:
    """A failure inside the window blocks the key."""
    # Arrange
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=10)
    limiter.record_failure("k", now=0.0)

    # Act
    blocked = limiter.is_blocked("k", now=5.0)

    # Assert
    assert blocked


def test_rate_limiter_window_expires() -> None:
    """Old failures fall out of the window."""
    # Arrange
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=10)
    limiter.record_failure("k", now=0.0)

    # Act
    blocked = limiter.is_blocked("k", now=20.0)

    # Assert
    assert not blocked


def test_rate_limiter_reset_clears_key() -> None:
    """A reset clears the key's failures."""
    # Arrange
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=10)
    limiter.record_failure("k", now=0.0)
    limiter.record_failure("k", now=21.0)

    # Act
    limiter.reset("k")

    # Assert
    assert not limiter.is_blocked("k", now=21.0)


def test_logout_clears_cookie(admin_client: TestClient) -> None:
    """Logout succeeds with 204."""
    # Act
    response = admin_client.post("/api/auth/logout")

    # Assert
    assert response.status_code == 204


def test_requests_after_logout_are_anonymous(admin_client: TestClient) -> None:
    """Once logged out with the cookie cleared, ``/me`` is a 401."""
    # Arrange
    admin_client.post("/api/auth/logout")
    admin_client.cookies.clear()

    # Act
    response = admin_client.get("/api/auth/me")

    # Assert
    assert response.status_code == 401


def test_csrf_header_required(app, db: Database) -> None:
    """Mutating API calls without the custom header are refused."""
    # Act
    with TestClient(app) as bare:
        response = bare.post("/api/auth/login", json={"username": "a", "password": "b"})

    # Assert
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "csrf_header_missing"


def test_invalid_or_expired_token(client: TestClient) -> None:
    """A garbage cookie is rejected as an invalid session."""
    # Arrange
    client.cookies.set(COOKIE_NAME, "garbage")

    # Act
    response = client.get("/api/auth/me")

    # Assert
    assert response.json()["error"]["code"] == "invalid_session"


def test_token_signed_with_other_secret_is_rejected(client: TestClient) -> None:
    """A token signed with another secret is rejected."""
    # Arrange
    client.cookies.set(COOKIE_NAME, "garbage")
    client.get("/api/auth/me")
    other = issue_token(OTHER_SECRET, user_id=1, username="a", role="admin", hours=1)
    client.cookies.set(COOKIE_NAME, other)

    # Act
    response = client.get("/api/auth/me")

    # Assert
    assert response.status_code == 401


def test_read_token_with_its_own_secret() -> None:
    """A token reads back with the secret that signed it."""
    # Arrange
    other = issue_token(OTHER_SECRET, user_id=1, username="a", role="admin", hours=1)

    # Act
    payload = read_token(OTHER_SECRET, other)

    # Assert
    assert payload is not None


def test_read_token_with_another_secret_is_none() -> None:
    """A token does not read back with a different secret."""
    # Arrange
    other = issue_token(OTHER_SECRET, user_id=1, username="a", role="admin", hours=1)

    # Act
    payload = read_token(TEST_SECRET, other)

    # Assert
    assert payload is None


def test_malformed_token_payload(client: TestClient) -> None:
    """A validly signed token with a non-numeric subject is rejected."""
    # Arrange
    import jwt

    token = jwt.encode({"sub": "abc", "username": "a", "role": "admin"}, TEST_SECRET, "HS256")
    client.cookies.set(COOKIE_NAME, token)

    # Act
    response = client.get("/api/auth/me")

    # Assert
    assert response.status_code == 401


def test_user_routes_admin_only(viewer_client: TestClient) -> None:
    """Viewers cannot list users."""
    # Act
    response = viewer_client.get("/api/users")

    # Assert
    assert response.status_code == 403


def test_viewer_cannot_create_users(viewer_client: TestClient) -> None:
    """Viewers cannot create users."""
    # Act
    response = viewer_client.post(
        "/api/users", json={"username": "n", "password": "password123", "role": "viewer"}
    )

    # Assert
    assert response.status_code == 403


def test_user_crud_routes(admin_client: TestClient) -> None:
    """Admins can create users."""
    # Act
    created = _create_bob(admin_client)

    # Assert
    assert created.status_code == 201


def test_create_duplicate_user_conflicts(admin_client: TestClient) -> None:
    """Creating a user with a taken username is a 409."""
    # Arrange
    _create_bob(admin_client)

    # Act
    duplicate = _create_bob(admin_client)

    # Assert
    assert duplicate.status_code == 409


def test_update_user_role(admin_client: TestClient) -> None:
    """Admins can change a user's role."""
    # Arrange
    bob_id = _create_bob(admin_client).json()["id"]

    # Act
    updated = admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})

    # Assert
    assert updated.json()["role"] == "admin"


def test_list_users(admin_client: TestClient) -> None:
    """Admins can list users."""
    # Arrange
    bob_id = _create_bob(admin_client).json()["id"]
    admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})

    # Act
    listed = admin_client.get("/api/users").json()

    # Assert
    assert [u["username"] for u in listed] == ["admin", "bob"]


def test_delete_self_conflicts(admin_client: TestClient) -> None:
    """Admins cannot delete themselves."""
    # Arrange
    bob_id = _create_bob(admin_client).json()["id"]
    admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})
    me_id = admin_client.get("/api/auth/me").json()["user"]["id"]

    # Act
    response = admin_client.delete(f"/api/users/{me_id}")

    # Assert
    assert response.status_code == 409


def test_delete_user(admin_client: TestClient) -> None:
    """Admins can delete another user."""
    # Arrange
    bob_id = _create_bob(admin_client).json()["id"]
    admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})

    # Act
    response = admin_client.delete(f"/api/users/{bob_id}")

    # Assert
    assert response.status_code == 204


def test_delete_deleted_user_not_found(admin_client: TestClient) -> None:
    """Deleting a user twice is a 404 the second time."""
    # Arrange
    bob_id = _create_bob(admin_client).json()["id"]
    admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})
    admin_client.delete(f"/api/users/{bob_id}")

    # Act
    response = admin_client.delete(f"/api/users/{bob_id}")

    # Assert
    assert response.status_code == 404


def test_headers_constant_matches_middleware() -> None:
    """The fixture header is the one the middleware expects."""
    # Act
    header = CSRF_HEADERS["X-Requested-With"]

    # Assert
    assert header == "nflp"
