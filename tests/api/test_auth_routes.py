"""Tests for login, sessions, CSRF, rate limiting, and user routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_predictor.api.auth.ratelimit import LoginRateLimiter
from nfl_predictor.api.auth.tokens import issue_token, read_token
from nfl_predictor.api.db import Database
from nfl_predictor.api.settings import COOKIE_NAME
from tests.api.conftest import ADMIN_PASSWORD, CSRF_HEADERS, TEST_SECRET

OTHER_SECRET = "other-secret-" * 4


def test_health_is_public(client: TestClient) -> None:
    """The health probe needs no session."""
    assert client.get("/api/health").json() == {"status": "ok"}


def test_me_requires_session(client: TestClient) -> None:
    """Anonymous requests get a structured 401."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_login_sets_cookie_and_me_works(admin_client: TestClient) -> None:
    """A successful login sets the cookie and ``/me`` reflects the user."""
    assert COOKIE_NAME in admin_client.cookies
    me = admin_client.get("/api/auth/me").json()["user"]
    assert me["username"] == "admin"
    assert me["role"] == "admin"


def test_login_rejects_bad_password(client: TestClient, db: Database) -> None:
    """Wrong credentials return 401 without a cookie."""
    from nfl_predictor.api.auth import users as user_store

    user_store.create_user(db, "admin", ADMIN_PASSWORD, "admin")
    response = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401
    assert COOKIE_NAME not in client.cookies


def test_login_rate_limited(client: TestClient, app) -> None:  # noqa: ANN001
    """After the failure budget is spent the login returns 429."""
    app.state.login_limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)
    for _ in range(2):
        client.post("/api/auth/login", json={"username": "x", "password": "y"})
    response = client.post("/api/auth/login", json={"username": "x", "password": "y"})
    assert response.status_code == 429


def test_rate_limiter_window_expires() -> None:
    """Old failures fall out of the window and a reset clears the key."""
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=10)
    limiter.record_failure("k", now=0.0)
    assert limiter.is_blocked("k", now=5.0)
    assert not limiter.is_blocked("k", now=20.0)
    limiter.record_failure("k", now=21.0)
    limiter.reset("k")
    assert not limiter.is_blocked("k", now=21.0)


def test_logout_clears_cookie(admin_client: TestClient) -> None:
    """Logout removes the cookie and later requests are anonymous."""
    response = admin_client.post("/api/auth/logout")
    assert response.status_code == 204
    admin_client.cookies.clear()
    assert admin_client.get("/api/auth/me").status_code == 401


def test_csrf_header_required(app, db: Database) -> None:  # noqa: ANN001
    """Mutating API calls without the custom header are refused."""
    with TestClient(app) as bare:
        response = bare.post("/api/auth/login", json={"username": "a", "password": "b"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "csrf_header_missing"


def test_invalid_or_expired_token(client: TestClient) -> None:
    """A garbage cookie or a token signed with another secret is rejected."""
    client.cookies.set(COOKIE_NAME, "garbage")
    assert client.get("/api/auth/me").json()["error"]["code"] == "invalid_session"
    other = issue_token(OTHER_SECRET, user_id=1, username="a", role="admin", hours=1)
    client.cookies.set(COOKIE_NAME, other)
    assert client.get("/api/auth/me").status_code == 401
    assert read_token(OTHER_SECRET, other) is not None
    assert read_token(TEST_SECRET, other) is None


def test_malformed_token_payload(client: TestClient) -> None:
    """A validly signed token with a non-numeric subject is rejected."""
    import jwt

    token = jwt.encode({"sub": "abc", "username": "a", "role": "admin"}, TEST_SECRET, "HS256")
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/api/auth/me").status_code == 401


def test_user_routes_admin_only(viewer_client: TestClient) -> None:
    """Viewers cannot list or create users."""
    assert viewer_client.get("/api/users").status_code == 403
    response = viewer_client.post(
        "/api/users", json={"username": "n", "password": "password123", "role": "viewer"}
    )
    assert response.status_code == 403


def test_user_crud_routes(admin_client: TestClient) -> None:
    """Admins can create, update, list, and delete users with guards enforced."""
    created = admin_client.post(
        "/api/users", json={"username": "bob", "password": "password123", "role": "viewer"}
    )
    assert created.status_code == 201
    bob_id = created.json()["id"]
    duplicate = admin_client.post(
        "/api/users", json={"username": "bob", "password": "password123", "role": "viewer"}
    )
    assert duplicate.status_code == 409
    updated = admin_client.patch(f"/api/users/{bob_id}", json={"role": "admin"})
    assert updated.json()["role"] == "admin"
    listed = admin_client.get("/api/users").json()
    assert [u["username"] for u in listed] == ["admin", "bob"]
    me_id = admin_client.get("/api/auth/me").json()["user"]["id"]
    assert admin_client.delete(f"/api/users/{me_id}").status_code == 409
    assert admin_client.delete(f"/api/users/{bob_id}").status_code == 204
    assert admin_client.delete(f"/api/users/{bob_id}").status_code == 404


def test_headers_constant_matches_middleware() -> None:
    """The fixture header is the one the middleware expects."""
    assert CSRF_HEADERS["X-Requested-With"] == "nflp"
