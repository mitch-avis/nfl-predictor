"""Login, logout, and session routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.auth.ratelimit import LoginRateLimiter
from nfl_predictor.api.auth.tokens import issue_token
from nfl_predictor.api.deps import CurrentUser, DbDep, SettingsDep
from nfl_predictor.api.errors import ApiError, UnauthorizedError
from nfl_predictor.api.schemas.auth import LoginIn, SessionOut, UserOut
from nfl_predictor.api.settings import COOKIE_NAME

router = APIRouter(prefix="/auth", tags=["auth"])


def _limiter(request: Request) -> LoginRateLimiter:
    """Return the application's login rate limiter."""
    limiter: LoginRateLimiter = request.app.state.login_limiter
    return limiter


def _client_key(request: Request, username: str) -> str:
    """Return the rate-limit key for a login attempt."""
    host = request.client.host if request.client else "unknown"
    return f"{host}:{username.strip().lower()}"


@router.post("/login", response_model=SessionOut)
def login(
    payload: LoginIn, request: Request, response: Response, db: DbDep, settings: SettingsDep
) -> SessionOut:
    """Verify credentials and set the session cookie."""
    limiter = _limiter(request)
    key = _client_key(request, payload.username)
    if limiter.is_blocked(key):
        raise ApiError(429, "too_many_attempts", "Too many failed logins; try again later")
    user = user_store.authenticate(db, payload.username, payload.password)
    if user is None:
        limiter.record_failure(key)
        raise UnauthorizedError("Invalid username or password", code="invalid_credentials")
    limiter.reset(key)
    token = issue_token(
        settings.resolve_jwt_secret(),
        user_id=user.id,
        username=user.username,
        role=user.role,
        hours=settings.session_hours,
    )
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return SessionOut(user=UserOut(**user.__dict__))


@router.post("/logout", status_code=204)
def logout(response: Response, settings: SettingsDep) -> Response:
    """Clear the session cookie."""
    response.delete_cookie(COOKIE_NAME, path="/", samesite="lax", secure=settings.cookie_secure)
    response.status_code = 204
    return response


@router.get("/me", response_model=SessionOut)
def me(user: CurrentUser, db: DbDep) -> SessionOut:
    """Return the current user from the database (so role changes take effect)."""
    fresh = user_store.get_user(db, user.id)
    return SessionOut(user=UserOut(**fresh.__dict__))
