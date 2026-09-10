"""FastAPI dependencies shared by every router."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from nfl_predictor.api.auth.tokens import read_token
from nfl_predictor.api.auth.users import User
from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import ForbiddenError, UnauthorizedError
from nfl_predictor.api.settings import COOKIE_NAME, Settings


def get_settings(request: Request) -> Settings:
    """Return the settings attached to the application."""
    settings: Settings = request.app.state.settings
    return settings


def get_db(request: Request) -> Database:
    """Return the database attached to the application."""
    db: Database = request.app.state.db
    return db


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Database, Depends(get_db)]


def current_user(request: Request, settings: SettingsDep) -> User:
    """Return the user for the session cookie or raise 401."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise UnauthorizedError()
    payload = read_token(settings.resolve_jwt_secret(), token)
    if payload is None:
        raise UnauthorizedError("Session expired or invalid", code="invalid_session")
    try:
        return User(
            id=int(str(payload["sub"])),
            username=str(payload["username"]),
            role=str(payload["role"]),
            created_at="",
        )
    except (KeyError, ValueError) as exc:
        raise UnauthorizedError("Malformed session", code="invalid_session") from exc


CurrentUser = Annotated[User, Depends(current_user)]


def require_admin(user: CurrentUser) -> User:
    """Return the user when they are an admin, else raise 403."""
    if not user.is_admin:
        raise ForbiddenError()
    return user


AdminUser = Annotated[User, Depends(require_admin)]
