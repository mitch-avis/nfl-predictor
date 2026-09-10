"""Auth and user payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["viewer", "admin"]


class UserOut(BaseModel):
    """A user as returned by the API."""

    id: int
    username: str
    role: Role
    created_at: str


class SessionOut(BaseModel):
    """The current session."""

    user: UserOut


class LoginIn(BaseModel):
    """Login credentials."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserCreateIn(BaseModel):
    """Payload for creating a user."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    role: Role = "viewer"


class UserUpdateIn(BaseModel):
    """Payload for changing a user's role or password."""

    role: Role | None = None
    password: str | None = Field(default=None, min_length=1, max_length=256)
