"""Signed session tokens (HS256 JWT)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

ALGORITHM = "HS256"


def issue_token(secret: str, *, user_id: int, username: str, role: str, hours: int) -> str:
    """Return a signed token carrying the user's id, name, role, and expiry."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=hours)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def read_token(secret: str, token: str) -> dict[str, object] | None:
    """Return the token payload, or ``None`` when it is invalid or expired."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return dict(payload)
