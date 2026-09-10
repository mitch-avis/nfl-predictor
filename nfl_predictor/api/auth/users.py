"""User records and the operations the API performs on them."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from nfl_predictor.api.auth.passwords import hash_password, verify_password
from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import BadRequestError, ConflictError, NotFoundError

ROLES = ("viewer", "admin")
MIN_PASSWORD_LENGTH = 8


@dataclass(frozen=True)
class User:
    """A stored user without the password hash."""

    id: int
    username: str
    role: str
    created_at: str

    @property
    def is_admin(self) -> bool:
        """Return whether the user holds the admin role."""
        return self.role == "admin"


def _row_to_user(row: sqlite3.Row) -> User:
    """Build a :class:`User` from a ``users`` row."""
    return User(
        id=int(row["id"]),
        username=str(row["username"]),
        role=str(row["role"]),
        created_at=str(row["created_at"]),
    )


def _validate(username: str, role: str, password: str | None) -> None:
    """Raise :class:`BadRequestError` when a username, role, or password is unacceptable."""
    if not username or not username.strip() or len(username) > 64:
        raise BadRequestError("Username must be 1-64 characters", code="invalid_username")
    if role not in ROLES:
        raise BadRequestError(f"Role must be one of {', '.join(ROLES)}", code="invalid_role")
    if password is not None and len(password) < MIN_PASSWORD_LENGTH:
        raise BadRequestError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters", code="weak_password"
        )


def list_users(db: Database) -> list[User]:
    """Return every user ordered by id."""
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [_row_to_user(row) for row in rows]


def get_user(db: Database, user_id: int) -> User:
    """Return the user with ``user_id`` or raise :class:`NotFoundError`."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"User {user_id} not found", code="user_not_found")
    return _row_to_user(row)


def create_user(db: Database, username: str, password: str, role: str) -> User:
    """Create a user; raise :class:`ConflictError` when the username is taken."""
    username = username.strip()
    _validate(username, role, password)
    created_at = datetime.now(UTC).isoformat()
    with db.connect() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), role, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(
                f"Username {username!r} already exists", code="username_taken"
            ) from exc
        user_id = int(cursor.lastrowid or 0)
    return User(id=user_id, username=username, role=role, created_at=created_at)


def authenticate(db: Database, username: str, password: str) -> User | None:
    """Return the user when the credentials match, else ``None``."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None or not verify_password(str(row["password_hash"]), password):
        return None
    return _row_to_user(row)


def count_admins(db: Database) -> int:
    """Return how many admin users exist."""
    with db.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'").fetchone()
    return int(row["n"]) if row is not None else 0


def update_user(
    db: Database,
    user_id: int,
    *,
    role: str | None = None,
    password: str | None = None,
) -> User:
    """Change a user's role and/or password, refusing to demote the last admin."""
    user = get_user(db, user_id)
    new_role = role or user.role
    _validate(user.username, new_role, password)
    if user.is_admin and new_role != "admin" and count_admins(db) <= 1:
        raise ConflictError("Cannot demote the last admin", code="last_admin")
    with db.connect() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
        if password is not None:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user_id),
            )
    return get_user(db, user_id)


def delete_user(db: Database, user_id: int, *, acting_user_id: int) -> None:
    """Delete a user, refusing to delete the caller or the last admin."""
    user = get_user(db, user_id)
    if user.id == acting_user_id:
        raise ConflictError("Cannot delete your own account", code="self_delete")
    if user.is_admin and count_admins(db) <= 1:
        raise ConflictError("Cannot delete the last admin", code="last_admin")
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
