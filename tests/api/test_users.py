"""Tests for the user store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.auth.passwords import hash_password, verify_password
from nfl_predictor.api.errors import BadRequestError, ConflictError, NotFoundError

if TYPE_CHECKING:
    from nfl_predictor.api.db import Database


def test_password_hash_roundtrip() -> None:
    """Hashes verify their password and reject others."""
    # Act
    digest = hash_password("correct horse")

    # Assert
    assert verify_password(digest, "correct horse")
    assert not verify_password(digest, "wrong")


def test_verify_password_rejects_malformed_hash() -> None:
    """A value that is not a hash never verifies."""
    # Act
    verified = verify_password("not-a-hash", "wrong")

    # Assert
    assert not verified


def test_create_and_authenticate(db: Database) -> None:
    """Created users authenticate with their password only."""
    # Act
    user = user_store.create_user(db, " mitch ", "password123", "admin")

    # Assert
    assert user.username == "mitch"
    assert user.is_admin
    assert user_store.authenticate(db, "mitch", "password123") == user
    assert user_store.authenticate(db, "mitch", "nope") is None
    assert user_store.authenticate(db, "ghost", "password123") is None
    assert user_store.list_users(db) == [user]
    assert user_store.get_user(db, user.id) == user


@pytest.mark.parametrize(
    ("username", "password", "role"),
    [
        ("  ", "password123", "viewer"),
        ("x", "password123", "root"),
        ("x", "short", "viewer"),
    ],
)
def test_validation_and_conflicts(db: Database, username: str, password: str, role: str) -> None:
    """Bad usernames, roles and short passwords are rejected."""
    # Act & Assert
    with pytest.raises(BadRequestError):
        user_store.create_user(db, username, password, role)


def test_create_duplicate_user_conflicts(db: Database) -> None:
    """A duplicate username is rejected."""
    # Arrange
    user_store.create_user(db, "dup", "password123", "viewer")

    # Act & Assert
    with pytest.raises(ConflictError):
        user_store.create_user(db, "dup", "password123", "viewer")


def test_get_unknown_user_not_found(db: Database) -> None:
    """Looking up an unknown user id raises not found."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        user_store.get_user(db, 999)


def test_update_and_last_admin_guard(db: Database) -> None:
    """The last admin cannot be demoted."""
    # Arrange
    admin = user_store.create_user(db, "admin", "password123", "admin")
    user_store.create_user(db, "viewer", "password123", "viewer")

    # Act & Assert
    with pytest.raises(ConflictError):
        user_store.update_user(db, admin.id, role="viewer")


def test_update_user_changes_role_and_password(db: Database) -> None:
    """Roles and passwords change on update."""
    # Arrange
    user_store.create_user(db, "admin", "password123", "admin")
    viewer = user_store.create_user(db, "viewer", "password123", "viewer")

    # Act
    promoted = user_store.update_user(db, viewer.id, role="admin", password="newpassword")

    # Assert
    assert promoted.is_admin
    assert user_store.authenticate(db, "viewer", "newpassword") is not None
    assert user_store.count_admins(db) == 2


def test_update_user_demotes_admin_when_another_admin_exists(db: Database) -> None:
    """An admin can be demoted once a second admin exists."""
    # Arrange
    admin = user_store.create_user(db, "admin", "password123", "admin")
    viewer = user_store.create_user(db, "viewer", "password123", "viewer")
    user_store.update_user(db, viewer.id, role="admin", password="newpassword")

    # Act
    demoted = user_store.update_user(db, admin.id, role="viewer")

    # Assert
    assert not demoted.is_admin


@pytest.mark.parametrize("acting_as", ["admin", "viewer"])
def test_delete_guards(db: Database, acting_as: str) -> None:
    """Deleting self or the last admin is refused."""
    # Arrange
    users = {
        "admin": user_store.create_user(db, "admin", "password123", "admin"),
        "viewer": user_store.create_user(db, "viewer", "password123", "viewer"),
    }

    # Act & Assert
    with pytest.raises(ConflictError):
        user_store.delete_user(db, users["admin"].id, acting_user_id=users[acting_as].id)


def test_delete_user_removes_viewer(db: Database) -> None:
    """An admin can delete another, non-admin user."""
    # Arrange
    admin = user_store.create_user(db, "admin", "password123", "admin")
    viewer = user_store.create_user(db, "viewer", "password123", "viewer")

    # Act
    user_store.delete_user(db, viewer.id, acting_user_id=admin.id)

    # Assert
    assert [u.username for u in user_store.list_users(db)] == ["admin"]
