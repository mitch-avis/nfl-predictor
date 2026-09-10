"""Tests for the user store."""

from __future__ import annotations

import pytest

from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.auth.passwords import hash_password, verify_password
from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import BadRequestError, ConflictError, NotFoundError


def test_password_hash_roundtrip() -> None:
    """Hashes verify their password and reject others."""
    digest = hash_password("correct horse")
    assert verify_password(digest, "correct horse")
    assert not verify_password(digest, "wrong")
    assert not verify_password("not-a-hash", "wrong")


def test_create_and_authenticate(db: Database) -> None:
    """Created users authenticate with their password only."""
    user = user_store.create_user(db, " mitch ", "password123", "admin")
    assert user.username == "mitch"
    assert user.is_admin
    assert user_store.authenticate(db, "mitch", "password123") == user
    assert user_store.authenticate(db, "mitch", "nope") is None
    assert user_store.authenticate(db, "ghost", "password123") is None
    assert user_store.list_users(db) == [user]
    assert user_store.get_user(db, user.id) == user


def test_validation_and_conflicts(db: Database) -> None:
    """Bad usernames, roles, short passwords, and duplicates are rejected."""
    with pytest.raises(BadRequestError):
        user_store.create_user(db, "  ", "password123", "viewer")
    with pytest.raises(BadRequestError):
        user_store.create_user(db, "x", "password123", "root")
    with pytest.raises(BadRequestError):
        user_store.create_user(db, "x", "short", "viewer")
    user_store.create_user(db, "dup", "password123", "viewer")
    with pytest.raises(ConflictError):
        user_store.create_user(db, "dup", "password123", "viewer")
    with pytest.raises(NotFoundError):
        user_store.get_user(db, 999)


def test_update_and_last_admin_guard(db: Database) -> None:
    """Roles and passwords change, but the last admin cannot be demoted."""
    admin = user_store.create_user(db, "admin", "password123", "admin")
    viewer = user_store.create_user(db, "viewer", "password123", "viewer")
    with pytest.raises(ConflictError):
        user_store.update_user(db, admin.id, role="viewer")
    promoted = user_store.update_user(db, viewer.id, role="admin", password="newpassword")
    assert promoted.is_admin
    assert user_store.authenticate(db, "viewer", "newpassword") is not None
    assert user_store.count_admins(db) == 2
    demoted = user_store.update_user(db, admin.id, role="viewer")
    assert not demoted.is_admin


def test_delete_guards(db: Database) -> None:
    """Deleting self or the last admin is refused."""
    admin = user_store.create_user(db, "admin", "password123", "admin")
    viewer = user_store.create_user(db, "viewer", "password123", "viewer")
    with pytest.raises(ConflictError):
        user_store.delete_user(db, admin.id, acting_user_id=admin.id)
    with pytest.raises(ConflictError):
        user_store.delete_user(db, admin.id, acting_user_id=viewer.id)
    user_store.delete_user(db, viewer.id, acting_user_id=admin.id)
    assert [u.username for u in user_store.list_users(db)] == ["admin"]
