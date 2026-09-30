"""Admin-only user management routes."""

from __future__ import annotations

from fastapi import APIRouter

from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.deps import AdminUser, DbDep
from nfl_predictor.api.schemas.auth import UserCreateIn, UserOut, UserUpdateIn

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
def list_users(_admin: AdminUser, db: DbDep) -> list[UserOut]:
    """List every user."""
    return [UserOut(**user.__dict__) for user in user_store.list_users(db)]


@router.post("", response_model=UserOut, status_code=201)
def create_user(payload: UserCreateIn, _admin: AdminUser, db: DbDep) -> UserOut:
    """Create a user."""
    user = user_store.create_user(db, payload.username, payload.password, payload.role)
    return UserOut(**user.__dict__)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UserUpdateIn, _admin: AdminUser, db: DbDep) -> UserOut:
    """Change a user's role or password."""
    user = user_store.update_user(db, user_id, role=payload.role, password=payload.password)
    return UserOut(**user.__dict__)


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, admin: AdminUser, db: DbDep) -> None:
    """Delete a user."""
    user_store.delete_user(db, user_id, acting_user_id=admin.id)
