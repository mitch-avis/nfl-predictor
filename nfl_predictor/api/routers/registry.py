"""Expose the column registry."""

from __future__ import annotations

from fastapi import APIRouter

from nfl_predictor.api.deps import CurrentUser
from nfl_predictor.api.registry import REGISTRY, group_map
from nfl_predictor.api.schemas.common import RegistryOut

router = APIRouter(prefix="/registry", tags=["registry"])


@router.get("", response_model=RegistryOut)
def registry(_user: CurrentUser) -> RegistryOut:
    """Return every registered column with its display metadata."""
    return RegistryOut(
        columns={key: meta.to_out() for key, meta in REGISTRY.items()},
        groups=group_map(REGISTRY.keys()),
    )
