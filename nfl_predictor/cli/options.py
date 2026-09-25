"""Option handling shared by the command-line entry points."""

from __future__ import annotations


def market_modes(mode: str) -> list[tuple[str, bool, bool]]:
    """Resolve which market modes to evaluate.

    Returns ``(label, include_market, market_anchor)`` for ``features``, ``anchor`` or
    ``hybrid``; any other value (``all``) returns all three.
    """
    if mode == "features":
        return [("features", True, False)]
    if mode == "anchor":
        return [("anchor", False, True)]
    if mode == "hybrid":
        return [("hybrid", True, True)]
    return [("features", True, False), ("anchor", False, True), ("hybrid", True, True)]


def parse_feature_groups(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature group list into a tuple of stripped, non-empty names."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())
