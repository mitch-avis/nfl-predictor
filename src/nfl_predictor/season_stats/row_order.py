"""The fixed row order of every season table, as ``nfl-sos-ratings`` writes its data files.

Every table is returned in one fixed row order, so two builds from the same inputs give
identical files and a rebuild's diff shows only real changes. Files read top-down keep their
published order (best first, ties broken by id); every other file is in key order: its first
identity column present, then week and game, then unit and win-probability bin for files that split
a game's plays into bins, then the compared team or passer for head-to-head pair files, and the
left-out team for the preseason prior's file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

PUBLISHED_ROW_ORDER: dict[str, tuple[tuple[str, bool], ...]] = {
    "ratings": (("team_rating", True), ("team", False)),
    "qb_ratings": (("adj_qb_epa_per_dropback", True), ("qb_id", False)),
    "rating_ranges": (("team_rank", False), ("team", False)),
    "qb_rating_ranges": (("qb_rank", False), ("qb_id", False)),
    "team_prior": (("excluded_team", False), ("team", False)),
}
ROW_IDENTITY_KEYS = ("qb_id", "team")
ROW_EVENT_KEYS = ("week", "game_id")
ROW_DETAIL_KEYS = ("wp_unit", "wp_bin")
ROW_PAIR_KEYS = ("other_qb_id", "other_team", "excluded_team")


def row_identity_keys(columns: Collection[str]) -> tuple[str, ...]:
    """Return the columns that identify a data file's rows, or ``()`` when it has none.

    The identity is the first of ``qb_id`` and ``team`` present, then ``week``, ``game_id``,
    ``wp_unit``, ``wp_bin``, ``other_qb_id``, ``other_team``, and ``excluded_team`` where present.
    """
    identity = next((key for key in ROW_IDENTITY_KEYS if key in columns), None)
    if identity is None:
        return ()
    return (
        identity,
        *(key for key in (*ROW_EVENT_KEYS, *ROW_DETAIL_KEYS, *ROW_PAIR_KEYS) if key in columns),
    )


def data_file_row_order(suffix: str, columns: Collection[str]) -> tuple[tuple[str, bool], ...]:
    """Return the ``(column, descending)`` sort that fixes a data file's row order.

    Args:
        suffix: The file name after the season, for example ``ratings_by_week``.
        columns: The file's columns.

    Raises:
        ValueError: The file has no published order and no identity column to sort by.

    """
    if suffix in PUBLISHED_ROW_ORDER:
        return PUBLISHED_ROW_ORDER[suffix]
    keys = row_identity_keys(columns)
    if not keys:
        msg = (
            f"Output {suffix} has no row order: it needs one of {', '.join(ROW_IDENTITY_KEYS)} "
            "or an entry in PUBLISHED_ROW_ORDER"
        )
        raise ValueError(msg)
    return tuple((key, False) for key in keys)


__all__ = [
    "PUBLISHED_ROW_ORDER",
    "ROW_DETAIL_KEYS",
    "ROW_EVENT_KEYS",
    "ROW_IDENTITY_KEYS",
    "ROW_PAIR_KEYS",
    "data_file_row_order",
    "row_identity_keys",
]
