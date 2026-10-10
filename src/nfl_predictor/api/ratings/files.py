"""The ratings data contract: where each season's tables live, and reading them.

Every table is one season of one kind, ``{season}_{kind}.parquet`` in the ratings directory
(``Settings.ratings_path``). Reads go through the API's file cache, so a table is parsed once until
its file changes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.api.readers.cache import cached

if TYPE_CHECKING:
    from pathlib import Path

SEASON_FILE = re.compile(r"^(?P<season>\d{4})_(?P<kind>[a-z0-9_]+)\.parquet$")
INDEX_KINDS = ("combined", "qb_combined")
"""The team and quarterback season tables; a season is listed once both exist."""
_ENTITY_ORDER = ("week", "game_id")


class RatingsTableMissingError(LookupError):
    """A season has no table of the requested kind."""

    def __init__(self, season: int, kind: str) -> None:
        """Name the season and kind, and the file that would hold them."""
        self.season = season
        self.kind = kind
        super().__init__(f"Season {season} has no {kind} table ({table_name(season, kind)})")


class RatingsTableUnreadableError(RuntimeError):
    """A season's table exists but cannot be read: empty, cut short, or not Parquet."""

    def __init__(self, season: int, kind: str, reason: str) -> None:
        """Name the season, kind and file, and say why the read failed."""
        self.season = season
        self.kind = kind
        super().__init__(
            f"Season {season}'s {kind} table ({table_name(season, kind)}) cannot be read: "
            f"{reason}. It may be cut short or still being written."
        )


class EntityNotFoundError(LookupError):
    """A table has no rows for the requested team or quarterback."""


def table_name(season: int, kind: str) -> str:
    """Return the file name of one season's table of ``kind``."""
    return f"{season}_{kind}.parquet"


def table_path(ratings_dir: Path, season: int, kind: str) -> Path:
    """Return the path of one season's table of ``kind`` in ``ratings_dir``."""
    return ratings_dir / table_name(season, kind)


def available_seasons(ratings_dir: Path) -> list[int]:
    """Return the seasons with both index tables (``INDEX_KINDS``), newest first."""
    if not ratings_dir.is_dir():
        return []
    kinds_by_season: dict[int, set[str]] = {}
    for path in ratings_dir.glob("*.parquet"):
        match = SEASON_FILE.match(path.name)
        if match is not None:
            kinds_by_season.setdefault(int(match["season"]), set()).add(match["kind"])
    return sorted(
        (season for season, kinds in kinds_by_season.items() if set(INDEX_KINDS) <= kinds),
        reverse=True,
    )


def read_table(ratings_dir: Path, season: int, kind: str) -> pl.DataFrame:
    """Return one season's table of ``kind``, cached until the file changes.

    Raises:
        RatingsTableMissingError: If the file does not exist, including one removed between the
            check and the read (the folder being rewritten).
        RatingsTableUnreadableError: If Polars cannot read the file (empty or cut short).

    """
    path = table_path(ratings_dir, season, kind)
    if not path.is_file():
        raise RatingsTableMissingError(season, kind)
    try:
        frame: pl.DataFrame = cached(path, pl.read_parquet)
    except FileNotFoundError as exc:
        raise RatingsTableMissingError(season, kind) from exc
    except pl.exceptions.PolarsError as exc:
        raise RatingsTableUnreadableError(season, kind, str(exc)) from exc
    return frame


def entity_rows(frame: pl.DataFrame, column: str, entity_id: str) -> pl.DataFrame:
    """Return the rows of one team or quarterback, in week (then game) order.

    Raises:
        EntityNotFoundError: If ``frame`` has no ``column`` or no row for ``entity_id``.

    """
    if column not in frame.columns:
        msg = f"The table has no {column} column to find {entity_id} in"
        raise EntityNotFoundError(msg)
    rows = frame.filter(pl.col(column) == entity_id)
    if rows.is_empty():
        msg = f"No rows for {entity_id}"
        raise EntityNotFoundError(msg)
    order = [key for key in _ENTITY_ORDER if key in rows.columns]
    return rows.sort(order) if order else rows
