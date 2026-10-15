"""ESPN's quarterback rating (QBR) from nflverse, for validation only.

ESPN QBR is a reference to check this repository's quarterback ratings against; it must never
feed the game dataset or the model. ``tests/test_espn_qbr.py`` checks two things: no other module
of the package imports this one anywhere, inside functions included, unless the test's
allow-list names it (none does); and importing every module the command line's commands run, and
the web app, in a fresh interpreter leaves this module unloaded.

nflreadpy has no loader for it, so ``load_espn_qbr`` downloads the season-level or week-level
Parquet asset nflverse publishes, which holds every season from 2006, and caches each season like
the other nflverse sources (``nflverse_sources.SeasonSource``): a historical season reads its
cached file, the current season is downloaded again, and a failure for the current season falls
back to the cache or skips the season. A season the asset holds no row for is such a failure, so
no empty season is ever cached; for a historical season it stops the load. Only regular-season
rows are kept, and the team codes are canonical (ESPN
writes the Raiders, Chargers and Rams under their former cities' codes in earlier seasons). ESPN's
``player_id`` is the players file's ``espn_id``.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal

import polars as pl
import requests

from nfl_predictor import constants
from nfl_predictor.utils.polars.nflverse_sources import SeasonSource, source_schema

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence

_DOWNLOAD_TIMEOUT_SECONDS = 60
_REGULAR_SEASON = "Regular"
_SCORES = frozenset({"qb_plays", "qbr_total", "qbr_raw"})
_FLAGS = frozenset({"qualified"})


def _download_parquet(url: str) -> pl.DataFrame:
    """Return the Parquet file at ``url``.

    Raises:
        ConnectionError: If the download fails, as nflreadpy reports a failed download.
        ValueError: If the file is not Parquet or has no ``season`` column.

    """
    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        msg = f"Failed to download {url}: {exc}"
        raise ConnectionError(msg) from exc
    try:
        frame = pl.read_parquet(io.BytesIO(response.content))
    except pl.exceptions.PolarsError as exc:
        msg = f"Failed to parse {url}: {exc}"
        raise ValueError(msg) from exc
    if "season" not in frame.columns:
        msg = f"{url} has no season column"
        raise ValueError(msg)
    return frame


class _ReleaseAsset:
    """One nflverse release asset holding every season, downloaded at most once."""

    def __init__(self, url: str) -> None:
        """Remember the asset's address; nothing is downloaded yet."""
        self._url = url
        self._frame: pl.DataFrame | None = None

    def season(self, season: int) -> pl.DataFrame:
        """Return the asset's rows for ``season``, downloading the asset on first use.

        Raises:
            ValueError: If the asset holds no row for ``season``, as nflreadpy reports a season
                it does not serve: the current season then falls back to its cache or is skipped,
                instead of caching an empty season.

        """
        if self._frame is None:
            self._frame = _download_parquet(self._url)
        rows = self._frame.filter(pl.col("season") == season)
        if rows.is_empty():
            msg = f"{self._url} holds no rows for season {season}"
            raise ValueError(msg)
        return rows


def _qbr_source(
    level: Literal["season", "week"], fetch: Callable[[int], pl.DataFrame]
) -> SeasonSource:
    """Return the cached source of one QBR level, fetching seasons through ``fetch``."""
    if level == "season":
        columns = constants.ESPN_QBR_SEASON_COLUMNS
        text = {"season_type", "game_week", "team_abb", "player_id", "name_display"}
        team_columns: tuple[str, ...] = ("team_abb",)
    else:
        columns = constants.ESPN_QBR_WEEK_COLUMNS
        text = {"season_type", "game_id", "team_abb", "opp_abb", "player_id", "name_display"}
        team_columns = ("team_abb", "opp_abb")
    return SeasonSource(
        label=f"ESPN QBR ({level} level)",
        file_stem=f"espn_qbr_{level}",
        schema=source_schema(columns, text=frozenset(text), floats=_SCORES, booleans=_FLAGS),
        fetch=fetch,
        first_season=constants.ESPN_QBR_MIN_SEASON,
        season_type_column="season_type",
        regular_season_value=_REGULAR_SEASON,
        team_columns=team_columns,
    )


def load_espn_qbr(
    seasons: Sequence[int],
    *,
    level: Literal["season", "week"] = "season",
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load ESPN QBR for ``seasons``: season totals or one row per quarterback and game.

    At most one download serves every season the call fetches. Seasons before 2006 have no QBR
    and return no rows.

    Args:
        seasons: Season years to load
        level: ``season`` for season totals or ``week`` for single games
        cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
        force_refresh: If True, download historical seasons again even when cached
        current_season: Optional current season override for cache decisions

    Returns:
        The regular-season rows of ``seasons`` (``constants.ESPN_QBR_SEASON_COLUMNS`` or
        ``constants.ESPN_QBR_WEEK_COLUMNS``)

    Raises:
        ValueError: If ``level`` is neither ``season`` nor ``week``.

    """
    if level not in constants.ESPN_QBR_RELEASE_URLS:
        valid = ", ".join(sorted(constants.ESPN_QBR_RELEASE_URLS))
        msg = f"Unknown QBR level {level!r}; expected one of: {valid}"
        raise ValueError(msg)
    asset = _ReleaseAsset(constants.ESPN_QBR_RELEASE_URLS[level])
    return _qbr_source(level, asset.season).load(
        seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
        regular_season_only=True,
    )
