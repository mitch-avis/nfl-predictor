"""Cached loaders for the nflverse player sources, one Parquet file per season.

The sources are weekly player stats, snap counts, the players file, and season and weekly
rosters. Each per-season source follows the play-by-play loader's cache contract
(``loaders.load_pbp``).
A season is fetched through nflreadpy, kept to its requested columns (lists in ``constants``),
filtered to the regular season where the source has a season type, given canonical team codes,
and cached as one Parquet file per season under the nflreadpy cache directory, which records the
columns it was written for:

- A historical season (before ``current_season``) reads its cached file when the file is
  current; a file that lacks a column the code now requests is fetched again (see
  ``loaders._read_cached_frame``). ``force_refresh`` fetches every season again.
- The current and any later season is always fetched again.
- A failed fetch for a historical season is re-raised. For the current or a later season it
  falls back to the cached file, or the season is skipped: nflverse publishes a season's files
  only once it has games, and nflreadpy reports a missing file as a ``ConnectionError`` and a
  season it does not serve yet as a ``ValueError``.
- A season before the source's first season (the ``*_MIN_SEASON`` constants) returns no rows
  without asking nflreadpy.

Every frame holds every requested column, in order and with a fixed type (null where a season's
file lacks it), so seasons stack and an empty result keeps the schema.

Two nflverse quirks are handled after the cache, so the cache keeps the stats as published:
weekly player stats leave null the stats nflverse never charted in a season
(``constants.NFLREADPY_PLAYER_STAT_GAPS``), and the games nflverse credits wholly to one team get
their team codes back from the season rosters (``player_team_repair``). The quarterback identity
of a season (``player_identity``) combines the players file with the weekly roster.

The ETL does not read these sources yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import player_identity, player_team_repair
from nfl_predictor.utils.polars.loaders import (
    _read_cached_frame,
    _resolve_cache_dir,
    _resolve_current_season,
    _write_cached_frame,
)
from nfl_predictor.utils.scraping_utils import normalize_team_column

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

# How nflreadpy reports a season it cannot serve: a file it could not download or parse
# (ConnectionError, ValueError) or a season outside its range (ValueError).
DOWNLOAD_ERRORS: tuple[type[Exception], ...] = (ConnectionError, ValueError)
_PLAYERS_CACHE_NAME = "players.parquet"


def conform(frame: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Return ``frame`` with exactly the columns of ``schema``, in order and with its types.

    A column ``frame`` lacks is added as nulls; a column ``schema`` does not name is dropped.
    """
    return frame.select(
        pl.col(column).cast(dtype)
        if column in frame.columns
        else pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in schema.items()
    )


def source_schema(
    columns: Sequence[str],
    *,
    text: frozenset[str],
    floats: frozenset[str] = frozenset(),
    booleans: frozenset[str] = frozenset(),
) -> pl.Schema:
    """Return the schema of requested ``columns``: text, floats, booleans, whole numbers else."""
    dtypes: dict[str, pl.DataType] = {}
    for column in columns:
        if column in text:
            dtypes[column] = pl.String()
        elif column in floats:
            dtypes[column] = pl.Float64()
        elif column in booleans:
            dtypes[column] = pl.Boolean()
        else:
            dtypes[column] = pl.Int64()
    return pl.Schema(dtypes)


@dataclass(frozen=True)
class SeasonSource:
    """How one nflverse source is fetched, prepared and cached, one season per file.

    Attributes:
        label: The source's name in log messages.
        file_stem: Cache file name before ``_<season>[_<reg|all>].parquet``.
        schema: The requested columns, in output order, with their types.
        fetch: Downloads one season's raw frame.
        first_season: The first season the source publishes.
        season_type_column: The column naming regular-season rows, when the source has one; the
            cache file name then says whether the postseason was kept (``reg`` or ``all``).
        regular_season_value: That column's value on regular-season rows.
        team_columns: Columns holding team codes, normalized to the canonical codes.
        team_codes: Source-specific codes replaced before that normalization.

    """

    label: str
    file_stem: str
    schema: pl.Schema
    fetch: Callable[[int], pl.DataFrame]
    first_season: int
    season_type_column: str | None = None
    regular_season_value: str = "REG"
    team_columns: tuple[str, ...] = ()
    team_codes: Mapping[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the requested columns in output order."""
        return tuple(self.schema.names())

    def empty(self) -> pl.DataFrame:
        """Return an empty frame with every requested column."""
        return pl.DataFrame(schema=self.schema)

    def cache_path(self, cache_dir: Path, season: int, *, regular_season_only: bool) -> Path:
        """Return the cache file of one season."""
        if self.season_type_column is None:
            return cache_dir / f"{self.file_stem}_{season}.parquet"
        suffix = "reg" if regular_season_only else "all"
        return cache_dir / f"{self.file_stem}_{season}_{suffix}.parquet"

    def prepare(self, raw: pl.DataFrame, *, regular_season_only: bool) -> pl.DataFrame:
        """Return one downloaded season filtered, conformed and with canonical team codes."""
        frame = raw
        if (
            regular_season_only
            and self.season_type_column is not None
            and self.season_type_column in frame.columns
        ):
            frame = frame.filter(pl.col(self.season_type_column) == self.regular_season_value)
        frame = conform(frame, self.schema)
        for column in self.team_columns:
            if self.team_codes:
                frame = frame.with_columns(pl.col(column).replace(dict(self.team_codes)))
            frame = normalize_team_column(frame, column)
        return frame

    def load(
        self,
        seasons: Sequence[int],
        *,
        cache_dir: os.PathLike[str] | str | None = None,
        force_refresh: bool = False,
        current_season: int | None = None,
        regular_season_only: bool = True,
    ) -> pl.DataFrame:
        """Load ``seasons`` of the source under the cache contract in the module docstring.

        Args:
            seasons: Season years to load
            cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
            force_refresh: If True, fetch historical seasons again even when cached
            current_season: Optional current season override for cache decisions
            regular_season_only: If True, keep only regular-season rows (when the source has a
                season type)

        Returns:
            The seasons' rows, stacked in season order, with every requested column

        """
        if not seasons:
            return self.empty()

        cache_root = _resolve_cache_dir(cache_dir)
        current = _resolve_current_season(current_season)
        log.info("Loading %s for seasons: %s", self.label, list(seasons))
        frames: list[pl.DataFrame] = []
        for season in seasons:
            frame = self._load_season(
                season,
                self.cache_path(cache_root, season, regular_season_only=regular_season_only),
                historical=season < current,
                force_refresh=force_refresh,
                regular_season_only=regular_season_only,
            )
            if frame is not None:
                frames.append(frame)
        if not frames:
            log.warning("No %s available for seasons: %s", self.label, list(seasons))
            return self.empty()
        return pl.concat(frames)

    def _load_season(
        self,
        season: int,
        path: Path,
        *,
        historical: bool,
        force_refresh: bool,
        regular_season_only: bool,
    ) -> pl.DataFrame | None:
        """Return one season from its cache file or a download, or None when it has no rows."""
        if season < self.first_season:
            log.info(
                "nflverse publishes no %s before %d; season %d has none.",
                self.label,
                self.first_season,
                season,
            )
            return None
        cached = (
            _read_cached_frame(path, self.columns) if historical and not force_refresh else None
        )
        if cached is not None:
            log.info("Using cached %s for season %d from %s", self.label, season, path)
            return conform(cached, self.schema)

        if historical and not force_refresh:
            log.info("%s cache miss for season %d; downloading from nflverse.", self.label, season)
        else:
            log.info("Refreshing %s from nflverse for season %d.", self.label, season)

        try:
            raw = self.fetch(season)
        except DOWNLOAD_ERRORS as exc:
            if historical:
                raise
            fallback = _read_cached_frame(path)
            if fallback is not None:
                log.warning(
                    "Current-season %s unavailable for season %d; using cached data from %s. "
                    "Error: %s",
                    self.label,
                    season,
                    path,
                    exc,
                )
                return conform(fallback, self.schema)
            log.warning(
                "Current-season %s not published yet for season %d; continuing without it. "
                "Error: %s",
                self.label,
                season,
                exc,
            )
            return None

        prepared = self.prepare(raw, regular_season_only=regular_season_only)
        _write_cached_frame(prepared, path, self.columns)
        return prepared


def _fetch_player_stats(season: int) -> pl.DataFrame:
    """Download one season of weekly player stats."""
    return nfl.load_player_stats(seasons=[season], summary_level="week")


def _fetch_snap_counts(season: int) -> pl.DataFrame:
    """Download one season of snap counts."""
    return nfl.load_snap_counts(seasons=[season])


def _fetch_rosters(season: int) -> pl.DataFrame:
    """Download one season roster."""
    return nfl.load_rosters(seasons=[season])


def _fetch_weekly_rosters(season: int) -> pl.DataFrame:
    """Download one season of weekly rosters."""
    return nfl.load_rosters_weekly(seasons=[season])


_ROSTER_SCHEMA = source_schema(
    constants.NFLREADPY_ROSTER_COLUMNS,
    text=frozenset({"game_type", "team", "position", "full_name", "gsis_id", "pfr_id"}),
)

PLAYER_STATS = SeasonSource(
    label="weekly player stats",
    file_stem="player_stats_week",
    schema=source_schema(
        constants.NFLREADPY_PLAYER_STATS_COLUMNS,
        text=frozenset(
            {
                "player_id",
                "player_name",
                "player_display_name",
                "position",
                "season_type",
                "game_id",
                "team",
                "opponent_team",
            }
        ),
        floats=frozenset({"passing_epa", "passing_cpoe", "rushing_epa", "def_sacks"}),
    ),
    fetch=_fetch_player_stats,
    first_season=constants.NFLREADPY_PLAYER_STATS_MIN_SEASON,
    season_type_column="season_type",
    team_columns=("team", "opponent_team"),
)
SNAP_COUNTS = SeasonSource(
    label="snap counts",
    file_stem="snap_counts",
    schema=source_schema(
        constants.NFLREADPY_SNAP_COUNT_COLUMNS,
        text=frozenset(
            {"game_id", "game_type", "player", "pfr_player_id", "position", "team", "opponent"}
        ),
        floats=frozenset({"offense_snaps", "offense_pct"}),
    ),
    fetch=_fetch_snap_counts,
    first_season=constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON,
    season_type_column="game_type",
    team_columns=("team", "opponent"),
)
ROSTERS = SeasonSource(
    label="season rosters",
    file_stem="rosters",
    schema=_ROSTER_SCHEMA,
    fetch=_fetch_rosters,
    first_season=constants.NFLREADPY_ROSTERS_MIN_SEASON,
    team_columns=("team",),
    team_codes=constants.NFLVERSE_ROSTER_TEAM_CODES,
)
WEEKLY_ROSTERS = SeasonSource(
    label="weekly rosters",
    file_stem="rosters_weekly",
    schema=_ROSTER_SCHEMA,
    fetch=_fetch_weekly_rosters,
    first_season=constants.NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON,
    season_type_column="game_type",
    team_columns=("team",),
    team_codes=constants.NFLVERSE_ROSTER_TEAM_CODES,
)
_PLAYERS_SCHEMA = source_schema(
    constants.NFLREADPY_PLAYERS_COLUMNS,
    text=frozenset(constants.NFLREADPY_PLAYERS_COLUMNS),
)


def blank_player_stat_gaps(stats: pl.DataFrame) -> pl.DataFrame:
    """Return weekly player stats with every stat nflverse never charted in a season set to null.

    nflverse publishes 0 for a stat it credits to no one in a season;
    ``constants.NFLREADPY_PLAYER_STAT_GAPS`` names each such stat and its seasons.
    """
    gaps = [
        pl.when(pl.col("season").is_between(first, last))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column, (first, last) in constants.NFLREADPY_PLAYER_STAT_GAPS.items()
        if column in stats.columns
    ]
    return stats.with_columns(gaps) if gaps else stats


def load_weekly_player_stats(
    seasons: Sequence[int],
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Load weekly player stats, one row per player and game (``constants`` column list).

    The stats nflverse never charted in a season are null (``blank_player_stat_gaps``). In a
    game nflverse credits wholly to one team, ``team`` and ``opponent_team`` are rebuilt from the
    season rosters (``load_rosters``); a player the rosters cannot place gets null for both.

    Args:
        seasons: Season years to load
        cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
        force_refresh: If True, fetch historical seasons again even when cached
        current_season: Optional current season override for cache decisions
        regular_season_only: If True, keep only regular-season rows

    Returns:
        The player-game rows of ``seasons``

    """
    stats = blank_player_stat_gaps(
        PLAYER_STATS.load(
            seasons,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            current_season=current_season,
            regular_season_only=regular_season_only,
        )
    )
    games = player_team_repair.one_team_games(stats, ["team"])
    if not games:
        return stats

    log.warning(
        "Rebuilding player teams from the rosters in %d game(s) nflverse credits to one team: %s",
        len(games),
        games,
    )
    roster_seasons = (
        player_team_repair.game_sides(games)
        .get_column("season")
        .unique(maintain_order=True)
        .sort()
        .to_list()
    )
    rosters = load_rosters(
        roster_seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
    )
    player_teams = player_team_repair.game_player_teams(
        games,
        rosters.select("season", pl.col("gsis_id").alias("player_id"), "team")
        .drop_nulls()
        .unique(maintain_order=True),
    )
    repaired = player_team_repair.repair_player_stats_teams(stats, games, player_teams)
    unplaced = repaired.filter(pl.col("game_id").is_in(games) & pl.col("team").is_null()).height
    if unplaced:
        log.warning(
            "%d player row(s) in those games stay without a team: the rosters place the player "
            "on neither or both teams.",
            unplaced,
        )
    return repaired


def load_snap_counts(
    seasons: Sequence[int],
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Load snap counts, one row per player and game, keyed by ``pfr_player_id``.

    See ``SeasonSource.load`` for the arguments and the cache contract.

    Returns:
        The player-game rows of ``seasons``; none before 2012 (and 2012's file is empty)

    """
    return SNAP_COUNTS.load(
        seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
        regular_season_only=regular_season_only,
    )


def load_rosters(
    seasons: Sequence[int],
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load season rosters, one row per player and team he was on that season.

    The league codes of the 2002-2015 rosters (``constants.NFLVERSE_ROSTER_TEAM_CODES``) become
    canonical codes like every other. See ``SeasonSource.load`` for the arguments.

    Returns:
        The roster rows of ``seasons``

    """
    return ROSTERS.load(
        seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
        regular_season_only=False,
    )


def load_weekly_rosters(
    seasons: Sequence[int],
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Load weekly rosters, one row per player, team and week; nflverse publishes them from 2002.

    See ``load_rosters`` for the team codes and ``SeasonSource.load`` for the arguments.

    Returns:
        The roster rows of ``seasons``

    """
    return WEEKLY_ROSTERS.load(
        seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
        regular_season_only=regular_season_only,
    )


def load_players(
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    degrade_on_failure: bool = False,
) -> pl.DataFrame:
    """Load the nflverse players file: one row per player with his latest position and ids.

    The file covers every season, so it is cached as one file (``players.parquet``) and read
    from there unless ``force_refresh``. A failed download is re-raised, except with
    ``degrade_on_failure``, the current-season contract: then the cached file is used even if
    stale, or an empty frame when there is none.

    Args:
        cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
        force_refresh: If True, download the file even when cached
        degrade_on_failure: If True, a failed download falls back instead of raising

    Returns:
        The players file's requested columns (``constants.NFLREADPY_PLAYERS_COLUMNS``)

    """
    path = _resolve_cache_dir(cache_dir) / _PLAYERS_CACHE_NAME
    columns = tuple(constants.NFLREADPY_PLAYERS_COLUMNS)
    cached = None if force_refresh else _read_cached_frame(path, columns)
    if cached is not None:
        log.info("Using the cached players file from %s", path)
        return conform(cached, _PLAYERS_SCHEMA)

    log.info("Downloading the players file from nflverse.")
    try:
        raw = nfl.load_players()
    except DOWNLOAD_ERRORS as exc:
        if not degrade_on_failure:
            raise
        fallback = _read_cached_frame(path)
        if fallback is not None:
            log.warning("Players file unavailable; using cached data from %s. Error: %s", path, exc)
            return conform(fallback, _PLAYERS_SCHEMA)
        log.warning("Players file unavailable; continuing without it. Error: %s", exc)
        return pl.DataFrame(schema=_PLAYERS_SCHEMA)

    players = conform(raw, _PLAYERS_SCHEMA)
    _write_cached_frame(players, path, columns)
    return players


def load_qb_identity(
    season: int,
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load one season's quarterback identity (``player_identity.qb_identity_crosswalk``).

    The current or a later season reads a fresh players file and survives its outage, as the
    per-season sources do; an earlier season reads the cached file. Before 2002 there is no
    weekly roster, and the players file stands alone.

    Args:
        season: The season
        cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
        force_refresh: If True, download the players file and the weekly roster again
        current_season: Optional current season override for cache decisions

    Returns:
        One row per GSIS id with the snap-count id, name and the season's position

    """
    current = _resolve_current_season(current_season)
    is_current = season >= current
    players = load_players(
        cache_dir=cache_dir,
        force_refresh=force_refresh or is_current,
        degrade_on_failure=is_current,
    )
    weekly_rosters = load_weekly_rosters(
        [season], cache_dir=cache_dir, force_refresh=force_refresh, current_season=current
    )
    return player_identity.qb_identity_crosswalk(players, weekly_rosters)
