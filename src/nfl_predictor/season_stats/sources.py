"""The nflverse inputs of one season's tables, read through this repository's loaders.

``nfl-sos-ratings`` reads nflverse straight through nflreadpy (``data_loader.py`` there). Here
each source comes through this repository's cached loaders, and this module adds what that
project's loaders did beyond them:

- play-by-play (``loaders.load_pbp``): every team code normalized, the Rams' ``LA`` and an empty
  string (old seasons' timeouts) included, in every team column and in the yard-line text
  (:func:`prepare_pbp`); the fields nflverse never charted in a season set to null
  (``PBP_FIELD_GAPS``); and the player teams of the games nflverse credits wholly to one team
  (Jacksonville's 2001-2002 home games) rebuilt from the season roster
  (:func:`repair_one_team_games`);
- the schedule (``loaders.load_schedule``): regular-season games only, under the team column
  names the builders read (:func:`prepare_schedule`);
- the official weekly team stats, whose separate passing and rushing first downs and lost fumbles
  this repository's team-stat loader sums away: their own cached source
  (:data:`OFFICIAL_TEAM_STATS`), on the same loader as the other nflverse player sources;
- weekly player stats, snap counts, and the season's quarterback identity
  (``nflverse_sources``), which already carry the stat gaps, the Jacksonville repair, and the
  quarterback rule.

:func:`load_season_inputs` reads one season's six inputs. It refuses play-by-play that lacks a
column the builders read (``SEASON_STATS_PBP_COLUMNS``): a missing column would otherwise turn
every stat built on it into zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.season_stats.game_logs import OFFICIAL_TEAM_STAT_COLUMNS
from nfl_predictor.utils.polars import loaders, nflverse_sources, player_team_repair

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence

# nflverse play-by-play starts in 1999.
PBP_START_SEASON = 1999
# The first season nflverse charts pass depth, air yards, and yards after catch, with their EPA
# splits and expected yards after catch: before it they are null, apart from 1999's yards after
# catch on 337 of 9,522 completions and pass depth on 344 of 16,651 passes.
PASS_DEPTH_START_SEASON = 2006
_BEFORE_PASS_DEPTH = range(PBP_START_SEASON, PASS_DEPTH_START_SEASON)
# Seasons in which nflverse has no value for a play-by-play field, so every stat built on it is
# unknown there: nulls replace them, including where nflverse writes 0. Play-by-play records a QB
# hit on no play in 2003-2005 (before 2003 only on sacks), flags 0 to 22 dropbacks and runs a
# season as no-huddle before 2003 (275 to 506 in 2003-2005, following the play text), and has no
# drive penalty yards before 2001.
PBP_FIELD_GAPS: dict[str, range] = {
    "air_yards": _BEFORE_PASS_DEPTH,
    "air_epa": _BEFORE_PASS_DEPTH,
    "pass_length": _BEFORE_PASS_DEPTH,
    "yards_after_catch": _BEFORE_PASS_DEPTH,
    "yac_epa": _BEFORE_PASS_DEPTH,
    "xyac_mean_yardage": _BEFORE_PASS_DEPTH,
    "qb_hit": range(2003, 2006),
    "no_huddle": range(PBP_START_SEASON, 2003),
    "drive_yards_penalized": range(PBP_START_SEASON, 2001),
}
# Play-by-play columns that hold a team code. nflverse writes the Rams as LA in every one of them,
# not only in posteam and defteam, so all are normalized before any code compares one team column
# with another.
PBP_TEAM_COLUMNS = (
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "side_of_field",
    "timeout_team",
    "td_team",
    "return_team",
    "penalty_team",
    "fumbled_1_team",
    "fumbled_2_team",
    "fumble_recovery_1_team",
    "fumble_recovery_2_team",
    "forced_fumble_player_1_team",
    "forced_fumble_player_2_team",
    "solo_tackle_1_team",
    "solo_tackle_2_team",
    "assist_tackle_1_team",
    "assist_tackle_2_team",
    "assist_tackle_3_team",
    "assist_tackle_4_team",
    "tackle_with_assist_1_team",
    "tackle_with_assist_2_team",
)
# Play-by-play yard lines, written "<team> <yards>" (LA 25), or 50 at midfield.
PBP_YARD_LINE_COLUMNS = ("yrdln", "drive_start_yard_line", "drive_end_yard_line", "end_yard_line")
# The play-by-play columns the builders read (team_stats, team_stats_expanded, qb_stats, wp_bins
# and their shared expressions), found by their names in that code; .agents/sos_port's season
# statistics audit checks that a season built from these columns alone equals one built from all.
_BUILDER_PBP_COLUMNS = (
    "air_epa",
    "air_yards",
    "complete_pass",
    "cpoe",
    "defensive_two_point_conv",
    "defteam",
    "down",
    "drive_first_downs",
    "drive_inside20",
    "drive_play_count",
    "drive_start_yard_line",
    "drive_time_of_possession",
    "drive_yards_penalized",
    "epa",
    "field_goal_attempt",
    "first_down",
    "first_down_penalty",
    "fixed_drive",
    "fixed_drive_result",
    "fourth_down_converted",
    "fourth_down_failed",
    "fumble",
    "fumble_forced",
    "fumble_lost",
    "fumbled_1_player_id",
    "fumbled_1_team",
    "fumbled_2_player_id",
    "game_id",
    "goal_to_go",
    "incomplete_pass",
    "interception",
    "no_huddle",
    "pass",
    "pass_attempt",
    "pass_defense_1_player_id",
    "pass_length",
    "pass_oe",
    "pass_touchdown",
    "passer_player_id",
    "passer_player_name",
    "passing_yards",
    "penalty",
    "penalty_team",
    "penalty_type",
    "penalty_yards",
    "play_type",
    "posteam",
    "posteam_score",
    "posteam_score_post",
    "punt_attempt",
    "qb_dropback",
    "qb_epa",
    "qb_hit",
    "qb_kneel",
    "qb_scramble",
    "qb_spike",
    "qtr",
    "receiver_player_id",
    "receiver_player_name",
    "return_yards",
    "rush",
    "rush_attempt",
    "rush_touchdown",
    "rusher_player_id",
    "rusher_player_name",
    "rushing_yards",
    "sack",
    "score_differential",
    "score_differential_post",
    "season",
    "season_type",
    "series",
    "series_result",
    "series_success",
    "shotgun",
    "special",
    "special_teams_play",
    "success",
    "tackled_for_loss",
    "td_team",
    "third_down_converted",
    "third_down_failed",
    "touchdown",
    "two_point_attempt",
    "two_point_conv_result",
    "week",
    "wp",
    "wpa",
    "xyac_mean_yardage",
    "yac_epa",
    "yards_after_catch",
    "yards_gained",
    "ydsnet",
    "ydstogo",
)
# Every play-by-play column the season tables need: the builders' reads, and the player team and
# id columns and play text the repair of one-team games reads.
SEASON_STATS_PBP_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {
            *_BUILDER_PBP_COLUMNS,
            *constants.PBP_PLAYER_TEAM_ID_COLUMNS,
            *constants.PBP_PLAYER_TEAM_ID_COLUMNS.values(),
            "desc",
        }
    )
)
# The schedule columns the builders read, under the names they read them.
SCHEDULE_COLUMNS = ("game_id", "week", "home_team", "away_team", "home_score", "away_score")
_SCHEDULE_RENAMES = {"home_abbr": "home_team", "away_abbr": "away_team"}
_REGULAR_SEASON = "REG"


def _fetch_official_team_stats(season: int) -> pl.DataFrame:
    """Download one season of official weekly team stats."""
    return nfl.load_team_stats(seasons=[season], summary_level="week")


OFFICIAL_TEAM_STATS = nflverse_sources.SeasonSource(
    label="official weekly team stats",
    file_stem="official_team_stats",
    schema=nflverse_sources.source_schema(
        (
            "season",
            "week",
            "season_type",
            "game_id",
            "team",
            "opponent_team",
            *OFFICIAL_TEAM_STAT_COLUMNS,
        ),
        text=frozenset({"season_type", "game_id", "team", "opponent_team"}),
        floats=frozenset({"passing_epa", "rushing_epa", "passing_cpoe"}),
    ),
    fetch=_fetch_official_team_stats,
    first_season=PBP_START_SEASON,
    season_type_column="season_type",
    team_columns=("team", "opponent_team"),
)


def load_official_team_stats(
    seasons: Sequence[int],
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load regular-season official weekly team stats, one row per team-game.

    The cache contract is ``nflverse_sources.SeasonSource.load``'s: one Parquet file per season
    (``official_team_stats_<season>_reg.parquet``), historical seasons read from it, the current
    season fetched again and degrading to the cache.

    Returns:
        The team-game rows of ``seasons`` with ``OFFICIAL_TEAM_STATS``' columns

    """
    return OFFICIAL_TEAM_STATS.load(
        seasons,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        current_season=current_season,
    )


def normalize_team_codes(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Return ``frame`` with each of ``columns`` it has in canonical team codes.

    An empty code becomes null: old play-by-play leaves ``posteam`` empty, not null, on non-plays
    such as timeouts, and code that drops null teams would otherwise treat ``""`` as a team.
    """
    exprs = [
        pl.when(pl.col(column) == "")
        .then(None)
        .otherwise(pl.col(column).replace(constants.ALIAS_TO_CANONICAL))
        .alias(column)
        for column in columns
        if column in frame.columns
    ]
    return frame.with_columns(exprs) if exprs else frame


def normalize_yard_line_teams(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Return ``frame`` with the team code that opens yard-line text in canonical form.

    Code that reads a yard line compares its team with ``posteam`` to tell the offense's own half
    from the opponent's, so the code has to match the normalized ``posteam``. Whatever follows the
    code is kept as written (one 2002 end yard line reads ``LA -10``); midfield (``50`` or
    ``MID 50``), other codes, and nulls pass through unchanged.
    """
    exprs: list[pl.Expr] = []
    for column in columns:
        if column not in frame.columns:
            continue
        parts = pl.col(column).str.extract_groups(r"^(?<team>[A-Z]+) (?<rest>.+)$")
        normalized = pl.concat_str(
            [
                parts.struct.field("team").replace(constants.ALIAS_TO_CANONICAL),
                parts.struct.field("rest"),
            ],
            separator=" ",
        )
        exprs.append(pl.coalesce(normalized, pl.col(column)).alias(column))
    return frame.with_columns(exprs) if exprs else frame


def blank_fields_missing_in_season(
    frame: pl.DataFrame, season: int, gaps: Mapping[str, range] = PBP_FIELD_GAPS
) -> pl.DataFrame:
    """Return ``frame`` with every field the season's source lacks (per ``gaps``) set to null."""
    exprs = [
        pl.lit(None, dtype=frame.schema[column]).alias(column)
        for column, seasons in gaps.items()
        if column in frame.columns and season in seasons
    ]
    return frame.with_columns(exprs) if exprs else frame


def prepare_pbp(pbp: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one season's play-by-play ready for the builders, apart from one-team games.

    The fields nflverse lacks in the season are null (``PBP_FIELD_GAPS``), and every team code is
    canonical, in the team columns (``PBP_TEAM_COLUMNS``) and the yard lines
    (``PBP_YARD_LINE_COLUMNS``). :func:`repair_one_team_games` rebuilds the player teams of the
    games nflverse credits to one team.
    """
    prepared = blank_fields_missing_in_season(pbp, season)
    prepared = normalize_team_codes(prepared, PBP_TEAM_COLUMNS)
    return normalize_yard_line_teams(prepared, PBP_YARD_LINE_COLUMNS)


def repair_one_team_games(pbp: pl.DataFrame, rosters: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Return ``pbp`` with the player teams of every game credited to one team rebuilt.

    In such a game (Jacksonville's 2001-2002 home games, nflverse-pbp issue 92) every player team
    column names the visiting team; each player's team comes from the season roster instead
    (``player_team_repair``). ``rosters`` returns the season roster (``season``, ``gsis_id``,
    ``team``) and is called only when a game needs it.
    """
    games = player_team_repair.one_team_games(pbp, list(constants.PBP_PLAYER_TEAM_ID_COLUMNS))
    if not games:
        return pbp
    roster = rosters()
    player_teams = player_team_repair.game_player_teams(
        games,
        roster.select("season", pl.col("gsis_id").alias("player_id"), "team")
        .drop_nulls()
        .unique(maintain_order=True),
    )
    return player_team_repair.repair_pbp_player_teams(pbp, games, player_teams)


def prepare_schedule(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return the regular-season games of a loaded schedule, in ``SCHEDULE_COLUMNS``.

    ``loaders.load_schedule`` names the teams ``home_abbr`` and ``away_abbr`` (canonical codes)
    and keeps every game type.
    """
    renamed = schedule.rename(
        {old: new for old, new in _SCHEDULE_RENAMES.items() if old in schedule.columns}
    )
    if "game_type" in renamed.columns:
        renamed = renamed.filter(pl.col("game_type") == _REGULAR_SEASON)
    return renamed.select([column for column in SCHEDULE_COLUMNS if column in renamed.columns])


def missing_pbp_columns(pbp: pl.DataFrame) -> list[str]:
    """Return the columns of ``SEASON_STATS_PBP_COLUMNS`` that ``pbp`` lacks, in that order."""
    return [column for column in SEASON_STATS_PBP_COLUMNS if column not in pbp.columns]


@dataclass(frozen=True, slots=True)
class SeasonInputs:
    """One regular season's inputs to the season tables, each from its loader.

    Attributes:
        season: The season.
        pbp: Play-by-play, prepared (:func:`prepare_pbp`) and repaired.
        player_stats: Weekly player stats (``nflverse_sources.load_weekly_player_stats``).
        schedule: The regular-season schedule (:func:`prepare_schedule`), every game, played or
            not.
        official_team_stats: Official weekly team stats (:func:`load_official_team_stats`).
        snap_counts: Snap counts, empty before 2013.
        qb_identity: The season's quarterback identity
            (``nflverse_sources.load_season_qb_identity``).

    """

    season: int
    pbp: pl.DataFrame
    player_stats: pl.DataFrame
    schedule: pl.DataFrame
    official_team_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    qb_identity: pl.DataFrame


def load_season_inputs(
    season: int,
    *,
    cache_dir: os.PathLike[str] | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> SeasonInputs:
    """Load one regular season's inputs through this repository's loaders.

    Args:
        season: The season
        cache_dir: Optional cache directory override (default: the nflreadpy cache directory)
        force_refresh: If True, fetch historical seasons again even when cached
        current_season: Optional current season override for cache decisions

    Returns:
        The season's inputs, play-by-play prepared and repaired

    Raises:
        ValueError: If the season has plays but lacks a column of ``SEASON_STATS_PBP_COLUMNS``.

    """
    pbp = loaders.load_pbp(
        [season], cache_dir=cache_dir, force_refresh=force_refresh, current_season=current_season
    )
    missing = missing_pbp_columns(pbp)
    if not pbp.is_empty() and missing:
        msg = f"play-by-play of {season} lacks columns the season statistics read: " + ", ".join(
            missing
        )
        raise ValueError(msg)

    def season_roster() -> pl.DataFrame:
        return nflverse_sources.load_rosters(
            [season],
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            current_season=current_season,
        )

    return SeasonInputs(
        season=season,
        pbp=repair_one_team_games(prepare_pbp(pbp, season), season_roster),
        player_stats=nflverse_sources.load_weekly_player_stats(
            [season],
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        schedule=prepare_schedule(
            loaders.load_schedule(
                [season],
                cache_dir=cache_dir,
                force_refresh=force_refresh,
                current_season=current_season,
            )
        ),
        official_team_stats=load_official_team_stats(
            [season],
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        snap_counts=nflverse_sources.load_snap_counts(
            [season],
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            current_season=current_season,
        ),
        qb_identity=nflverse_sources.load_season_qb_identity(
            season, cache_dir=cache_dir, force_refresh=force_refresh, current_season=current_season
        ),
    )


__all__ = [
    "OFFICIAL_TEAM_STATS",
    "PBP_FIELD_GAPS",
    "PBP_TEAM_COLUMNS",
    "PBP_YARD_LINE_COLUMNS",
    "SCHEDULE_COLUMNS",
    "SEASON_STATS_PBP_COLUMNS",
    "SeasonInputs",
    "blank_fields_missing_in_season",
    "load_official_team_stats",
    "load_season_inputs",
    "missing_pbp_columns",
    "normalize_team_codes",
    "normalize_yard_line_teams",
    "prepare_pbp",
    "prepare_schedule",
    "repair_one_team_games",
]
