"""Adapter from this repository's play-by-play and nflverse sources to the ratings' input rows.

The ratings reproduce ``nfl-sos-ratings``' published files only when their input rows follow that
project's definitions, which differ in places from this repository's own play-by-play features
(``utils/polars/pbp.py``). This module builds the rows of :mod:`nfl_predictor.ratings.inputs` from
the frames this repository's loaders return (``utils/polars/loaders.py`` and
``utils/polars/nflverse_sources.py``, canonical team codes), with that project's definitions:

Team-game rows (:func:`team_games_from_pbp`), one per team and regular-season game with a snap:

- A scrimmage snap is a dropback, a run (``rush``), a kneel-down or a spike, on a play with both
  teams known (an empty team code counts as unknown). A dropback is a play flagged
  ``qb_dropback``, or a run play flagged ``qb_scramble``: nflverse leaves most scrambles before
  2006 without the dropback flag.
- ``offensive_epa`` sums the snaps' EPA, a missing EPA counting as 0, in the order the plays come;
  the sums depend on that order in their last bits, so the plays keep nflverse's order.
- Special-teams plays are the plays flagged ``special``, credited to the team with the ball.
- The margin and the home side come from the schedule. The schedule's home team is the home team
  even at a neutral site, as in that project, so its home-field term reaches those games too.

Quarterback rows, one per passer and game, built in two steps. :func:`passer_games_from_pbp`
returns every passer-game with what the quarterback test and the official values need, and
:func:`quarterback_rows` applies them, for the whole season or as known before a week:

- A passer's dropbacks are his plays flagged ``qb_dropback`` (sacks included; scrambles carry no
  passer, so they count for the team only). His attempts are the pass attempts among them. His
  passing EPA is the ``qb_epa`` of those plays plus his spikes', and his dropback EPA leaves the
  spikes' out again (a spike is not a dropback).
- Snap counts add every quarterback who played offensive snaps without a dropback, matched to a
  passer through his snap-count id.
- Weekly player stats replace a passer's attempts and passing EPA when they list him at
  quarterback in that game, or when the identity does.
- The identity (``player_identity.qb_identity_crosswalk``) names each passer, players file first,
  then the weekly roster, then the play-by-play. A passer stays when the players file or a
  regular-season week of the weekly roster lists him at quarterback, or when no source places
  him at any position; a passer placed only at other positions is left out.
- For the whole season (``before_week`` of ``None``) every roster week counts, as in that project;
  rows are in its fit order, team, week, game and name.
- Before week ``w`` only the roster weeks before ``w`` count, for the quarterback test and for the
  identity's part in choosing the official values, and only the games before ``w`` are returned,
  sorted by team, week, game and passer id, an order no later information can change. The
  players file, a single current snapshot, counts in both.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.ratings.inputs import QB_GAME_SCHEMA, TEAM_GAME_SCHEMA
from nfl_predictor.utils.polars.player_identity import qb_identity_crosswalk

if TYPE_CHECKING:
    from collections.abc import Iterable

# Every passer-game, before the quarterback test, with the inputs the test and the official
# values read: both versions of the attempts and the passing EPA (spikes included), the spikes'
# EPA, whether the weekly player stats list him at quarterback in the game, and what the players
# file and the season's weekly roster say about his position.
PASSER_GAME_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "game_id": pl.String(),
    "week": pl.Int64(),
    "team": pl.String(),
    "opponent_team": pl.String(),
    "is_home": pl.Boolean(),
    "qb_id": pl.String(),
    "qb_name": pl.String(),
    "qb_dropbacks": pl.Int64(),
    "pbp_attempts": pl.Int64(),
    "official_attempts": pl.Int64(),
    "pbp_passing_epa": pl.Float64(),
    "official_passing_epa": pl.Float64(),
    "qb_spike_epa": pl.Float64(),
    "stats_list_qb": pl.Boolean(),
    "players_file_qb": pl.Boolean(),
    "players_file_placed": pl.Boolean(),
    "first_qb_roster_week": pl.Int64(),
    "first_roster_week": pl.Int64(),
}

_REGULAR_SEASON = "REG"
_RUN = "run"
_TEAM_PLAY_COLUMNS = (
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "qb_dropback",
    "qb_scramble",
    "rush",
    "qb_kneel",
    "qb_spike",
    "epa",
)
_PASSER_PLAY_COLUMNS = (
    "game_id",
    "week",
    "posteam",
    "qb_dropback",
    "qb_spike",
    "pass",
    "qb_epa",
    "passer_player_id",
    "passer_player_name",
)
_SCHEDULE_COLUMNS = ("game_id", "week", "away_abbr", "home_abbr", "away_score", "home_score")
_GAME_KEY = ("game_id", "week", "team")
_PASSER_KEY = (*_GAME_KEY, "qb_id")


@dataclass(frozen=True, slots=True)
class SeasonSources:
    """One season's inputs for the quarterback rows, as this repository's loaders return them.

    Attributes:
        season: The season.
        plays: Its regular-season play-by-play (``loaders.load_pbp``).
        player_stats: Its weekly player stats (``nflverse_sources.load_weekly_player_stats``).
        snap_counts: Its snap counts (``nflverse_sources.load_snap_counts``), empty before 2013.
        players: The players file (``nflverse_sources.load_players``).
        weekly_rosters: Its regular-season weekly roster (``load_weekly_rosters``), empty
            before 2002.

    """

    season: int
    plays: pl.DataFrame
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    players: pl.DataFrame
    weekly_rosters: pl.DataFrame


def _require(frame: pl.DataFrame, columns: Iterable[str], what: str) -> None:
    """Raise ``ValueError`` naming every column of ``columns`` that ``frame`` lacks."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        msg = f"{what} are missing columns: {', '.join(missing)}"
        raise ValueError(msg)


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing."""
    return pl.col(column).fill_null(0) > 0


def _known_team(column: str) -> pl.Expr:
    """Return whether a team-code column holds a team: old play-by-play leaves some empty."""
    return pl.col(column).is_not_null() & (pl.col(column) != "")


def _regular_plays(plays: pl.DataFrame) -> pl.DataFrame:
    """Return the regular-season plays of ``plays`` (all of them without a season type)."""
    if "season_type" in plays.columns:
        return plays.filter(pl.col("season_type") == _REGULAR_SEASON)
    return plays


def _special_teams_play(columns: Iterable[str]) -> pl.Expr:
    """Return the special-teams flag, from the first flag column the play-by-play has."""
    present = set(columns)
    for column in constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES:
        if column in present:
            return _flag(column)
    return pl.lit(value=False)


def _scrimmage_snap() -> pl.Expr:
    """Return the scrimmage-snap flag: a dropback, a run, a kneel-down or a spike."""
    scramble_run = _flag("qb_scramble") & (pl.col("play_type") == _RUN).fill_null(value=False)
    return (
        _flag("qb_dropback") | scramble_run | _flag("rush") | _flag("qb_kneel") | _flag("qb_spike")
    )


def _team_scores(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return one row per team and regular-season game: home side and points for and against."""
    games = (
        schedule.filter(pl.col("game_type") == _REGULAR_SEASON)
        if "game_type" in schedule.columns
        else schedule
    ).with_columns(pl.col("week").cast(pl.Int64))
    sides = [
        games.select(
            "game_id",
            "week",
            pl.col(team).alias("team"),
            pl.col(opponent).alias("opponent_team"),
            pl.lit(value=is_home).alias("is_home"),
            pl.col(scored).cast(pl.Int64).alias("points_for"),
            pl.col(allowed).cast(pl.Int64).alias("points_allowed"),
        )
        for team, opponent, is_home, scored, allowed in (
            ("home_abbr", "away_abbr", True, "home_score", "away_score"),
            ("away_abbr", "home_abbr", False, "away_score", "home_score"),
        )
    ]
    return pl.concat(sides)


def team_games_from_pbp(plays: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Return the team-game rows of every season in ``plays``.

    Args:
        plays: Play-by-play as ``loaders.load_pbp`` returns it; regular-season plays are used.
        schedule: The schedule of the same seasons as ``loaders.load_schedule`` returns it.

    Returns:
        ``season`` and the columns of ``TEAM_GAME_SCHEMA``, one row per team and game with a
        scrimmage snap, sorted by season, team, week and game.

    Raises:
        ValueError: If ``plays`` or ``schedule`` lacks a column the rows need.

    """
    _require(plays, _TEAM_PLAY_COLUMNS, "play-by-play rows")
    _require(schedule, _SCHEDULE_COLUMNS, "schedule rows")
    epa = pl.col("epa").fill_null(0.0)
    snap = _scrimmage_snap()
    special = _special_teams_play(plays.columns)
    rows = (
        _regular_plays(plays)
        .filter(_known_team("posteam") & _known_team("defteam"))
        .group_by(["season", "game_id", "week", "posteam", "defteam"], maintain_order=True)
        .agg(
            snap.cast(pl.Int64).sum().alias("offensive_snaps"),
            epa.filter(snap).sum().fill_null(0.0).alias("offensive_epa"),
            special.cast(pl.Int64).sum().alias("st_plays"),
            epa.filter(special).sum().fill_null(0.0).alias("st_epa"),
        )
        .filter(pl.col("offensive_snaps") > 0)
        .rename({"posteam": "team", "defteam": "opponent_team"})
        .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
    )
    return (
        rows.join(
            _team_scores(schedule), on=["game_id", "week", "team", "opponent_team"], how="left"
        )
        .with_columns((pl.col("points_for") - pl.col("points_allowed")).alias("point_margin"))
        .select(
            pl.col("season"),
            *(pl.col(name).cast(dtype) for name, dtype in TEAM_GAME_SCHEMA.items()),
        )
        .sort("season", "team", "week", "game_id")
    )


def _passer_groups(plays: pl.DataFrame, *aggregations: pl.Expr) -> pl.DataFrame:
    """Return one row per passer and team-game of ``plays``, keyed by id or else by name.

    A passer is his id, or his name when the play has no id, so a passer the play-by-play names
    two ways in a game stays one row.
    """
    passer = pl.coalesce(pl.col("passer_player_id"), pl.col("passer_player_name"))
    return (
        plays.with_columns(passer.alias("_passer"))
        .group_by(["game_id", "week", "posteam", "_passer"], maintain_order=True)
        .agg(pl.col("passer_player_id").drop_nulls().first().alias("qb_id"), *aggregations)
        .drop("_passer")
        .rename({"posteam": "team"})
        .with_columns(pl.col("week").cast(pl.Int64))
    )


def _dropback_passers(plays: pl.DataFrame) -> pl.DataFrame:
    """Return each passer-game's dropbacks, pass attempts among them, and their ``qb_epa``."""
    return _passer_groups(
        plays.filter(_flag("qb_dropback")),
        pl.col("passer_player_name").drop_nulls().sort().first().alias("pbp_name"),
        pl.col("qb_dropback").sum().cast(pl.Int64).alias("qb_dropbacks"),
        pl.col("pass").fill_null(0).sum().cast(pl.Int64).alias("pbp_attempts"),
        pl.col("qb_epa").fill_null(0.0).sum().alias("dropback_qb_epa"),
    )


def _spike_epa(plays: pl.DataFrame) -> pl.DataFrame:
    """Return each identified passer-game's spikes' ``qb_epa``."""
    return (
        _passer_groups(
            plays.filter(_flag("qb_spike")),
            pl.col("qb_epa").fill_null(0.0).sum().alias("spike_epa"),
        )
        .filter(pl.col("qb_id").is_not_null())
        .select(*_PASSER_KEY, "spike_epa")
    )


def _snap_count_quarterbacks(snap_counts: pl.DataFrame) -> pl.DataFrame:
    """Return each quarterback-game in the snap counts with an offensive snap, by snap-count id."""
    if snap_counts.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String(),
                "week": pl.Int64(),
                "team": pl.String(),
                "snap_player_id": pl.String(),
                "snap_name": pl.String(),
            }
        )
    return (
        snap_counts.filter(
            (pl.col("position") == constants.QB_POSITION)
            & (pl.col("offense_snaps").fill_null(0) > 0)
        )
        .select(
            "game_id",
            pl.col("week").cast(pl.Int64),
            "team",
            pl.col("pfr_player_id").alias("snap_player_id"),
            pl.col("player").alias("snap_name"),
        )
        .unique(maintain_order=True)
    )


def _merged_passers(
    dropbacks: pl.DataFrame, snaps: pl.DataFrame, identity: pl.DataFrame
) -> pl.DataFrame:
    """Return one row per passer-game: the dropback passers and the snap-count quarterbacks.

    Both are named through the identity; a snap-count row joins its passer's dropback row when
    the identity maps its snap-count id to his id. Without any identity rows they merge by name.
    """
    by_id = identity.select("qb_id", "snap_player_id", pl.col("qb_name").alias("identity_name"))
    by_snap_id = (
        identity.filter(pl.col("snap_player_id").is_not_null())
        .sort("qb_id")
        .unique(subset=["snap_player_id"], keep="first", maintain_order=True)
        .select("snap_player_id", "qb_id", pl.col("qb_name").alias("identity_name"))
    )
    passers = dropbacks.join(by_id, on="qb_id", how="left").select(
        *_GAME_KEY,
        "qb_id",
        "snap_player_id",
        pl.coalesce("identity_name", "pbp_name").alias("qb_name"),
        "qb_dropbacks",
    )
    quarterbacks = snaps.join(by_snap_id, on="snap_player_id", how="left").select(
        *_GAME_KEY,
        "qb_id",
        "snap_player_id",
        pl.coalesce("identity_name", "snap_name").alias("qb_name"),
        pl.lit(0, dtype=pl.Int64).alias("qb_dropbacks"),
    )
    key = (
        pl.coalesce("qb_id", "snap_player_id", "qb_name") if identity.height else pl.col("qb_name")
    )
    return (
        pl.concat([passers, quarterbacks])
        .with_columns(key.alias("_passer"))
        .group_by([*_GAME_KEY, "_passer"], maintain_order=True)
        .agg(
            pl.col("qb_id").drop_nulls().first(),
            pl.col("snap_player_id").drop_nulls().first(),
            pl.col("qb_name").drop_nulls().first(),
            pl.col("qb_dropbacks").sum(),
        )
        .drop("_passer")
    )


def _official_values(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Return the weekly player stats' attempts, passing EPA and position per player-game."""
    return player_stats.filter(
        pl.col("team").is_not_null()
        & pl.col("week").is_not_null()
        & pl.col("player_id").is_not_null()
    ).select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("player_id").cast(pl.String).alias("qb_id"),
        (pl.col("position") == constants.QB_POSITION).fill_null(value=False).alias("stats_list_qb"),
        pl.col("attempts").cast(pl.Int64).alias("official_attempts"),
        pl.col("passing_epa").cast(pl.Float64).alias("official_passing_epa"),
    )


def _position_facts(players: pl.DataFrame, weekly_rosters: pl.DataFrame) -> pl.DataFrame:
    """Return what the players file and the weekly roster say about each player's position."""
    from_players = (
        players.filter(pl.col("gsis_id").is_not_null())
        .group_by("gsis_id", maintain_order=True)
        .agg(
            (pl.col("position") == constants.QB_POSITION).any().alias("players_file_qb"),
            pl.col("position").is_not_null().any().alias("players_file_placed"),
        )
    )
    rosters = (
        weekly_rosters.filter(pl.col("game_type") == _REGULAR_SEASON)
        if "game_type" in weekly_rosters.columns
        else weekly_rosters
    )
    week = pl.col("week").cast(pl.Int64)
    from_rosters = (
        rosters.filter(pl.col("gsis_id").is_not_null())
        .group_by("gsis_id", maintain_order=True)
        .agg(
            week.filter(pl.col("position") == constants.QB_POSITION)
            .min()
            .alias("first_qb_roster_week"),
            week.filter(pl.col("position").is_not_null()).min().alias("first_roster_week"),
        )
    )
    return from_players.join(from_rosters, on="gsis_id", how="full", coalesce=True).rename(
        {"gsis_id": "qb_id"}
    )


def passer_games_from_pbp(sources: SeasonSources, team_games: pl.DataFrame) -> pl.DataFrame:
    """Return every passer-game of one season, before the quarterback test.

    Args:
        sources: The season's play-by-play, player sources and identity sources.
        team_games: The season's team-game rows (:func:`team_games_from_pbp`), for each game's
            opponent and home side.

    Returns:
        ``PASSER_GAME_SCHEMA`` rows, sorted by team, week, game, name and id; turn them into
        quarterback-game rows with :func:`quarterback_rows`.

    Raises:
        ValueError: If the play-by-play lacks a column the rows need.

    """
    _require(sources.plays, _PASSER_PLAY_COLUMNS, "play-by-play rows")
    plays = _regular_plays(sources.plays).filter(
        _known_team("posteam") & pl.col("passer_player_name").is_not_null()
    )
    identity = qb_identity_crosswalk(sources.players, sources.weekly_rosters)
    dropbacks = _dropback_passers(plays)
    merged = _merged_passers(dropbacks, _snap_count_quarterbacks(sources.snap_counts), identity)
    context = team_games.select("team", pl.col("week").cast(pl.Int64), "opponent_team", "is_home")
    rows = (
        merged.join(
            dropbacks.select(*_PASSER_KEY, "pbp_attempts", "dropback_qb_epa").filter(
                pl.col("qb_id").is_not_null()
            ),
            on=list(_PASSER_KEY),
            how="left",
        )
        .join(_spike_epa(plays), on=list(_PASSER_KEY), how="left")
        .join(_official_values(sources.player_stats), on=list(_PASSER_KEY), how="left")
        .join(context, on=["team", "week"], how="left")
        .with_columns(
            pl.coalesce("qb_id", "snap_player_id", "qb_name").alias("qb_id"),
            pl.col("pbp_attempts").fill_null(0),
            (pl.col("dropback_qb_epa").fill_null(0.0) + pl.col("spike_epa").fill_null(0.0)).alias(
                "pbp_passing_epa"
            ),
            pl.col("spike_epa").fill_null(0.0).alias("qb_spike_epa"),
            pl.col("stats_list_qb").fill_null(value=False),
            pl.lit(sources.season, dtype=pl.Int64).alias("season"),
        )
        .join(_position_facts(sources.players, sources.weekly_rosters), on="qb_id", how="left")
        .with_columns(pl.col("players_file_qb", "players_file_placed").fill_null(value=False))
    )
    return rows.select(pl.col(name).cast(dtype) for name, dtype in PASSER_GAME_SCHEMA.items()).sort(
        "team", "week", "game_id", "qb_name", "qb_id"
    )


def quarterback_rows(passer_games: pl.DataFrame, *, before_week: int | None = None) -> pl.DataFrame:
    """Return one season's quarterback-game rows, for the whole season or as known before a week.

    Args:
        passer_games: The season's rows from :func:`passer_games_from_pbp`.
        before_week: ``None`` for the whole season (every roster week counts, rows in the fit
            order of ``nfl-sos-ratings``); a week to keep only the games before it, judged with
            the roster weeks before it, sorted by team, week, game and passer id.

    Returns:
        The columns of ``QB_GAME_SCHEMA``.

    """
    if before_week is None:
        listed_by_roster = pl.col("first_qb_roster_week").is_not_null()
        placed_by_roster = pl.col("first_roster_week").is_not_null()
        rows = passer_games
        order = ["team", "week", "game_id", "qb_name", "qb_id"]
    else:
        listed_by_roster = (pl.col("first_qb_roster_week") < before_week).fill_null(value=False)
        placed_by_roster = (pl.col("first_roster_week") < before_week).fill_null(value=False)
        rows = passer_games.filter(pl.col("week") < before_week)
        order = ["team", "week", "game_id", "qb_id"]
    quarterback = pl.col("players_file_qb") | listed_by_roster
    placed = quarterback | pl.col("players_file_placed") | placed_by_roster
    listed = pl.col("stats_list_qb") | quarterback
    attempts = (
        pl.when(listed & pl.col("official_attempts").is_not_null())
        .then(pl.col("official_attempts"))
        .otherwise(pl.col("pbp_attempts"))
    )
    passing_epa = (
        pl.when(listed & pl.col("official_passing_epa").is_not_null())
        .then(pl.col("official_passing_epa"))
        .otherwise(pl.col("pbp_passing_epa"))
    )
    return (
        rows.filter(quarterback | ~placed)
        .with_columns(
            attempts.alias("qb_attempts"),
            (passing_epa - pl.col("qb_spike_epa")).alias("qb_dropback_epa"),
        )
        .sort(order)
        .select(pl.col(name).cast(dtype) for name, dtype in QB_GAME_SCHEMA.items())
    )


__all__ = [
    "PASSER_GAME_SCHEMA",
    "SeasonSources",
    "passer_games_from_pbp",
    "quarterback_rows",
    "team_games_from_pbp",
]
