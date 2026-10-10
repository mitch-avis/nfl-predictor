"""The starting quarterback of a completed game, from its play-by-play.

Every quarterback feature keys on one starter definition, ported from ``nfl-sos-ratings`` (its
quarterback protocol's starter, ``first_qb_snaps`` in its ``qb_protocol_checks.py``): the
starter of a team in a game is the quarterback on the team's first first-quarter snap, the first
play of the first quarter, by ``play_id``, whose passer or rusher has a quarterback-game row for
that team in that game. On a play with both, the passer comes first.

A quarterback-game row (``qb_stats.aggregate_qb_game_stats``) means at least one dropback is
credited to the player in the game, as the passer or as the rusher of a scramble, and the player
counts only when he is a quarterback that season or no identity source lists him: the season
identity (``player_identity.qb_identity_crosswalk``: the players file or any regular-season week
of the weekly roster lists him at quarterback) leaves out a running back or receiver who threw.
So the first snap skips a running back's carry, a gadget passer, and a passer whose only pass was
a two-point try, and it picks a quarterback who ran on the first snap. A team-game without such
a snap in the first quarter has no play-by-play starter.

The starter is known at kickoff, not before: a completed game is rated for the quarterback who
started it. A game without play-by-play (an upcoming game, and every postseason game, since the
cache holds the regular season) takes the starter the ETL names for it (``away_qb``/``home_qb``).
nflverse's schedule columns ``away_qb_id``/``home_qb_id`` are not read: ``nfl-sos-ratings`` found
them naming a player without a quarterback row in 42 team-games of 2006-2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import pbp, qb_stats

if TYPE_CHECKING:
    from collections.abc import Mapping

POSITIONS_SCHEMA = pl.Schema(
    {"season": pl.Int64(), "qb_id": pl.String(), "qb_position": pl.String()}
)
STARTERS_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "season": pl.Int64(),
        "week": pl.Int64(),
        "team_abbr": pl.String(),
        "qb_id": pl.String(),
    }
)
_FIRST_QUARTER = 1
# The order of a play's candidates: the passer, then the rusher.
_CANDIDATE_COLUMNS = ("passer_player_id", "rusher_player_id")
_REQUIRED_COLUMNS = frozenset({"game_id", "season", "week", "posteam", "qtr", "play_id"})


@dataclass(frozen=True, eq=False)
class QuarterbackHistory:
    """The play-by-play inputs every quarterback feature reads, built once per run.

    Attributes:
        plays: Regular-season play-by-play from the first season through the run's last.
        qb_games: ``qb_stats.aggregate_qb_game_stats`` over ``plays``.
        positions: Each season's positions (``POSITIONS_SCHEMA``).
        starters: ``play_by_play_starters`` over ``plays``.

    """

    plays: pl.DataFrame
    qb_games: pl.DataFrame
    positions: pl.DataFrame
    starters: pl.DataFrame


def quarterback_history(plays: pl.DataFrame, positions: pl.DataFrame) -> QuarterbackHistory:
    """Return the quarterback games and the play-by-play starters of ``plays``."""
    qb_games = qb_stats.aggregate_qb_game_stats(plays)
    starters = play_by_play_starters(plays, qb_games=qb_games, positions=positions)
    return QuarterbackHistory(plays, qb_games, positions, starters)


def season_positions(crosswalks: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
    """Return each season's position of every identified player, from the season crosswalks.

    Args:
        crosswalks: ``player_identity.qb_identity_crosswalk`` output (``qb_id``,
            ``qb_position``) by season.

    Returns:
        ``POSITIONS_SCHEMA`` rows, in season order and each crosswalk's row order.

    """
    frames = [
        crosswalk.select(
            pl.lit(season, dtype=pl.Int64).alias("season"),
            pl.col("qb_id").cast(pl.String),
            pl.col("qb_position").cast(pl.String),
        )
        for season, crosswalk in sorted(crosswalks.items())
    ]
    if not frames:
        return pl.DataFrame(schema=POSITIONS_SCHEMA)
    return pl.concat(frames).cast(POSITIONS_SCHEMA)


def play_by_play_starters(
    plays: pl.DataFrame, *, qb_games: pl.DataFrame, positions: pl.DataFrame
) -> pl.DataFrame:
    """Return the starter of every team-game the play-by-play names one for.

    Args:
        plays: Play-by-play rows (``game_id``, ``season``, ``week``, ``posteam``, ``qtr``,
            ``play_id`` and the passer and rusher ids); postseason plays and plays without
            both teams are left out (``pbp.regular_season_plays``).
        qb_games: ``qb_stats.aggregate_qb_game_stats`` output for the same plays.
        positions: ``POSITIONS_SCHEMA`` rows; a player listed at a position other than
            ``constants.QB_POSITION`` in a season is no candidate that season.

    Returns:
        ``STARTERS_SCHEMA`` rows, one per team-game with a starter, sorted by game and team. The
        rule is in the module docstring.

    """
    missing = sorted(_REQUIRED_COLUMNS - set(plays.columns))
    if missing:
        if plays.height:
            log.warning("Play-by-play lacks %s; no game has a play-by-play starter", missing)
        return pl.DataFrame(schema=STARTERS_SCHEMA)
    regular = pbp.regular_season_plays(plays).filter(
        (pl.col("qtr").cast(pl.Int64, strict=False) == _FIRST_QUARTER).fill_null(value=False)
    )
    candidates = pl.concat(
        [
            regular.select(
                pl.col("game_id").cast(pl.String),
                pl.col("season").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.col("posteam").cast(pl.String).alias("team_abbr"),
                pl.col("play_id").cast(pl.Float64),
                pl.lit(order, dtype=pl.Int64).alias("_order"),
                (
                    pl.col(column).cast(pl.String)
                    if column in regular.columns
                    else pl.lit(None, dtype=pl.String)
                ).alias("qb_id"),
            )
            for order, column in enumerate(_CANDIDATE_COLUMNS)
        ]
    ).drop_nulls("qb_id")
    with_rows = qb_games.select(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team_abbr").cast(pl.String),
        pl.col("qb_id").cast(pl.String),
    ).unique(maintain_order=True)
    elsewhere = positions.filter(
        pl.col("qb_position").is_not_null() & (pl.col("qb_position") != constants.QB_POSITION)
    ).select("season", "qb_id")
    return (
        candidates.join(with_rows, on=["season", "week", "team_abbr", "qb_id"], how="semi")
        .join(elsewhere, on=["season", "qb_id"], how="anti")
        .sort("game_id", "team_abbr", "play_id", "_order")
        .unique(subset=["game_id", "team_abbr"], keep="first", maintain_order=True)
        .select(STARTERS_SCHEMA.names())
        .cast(STARTERS_SCHEMA)
    )
