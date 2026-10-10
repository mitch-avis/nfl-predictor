"""The rows every rating fit consumes: one per team-game and one per quarterback-game.

Both are regular-season rows only: playoff games never feed a rating. The fits read nothing else,
so an adapter from any play-by-play source only has to produce these columns.

Team-game rows (``TEAM_GAME_SCHEMA``), one per team and game, the team on offense:

- ``game_id``: the game, which is also the unit of the cross-validation folds and the bootstrap.
- ``week``: the regular-season week of the game.
- ``team``: the offense, and the possession unit on special teams.
- ``opponent_team``: the defense, and the coverage unit on special teams.
- ``is_home``: true at home, false away, null at a neutral site (no home field).
- ``offensive_snaps``: the team's scrimmage plays with both teams known: dropbacks (pass attempts,
  sacks, and scrambles; a run play flagged as a scramble counts as a dropback in every season,
  because nflverse leaves most scrambles before 2006 unflagged), rushes, kneel-downs, and spikes.
- ``offensive_epa``: the expected points added on those plays.
- ``st_plays``: special-teams plays (kickoffs, punts, field goals, extra points) with the team in
  possession: kicking on punts, field goals, and extra points, receiving on kickoffs.
- ``st_epa``: the expected points added on those plays, from the possession team's side.
- ``point_margin``: the team's points minus its opponent's, for the score-based SRS reference.

Quarterback-game rows (``QB_GAME_SCHEMA``), one per passer and game:

- ``game_id``, ``week``, ``team`` (the passer's team), ``opponent_team`` (the defense he faced),
  ``is_home``: as for team-games.
- ``qb_id``: the passer's player id; ``qb_name``: his display name.
- ``qb_dropbacks``: his pass attempts and sacks; spikes and kneel-downs are not dropbacks. A row
  with none still counts as a game played (a quarterback on the field without a dropback).
- ``qb_dropback_epa``: the expected points added on those dropbacks, spikes left out.
- ``qb_attempts``: his pass attempts, for the rankings qualifier.

The checks refuse a null ``game_id``, ``week``, ``team``, ``opponent_team`` or ``qb_id``, an empty
or blank text id or team (play-by-play has rows whose ``posteam`` is an empty string), and two rows
for one team or passer in a game. Row order is kept: the fits sum over rows in the order given, so
the same rows in another order can move a rating in its last bits (see
:mod:`nfl_predictor.ratings.ridge`).
"""

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

TEAM_GAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "week": pl.Int64(),
    "team": pl.String(),
    "opponent_team": pl.String(),
    "is_home": pl.Boolean(),
    "offensive_snaps": pl.Int64(),
    "offensive_epa": pl.Float64(),
    "st_plays": pl.Int64(),
    "st_epa": pl.Float64(),
    "point_margin": pl.Int64(),
}
QB_GAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "week": pl.Int64(),
    "team": pl.String(),
    "opponent_team": pl.String(),
    "is_home": pl.Boolean(),
    "qb_id": pl.String(),
    "qb_name": pl.String(),
    "qb_dropbacks": pl.Int64(),
    "qb_dropback_epa": pl.Float64(),
    "qb_attempts": pl.Int64(),
}
_TEAM_GAME_KEY = ("game_id", "team")
_QB_GAME_KEY = ("game_id", "qb_id")
_REQUIRED_VALUES = ("game_id", "week", "team", "opponent_team")


def _conformed(
    frame: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
    key: tuple[str, ...],
    kind: str,
) -> pl.DataFrame:
    """Return ``frame`` reduced to ``schema``'s columns and types, after checking its rows.

    Raises:
        ValueError: If a schema column is missing, a key or identity value is null or (for text)
            empty, or two rows share a key.
        polars.exceptions.InvalidOperationError: If a value does not convert to its column type.

    """
    missing = sorted(set(schema) - set(frame.columns))
    if missing:
        msg = f"{kind} rows are missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    rows = frame.select(pl.col(name).cast(dtype, strict=True) for name, dtype in schema.items())
    for column in dict.fromkeys((*_REQUIRED_VALUES, *key)):
        values = rows.get_column(column)
        nulls = values.null_count()
        if nulls:
            msg = f"{kind} rows have {nulls} null {column} values"
            raise ValueError(msg)
        if values.dtype == pl.String:
            blank = int((values.str.strip_chars() == "").sum())
            if blank:
                msg = f"{kind} rows have {blank} empty {column} values"
                raise ValueError(msg)
    duplicated = rows.select(key).is_duplicated()
    if duplicated.any():
        examples = rows.filter(duplicated).select(key).unique(maintain_order=True).head(3).rows()
        msg = f"{kind} rows have more than one row for {', '.join(key)}: {examples}"
        raise ValueError(msg)
    return rows


def team_game_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame``'s team-game rows with exactly ``TEAM_GAME_SCHEMA``'s columns and types.

    Rows keep their order; other columns are dropped.

    Raises:
        ValueError: If a column is missing, ``game_id``, ``week``, ``team``, or
            ``opponent_team`` is null, an id or team is empty, or a team has two rows for one game.

    """
    return _conformed(frame, TEAM_GAME_SCHEMA, _TEAM_GAME_KEY, "team-game")


def qb_game_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame``'s quarterback-game rows with exactly ``QB_GAME_SCHEMA``'s columns and types.

    Rows keep their order; other columns are dropped.

    Raises:
        ValueError: If a column is missing, ``game_id``, ``week``, ``team``, ``opponent_team``, or
            ``qb_id`` is null, an id or team is empty, or a passer has two rows for one game.

    """
    return _conformed(frame, QB_GAME_SCHEMA, _QB_GAME_KEY, "quarterback-game")


__all__ = ["QB_GAME_SCHEMA", "TEAM_GAME_SCHEMA", "qb_game_rows", "team_game_rows"]
