"""Adapter from the game logs ``nfl-sos-ratings`` published to this package's input rows.

That project writes one ``{season}_team_game_logs`` and one ``{season}_qb_game_logs`` Parquet file
per season. These functions turn a frame read from either into the rows of
:mod:`nfl_predictor.ratings.inputs`, in the row order that project's fits used, so a fit of the
adapted rows reproduces its published ratings bit for bit (the fits' floating-point sums depend on
row order):

- Team game logs are written sorted by team, week, and game, the order its season fit and its
  earlier-season fits read them in; they keep that order.
- Quarterback game logs are written sorted by passer, week, and game, but its quarterback fits
  read them sorted by team, week, game, and passer name; they are re-sorted to that order.

The quarterback rows' ``qb_dropback_epa`` is ``qb_passing_epa - qb_spike_epa`` (nflverse's official
passing EPA, which counts spikes, less the spikes' EPA), the numerator of that project's per-game
``qb_epa_per_dropback``.
"""

import polars as pl

from nfl_predictor.ratings.inputs import qb_game_rows, team_game_rows

# The order nfl-sos-ratings' quarterback fits read a season's passer-game rows in.
_QB_FIT_ORDER = ("team", "week", "game_id", "qb_name")


def team_games_from_published(team_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return the team-game rows of one published ``{season}_team_game_logs`` frame.

    Raises:
        ValueError: If an input column is missing, a key is null, or a row is duplicated.

    """
    return team_game_rows(team_game_logs)


def qb_games_from_published(qb_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return the quarterback-game rows of one published ``{season}_qb_game_logs`` frame.

    Raises:
        ValueError: If an input column is missing, a key is null, or a row is duplicated.

    """
    return qb_game_rows(
        qb_game_logs.with_columns(
            (pl.col("qb_passing_epa") - pl.col("qb_spike_epa")).alias("qb_dropback_epa")
        ).sort(list(_QB_FIT_ORDER), maintain_order=True)
    )


__all__ = ["qb_games_from_published", "team_games_from_published"]
