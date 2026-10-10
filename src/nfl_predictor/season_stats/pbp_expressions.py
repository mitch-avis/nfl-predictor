"""Shared Polars expressions used by the team and QB stat builders.

These definitions are correctness invariants: the scrimmage-snap definition
here is the single definition of an offensive snap for the whole project, and
the passer-rating formula the single definition of passer rating.
"""

import polars as pl


def _flag_expr(columns: list[str], column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing or the column is absent."""
    if column in columns:
        return pl.col(column).fill_null(0) > 0
    return pl.lit(value=False)


def dropback_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that flags dropbacks: pass attempts, sacks, and scrambles that stood.

    nflverse codes a scramble as a pass play (``rush`` is 0) and, from 2006 on, flags it with
    ``qb_dropback``; before 2006 it leaves ``qb_dropback`` at 0 on most scrambles. So a run play
    flagged ``qb_scramble`` also counts, which adds those scrambles and changes nothing from 2006
    on. A scramble a penalty wiped out (``play_type`` ``no_play``) is not a dropback in any season.
    """
    scramble = (
        _flag_expr(columns, "qb_scramble") & (pl.col("play_type") == "run").fill_null(value=False)
        if "play_type" in columns
        else pl.lit(value=False)
    )
    return _flag_expr(columns, "qb_dropback") | scramble


def scrimmage_snap_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that flags offensive scrimmage snaps in PBP data.

    A scrimmage snap is a dropback (:func:`dropback_expr`), rush, kneel, or spike.
    """
    return (
        dropback_expr(columns)
        | _flag_expr(columns, "rush")
        | _flag_expr(columns, "qb_kneel")
        | _flag_expr(columns, "qb_spike")
    )


def special_teams_play_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that flags special-teams plays in PBP data.

    nflverse marks kickoffs, punts, field goals, and extra points with ``special`` (older files
    name it ``special_teams_play``); a file with neither flags no play.
    """
    flag = "special" if "special" in columns else "special_teams_play"
    return value_expr(columns, flag) > 0


def value_expr(columns: list[str], column: str, default: float = 0) -> pl.Expr:
    """Return a null-safe column expression or a literal default when absent."""
    if column in columns:
        return pl.col(column).fill_null(default)
    return pl.lit(default)


def lost_fumble_team_expr(columns: list[str]) -> pl.Expr:
    """Return the team that lost the play's fumble, or null when no fumble was lost.

    nflverse's ``fumble_lost`` flags a play on which a fumble was lost, whichever team fumbled, and
    ``fumbled_1_team`` names the fumbling team. ``posteam`` is not always that team: on a punt it is
    the punting team, so a returner's muff the punting team recovers is a lost fumble on a
    punting-team row, and after an interception the intercepting team's fumble is one on the
    passing team's row. When ``fumbled_1_team`` is missing, the team that fumbles on such plays
    almost always stands in: the receiving team on a punt, the intercepting team after an
    interception, and the team with the ball otherwise.
    """
    defteam = pl.col("defteam") if "defteam" in columns else pl.lit(None, dtype=pl.String)
    usual = (
        pl.when(
            (value_expr(columns, "punt_attempt") > 0) | (value_expr(columns, "interception") > 0)
        )
        .then(defteam)
        .otherwise(pl.col("posteam"))
    )
    fumbler = (
        pl.col("fumbled_1_team") if "fumbled_1_team" in columns else pl.lit(None, dtype=pl.String)
    )
    return pl.when(value_expr(columns, "fumble_lost") > 0).then(pl.coalesce(fumbler, usual))


def giveaway_team_expr(columns: list[str]) -> pl.Expr:
    """Return the first team to give the ball away on the play, or null when neither did.

    An interception is the passing team's giveaway, even when the intercepting team fumbles the
    ball back; otherwise a lost fumble is the fumbling team's (``lost_fumble_team_expr``). Each
    play is one team's giveaway at most, so its EPA counts once.
    """
    return (
        pl.when(value_expr(columns, "interception") > 0)
        .then(pl.col("posteam"))
        .otherwise(lost_fumble_team_expr(columns))
    )


def passer_rating_from_rates(
    completion_rate: pl.Expr,
    yards_per_attempt: pl.Expr,
    touchdown_rate: pl.Expr,
    interception_rate: pl.Expr,
) -> pl.Expr:
    """Return the official NFL passer rating from its four per-attempt rates.

    Null when any rate is null (no attempts).
    """

    def _clamp(component: pl.Expr) -> pl.Expr:
        return component.clip(0.0, 2.375)

    a = _clamp((completion_rate - 0.3) * 5.0)
    b = _clamp((yards_per_attempt - 3.0) * 0.25)
    c = _clamp(touchdown_rate * 20.0)
    d = _clamp(2.375 - (interception_rate * 25.0))
    return (a + b + c + d) / 6.0 * 100.0
