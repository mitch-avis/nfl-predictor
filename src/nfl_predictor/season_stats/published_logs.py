"""Season-table inputs rebuilt from the game logs ``nfl-sos-ratings`` published.

The season rows and opponent profiles pool each rate over games through its hidden numerator and
denominator (``pooled_rates``), and the published game logs leave those parts out. This adapter
turns published logs back into the rows the season tables read, rebuilding the parts it can read
off the published columns, as the code that wrote them defines them:

- a rate whose numerator and denominator are both published columns (``team_stats.PER_SNAP_RATES``,
  ``team_stats_expanded.OFFENSE_RATIO_SPECS`` and ``GIVEAWAYS_PER_DRIVE_SPEC``), the build's
  working count of scrimmage snaps being the published ``offensive_snaps``;
- a rate of a whole-number total over a published count (explosive passes per dropback, points
  per drive): the published rate times the count rounds back to the total;
- the defensive mirror of each (``team_stats_expanded.DEFENSE_MIRROR_RENAMES`` and
  ``DEFENSE_MIRROR_PARTS_ONLY``), whose parts are the opponent's in the same game.

The adapter does not rebuild the other rates' parts: a numerator that is neither published nor a
whole number (the dropbacks' EPA behind EPA per dropback), or a denominator the logs do not carry
(the passes with a completion probability behind the completion percentage over expected). A
season row built from these inputs averages such a rate per game instead, so it is a check of that
rate only where :func:`rebuilt_team_rates` lists it. The quarterback rows rebuild every pooled
rate from published totals (``qb_stats.QB_RATES``) except the completion percentage over expected,
whose parts the logs drop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.season_stats.game_logs import (
    QB_GAME_ORDER,
    QB_LOG_CONTEXT_COLUMNS,
    TEAM_GAME_ORDER,
)
from nfl_predictor.season_stats.pooled_rates import denominator_column, numerator_column
from nfl_predictor.season_stats.team_stats import PER_SNAP_RATES
from nfl_predictor.season_stats.team_stats_expanded import (
    DEFENSE_MIRROR_PARTS_ONLY,
    DEFENSE_MIRROR_RENAMES,
    GIVEAWAYS_PER_DRIVE_SPEC,
    OFFENSE_RATIO_SPECS,
)

if TYPE_CHECKING:
    from collections.abc import Collection

_GAME_KEYS = ("game_id", "team", "opponent_team")

# The build's working count of a team's scrimmage snaps counts the same plays as its published
# offensive snaps (``team_stats.compute_team_snap_counts_from_pbp``).
_PUBLISHED_COUNTS = {"aux_off_snaps": "offensive_snaps"}

# The build's working totals that are whole numbers (counts of plays and events, sums of whole
# yards) and that the logs drop: a published rate of one times its denominator rounds back to it.
_WHOLE_NUMBER_TOTALS = frozenset(
    {
        "aux_sacks",
        "aux_pass_yards",
        "aux_pass_tds",
        "aux_interceptions",
        "aux_explosive_passes",
        "aux_deep_attempts",
        "aux_rush_yards",
        "aux_explosive_rushes",
        "aux_stuffed_rushes",
        "aux_total_yards",
        "aux_early_dropbacks",
        "aux_fourth_short_go",
        "aux_presnap_penalties",
        "aux_havoc_events",
    }
)

# Rates of whole-number totals that no ratio spec lists, each with the published count it divides
# by: sums of counts per snap or dropback, and means over plays, series, or drives of a value that
# is never missing (a success or shotgun flag, a series' conversion, a drive's yards, plays, first
# downs, or points), whose count is that of the plays, series, or drives.
_WHOLE_NUMBER_RATES: tuple[tuple[str, str], ...] = (
    ("explosive_play_rate", "offensive_snaps"),
    ("aux_pressure_rate", "dropbacks"),
    ("success_rate", "offensive_snaps"),
    ("pass_success_rate", "dropbacks"),
    ("rush_success_rate", "designed_carries"),
    ("shotgun_rate", "offensive_snaps"),
    ("series_conversion_rate", "series"),
    ("yards_per_drive", "drives"),
    ("plays_per_drive", "drives"),
    ("first_downs_per_drive", "drives"),
    ("points_per_drive", "drives"),
    ("points_per_red_zone_trip", "red_zone_trips"),
)

# A rate's parts in the published columns: (numerator, denominator, rate). A numerator of None is
# a whole-number total, the rate times the denominator rounded.
type _Spec = tuple[str | None, str, str]

_SPECS: tuple[_Spec, ...] = (
    *(
        (
            None if numerator in _WHOLE_NUMBER_TOTALS else numerator,
            _PUBLISHED_COUNTS.get(denominator, denominator),
            rate,
        )
        for numerator, denominator, rate in (
            *PER_SNAP_RATES,
            *OFFENSE_RATIO_SPECS,
            GIVEAWAYS_PER_DRIVE_SPEC,
        )
    ),
    *((None, denominator, rate) for rate, denominator in _WHOLE_NUMBER_RATES),
)
_SPEC_BY_RATE = {spec[2]: spec for spec in _SPECS}
_MIRRORS = {**DEFENSE_MIRROR_RENAMES, **DEFENSE_MIRROR_PARTS_ONLY}
_PARTS_ONLY_MIRRORS = frozenset(DEFENSE_MIRROR_PARTS_ONLY.values())


def _rounded_total(rate: str, denominator: str) -> pl.Expr:
    """Return a rate times its denominator, rounded: the whole-number total it divides."""
    return (pl.col(rate) * pl.col(denominator)).round(0)


def _own_specs(columns: Collection[str]) -> list[_Spec]:
    """Return the specs a team's own published row determines, the rate itself published."""
    available = set(columns)
    return [
        (numerator, denominator, rate)
        for numerator, denominator, rate in _SPECS
        if {rate, denominator} <= available and (numerator is None or numerator in available)
    ]


def _mirror_specs(columns: Collection[str]) -> list[tuple[str, _Spec]]:
    """Return (mirrored rate, the mirrored source's spec) for each mirror the logs determine.

    A mirror's parts are its source's parts on the opponent's row in the same game. A whole-number
    numerator is the opponent's published source rate, or else the team's published mirror, times
    the opponent's denominator, rounded. A mirror only the parts carry
    (``DEFENSE_MIRROR_PARTS_ONLY``) needs no published column of its own.
    """
    available = set(columns)
    mirrors: list[tuple[str, _Spec]] = []
    for source, target in _MIRRORS.items():
        spec = _SPEC_BY_RATE.get(source)
        if spec is None or spec[1] not in available:
            continue
        numerator = spec[0]
        published_target = target in available
        if not (published_target or target in _PARTS_ONLY_MIRRORS):
            continue
        if numerator is not None:
            known = numerator in available
        else:
            known = source in available or published_target
        if known:
            mirrors.append((target, spec))
    return mirrors


def rebuilt_team_rates(columns: Collection[str]) -> tuple[str, ...]:
    """Return the rates whose hidden parts :func:`team_stat_rows_from_published` rebuilds."""
    return (
        *(rate for _, _, rate in _own_specs(columns)),
        *(target for target, _ in _mirror_specs(columns)),
    )


def _own_parts(columns: Collection[str]) -> list[pl.Expr]:
    """Return the numerator and denominator of every rate a team's own row determines."""
    return [
        expr
        for numerator, denominator, rate in _own_specs(columns)
        for expr in (
            (_rounded_total(rate, denominator) if numerator is None else pl.col(numerator)).alias(
                numerator_column(rate)
            ),
            pl.col(denominator).alias(denominator_column(rate)),
        )
    ]


def _opponent_parts(
    logs: pl.DataFrame, mirrors: list[tuple[str, _Spec]]
) -> tuple[pl.DataFrame, list[str]]:
    """Return each mirror's parts from the opponent's row, keyed to the team's row.

    Also returns the mirrors whose numerator the team's own published mirror gives instead, rounded
    once the opponent's denominator is joined.
    """
    exprs: list[pl.Expr] = []
    from_own_row: list[str] = []
    for target, (numerator, denominator, source) in mirrors:
        exprs.append(pl.col(denominator).alias(denominator_column(target)))
        if numerator is not None:
            exprs.append(pl.col(numerator).alias(numerator_column(target)))
        elif source in logs.columns:
            exprs.append(_rounded_total(source, denominator).alias(numerator_column(target)))
        else:
            from_own_row.append(target)
    frame = logs.select(
        "game_id",
        pl.col("opponent_team").alias("team"),
        pl.col("team").alias("opponent_team"),
        *exprs,
    )
    return frame, from_own_row


def team_stat_rows_from_published(logs: pl.DataFrame) -> pl.DataFrame:
    """Return published team game logs as season-table input rows, recoverable parts rebuilt.

    Args:
        logs: One season's published team game logs (``{season}_team_game_logs``).

    Returns:
        ``logs`` with the hidden numerator and denominator of every rate of
        :func:`rebuilt_team_rates`, in the builders' row order (``game_logs.TEAM_GAME_ORDER``)

    """
    # Sorted as the builders sort, not only ordered so: Polars sums a group's floats in another
    # order when the frame is flagged sorted by the group key, as the builders' frames are.
    games = logs.with_columns(_own_parts(logs.columns)).sort(TEAM_GAME_ORDER)
    mirrors = _mirror_specs(logs.columns)
    if not mirrors:
        return games
    opponent, from_own_row = _opponent_parts(logs, mirrors)
    joined = games.join(opponent, on=list(_GAME_KEYS), how="left", maintain_order="left")
    return joined.with_columns(
        _rounded_total(target, denominator_column(target)).alias(numerator_column(target))
        for target in from_own_row
    ).sort(TEAM_GAME_ORDER)


def qb_stat_rows_from_published(logs: pl.DataFrame) -> pl.DataFrame:
    """Return published QB game logs as the season tables read passer-game rows.

    The passer's team is ``team_abbr``, and the game's opponent, venue, and result, which the logs
    join from the team rows (``game_logs.QB_LOG_CONTEXT_COLUMNS``), are left out. The rows are in
    the builders' order (``game_logs.QB_GAME_ORDER``).
    """
    return (
        logs.drop(QB_LOG_CONTEXT_COLUMNS, strict=False)
        .rename({"team": "team_abbr"})
        .sort(QB_GAME_ORDER)
    )


__all__ = [
    "qb_stat_rows_from_published",
    "rebuilt_team_rates",
    "team_stat_rows_from_published",
]
