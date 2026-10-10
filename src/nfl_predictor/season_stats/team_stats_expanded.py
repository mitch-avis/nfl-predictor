"""Expanded team metrics derived from play-by-play data.

Every published column here matches a metric-registry entry; formulas follow the
stats catalog beside this package (``stats-catalog.md``). The frame is keyed by (game keys,
team, opponent_team)
and is joined onto the core weekly team frame in ``team_stats``.

Defensive mirrors are built by mirroring each offense row onto its opponent
(what the offense produced is exactly what the defense allowed), so both
sides always agree by construction.

Each rate carries its numerator and denominator as hidden columns, mirrored
with it, so rows for several games pool it (``pooled_rates``); a rate built
from other rates is listed in ``TEAM_FORMULA_RATES`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.season_stats.pbp_expressions import (
    dropback_expr,
    giveaway_team_expr,
    lost_fumble_team_expr,
    passer_rating_from_rates,
    scrimmage_snap_expr,
    special_teams_play_expr,
    value_expr,
)
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    expression_ratio_with_parts,
    is_rate_part,
    mean_with_parts,
    numerator_column,
    part_rate,
    ratio_with_parts,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_GROUP_KEY_CANDIDATES = ("game_id", "season", "season_type", "week")

_PRESNAP_PENALTY_TYPES = ("False Start", "Delay of Game")
_THIRD_DOWN = 3
_FOURTH_DOWN = 4
# First and second down are the "early" downs.
_LAST_EARLY_DOWN = 2
# A fourth down with this many yards to go or fewer is short yardage.
_SHORT_YARDAGE = 2
# Explosive-play thresholds: completions of 20+ yards and rushes of 10+ yards.
_EXPLOSIVE_PASS_YARDS = 20
_EXPLOSIVE_RUSH_YARDS = 10
# Drives that start at or inside the offense's own 25 face a long field.
_LONG_FIELD_START_YARDLINE = 25
# nflverse play-by-play names the intended receiver on most incomplete passes (the rest are
# throwaways, spikes, and batted balls), except in 2003-2008, where it names almost none. Below
# this share of incompletions with a named receiver, a season's targets are unknown.
_MIN_INCOMPLETION_RECEIVER_SHARE = 0.5
_RECEIVER_COLUMNS = ("receiver_player_id", "receiver_player_name")
# nflverse records up to two fumblers per play.
_FUMBLER_COLUMNS = ("fumbled_1_player_id", "fumbled_2_player_id")

# Offense-row column -> opponent's defense-row column.
DEFENSE_MIRROR_RENAMES = {
    "attempts": "attempts_faced",
    "completions": "completions_allowed",
    "completion_pct": "completion_pct_allowed",
    "targets": "targets_faced",
    "catch_rate": "catch_rate_allowed",
    "net_passing_yards": "net_passing_yards_allowed",
    "passing_air_yards": "air_yards_allowed",
    "passing_yards_after_catch": "yac_allowed",
    "epa_per_dropback": "epa_per_dropback_allowed",
    "adjusted_net_yards_per_attempt": "any_a_allowed",
    "explosive_pass_rate": "explosive_pass_rate_allowed",
    "team_passer_rating": "team_passer_rating_allowed",
    "deep_attempt_rate": "deep_attempt_rate_faced",
    "sack_yards_lost": "def_sack_yards",
    "sack_rate_per_dropback": "def_sack_rate_per_dropback",
    "aux_pressure_rate": "qb_pressure_events_rate",
    "carries": "carries_faced",
    "yards_per_carry": "yards_per_carry_allowed",
    "rush_success_rate": "rush_success_rate_allowed",
    "explosive_rush_rate": "explosive_rush_rate_allowed",
    "stuffed_run_rate": "stuff_rate",
    "success_rate": "success_rate_allowed",
    "explosive_play_rate": "explosive_play_rate_allowed",
    "epa_per_offensive_snap": "epa_per_defensive_snap_allowed",
    "yards_per_offensive_snap": "yards_per_defensive_snap_allowed",
    "first_downs": "first_downs_allowed",
    "first_downs_penalty": "penalty_first_downs_allowed",
    "third_down_pct": "third_down_pct_allowed",
    "fourth_down_pct": "fourth_down_pct_allowed",
    "series_conversion_rate": "series_conversion_rate_allowed",
    "three_and_out_rate": "three_and_outs_forced_rate",
    "score_pct_per_drive": "score_pct_per_drive_allowed",
    "punt_pct_per_drive": "punts_forced_pct",
    "points_per_drive": "points_per_drive_allowed",
    "red_zone_td_pct": "red_zone_td_pct_allowed",
    "goal_to_go_td_pct": "goal_to_go_td_pct_allowed",
    "avg_starting_field_position": "avg_starting_field_position_allowed",
    "two_pt_conversion_rate": "two_pt_conversion_rate_allowed",
    "giveaways": "takeaways",
    "fumbles_lost": "fumble_recovery_opp",
    "giveaway_rate_per_offensive_snap": "takeaway_rate_per_defensive_snap",
    "giveaways_per_drive": "takeaways_per_drive",
    "aux_int_return_yards": "def_interception_yards",
    "aux_def_tds": "def_tds",
    "aux_fumble_recovery_tds": "fumble_recovery_tds",
    "aux_def_2pt": "defensive_2pt_conversions",
    "aux_havoc_rate": "havoc_rate",
    "aux_def_pen_count": "defensive_penalties",
    "aux_def_pen_yards": "defensive_penalty_yards",
    "aux_dpi": "defensive_pass_interference",
    "aux_total_yards": "aux_total_yards_allowed",
}
# Offense rates without a published defensive mirror whose mirror still feeds a published rate
# over several games (passer rating allowed, ``TEAM_FORMULA_RATES``): only their numerator and
# denominator are mirrored.
DEFENSE_MIRROR_PARTS_ONLY = {
    "yards_per_attempt": "yards_per_attempt_allowed",
    "passing_td_rate_per_attempt": "passing_td_rate_per_attempt_allowed",
    "int_rate_per_attempt": "int_rate_per_attempt_allowed",
}


# Offense ratios of game totals: (numerator, denominator, rate). Each carries its numerator
# and denominator as hidden parts, so rows for several games pool it (``pooled_rates``).
OFFENSE_RATIO_SPECS: tuple[tuple[str, str, str], ...] = (
    ("completions", "attempts", "completion_pct"),
    ("completions", "targets", "catch_rate"),
    ("aux_sacks", "dropbacks", "sack_rate_per_dropback"),
    ("passing_air_yards", "attempts", "air_yards_per_attempt"),
    ("passing_yards_after_catch", "completions", "yac_per_completion"),
    ("aux_pass_yards", "attempts", "yards_per_attempt"),
    ("aux_pass_yards", "dropbacks", "yards_per_dropback"),
    ("aux_pass_epa", "dropbacks", "epa_per_dropback"),
    ("aux_pass_tds", "attempts", "passing_td_rate_per_attempt"),
    ("aux_interceptions", "attempts", "int_rate_per_attempt"),
    ("aux_explosive_passes", "dropbacks", "explosive_pass_rate"),
    ("aux_deep_attempts", "attempts", "deep_attempt_rate"),
    ("aux_rush_yards", "carries", "yards_per_carry"),
    ("aux_rush_epa", "carries", "epa_per_carry"),
    ("aux_explosive_rushes", "carries", "explosive_rush_rate"),
    ("aux_stuffed_rushes", "aux_carries_without_kneels", "stuffed_run_rate"),
    ("offensive_epa", "aux_off_snaps", "epa_per_offensive_snap"),
    ("aux_total_yards", "aux_off_snaps", "yards_per_offensive_snap"),
    ("dropbacks", "aux_off_snaps", "pass_rate"),
    ("aux_early_dropbacks", "aux_early_snaps", "early_down_pass_rate"),
    ("third_down_conversions", "third_down_attempts", "third_down_pct"),
    ("fourth_down_conversions", "fourth_down_attempts", "fourth_down_pct"),
    ("fourth_down_attempts", "aux_fourth_downs_faced", "fourth_down_go_rate"),
    ("aux_fourth_short_go", "aux_fourth_short_faced", "fourth_down_aggressiveness"),
    ("two_pt_conversions", "two_pt_attempts", "two_pt_conversion_rate"),
    ("giveaways", "aux_off_snaps", "giveaway_rate_per_offensive_snap"),
    ("offensive_penalties", "aux_off_snaps", "penalty_rate_per_offensive_snap"),
    ("aux_presnap_penalties", "aux_off_snaps", "presnap_penalty_rate"),
    ("aux_havoc_events", "aux_off_snaps", "aux_havoc_rate"),
)
# Giveaways per drive, built once the drive totals are joined.
GIVEAWAYS_PER_DRIVE_SPEC = ("giveaways", "drives", "giveaways_per_drive")


def compute_expanded_team_game_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Derive the expanded team metric surface, one row per team-game.

    Expects play-by-play as ``sources.prepare_pbp`` leaves it, every team code normalized: the
    penalty, touchdown, and drive-start credits compare ``penalty_team``, ``td_team``, and the
    team in ``drive_start_yard_line`` with ``posteam`` and ``defteam``.
    """
    if pbp_df.is_empty() or not {"posteam", "defteam"}.issubset(pbp_df.columns):
        return pl.DataFrame(schema={"team": pl.String, "opponent_team": pl.String})

    keys = [key for key in _GROUP_KEY_CANDIDATES if key in pbp_df.columns]
    plays = pbp_df.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    if plays.is_empty():
        return pl.DataFrame(schema={"team": pl.String, "opponent_team": pl.String})

    frame = _aggregate_play_stats(plays, keys)
    for extra in (
        _aggregate_series_stats(plays, keys),
        _aggregate_drive_stats(plays, keys),
    ):
        if extra is not None:
            frame = frame.join(extra, on=[*keys, "team"], how="left")

    frame = _add_offense_ratios(frame)
    frame = _join_defense_mirrors(frame, keys)
    frame = _join_committed_penalties(frame, plays, keys)
    frame = _join_touchdown_totals(frame, plays, keys)
    frame = _add_cross_side_margins(frame)

    aux_columns = [column for column in frame.columns if _is_auxiliary(column)]
    sort_keys = [key for key in ("team", "week", "game_id") if key in frame.columns]
    return frame.drop(aux_columns).sort(sort_keys)


def _is_auxiliary(column: str) -> bool:
    """Return whether ``column`` is a working column (``aux_``) or a rate part of one."""
    return (part_rate(column) if is_rate_part(column) else column).startswith("aux_")


def _aggregate_play_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Aggregate play-level counts per team-game (offense perspective)."""
    columns = plays.columns
    scrimmage = scrimmage_snap_expr(columns)
    is_special = special_teams_play_expr(columns)
    is_pass_attempt = value_expr(columns, "pass_attempt") > 0
    is_two_point = value_expr(columns, "two_point_attempt") > 0
    is_sack = value_expr(columns, "sack") > 0
    is_complete = value_expr(columns, "complete_pass") > 0
    is_dropback = dropback_expr(columns)
    is_rush_attempt = value_expr(columns, "rush_attempt") > 0
    # Carries (rush attempts that stood, not two-point tries) split into scrambles, kneel-downs,
    # and designed runs. nflverse keeps the scramble and run flags on plays a penalty wiped out,
    # so the split starts from rush attempts.
    is_carry = is_rush_attempt & ~is_two_point
    is_scramble = is_carry & (value_expr(columns, "qb_scramble") > 0)
    is_kneel = is_carry & (value_expr(columns, "qb_kneel") > 0)
    is_designed_run = is_carry & ~is_scramble & ~is_kneel
    is_interception = value_expr(columns, "interception") > 0
    # A lost fumble counts against the team with the ball only when it fumbled
    # (``lost_fumble_team_expr``): a returner's muff or an intercepting defender's fumble does not.
    lost_by_offense = (lost_fumble_team_expr(columns) == pl.col("posteam")).fill_null(value=False)
    giveaway_team = giveaway_team_expr(columns)
    yards = value_expr(columns, "yards_gained", 0.0)
    two_pt_success = (
        pl.col("two_point_conv_result") == "success"
        if "two_point_conv_result" in columns
        else pl.lit(False)
    )
    td_team = pl.col("td_team") if "td_team" in columns else pl.lit(None, dtype=pl.String)
    is_touchdown = value_expr(columns, "touchdown") > 0
    penalty_team = (
        pl.col("penalty_team") if "penalty_team" in columns else pl.lit(None, dtype=pl.String)
    )
    penalty_type = (
        pl.col("penalty_type") if "penalty_type" in columns else pl.lit(None, dtype=pl.String)
    )
    is_penalty = value_expr(columns, "penalty") > 0
    down = value_expr(columns, "down", 0)
    ydstogo = value_expr(columns, "ydstogo", 0)
    is_go_try = (
        value_expr(columns, "fourth_down_converted") + value_expr(columns, "fourth_down_failed")
    ) > 0
    fourth_down_faced = (down == _FOURTH_DOWN) & (
        scrimmage
        | (value_expr(columns, "punt_attempt") > 0)
        | (value_expr(columns, "field_goal_attempt") > 0)
    )
    xyac = (
        pl.col("xyac_mean_yardage")
        if "xyac_mean_yardage" in columns
        else pl.lit(None, dtype=pl.Float64)
    )
    yac = (
        pl.col("yards_after_catch")
        if "yards_after_catch" in columns
        else pl.lit(None, dtype=pl.Float64)
    )
    receiver_fumbled = _receiver_fumbled_expr(columns)
    # A season without the no-huddle flag leaves the rate's parts empty, so it pools to null.
    no_huddle = (
        value_expr(columns, "no_huddle", 0)
        if _is_charted(plays, "no_huddle")
        else pl.lit(None, dtype=pl.Float64)
    )

    def _count(condition: pl.Expr, name: str) -> pl.Expr:
        return condition.cast(pl.Int64).sum().alias(name)

    def _if_charted(field: str, expr: pl.Expr, dtype: type[pl.DataType]) -> pl.Expr:
        return _blank_unless_charted(plays, field, expr, dtype)

    # Targets are official attempts thrown to a named receiver (not throwaways or spikes).
    receiver_named = _target_receiver_expr(plays)
    targets = (
        pl.lit(None, dtype=pl.Int64).alias("targets")
        if receiver_named is None
        else _count(is_pass_attempt & ~is_sack & ~is_two_point & receiver_named, "targets")
    )

    return (
        plays.group_by([*keys, "posteam", "defteam"])
        .agg(
            # Passing volume (official attempts exclude sacks and two-point tries).
            _count(is_pass_attempt & ~is_sack & ~is_two_point, "attempts"),
            _count(is_complete & ~is_two_point, "completions"),
            targets,
            _count(is_dropback, "dropbacks"),
            (-yards).filter(is_sack).sum().fill_null(0.0).alias("sack_yards_lost"),
            _count(is_scramble, "scrambles"),
            value_expr(columns, "rushing_yards", 0.0)
            .filter(is_scramble)
            .sum()
            .fill_null(0.0)
            .alias("scramble_yards"),
            _if_charted(
                "air_yards",
                value_expr(columns, "air_yards", 0.0)
                .filter(is_pass_attempt & ~is_two_point)
                .sum()
                .fill_null(0.0)
                .alias("passing_air_yards"),
                pl.Float64,
            ),
            _if_charted(
                "yards_after_catch",
                value_expr(columns, "yards_after_catch", 0.0)
                .filter(is_complete)
                .sum()
                .fill_null(0.0)
                .alias("passing_yards_after_catch"),
                pl.Float64,
            ),
            value_expr(columns, "passing_yards", 0.0)
            .filter(is_complete)
            .max()
            .alias("longest_pass"),
            _count((value_expr(columns, "fumble") > 0) & is_sack, "sack_fumbles"),
            _count(is_pass_attempt & two_pt_success & is_two_point, "passing_2pt_conversions"),
            _if_charted(
                "air_epa",
                value_expr(columns, "air_epa", 0.0)
                .filter(is_pass_attempt)
                .sum()
                .fill_null(0.0)
                .alias("air_epa_total"),
                pl.Float64,
            ),
            _if_charted(
                "yac_epa",
                value_expr(columns, "yac_epa", 0.0)
                .filter(is_complete)
                .sum()
                .fill_null(0.0)
                .alias("yac_epa_total"),
                pl.Float64,
            ),
            *mean_with_parts("xyac_per_completion", xyac.filter(is_complete)),
            *mean_with_parts("yac_over_expected_per_completion", (yac - xyac).filter(is_complete)),
            # Rushing (official carries exclude two-point tries).
            _count(is_carry, "carries"),
            _count(is_designed_run, "designed_carries"),
            value_expr(columns, "rushing_yards", 0.0)
            .filter(is_rush_attempt)
            .max()
            .alias("longest_rush"),
            _count((value_expr(columns, "fumble") > 0) & is_rush_attempt, "rushing_fumbles"),
            _count(is_rush_attempt & two_pt_success & is_two_point, "rushing_2pt_conversions"),
            # Overall offense (scrimmage snaps only).
            value_expr(columns, "epa", 0.0)
            .filter(scrimmage)
            .sum()
            .fill_null(0.0)
            .alias("offensive_epa"),
            value_expr(columns, "wpa", 0.0)
            .filter(scrimmage)
            .sum()
            .fill_null(0.0)
            .alias("offensive_wpa"),
            # Special teams from the possession team's side (kicking on punts, field goals, and
            # extra points; receiving on kickoffs), the input to the special-teams rating.
            _count(is_special, "st_plays"),
            value_expr(columns, "epa", 0.0).filter(is_special).sum().fill_null(0.0).alias("st_epa"),
            *mean_with_parts("success_rate", value_expr(columns, "success", 0).filter(scrimmage)),
            *mean_with_parts(
                "pass_success_rate", value_expr(columns, "success", 0).filter(is_dropback)
            ),
            *mean_with_parts(
                "rush_success_rate", value_expr(columns, "success", 0).filter(is_designed_run)
            ),
            *mean_with_parts("shotgun_rate", value_expr(columns, "shotgun", 0).filter(scrimmage)),
            *mean_with_parts("no_huddle_rate", no_huddle.filter(scrimmage)),
            *mean_with_parts(
                "pass_rate_over_expected",
                (
                    pl.col("pass_oe") if "pass_oe" in columns else pl.lit(None, dtype=pl.Float64)
                ).filter(scrimmage),
            ),
            _count(scrimmage, "aux_off_snaps"),
            _count(scrimmage & (down <= _LAST_EARLY_DOWN), "aux_early_snaps"),
            _count(is_dropback & (down <= _LAST_EARLY_DOWN), "aux_early_dropbacks"),
            _count(is_complete & (yards >= _EXPLOSIVE_PASS_YARDS), "aux_explosive_passes"),
            _count(
                is_rush_attempt & ~is_two_point & (yards >= _EXPLOSIVE_RUSH_YARDS),
                "aux_explosive_rushes",
            ),
            # A kneel-down is a carry but never a stuff, so stuff rate leaves kneel-downs out.
            _count(is_carry & ~is_kneel, "aux_carries_without_kneels"),
            _count(is_carry & ~is_kneel & (yards <= 0), "aux_stuffed_rushes"),
            _if_charted(
                "pass_length",
                _count(
                    is_pass_attempt & ~is_two_point & (pl.col("pass_length") == "deep"),
                    "aux_deep_attempts",
                ),
                pl.Int64,
            ),
            # Turnovers.
            _count((value_expr(columns, "fumble") > 0) & scrimmage, "fumbles"),
            _count(is_complete & receiver_fumbled, "receiving_fumbles"),
            _count(is_complete & receiver_fumbled & lost_by_offense, "receiving_fumbles_lost"),
            _count(lost_by_offense & scrimmage, "fumbles_lost"),
            _count(is_interception, "aux_interceptions"),
            # Giveaway EPA from each side of the ball: the team with the ball's own, and the
            # defending team's (a returner's lost fumble, from the receiving team's side).
            value_expr(columns, "epa", 0.0)
            .filter(giveaway_team == pl.col("posteam"))
            .sum()
            .fill_null(0.0)
            .alias("turnover_epa"),
            (-value_expr(columns, "epa", 0.0))
            .filter(giveaway_team == pl.col("defteam"))
            .sum()
            .fill_null(0.0)
            .alias("aux_defense_giveaway_epa"),
            value_expr(columns, "return_yards", 0.0)
            .filter(is_interception)
            .sum()
            .fill_null(0.0)
            .alias("aux_int_return_yards"),
            # Downs and conversions.
            _count(value_expr(columns, "first_down") > 0, "first_downs"),
            _count(value_expr(columns, "first_down_penalty") > 0, "first_downs_penalty"),
            _count(value_expr(columns, "third_down_converted") > 0, "third_down_conversions"),
            _count(
                (
                    value_expr(columns, "third_down_converted")
                    + value_expr(columns, "third_down_failed")
                )
                > 0,
                "third_down_attempts",
            ),
            *mean_with_parts(
                "third_down_avg_distance", ydstogo.filter(scrimmage & (down == _THIRD_DOWN))
            ),
            _count(value_expr(columns, "fourth_down_converted") > 0, "fourth_down_conversions"),
            _count(is_go_try, "fourth_down_attempts"),
            _count(fourth_down_faced, "aux_fourth_downs_faced"),
            _count(is_go_try & (ydstogo <= _SHORT_YARDAGE), "aux_fourth_short_go"),
            _count(fourth_down_faced & (ydstogo <= _SHORT_YARDAGE), "aux_fourth_short_faced"),
            _count(value_expr(columns, "fourth_down_failed") > 0, "turnovers_on_downs"),
            # Scoring extras.
            _count(is_two_point, "two_pt_attempts"),
            _count(is_two_point & two_pt_success, "two_pt_conversions"),
            _count(value_expr(columns, "pass_touchdown") > 0, "aux_pass_tds"),
            _count(value_expr(columns, "rush_touchdown") > 0, "aux_rush_tds"),
            # Aux inputs for ratios and defensive mirrors.
            value_expr(columns, "passing_yards", 0.0).sum().fill_null(0.0).alias("aux_pass_yards"),
            value_expr(columns, "rushing_yards", 0.0).sum().fill_null(0.0).alias("aux_rush_yards"),
            value_expr(columns, "epa", 0.0)
            .filter(is_dropback)
            .sum()
            .fill_null(0.0)
            .alias("aux_pass_epa"),
            value_expr(columns, "epa", 0.0)
            .filter(is_carry)
            .sum()
            .fill_null(0.0)
            .alias("aux_rush_epa"),
            _count(is_sack, "aux_sacks"),
            _if_charted("qb_hit", _count(pl.col("qb_hit") > 0, "aux_qb_hits"), pl.Int64),
            (
                _count(
                    (value_expr(columns, "tackled_for_loss") > 0)
                    | (value_expr(columns, "fumble_forced") > 0)
                    | is_interception
                    | (
                        pl.col("pass_defense_1_player_id").is_not_null()
                        if "pass_defense_1_player_id" in columns
                        else pl.lit(False)
                    ),
                    "aux_havoc_events",
                )
            ),
            _count(is_touchdown & (td_team == pl.col("defteam")), "aux_def_tds"),
            _count(
                is_touchdown & (td_team == pl.col("defteam")) & (value_expr(columns, "fumble") > 0),
                "aux_fumble_recovery_tds",
            ),
            value_expr(columns, "defensive_two_point_conv", 0)
            .sum()
            .cast(pl.Int64)
            .alias("aux_def_2pt"),
            # Penalties observed from this offense's plays.
            _count(is_penalty & (penalty_team == pl.col("posteam")), "offensive_penalties"),
            value_expr(columns, "penalty_yards", 0.0)
            .filter(is_penalty & (penalty_team == pl.col("posteam")))
            .sum()
            .fill_null(0.0)
            .alias("offensive_penalty_yards"),
            _count(
                is_penalty
                & (penalty_team == pl.col("posteam"))
                & penalty_type.is_in(list(_PRESNAP_PENALTY_TYPES)),
                "aux_presnap_penalties",
            ),
            _count(is_penalty & (penalty_team == pl.col("defteam")), "aux_def_pen_count"),
            value_expr(columns, "penalty_yards", 0.0)
            .filter(is_penalty & (penalty_team == pl.col("defteam")))
            .sum()
            .fill_null(0.0)
            .alias("aux_def_pen_yards"),
            _count(
                is_penalty
                & (penalty_team == pl.col("defteam"))
                & (penalty_type == "Defensive Pass Interference"),
                "aux_dpi",
            ),
        )
        .rename({"posteam": "team", "defteam": "opponent_team"})
    )


def _blank_unless_charted(
    plays: pl.DataFrame, field: str, expr: pl.Expr, dtype: type[pl.DataType]
) -> pl.Expr:
    """Return ``expr``, or a null of ``dtype`` in its place when the season lacks ``field``.

    ``plays`` is one season's play-by-play. A field nflverse did not chart that season (absent,
    or null on every play, as ``sources.prepare_pbp`` leaves it) makes every stat built on it
    unknown, not zero.
    """
    if _is_charted(plays, field):
        return expr
    return pl.lit(None, dtype=dtype).alias(expr.meta.output_name())


def _is_charted(plays: pl.DataFrame, field: str) -> bool:
    """Return whether the season's play-by-play has any value for ``field``."""
    return field in plays.columns and plays.get_column(field).is_not_null().any()


def _receiver_fumbled_expr(columns: list[str]) -> pl.Expr:
    """Return an expression that is true when the play's targeted receiver fumbled.

    Matched by player id, so a quarterback's fumbled snap before a completion, or a fumble by
    a teammate after a lateral, is not the receiver's fumble.
    """
    fumblers = [column for column in _FUMBLER_COLUMNS if column in columns]
    if "receiver_player_id" not in columns or not fumblers:
        return pl.lit(False)
    receiver = pl.col("receiver_player_id")
    return pl.any_horizontal(
        [(pl.col(column) == receiver).fill_null(value=False) for column in fumblers]
    )


def _target_receiver_expr(plays: pl.DataFrame) -> pl.Expr | None:
    """Return a named-receiver flag for counting targets, or None when targets are unknown.

    ``plays`` is one season's play-by-play. Completions always name their receiver, so the test
    is how many incomplete passes do: where almost none do, counted targets would be little more
    than completions, so they stay unknown, as they do without receiver columns.
    """
    columns = plays.columns
    present = [pl.col(column) for column in _RECEIVER_COLUMNS if column in columns]
    if not present:
        return None
    receiver_named = pl.coalesce(present).is_not_null()
    named_share = (
        plays.filter(
            (value_expr(columns, "pass_attempt") > 0)
            & (value_expr(columns, "incomplete_pass") > 0)
            & ~(value_expr(columns, "sack") > 0)
            & ~(value_expr(columns, "two_point_attempt") > 0)
        )
        .select(receiver_named.mean())
        .item()
    )
    if named_share is not None and named_share < _MIN_INCOMPLETION_RECEIVER_SHARE:
        return None
    return receiver_named


def _aggregate_series_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame | None:
    """Aggregate first-down series outcomes per team-game."""
    if not {"series", "series_success"}.issubset(plays.columns):
        return None

    columns = plays.columns
    per_series = plays.group_by([*keys, "posteam", "series"]).agg(
        value_expr(columns, "series_success", 0).max().alias("converted"),
        value_expr(columns, "goal_to_go", 0).max().alias("goal_to_go"),
        (
            (pl.col("series_result") == "Touchdown").max()
            if "series_result" in columns
            else pl.lit(False).max()
        ).alias("touchdown"),
    )
    return (
        per_series.group_by([*keys, "posteam"])
        .agg(
            pl.len().cast(pl.Int64).alias("series"),
            *mean_with_parts("series_conversion_rate", pl.col("converted")),
            *mean_with_parts(
                "goal_to_go_td_pct",
                pl.col("touchdown").filter(pl.col("goal_to_go") > 0).cast(pl.Int64),
            ),
        )
        .rename({"posteam": "team"})
    )


def _aggregate_drive_stats(plays: pl.DataFrame, keys: list[str]) -> pl.DataFrame | None:
    """Aggregate drive-level outcomes and field position per team-game."""
    if not {"fixed_drive", "fixed_drive_result"}.issubset(plays.columns):
        return None

    columns = plays.columns
    per_drive = plays.group_by([*keys, "posteam", "fixed_drive"]).agg(
        pl.col("fixed_drive_result").first().alias("result"),
        value_expr(columns, "drive_play_count", 0).first().alias("play_count"),
        value_expr(columns, "drive_first_downs", 0).first().alias("first_downs"),
        value_expr(columns, "drive_inside20", 0).max().alias("inside_20"),
        value_expr(columns, "ydsnet", 0).first().alias("net_yards"),
        value_expr(columns, "drive_yards_penalized", 0).first().alias("yards_penalized"),
        (giveaway_team_expr(columns) == pl.col("posteam"))
        .fill_null(value=False)
        .any()
        .alias("giveaway_play"),
        (
            pl.col("drive_time_of_possession").first()
            if "drive_time_of_possession" in columns
            else pl.lit(None, dtype=pl.String).first()
        ).alias("possession_clock"),
        (
            pl.col("drive_start_yard_line").first()
            if "drive_start_yard_line" in columns
            else pl.lit(None, dtype=pl.String).first()
        ).alias("start_yard_line"),
        (
            value_expr(columns, "posteam_score_post", 0).last()
            - value_expr(columns, "posteam_score", 0).first()
        ).alias("points"),
    )

    start_side = pl.col("start_yard_line").str.extract(r"^([A-Z]{2,3})", 1)
    start_number = pl.col("start_yard_line").str.extract(r"(\d+)$", 1).cast(pl.Int64)
    per_drive = per_drive.with_columns(
        pl.when(pl.col("start_yard_line") == "50")
        .then(50)
        .when(start_side == pl.col("posteam"))
        .then(start_number)
        .otherwise(100 - start_number)
        .alias("start_from_own_goal"),
        (
            pl.col("possession_clock").str.extract(r"^(\d+):", 1).cast(pl.Int64) * 60
            + pl.col("possession_clock").str.extract(r":(\d+)$", 1).cast(pl.Int64)
        ).alias("possession_seconds"),
        pl.col("result").is_in(["Touchdown", "Field goal"]).alias("scored"),
        (pl.col("result") == "Punt").alias("punted"),
        # nflverse ends a drive lost to an interception or fumble as "Turnover"; "Opp touchdown"
        # also covers punts and kicks returned for a score, so it counts only with a giveaway.
        (
            (pl.col("result") == "Turnover")
            | ((pl.col("result") == "Opp touchdown") & pl.col("giveaway_play"))
        ).alias("turned_over"),
    )

    return (
        per_drive.group_by([*keys, "posteam"])
        .agg(
            pl.len().cast(pl.Int64).alias("drives"),
            *mean_with_parts("yards_per_drive", pl.col("net_yards")),
            *mean_with_parts("plays_per_drive", pl.col("play_count")),
            *mean_with_parts("time_per_drive", pl.col("possession_seconds")),
            *mean_with_parts("first_downs_per_drive", pl.col("first_downs")),
            *mean_with_parts("score_pct_per_drive", pl.col("scored").cast(pl.Int64)),
            *mean_with_parts("punt_pct_per_drive", pl.col("punted").cast(pl.Int64)),
            *mean_with_parts("turnover_pct_per_drive", pl.col("turned_over").cast(pl.Int64)),
            *mean_with_parts(
                "three_and_out_rate",
                (pl.col("punted") & (pl.col("first_downs") == 0)).cast(pl.Int64),
            ),
            *mean_with_parts("points_per_drive", pl.col("points")),
            (pl.col("inside_20") > 0).cast(pl.Int64).sum().alias("red_zone_trips"),
            *mean_with_parts(
                "red_zone_td_pct",
                (pl.col("result") == "Touchdown").filter(pl.col("inside_20") > 0).cast(pl.Int64),
            ),
            *mean_with_parts(
                "points_per_red_zone_trip", pl.col("points").filter(pl.col("inside_20") > 0)
            ),
            *mean_with_parts("avg_starting_field_position", pl.col("start_from_own_goal")),
            *mean_with_parts(
                "long_field_score_pct",
                pl.col("scored")
                .filter(pl.col("start_from_own_goal") <= _LONG_FIELD_START_YARDLINE)
                .cast(pl.Int64),
            ),
            _blank_unless_charted(
                plays,
                "drive_yards_penalized",
                pl.col("yards_penalized").sum().alias("drive_penalty_yards"),
                pl.Float64,
            ),
        )
        .rename({"posteam": "team"})
    )


def _join_committed_penalties(
    frame: pl.DataFrame, plays: pl.DataFrame, keys: list[str]
) -> pl.DataFrame:
    """Join all-unit committed penalties and the game penalty differentials."""
    if "penalty_team" not in plays.columns or "penalty" not in plays.columns:
        return frame

    columns = plays.columns
    committed = (
        plays.filter((value_expr(columns, "penalty") > 0) & pl.col("penalty_team").is_not_null())
        .group_by([*keys, "penalty_team"])
        .agg(
            pl.len().cast(pl.Int64).alias("penalties"),
            value_expr(columns, "penalty_yards", 0.0).sum().fill_null(0.0).alias("penalty_yards"),
        )
        .rename({"penalty_team": "team"})
        .with_columns(pl.col("team").cast(pl.String))
    )

    frame = frame.join(committed, on=[*keys, "team"], how="left").with_columns(
        pl.col("penalties").fill_null(0),
        pl.col("penalty_yards").fill_null(0.0),
    )
    opponent_committed = committed.rename(
        {
            "team": "opponent_team",
            "penalties": "aux_opp_penalties",
            "penalty_yards": "aux_opp_penalty_yards",
        }
    )
    return frame.join(opponent_committed, on=[*keys, "opponent_team"], how="left").with_columns(
        (pl.col("aux_opp_penalties").fill_null(0) - pl.col("penalties")).alias(
            "penalty_differential"
        ),
        (pl.col("aux_opp_penalty_yards").fill_null(0.0) - pl.col("penalty_yards")).alias(
            "penalty_yards_differential"
        ),
    )


def _join_touchdown_totals(
    frame: pl.DataFrame, plays: pl.DataFrame, keys: list[str]
) -> pl.DataFrame:
    """Join total touchdowns credited to each team via td_team."""
    if "td_team" not in plays.columns or "touchdown" not in plays.columns:
        return frame

    columns = plays.columns
    touchdowns = (
        plays.filter((value_expr(columns, "touchdown") > 0) & pl.col("td_team").is_not_null())
        .group_by([*keys, "td_team"])
        .agg(pl.len().cast(pl.Int64).alias("total_tds"))
        .rename({"td_team": "team"})
        .with_columns(pl.col("team").cast(pl.String))
    )
    return frame.join(touchdowns, on=[*keys, "team"], how="left").with_columns(
        pl.col("total_tds").fill_null(0)
    )


def _passer_rating_expr(
    completions: str, attempts: str, yards: str, touchdowns: str, interceptions: str
) -> pl.Expr:
    """Return the official NFL passer-rating formula over one row's passing totals."""
    attempts_col = pl.col(attempts)
    rating = passer_rating_from_rates(
        pl.col(completions) / attempts_col,
        pl.col(yards) / attempts_col,
        pl.col(touchdowns) / attempts_col,
        pl.col(interceptions) / attempts_col,
    )
    return pl.when(attempts_col > 0).then(rating).otherwise(None)


def _difference(own: pl.Expr, allowed: pl.Expr) -> pl.Expr:
    """Return a team's own rate minus the same rate allowed: a margin."""
    return own - allowed


@dataclass(frozen=True, slots=True)
class FormulaRate:
    """A rate built from other rates, so rows for several games rebuild it from their pooled values.

    ``combine`` takes the ``inputs`` in order.
    """

    inputs: tuple[str, ...]
    combine: Callable[..., pl.Expr]


# Published rates that are a formula of other rates rather than one ratio: passer rating from its
# four per-attempt rates (the defense's from hidden mirrors of the opponent's), and the margins.
TEAM_FORMULA_RATES: dict[str, FormulaRate] = {
    "team_passer_rating": FormulaRate(
        (
            "completion_pct",
            "yards_per_attempt",
            "passing_td_rate_per_attempt",
            "int_rate_per_attempt",
        ),
        passer_rating_from_rates,
    ),
    "team_passer_rating_allowed": FormulaRate(
        (
            "completion_pct_allowed",
            "yards_per_attempt_allowed",
            "passing_td_rate_per_attempt_allowed",
            "int_rate_per_attempt_allowed",
        ),
        passer_rating_from_rates,
    ),
    "epa_margin_per_play": FormulaRate(
        ("epa_per_offensive_snap", "epa_per_defensive_snap_allowed"), _difference
    ),
    "success_rate_margin": FormulaRate(("success_rate", "success_rate_allowed"), _difference),
}


def _add_offense_ratios(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive offense-side ratio columns from the aggregated counts."""
    frame = frame.with_columns(
        (pl.col("aux_pass_yards") + pl.col("aux_rush_yards")).alias("aux_total_yards"),
        (pl.col("aux_interceptions") + pl.col("fumbles_lost")).alias("giveaways"),
        (pl.col("aux_pass_tds") + pl.col("aux_rush_tds")).alias("scrimmage_tds"),
        (pl.col("aux_pass_yards") - pl.col("sack_yards_lost")).alias("net_passing_yards"),
    )
    frame = frame.with_columns(pl.col("scrimmage_tds").alias("offensive_tds"))

    frame = frame.with_columns(
        [
            expr
            for numerator, denominator, output in OFFENSE_RATIO_SPECS
            if {numerator, denominator}.issubset(frame.columns)
            for expr in ratio_with_parts(output, numerator, denominator)
        ]
    )

    pass_plays = pl.col("attempts") + pl.col("aux_sacks")
    derived = [
        *expression_ratio_with_parts(
            "net_yards_per_attempt",
            pl.col("aux_pass_yards") - pl.col("sack_yards_lost"),
            pass_plays,
        ),
        *expression_ratio_with_parts(
            "adjusted_net_yards_per_attempt",
            pl.col("aux_pass_yards")
            + 20.0 * pl.col("aux_pass_tds")
            - 45.0 * pl.col("aux_interceptions")
            - pl.col("sack_yards_lost"),
            pass_plays,
        ),
        _passer_rating_expr(
            "completions", "attempts", "aux_pass_yards", "aux_pass_tds", "aux_interceptions"
        ).alias("team_passer_rating"),
        *expression_ratio_with_parts(
            "explosive_play_rate",
            pl.col("aux_explosive_passes") + pl.col("aux_explosive_rushes"),
            pl.col("aux_off_snaps"),
        ),
        *expression_ratio_with_parts(
            "aux_pressure_rate",
            pl.col("aux_sacks") + pl.col("aux_qb_hits"),
            pl.col("dropbacks"),
        ),
    ]
    frame = frame.with_columns(derived)

    if "drives" in frame.columns:
        numerator, denominator, rate = GIVEAWAYS_PER_DRIVE_SPEC
        frame = frame.with_columns(ratio_with_parts(rate, numerator, denominator))
    return frame


def _join_defense_mirrors(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Mirror each offense row onto the opponent as its defensive surface.

    A full join keeps teams that only appear on defense in a fixture; in real
    games both teams run offense, so this is a no-op there.
    """
    available = {
        source: target
        for source, target in DEFENSE_MIRROR_RENAMES.items()
        if source in frame.columns
    }
    # A mirrored rate's numerator and denominator are the opponent's, so its defense pools too.
    part_mirrors = [
        pl.col(part(source)).alias(part(target))
        for source, target in {**available, **DEFENSE_MIRROR_PARTS_ONLY}.items()
        for part in (numerator_column, denominator_column)
        if part(source) in frame.columns
    ]
    mirror_columns = [
        *(pl.col(source).alias(target) for source, target in available.items()),
        *part_mirrors,
        pl.col("turnover_epa").alias("aux_opponent_turnover_epa"),
        pl.col("aux_defense_giveaway_epa").alias("aux_own_defense_giveaway_epa"),
    ]
    mirror = frame.select(
        [
            *keys,
            pl.col("opponent_team").alias("team"),
            pl.col("team").alias("opponent_team"),
            *mirror_columns,
        ]
    )
    joined = frame.join(mirror, on=[*keys, "team", "opponent_team"], how="full", coalesce=True)
    # A team's giveaway EPA is its own with the ball plus its giveaways while the opponent had the
    # ball (a muffed punt); its takeaway EPA is the opponent's giveaway EPA, both sides, reversed.
    return joined.with_columns(
        (
            pl.col("turnover_epa").fill_null(0.0)
            + pl.col("aux_own_defense_giveaway_epa").fill_null(0.0)
        ).alias("turnover_epa"),
        (
            -pl.col("aux_opponent_turnover_epa").fill_null(0.0)
            - pl.col("aux_defense_giveaway_epa").fill_null(0.0)
        ).alias("takeaway_epa"),
    )


def _add_cross_side_margins(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive whole-team margins that need both offense and defense values."""
    margin_specs = [
        # Each takeaway is the opponent's giveaway, so the league's margins sum to zero.
        ("takeaways", "giveaways", "turnover_margin"),
        ("aux_total_yards", "aux_total_yards_allowed", "total_yards_differential"),
        ("epa_per_offensive_snap", "epa_per_defensive_snap_allowed", "epa_margin_per_play"),
        ("success_rate", "success_rate_allowed", "success_rate_margin"),
    ]
    margins = [
        (pl.col(own) - pl.col(allowed)).alias(output)
        for own, allowed, output in margin_specs
        if {own, allowed}.issubset(frame.columns)
    ]
    return frame.with_columns(margins) if margins else frame
