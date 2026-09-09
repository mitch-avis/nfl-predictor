"""Aggregate play-by-play rows into one team-game record per team and opponent.

This module turns raw nflverse play-by-play into a single row per
``(season, week, team_abbr, opponent_abbr)`` containing **counts and sums only**.

Rates are deliberately not emitted here. Season-to-date aggregation happens
downstream, and a rate is only correct when it is computed as a ratio of summed
components after that aggregation (a mean of per-game rates is not the same
number). Emitting rates at the team-game level would therefore be wrong, so this
module carries the numerators and denominators and lets the downstream derived
metrics divide them.

Definitions used throughout:

- Scrimmage snap (the single definition of an offensive snap):
  ``(qb_dropback + rush + qb_kneel + qb_spike) > 0`` with nulls filled to 0.
- Dropback: a scrimmage snap with ``qb_dropback > 0`` that is neither a kneel
  nor a spike.
- Carry: a scrimmage snap with ``rush > 0`` that is neither a kneel nor a spike.
  Kneels and spikes are therefore offensive snaps but are neither dropbacks nor
  carries.
- Offense perspective: the team is ``posteam`` on the play.
- Allowed perspective: the team is ``defteam`` on the same play, so every
  ``*_allowed_*`` column of a team-game equals the matching offensive column of
  its opponent's team-game.
- Allowed denominators: ``dropbacks_allowed`` and ``carries_allowed`` are the
  dropbacks and carries a team faced, using the same dropback and carry
  definitions from the defending side. They are the correct denominators for
  defensive efficiency rates (EPA per dropback allowed, EPA per carry allowed,
  success rates allowed, and stuff rate), because ``defensive_snaps`` counts
  every scrimmage snap faced rather than only pass or only run snaps.

Deviation from the ``nfl-sos-ratings`` reference implementation (read-only
reference material, not a dependency):

- That repo counts an explosive rush at 10 or more yards; this repo uses 12 or
  more yards, which is the stricter threshold preferred for these features.
- That repo defines an explosive pass on *completed* passes; this repo defines
  it on *dropbacks*, so sacks and incompletions stay in the denominator family
  and the explosive-pass count shares the ``dropbacks`` denominator.

Both divergences are intentional; keep them in mind when comparing outputs
against that reference.

Missing-source-column contract: every source column read here is
existence-guarded. A missing counting flag or value column contributes its
documented default (0 for flags and counts, 0.0 for EPA and yardage sums, and
"not an early down" / "not in the red zone" for missing ``down`` and
``yardline_100``) instead of raising. Two consequences of those defaults are
worth stating outright: a missing ``success`` column means no play counts as a
success, and a missing ``yards_gained`` column means every play gained 0 yards,
so nothing is explosive and every rush counts as stuffed. A missing
``play_type`` column leaves all of the situational counts at 0.

The special-teams columns are the one exception to the zero-default rule: when
no special-teams flag column is present, ``st_epa_for``, ``st_epa_against``, and
``st_plays`` are emitted as nulls so downstream code can tell "no special-teams
source" apart from "no special-teams value".
"""

import polars as pl
from polars.datatypes.classes import DataTypeClass

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

# Yards gained on a dropback that make the play explosive.
# Source: nfl-sos-ratings explosive-pass family, retargeted from completions to dropbacks.
EXPLOSIVE_PASS_YARDS = 20
# Yards gained on a rush that make the play explosive.
# Source: nfl-sos-ratings uses 10; this repo deliberately uses the stricter 12.
EXPLOSIVE_RUSH_YARDS = 12
# Yards gained at or below which a rush is considered stuffed at or behind the line.
STUFFED_RUSH_YARDS = 0
# Highest down that still counts as an early down (first and second down).
EARLY_DOWN_MAX = 2
# Distance to the opponent goal line that defines the red zone.
# Source: existing red-zone aggregation in this repo's Polars loaders.
RED_ZONE_YARDLINE = 20
# Play types counted by the situational aggregation.
# Source: existing red-zone/down/two-point aggregation in this repo's Polars loaders.
SITUATIONAL_PLAY_TYPES = ("run", "pass", "qb_kneel", "qb_spike")

_PBP_TEAM_GAME_SCHEMA: dict[str, DataTypeClass] = {
    # Identity keys.
    "season": pl.Int64,
    "week": pl.Int64,
    "team_abbr": pl.String,
    "opponent_abbr": pl.String,
    # Snap volume.
    "offensive_snaps": pl.Int64,
    "defensive_snaps": pl.Int64,
    "dropbacks": pl.Int64,
    "carries": pl.Int64,
    "dropbacks_allowed": pl.Int64,
    "carries_allowed": pl.Int64,
    # EPA sums.
    "pass_epa_sum": pl.Float64,
    "rush_epa_sum": pl.Float64,
    "pass_epa_allowed_sum": pl.Float64,
    "rush_epa_allowed_sum": pl.Float64,
    # Success counts.
    "pass_success_count": pl.Int64,
    "rush_success_count": pl.Int64,
    "pass_success_allowed_count": pl.Int64,
    "rush_success_allowed_count": pl.Int64,
    # Explosive and stuffed counts.
    "explosive_pass_count": pl.Int64,
    "explosive_rush_count": pl.Int64,
    "stuffed_rush_count": pl.Int64,
    "explosive_pass_allowed_count": pl.Int64,
    "explosive_rush_allowed_count": pl.Int64,
    "stuffed_rush_allowed_count": pl.Int64,
    # Early-down tendency.
    "early_down_plays": pl.Int64,
    "early_down_passes": pl.Int64,
    # Special teams.
    "st_epa_for": pl.Float64,
    "st_epa_against": pl.Float64,
    "st_plays": pl.Int64,
    # Situational counts.
    "third_down_conversions": pl.Int64,
    "third_down_fails": pl.Int64,
    "third_down_attempts": pl.Int64,
    "fourth_down_conversions": pl.Int64,
    "fourth_down_fails": pl.Int64,
    "fourth_down_attempts": pl.Int64,
    "red_zone_plays": pl.Int64,
    "red_zone_tds": pl.Int64,
    "two_point_attempts": pl.Int64,
    "two_point_successes": pl.Int64,
    "total_plays": pl.Int64,
}

PBP_TEAM_GAME_COLUMNS: list[str] = list(_PBP_TEAM_GAME_SCHEMA)

_IDENTITY_COLUMNS = ("season", "week", "team_abbr", "opponent_abbr")
_REQUIRED_PBP_COLUMNS = ("season", "week", "posteam", "defteam")
_ST_COLUMNS = ("st_epa_for", "st_epa_against", "st_plays")
_ST_PLAYS_OFFENSE = "_st_plays_offense"
_ST_PLAYS_DEFENSE = "_st_plays_defense"


def _flag_expr(columns: list[str], column: str) -> pl.Expr:
    """Return a null-safe 0/1 expression for a play flag, or literal 0 when absent."""
    if column in columns:
        return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)
    return pl.lit(0.0)


def _value_expr(columns: list[str], column: str, default: float = 0.0) -> pl.Expr:
    """Return a null-safe float expression for a value column, or the default when absent."""
    if column in columns:
        return pl.col(column).cast(pl.Float64, strict=False).fill_null(default)
    return pl.lit(default, dtype=pl.Float64)


def _nullable_expr(columns: list[str], column: str, dtype: DataTypeClass) -> pl.Expr:
    """Return the column as-is when present, or a typed null literal when absent."""
    if column in columns:
        return pl.col(column).cast(dtype, strict=False)
    return pl.lit(None, dtype=dtype)


def _scrimmage_snap_expr(columns: list[str]) -> pl.Expr:
    """Return the single offensive-snap definition.

    Formula: ``(qb_dropback + rush + qb_kneel + qb_spike) > 0`` with nulls filled
    to 0, so a play counts as a scrimmage snap when any of those flags is set.
    """
    return (
        _flag_expr(columns, "qb_dropback")
        + _flag_expr(columns, "rush")
        + _flag_expr(columns, "qb_kneel")
        + _flag_expr(columns, "qb_spike")
    ) > 0


def _count(condition: pl.Expr, name: str) -> pl.Expr:
    """Return an aggregation counting the rows in the group where the condition holds."""
    return pl.when(condition).then(1).otherwise(0).sum().cast(pl.Int64).alias(name)


def _sum_when(condition: pl.Expr, value: pl.Expr, name: str) -> pl.Expr:
    """Return an aggregation summing a value over the rows where the condition holds."""
    return pl.when(condition).then(value).otherwise(0.0).sum().cast(pl.Float64).alias(name)


def special_teams_flag_column(columns: list[str]) -> str | None:
    """Return the first special-teams flag column present, or None when none exists.

    Candidates come from ``constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES`` in
    preference order, because nflverse has published the flag under more than
    one name.
    """
    for candidate in constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def empty_team_game_frame() -> pl.DataFrame:
    """Return a typed empty frame with exactly ``PBP_TEAM_GAME_COLUMNS``."""
    return pl.DataFrame(
        schema={name: _PBP_TEAM_GAME_SCHEMA[name] for name in PBP_TEAM_GAME_COLUMNS}
    )


def _play_conditions(columns: list[str]) -> dict[str, pl.Expr]:
    """Build the shared per-play condition and value expressions.

    Formulas:
        - ``scrimmage``: ``(qb_dropback + rush + qb_kneel + qb_spike) > 0``.
        - ``dropback``: scrimmage snap with ``qb_dropback > 0``, excluding kneels,
          spikes, and two-point tries.
        - ``rush``: scrimmage snap with ``rush > 0``, excluding kneels, spikes, and
          two-point tries. Two-point tries are untimed conversion attempts rather than
          scrimmage downs, so counting them would contaminate the per-attempt EPA and
          success denominators.
        - ``success``: ``success > 0`` (missing column counts as no success).
        - ``early_down``: ``down <= EARLY_DOWN_MAX`` (missing/null down is not early).
        - ``special``: the special-teams flag column is set (missing column is never special).
    """
    scrimmage = _scrimmage_snap_expr(columns)
    is_kneel_or_spike = (_flag_expr(columns, "qb_kneel") + _flag_expr(columns, "qb_spike")) > 0
    is_two_point = _flag_expr(columns, "two_point_attempt") > 0
    dropback = (
        scrimmage & (_flag_expr(columns, "qb_dropback") > 0) & ~is_kneel_or_spike & ~is_two_point
    )
    rush = scrimmage & (_flag_expr(columns, "rush") > 0) & ~is_kneel_or_spike & ~is_two_point
    yards = _value_expr(columns, "yards_gained")
    down = _nullable_expr(columns, "down", pl.Int64)
    st_flag = special_teams_flag_column(columns)
    special = _flag_expr(columns, st_flag) > 0 if st_flag is not None else pl.lit(False)
    return {
        "scrimmage": scrimmage,
        "dropback": dropback,
        "rush": rush,
        "epa": _value_expr(columns, "epa"),
        "success": _flag_expr(columns, "success") > 0,
        "explosive_pass": dropback & (yards >= EXPLOSIVE_PASS_YARDS),
        "explosive_rush": rush & (yards >= EXPLOSIVE_RUSH_YARDS),
        "stuffed_rush": rush & (yards <= STUFFED_RUSH_YARDS),
        "early_down": (down <= EARLY_DOWN_MAX).fill_null(False),
        "special": special,
    }


def _situational_aggregations(columns: list[str]) -> list[pl.Expr]:
    """Build the situational count aggregations for the offense perspective.

    Formulas (all restricted to plays whose ``play_type`` is in
    ``SITUATIONAL_PLAY_TYPES``, matching the existing pipeline definition):
        - ``third_down_conversions``: plays with ``third_down_converted`` set.
        - ``third_down_fails``: plays with ``third_down_failed`` set.
        - ``fourth_down_conversions``: plays with ``fourth_down_converted`` set.
        - ``fourth_down_fails``: plays with ``fourth_down_failed`` set.
        - ``red_zone_plays``: plays with ``yardline_100 <= RED_ZONE_YARDLINE``.
        - ``red_zone_tds``: red-zone plays whose ``td_team`` is not null.
        - ``two_point_attempts``: plays with ``two_point_attempt`` set.
        - ``two_point_successes``: plays with ``two_point_conv_result == "success"``.
        - ``total_plays``: the number of such plays.
    Third- and fourth-down attempts are the conversion plus fail counts and are
    added after aggregation.
    """
    if "play_type" in columns:
        counted = pl.col("play_type").is_in(list(SITUATIONAL_PLAY_TYPES)).fill_null(False)
    else:
        counted = pl.lit(False)
    yardline = _nullable_expr(columns, "yardline_100", pl.Float64)
    in_red_zone = (yardline <= RED_ZONE_YARDLINE).fill_null(False)
    scored_td = pl.col("td_team").is_not_null() if "td_team" in columns else pl.lit(False)
    if "two_point_conv_result" in columns:
        two_point_success = (pl.col("two_point_conv_result") == "success").fill_null(False)
    else:
        two_point_success = pl.lit(False)
    return [
        _count(
            counted & (_flag_expr(columns, "third_down_converted") > 0), "third_down_conversions"
        ),
        _count(counted & (_flag_expr(columns, "third_down_failed") > 0), "third_down_fails"),
        _count(
            counted & (_flag_expr(columns, "fourth_down_converted") > 0),
            "fourth_down_conversions",
        ),
        _count(counted & (_flag_expr(columns, "fourth_down_failed") > 0), "fourth_down_fails"),
        _count(counted & in_red_zone, "red_zone_plays"),
        _count(counted & in_red_zone & scored_td, "red_zone_tds"),
        _count(counted & (_flag_expr(columns, "two_point_attempt") > 0), "two_point_attempts"),
        _count(counted & two_point_success, "two_point_successes"),
        _count(counted, "total_plays"),
    ]


def _aggregate_offense(plays: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the offense perspective (the team is ``posteam``) per team-game.

    Formulas:
        - ``offensive_snaps``: scrimmage snaps taken by the team.
        - ``dropbacks`` / ``carries``: dropback and rush snaps (kneels and spikes excluded).
        - ``pass_epa_sum`` / ``rush_epa_sum``: sum of ``epa`` over dropbacks / rushes.
        - ``pass_success_count`` / ``rush_success_count``: dropbacks / rushes with ``success > 0``.
        - ``explosive_pass_count``: dropbacks gaining at least ``EXPLOSIVE_PASS_YARDS`` yards.
        - ``explosive_rush_count``: rushes gaining at least ``EXPLOSIVE_RUSH_YARDS`` yards.
        - ``stuffed_rush_count``: rushes gaining at most ``STUFFED_RUSH_YARDS`` yards.
        - ``early_down_plays``: scrimmage snaps with ``down <= EARLY_DOWN_MAX``.
        - ``early_down_passes``: dropbacks with ``down <= EARLY_DOWN_MAX``.
        - ``st_epa_for``: sum of ``epa`` on special-teams plays where the team has possession.
        - ``_st_plays_offense``: those special-teams plays.
    """
    columns = plays.columns
    play = _play_conditions(columns)
    epa = play["epa"]
    return (
        plays.group_by(["season", "week", "posteam", "defteam"])
        .agg(
            _count(play["scrimmage"], "offensive_snaps"),
            _count(play["dropback"], "dropbacks"),
            _count(play["rush"], "carries"),
            _sum_when(play["dropback"], epa, "pass_epa_sum"),
            _sum_when(play["rush"], epa, "rush_epa_sum"),
            _count(play["dropback"] & play["success"], "pass_success_count"),
            _count(play["rush"] & play["success"], "rush_success_count"),
            _count(play["explosive_pass"], "explosive_pass_count"),
            _count(play["explosive_rush"], "explosive_rush_count"),
            _count(play["stuffed_rush"], "stuffed_rush_count"),
            _count(play["scrimmage"] & play["early_down"], "early_down_plays"),
            _count(play["dropback"] & play["early_down"], "early_down_passes"),
            _sum_when(play["special"], epa, "st_epa_for"),
            _count(play["special"], _ST_PLAYS_OFFENSE),
            *_situational_aggregations(columns),
        )
        .rename({"posteam": "team_abbr", "defteam": "opponent_abbr"})
        .with_columns(
            (pl.col("third_down_conversions") + pl.col("third_down_fails")).alias(
                "third_down_attempts"
            ),
            (pl.col("fourth_down_conversions") + pl.col("fourth_down_fails")).alias(
                "fourth_down_attempts"
            ),
        )
    )


def _aggregate_allowed(plays: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the allowed perspective (the team is ``defteam``) per team-game.

    Every column here is the offense formula of ``_aggregate_offense`` evaluated
    over the plays where the team is defending, so a team's allowed column always
    equals its opponent's matching offensive column for that game:
        - ``defensive_snaps``: scrimmage snaps faced.
        - ``dropbacks_allowed`` / ``carries_allowed``: dropback and rush snaps
          faced (kneels and spikes excluded), the denominators for the
          per-dropback and per-carry rates allowed.
        - ``pass_epa_allowed_sum`` / ``rush_epa_allowed_sum``: ``epa`` allowed on
          dropbacks / rushes.
        - ``pass_success_allowed_count`` / ``rush_success_allowed_count``:
          dropbacks / rushes faced with ``success > 0``.
        - ``explosive_pass_allowed_count`` / ``explosive_rush_allowed_count`` /
          ``stuffed_rush_allowed_count``: the explosive and stuffed thresholds
          applied to the plays faced.
        - ``st_epa_against``: sum of ``epa`` on special-teams plays where the
          other team has possession.
        - ``_st_plays_defense``: those special-teams plays.
    """
    columns = plays.columns
    play = _play_conditions(columns)
    epa = play["epa"]
    return (
        plays.group_by(["season", "week", "defteam", "posteam"])
        .agg(
            _count(play["scrimmage"], "defensive_snaps"),
            _count(play["dropback"], "dropbacks_allowed"),
            _count(play["rush"], "carries_allowed"),
            _sum_when(play["dropback"], epa, "pass_epa_allowed_sum"),
            _sum_when(play["rush"], epa, "rush_epa_allowed_sum"),
            _count(play["dropback"] & play["success"], "pass_success_allowed_count"),
            _count(play["rush"] & play["success"], "rush_success_allowed_count"),
            _count(play["explosive_pass"], "explosive_pass_allowed_count"),
            _count(play["explosive_rush"], "explosive_rush_allowed_count"),
            _count(play["stuffed_rush"], "stuffed_rush_allowed_count"),
            _sum_when(play["special"], epa, "st_epa_against"),
            _count(play["special"], _ST_PLAYS_DEFENSE),
        )
        .rename({"defteam": "team_abbr", "posteam": "opponent_abbr"})
    )


def _prepare_plays(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Filter play-by-play rows to the regular-season plays with both teams known.

    Rows are kept when ``season_type == "REG"``; when the column is absent every
    row is kept. Plays whose ``posteam`` or ``defteam`` is missing are dropped because
    they cannot be credited to a team-game. Missing means null *or* an empty string:
    early nflverse seasons use an empty string, and grouping on it would invent a
    phantom team-game row and duplicate the `(season, week, team_abbr)` key that the
    downstream join depends on.
    """
    plays = pbp_df
    if "season_type" in plays.columns:
        plays = plays.filter(pl.col("season_type") == "REG")
    for col in ("posteam", "defteam"):
        plays = plays.filter(
            pl.col(col).is_not_null() & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "")
        )
    return plays


def aggregate_pbp_team_game_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate play-by-play rows into one team-game row of counts and sums.

    The output has one row per ``(season, week, team_abbr, opponent_abbr)`` with
    exactly the columns in ``PBP_TEAM_GAME_COLUMNS``. Only counts and sums are
    produced; rates are derived downstream as ratios of season-to-date sums.

    Args:
        pbp_df: Raw play-by-play rows. Optional source columns may be missing;
            see the module docstring for the per-column defaults.

    Returns:
        Team-game frame with the columns and dtypes of ``PBP_TEAM_GAME_COLUMNS``,
        sorted by season, week, and team. Empty input returns a typed empty frame.

    Raises:
        ValueError: If an identity column (``season``, ``week``, ``posteam``,
            ``defteam``) is missing, because team-games cannot be keyed without them.

    """
    if pbp_df.height == 0:
        return empty_team_game_frame()

    missing = [column for column in _REQUIRED_PBP_COLUMNS if column not in pbp_df.columns]
    if missing:
        raise ValueError(f"pbp_df is missing required columns: {', '.join(missing)}")

    plays = _prepare_plays(pbp_df)
    if plays.height == 0:
        log.warning("No play-by-play rows remain after filtering; returning empty team-game frame")
        return empty_team_game_frame()

    has_special_teams = special_teams_flag_column(plays.columns) is not None
    offense = _aggregate_offense(plays)
    allowed = _aggregate_allowed(plays)
    combined = offense.join(allowed, on=list(_IDENTITY_COLUMNS), how="full", coalesce=True)

    count_columns = [
        name
        for name, dtype in _PBP_TEAM_GAME_SCHEMA.items()
        if dtype is pl.Int64 and name not in _IDENTITY_COLUMNS
    ]
    sum_columns = [name for name, dtype in _PBP_TEAM_GAME_SCHEMA.items() if dtype is pl.Float64]
    combined = combined.with_columns(
        (pl.col(_ST_PLAYS_OFFENSE).fill_null(0) + pl.col(_ST_PLAYS_DEFENSE).fill_null(0)).alias(
            "st_plays"
        )
    ).with_columns(
        *[pl.col(name).fill_null(0) for name in count_columns],
        *[pl.col(name).fill_null(0.0) for name in sum_columns],
    )

    if not has_special_teams:
        log.warning(
            "No special-teams flag column in play-by-play; emitting null special-teams stats"
        )
        combined = combined.with_columns(
            *[pl.lit(None, dtype=_PBP_TEAM_GAME_SCHEMA[name]).alias(name) for name in _ST_COLUMNS]
        )

    return combined.select(
        pl.col(name).cast(_PBP_TEAM_GAME_SCHEMA[name]) for name in PBP_TEAM_GAME_COLUMNS
    ).sort(["season", "week", "team_abbr"])
