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
    # Context flag, not a count: never averaged, mirrored, or published.
    "is_home": pl.Boolean,
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
    "third_down_conversions_allowed": pl.Int64,
    "third_down_fails_allowed": pl.Int64,
    "third_down_attempts_allowed": pl.Int64,
    "fourth_down_conversions": pl.Int64,
    "fourth_down_fails": pl.Int64,
    "fourth_down_attempts": pl.Int64,
    "fourth_down_conversions_allowed": pl.Int64,
    "fourth_down_fails_allowed": pl.Int64,
    "fourth_down_attempts_allowed": pl.Int64,
    "red_zone_plays": pl.Int64,
    "red_zone_tds": pl.Int64,
    "red_zone_plays_allowed": pl.Int64,
    "red_zone_tds_allowed": pl.Int64,
    "red_zone_trips": pl.Int64,
    "red_zone_td_drives": pl.Int64,
    "red_zone_trips_allowed": pl.Int64,
    "red_zone_td_drives_allowed": pl.Int64,
    "two_point_attempts": pl.Int64,
    "two_point_successes": pl.Int64,
    "two_point_attempts_allowed": pl.Int64,
    "two_point_successes_allowed": pl.Int64,
    "total_plays": pl.Int64,
}

PBP_TEAM_GAME_COLUMNS: list[str] = list(_PBP_TEAM_GAME_SCHEMA)

_IDENTITY_COLUMNS = ("season", "week", "team_abbr", "opponent_abbr")
_REQUIRED_PBP_COLUMNS = ("season", "week", "posteam", "defteam")
_ST_COLUMNS = ("st_epa_for", "st_epa_against", "st_plays")
_ST_PLAYS_OFFENSE = "_st_plays_offense"
_ST_PLAYS_DEFENSE = "_st_plays_defense"
_IS_HOME_OFFENSE = "_is_home_offense"
_IS_HOME_DEFENSE = "_is_home_defense"
_TEAM_BOX_SCORE_SCHEMA: dict[str, DataTypeClass] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "season_type": pl.String,
    "team_abbr": pl.String,
    "opponent_abbr": pl.String,
    "pass_completions": pl.Float64,
    "pass_attempts": pl.Float64,
    "pass_yards": pl.Float64,
    "pass_touchdowns": pl.Float64,
    "interceptions_thrown": pl.Float64,
    "times_sacked": pl.Float64,
    "passing_epa": pl.Float64,
    "passing_cpoe": pl.Float64,
    "rushing_epa": pl.Float64,
    "rush_attempts": pl.Float64,
    "rush_yards": pl.Float64,
    "rush_touchdowns": pl.Float64,
    "fumbles": pl.Float64,
    "fumbles_lost": pl.Float64,
    "first_downs": pl.Float64,
    "2pt_conversions": pl.Float64,
    "total_yards": pl.Float64,
    "penalties": pl.Float64,
    "penalty_yards": pl.Float64,
    "def_sacks": pl.Float64,
    "def_interceptions": pl.Float64,
}
PBP_TEAM_BOX_SCORE_COLUMNS: list[str] = list(_TEAM_BOX_SCORE_SCHEMA)
_PBP_PASSING_CPOE_SUM = "_passing_cpoe_sum"
_PBP_PASSING_CPOE_COUNT = "_passing_cpoe_count"
_PBP_SACK_YARDS_LOST = "_sack_yards_lost"


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


def _possession_side_expr(columns: list[str], side: str) -> pl.Expr:
    """Return a group-constant flag for whether ``posteam_type`` equals ``side``.

    Formula: ``max(lower(posteam_type) == side)`` over the group. ``posteam_type``
    is constant within a ``(season, week, posteam, defteam)`` group because the
    possessing team is fixed there, so the maximum simply lifts that constant out
    of the group while ignoring rows where the column is null. When
    ``posteam_type`` is absent the result is a typed null, matching the
    missing-source contract in the module docstring.
    """
    if "posteam_type" not in columns:
        return pl.lit(None, dtype=pl.Boolean)
    return pl.col("posteam_type").cast(pl.Utf8, strict=False).str.to_lowercase().eq(side).max()


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


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, name: str) -> pl.Expr:
    """Return a float ratio expression that stays null when the denominator is zero."""
    return (
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(None)
        .cast(pl.Float64)
        .alias(name)
    )


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


def empty_team_box_score_frame() -> pl.DataFrame:
    """Return a typed empty frame with exactly ``PBP_TEAM_BOX_SCORE_COLUMNS``."""
    return pl.DataFrame(
        schema={name: _TEAM_BOX_SCORE_SCHEMA[name] for name in PBP_TEAM_BOX_SCORE_COLUMNS}
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


def _red_zone_drive_aggregations(
    columns: list[str], in_red_zone: pl.Expr, suffix: str
) -> list[pl.Expr]:
    """Build the drive-level red-zone counts for one perspective.

    A red-zone trip is a drive that reached the red zone, so it is counted once however
    many snaps it took inside the 20. A trip converts when the drive ends in a touchdown,
    which ``fixed_drive_result`` records for every play of the drive. Counting drives
    rather than snaps is what makes ``red_zone_td_pct`` comparable to the scraped column
    of the same name; dividing touchdowns by red-zone snaps measures a different quantity.

    Both counts are zero when ``fixed_drive`` is absent, so a season without drive ids
    still emits the invariant schema.

    Args:
        columns: Columns present on the prepared play frame.
        in_red_zone: Per-play condition selecting snaps inside the red zone.
        suffix: ``"_allowed"`` for the defensive perspective, otherwise empty.

    Returns:
        Aggregation expressions for the trip and touchdown-drive counts.

    """
    if "fixed_drive" not in columns:
        return [
            pl.lit(0, dtype=pl.Int64).alias(f"red_zone_trips{suffix}"),
            pl.lit(0, dtype=pl.Int64).alias(f"red_zone_td_drives{suffix}"),
        ]
    drive = pl.col("fixed_drive")
    trips = drive.filter(in_red_zone).n_unique().cast(pl.Int64).alias(f"red_zone_trips{suffix}")
    if "fixed_drive_result" in columns:
        scored = (pl.col("fixed_drive_result") == "Touchdown").fill_null(False)
        td_drives = drive.filter(in_red_zone & scored).n_unique().cast(pl.Int64)
    else:
        td_drives = pl.lit(0, dtype=pl.Int64)
    return [trips, td_drives.alias(f"red_zone_td_drives{suffix}")]


def _situational_aggregations(columns: list[str], *, allowed: bool = False) -> list[pl.Expr]:
    """Build the situational count aggregations for one perspective.

    Formulas (all restricted to plays whose ``play_type`` is in
    ``SITUATIONAL_PLAY_TYPES``, matching the existing pipeline definition):
        - ``third_down_conversions``: plays with ``third_down_converted`` set.
        - ``third_down_fails``: plays with ``third_down_failed`` set.
        - ``fourth_down_conversions``: plays with ``fourth_down_converted`` set.
        - ``fourth_down_fails``: plays with ``fourth_down_failed`` set.
        - ``red_zone_plays``: plays with ``yardline_100 <= RED_ZONE_YARDLINE``.
        - ``red_zone_tds``: red-zone plays whose ``td_team == posteam``.
        - ``red_zone_trips``: distinct ``fixed_drive`` values with at least one red-zone
          play, i.e. drives that reached the red zone rather than snaps taken inside it.
        - ``red_zone_td_drives``: those trips whose ``fixed_drive_result`` is
          ``"Touchdown"``. ``red_zone_td_drives / red_zone_trips`` is the per-trip
          conversion rate that the published ``red_zone_td_pct`` reports.
        - ``two_point_attempts``: plays with ``two_point_attempt`` set.
        - ``two_point_successes``: plays with ``two_point_conv_result == "success"``.
        - ``total_plays``: the number of such plays.
    Third- and fourth-down attempts are the conversion plus fail counts and are
    added after aggregation. When ``allowed`` is true, the same offense-side events are
    counted again but written to ``*_allowed`` columns for the team that defended them.
    """
    if "play_type" in columns:
        counted = pl.col("play_type").is_in(list(SITUATIONAL_PLAY_TYPES)).fill_null(False)
    else:
        counted = pl.lit(False)
    yardline = _nullable_expr(columns, "yardline_100", pl.Float64)
    in_red_zone = (yardline <= RED_ZONE_YARDLINE).fill_null(False)
    if {"td_team", "posteam"} <= set(columns):
        scored_td = (pl.col("td_team") == pl.col("posteam")).fill_null(False)
    else:
        scored_td = pl.lit(False)
    if "two_point_conv_result" in columns:
        two_point_success = (pl.col("two_point_conv_result") == "success").fill_null(False)
    else:
        two_point_success = pl.lit(False)
    suffix = "_allowed" if allowed else ""
    return [
        _count(
            counted & (_flag_expr(columns, "third_down_converted") > 0),
            f"third_down_conversions{suffix}",
        ),
        _count(
            counted & (_flag_expr(columns, "third_down_failed") > 0),
            f"third_down_fails{suffix}",
        ),
        _count(
            counted & (_flag_expr(columns, "fourth_down_converted") > 0),
            f"fourth_down_conversions{suffix}",
        ),
        _count(
            counted & (_flag_expr(columns, "fourth_down_failed") > 0),
            f"fourth_down_fails{suffix}",
        ),
        _count(counted & in_red_zone, f"red_zone_plays{suffix}"),
        _count(counted & in_red_zone & scored_td, f"red_zone_tds{suffix}"),
        *_red_zone_drive_aggregations(columns, in_red_zone, suffix),
        _count(
            counted & (_flag_expr(columns, "two_point_attempt") > 0),
            f"two_point_attempts{suffix}",
        ),
        _count(counted & two_point_success, f"two_point_successes{suffix}"),
        *([] if allowed else [_count(counted, "total_plays")]),
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
            _possession_side_expr(columns, "home").alias(_IS_HOME_OFFENSE),
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
            _possession_side_expr(columns, "away").alias(_IS_HOME_DEFENSE),
            *_situational_aggregations(columns, allowed=True),
        )
        .rename({"defteam": "team_abbr", "posteam": "opponent_abbr"})
        .with_columns(
            (pl.col("third_down_conversions_allowed") + pl.col("third_down_fails_allowed")).alias(
                "third_down_attempts_allowed"
            ),
            (pl.col("fourth_down_conversions_allowed") + pl.col("fourth_down_fails_allowed")).alias(
                "fourth_down_attempts_allowed"
            ),
        )
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


def regular_season_plays(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return the regular-season plays with both teams known (see ``_prepare_plays``)."""
    return _prepare_plays(pbp_df)


def dropback_condition(columns: list[str]) -> pl.Expr:
    """Return the dropback definition shared by every play-by-play family.

    Formula: a scrimmage snap with ``qb_dropback > 0`` that is not a kneel, a spike, or a
    two-point try (see ``_play_conditions``).
    """
    return _play_conditions(columns)["dropback"]


def aggregate_pbp_team_game_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate play-by-play rows into one team-game row of counts and sums.

    The output has one row per ``(season, week, team_abbr, opponent_abbr)`` with
    exactly the columns in ``PBP_TEAM_GAME_COLUMNS``. Apart from ``is_home`` only
    counts and sums are produced; rates are derived downstream as ratios of
    season-to-date sums.

    ``is_home`` is the one non-count column: a boolean saying whether the team
    hosted the game, taken from ``posteam_type`` on either perspective. It is
    context for opponent-adjusted solves, not a statistic, so it is deliberately
    absent from ``constants.PBP_COUNT_COLUMNS`` and ``constants.PBP_STATS`` and is
    never averaged, mirrored, or published.

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
        ),
        # Either perspective identifies the same team-game, so take whichever is
        # present: a team with no offensive snaps still has its side recorded on
        # the plays it defended, and vice versa.
        pl.coalesce(pl.col(_IS_HOME_OFFENSE), pl.col(_IS_HOME_DEFENSE))
        .cast(pl.Boolean)
        .alias("is_home"),
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


def aggregate_pbp_team_box_score_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw play rows into nflreadpy-style per-team-game box-score stats.

    The output uses the nflreadpy team-stat names for the box-score families that can be
    derived directly from play-by-play. When a required source column is absent, the affected
    derived stat is left null so callers can fall back to another source. Counts are zero when
    the source column exists but no play in the game satisfies the condition.

    Formulas, verified against nflverse team stats over 1999-2025 (a four-season sample
    unless noted): every match rate below is the share of team-games where the derived value
    equals nflverse's to within 1e-4.
        - ``pass_attempts``: ``pass_attempt`` set, excluding sacks and two-point tries (100%
          match). ``pass_attempt`` is nflverse's own canonical flag; a sack carries
          ``pass_attempt = 1`` too, so nflverse's own ``pass_attempts`` explicitly excludes
          sacks, unlike the looser ``play_type == "pass"`` this module used before.
        - ``pass_completions``: pass attempts with ``complete_pass`` set (100%).
        - ``pass_yards``: ``yards_gained`` summed over pass attempts (99.89%).
        - ``pass_touchdowns`` / ``interceptions_thrown``: pass attempts with those flags set
          (100%).
        - ``times_sacked``: plays with ``sack`` set (100%).
        - ``passing_epa``: ``qb_epa`` (not ``epa``) over every ``pass_attempt`` play, sacks and
          two-point tries included (100% on a single season, 98.84% over four). ``qb_epa``
          attributes EPA to the quarterback's passing line the way nflverse's own team stat
          does; summing ``epa`` over the same plays matched only 69.54% before this fix.
        - ``passing_cpoe``: mean ``cpoe`` over pass attempts with a published CPOE.
        - ``rush_attempts``: ``rush_attempt`` set, excluding two-point tries (100%).
          ``rush_attempt`` is nflverse's own canonical flag; a kneel carries
          ``rush_attempt = 1`` despite ``rush = 0``, so the looser ``play_type in {"run",
          "qb_kneel"}`` this module used before agreed with it by coincidence on ordinary
          plays but not universally.
        - ``rush_yards``: summed over rush attempts (99.94%). ``rush_touchdowns``: rush
          attempts with that flag set (100%).
        - ``rushing_epa``: ``epa`` over every ``rush_attempt`` play, two-point tries included
          (99.78%).
        - ``fumbles`` / ``fumbles_lost``: plays with those flags set, excluding special-teams
          plays (87.49% / 98.03%, up from 73.63% / 90.35% when special-teams fumbles were
          included). nflverse's team-level ``fumbles`` is an offense-only stat (the sum of a
          player's sack, rushing and receiving fumbles); a residual gap remains for fumbles on
          aborted snaps, which nflverse's own player-level fumble categories also do not
          cleanly attribute.
        - ``first_downs``: ``first_down_pass + first_down_rush`` when those component flags exist.
        - ``2pt_conversions``: plays whose ``two_point_conv_result == "success"`` (94.35%). No
          simple redefinition closed the residual gap; tracing individual mismatches found
          nflverse team stats itself disagreeing with its own play-by-play on rare plays (see
          ``models/pbp_vs_nflverse_m54_2/COMPARISON.md``), which is not fixable from this side.
        - ``total_yards``: ``pass_yards + rush_yards + sack_yards_lost``. nflverse defines
          ``total_yards`` as ``pass_yards + rush_yards - yards_lost_from_sacks`` and stores
          the sack losses as a negative number, so the sack yardage is added back rather
          than deducted. Verified against every nflverse team-game of 2000-2025: the
          identity holds on 13418 of 13418 rows.
        - ``penalties`` / ``penalty_yards``: grouped by ``penalty_team`` across offensive and
          defensive plays.
        - ``def_sacks`` / ``def_interceptions``: sacks and interceptions faced by the opponent.

    Args:
        pbp_df: Raw play-by-play rows. Optional source columns may be absent.

    Returns:
        One row per ``(season, week, team_abbr, opponent_abbr)`` with the typed columns in
        ``PBP_TEAM_BOX_SCORE_COLUMNS``.

    Raises:
        ValueError: If an identity column required to key team-games is missing.

    """
    if pbp_df.height == 0:
        return empty_team_box_score_frame()

    missing = [column for column in _REQUIRED_PBP_COLUMNS if column not in pbp_df.columns]
    if missing:
        raise ValueError(f"pbp_df is missing required columns: {', '.join(missing)}")

    plays = _prepare_plays(pbp_df)
    if plays.height == 0:
        return empty_team_box_score_frame()

    columns = plays.columns
    is_two_point = _flag_expr(columns, "two_point_attempt") > 0
    is_sack = _flag_expr(columns, "sack") > 0
    is_special = _flag_expr(columns, "special") > 0
    # ``pass_attempt``/``rush_attempt`` are nflverse's own canonical stat-counting flags,
    # verified against nflverse team stats over 1999-2025 (four-season sample): they differ
    # from the looser ``pass``/``rush`` indicators, most visibly that a sack carries
    # ``pass_attempt = 1`` (excluded below to match nflverse's ``pass_attempts``) and a kneel
    # carries ``rush_attempt = 1`` despite ``rush = 0``.
    has_attempt_flags = {"pass_attempt", "rush_attempt"} <= set(columns)
    is_pass_flagged = _flag_expr(columns, "pass_attempt") > 0
    is_rush_flagged = _flag_expr(columns, "rush_attempt") > 0
    # Counting/yardage attempts exclude sacks (for passes) and two-point tries (both sides);
    # EPA attempts keep sacks and two-point tries in, which is what reproduces nflverse's
    # ``passing_epa``/``rushing_epa`` (100% and 99.78% match on the verification sample).
    is_pass_attempt = is_pass_flagged & ~is_sack & ~is_two_point
    is_pass_epa_play = is_pass_flagged
    is_rush_attempt = is_rush_flagged & ~is_two_point
    is_rush_epa_play = is_rush_flagged
    yards = _value_expr(columns, "yards_gained")
    epa = _value_expr(columns, "epa")
    qb_epa = _value_expr(columns, "qb_epa")

    offense_aggs: list[pl.Expr] = []
    if "season_type" in columns:
        offense_aggs.append(pl.col("season_type").drop_nulls().first().alias("season_type"))
    if "complete_pass" in columns and has_attempt_flags:
        offense_aggs.append(
            _count(is_pass_attempt & (_flag_expr(columns, "complete_pass") > 0), "pass_completions")
        )
    if has_attempt_flags:
        offense_aggs.append(_count(is_pass_attempt, "pass_attempts"))
        offense_aggs.append(_sum_when(is_pass_attempt, yards, "pass_yards"))
        offense_aggs.append(_count(is_rush_attempt, "rush_attempts"))
        offense_aggs.append(_sum_when(is_rush_attempt, yards, "rush_yards"))
    if "pass_touchdown" in columns and has_attempt_flags:
        offense_aggs.append(
            _count(is_pass_attempt & (_flag_expr(columns, "pass_touchdown") > 0), "pass_touchdowns")
        )
    if "interception" in columns and has_attempt_flags:
        offense_aggs.append(
            _count(
                is_pass_attempt & (_flag_expr(columns, "interception") > 0),
                "interceptions_thrown",
            )
        )
    if "sack" in columns:
        offense_aggs.append(_count(is_sack, "times_sacked"))
        offense_aggs.append(
            _sum_when(
                is_sack,
                pl.when(yards < 0).then(-yards).otherwise(0.0),
                _PBP_SACK_YARDS_LOST,
            )
        )
    if "qb_epa" in columns and "pass_attempt" in columns:
        offense_aggs.append(_sum_when(is_pass_epa_play, qb_epa, "passing_epa"))
    if "epa" in columns and "rush_attempt" in columns:
        offense_aggs.append(_sum_when(is_rush_epa_play, epa, "rushing_epa"))
    if "cpoe" in columns and has_attempt_flags:
        has_cpoe = is_pass_attempt & pl.col("cpoe").is_not_null()
        offense_aggs.append(
            _sum_when(has_cpoe, pl.col("cpoe").cast(pl.Float64), _PBP_PASSING_CPOE_SUM)
        )
        offense_aggs.append(_count(has_cpoe, _PBP_PASSING_CPOE_COUNT))
    if "rush_touchdown" in columns and has_attempt_flags:
        offense_aggs.append(
            _count(is_rush_attempt & (_flag_expr(columns, "rush_touchdown") > 0), "rush_touchdowns")
        )
    if "fumble" in columns:
        # nflverse's team-level ``fumbles`` is an offense-only stat (rushing, receiving and
        # sack fumbles); special-teams fumbles (kickoff/punt) are excluded to match it
        # (87.5% match, up from 73.6% when special-teams plays were included).
        offense_aggs.append(_count((_flag_expr(columns, "fumble") > 0) & ~is_special, "fumbles"))
    if "fumble_lost" in columns:
        offense_aggs.append(
            _count((_flag_expr(columns, "fumble_lost") > 0) & ~is_special, "fumbles_lost")
        )
    if {"first_down_pass", "first_down_rush"} & set(columns):
        first_down_pass = _flag_expr(columns, "first_down_pass")
        first_down_rush = _flag_expr(columns, "first_down_rush")
        offense_aggs.append(
            (first_down_pass + first_down_rush).sum().cast(pl.Float64).alias("first_downs")
        )
    if "two_point_conv_result" in columns:
        offense_aggs.append(
            _count(
                (pl.col("two_point_conv_result") == "success").fill_null(False),
                "2pt_conversions",
            )
        )

    offense = (
        plays.group_by(["season", "week", "posteam", "defteam"])
        .agg(offense_aggs)
        .rename({"posteam": "team_abbr", "defteam": "opponent_abbr"})
        if offense_aggs
        else empty_team_box_score_frame()
    )

    defense_aggs: list[pl.Expr] = []
    if "sack" in columns:
        defense_aggs.append(_count(is_sack, "def_sacks"))
    if "interception" in columns:
        defense_aggs.append(
            _count(is_pass_attempt & (_flag_expr(columns, "interception") > 0), "def_interceptions")
        )

    defense = (
        plays.group_by(["season", "week", "defteam", "posteam"])
        .agg(defense_aggs)
        .rename({"defteam": "team_abbr", "posteam": "opponent_abbr"})
        if defense_aggs
        else empty_team_box_score_frame()
    )

    penalty_frame = empty_team_box_score_frame()
    if {"penalty", "penalty_team"} <= set(columns):
        penalty_rows = plays.filter(
            (_flag_expr(columns, "penalty") > 0)
            & pl.col("penalty_team").is_not_null()
            & (pl.col("penalty_team").cast(pl.Utf8).str.strip_chars() != "")
        ).with_columns(
            pl.col("penalty_team").cast(pl.String).alias("team_abbr"),
            pl.when(pl.col("penalty_team") == pl.col("posteam"))
            .then(pl.col("defteam"))
            .otherwise(pl.col("posteam"))
            .alias("opponent_abbr")
            .cast(pl.String),
        )
        penalty_aggs: list[pl.Expr] = []
        penalty_aggs.append(pl.len().cast(pl.Float64).alias("penalties"))
        if "penalty_yards" in columns:
            penalty_aggs.append(
                pl.col("penalty_yards")
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .sum()
                .alias("penalty_yards")
            )
        penalty_frame = penalty_rows.group_by(["season", "week", "team_abbr", "opponent_abbr"]).agg(
            penalty_aggs
        )

    keys = ["season", "week", "team_abbr", "opponent_abbr"]
    # Join the perspectives that actually produced rows. Seeding `combined` from one of them
    # and then joining that same frame again would collide every non-key column with itself.
    parts = [frame for frame in (offense, defense, penalty_frame) if frame.height]
    if not parts:
        return empty_team_box_score_frame()
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.join(part, on=keys, how="full", coalesce=True)

    if combined.height == 0:
        return empty_team_box_score_frame()

    present_numeric = [
        column
        for column in (
            "pass_completions",
            "pass_attempts",
            "pass_yards",
            "pass_touchdowns",
            "interceptions_thrown",
            "times_sacked",
            "passing_epa",
            "rushing_epa",
            "rush_attempts",
            "rush_yards",
            "rush_touchdowns",
            "fumbles",
            "fumbles_lost",
            "first_downs",
            "2pt_conversions",
            "penalties",
            "penalty_yards",
            "def_sacks",
            "def_interceptions",
            _PBP_PASSING_CPOE_SUM,
            _PBP_PASSING_CPOE_COUNT,
            _PBP_SACK_YARDS_LOST,
        )
        if column in combined.columns
    ]
    if present_numeric:
        combined = combined.with_columns(
            pl.col(column).fill_null(0.0) for column in present_numeric
        )

    derived: list[pl.Expr] = []
    if {_PBP_PASSING_CPOE_SUM, _PBP_PASSING_CPOE_COUNT} <= set(combined.columns):
        derived.append(
            _safe_ratio(
                pl.col(_PBP_PASSING_CPOE_SUM),
                pl.col(_PBP_PASSING_CPOE_COUNT),
                "passing_cpoe",
            )
        )
    if {"pass_yards", "rush_yards", _PBP_SACK_YARDS_LOST} <= set(combined.columns):
        derived.append(
            (pl.col("pass_yards") + pl.col("rush_yards") + pl.col(_PBP_SACK_YARDS_LOST))
            .cast(pl.Float64)
            .alias("total_yards")
        )
    if derived:
        combined = combined.with_columns(derived)

    combined = combined.drop(
        [
            column
            for column in (
                _PBP_PASSING_CPOE_SUM,
                _PBP_PASSING_CPOE_COUNT,
                _PBP_SACK_YARDS_LOST,
            )
            if column in combined.columns
        ]
    )

    non_stat_columns = set(keys) | {"season_type"}
    missing_stats = [
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column in PBP_TEAM_BOX_SCORE_COLUMNS
        if column not in non_stat_columns and column not in combined.columns
    ]
    if missing_stats:
        combined = combined.with_columns(missing_stats)
    if "season_type" not in combined.columns:
        combined = combined.with_columns(pl.lit(None, dtype=pl.String).alias("season_type"))

    return combined.select(
        pl.col(name).cast(_TEAM_BOX_SCORE_SCHEMA[name]) for name in PBP_TEAM_BOX_SCORE_COLUMNS
    ).sort(["season", "week", "team_abbr"])
