"""Tests for play-by-play team-game aggregation."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nfl_predictor.utils.polars import pbp

# Default values for one fixture play. Every fixture row starts from these and
# overrides only the fields that matter for the hand-computed expectations.
_PLAY_DEFAULTS: dict[str, Any] = {
    "season": 2024,
    "week": 1,
    "season_type": "REG",
    "posteam": "",
    "defteam": "",
    "play_type": "no_play",
    "qb_dropback": 0,
    "rush": 0,
    "qb_kneel": 0,
    "qb_spike": 0,
    "down": None,
    "yards_gained": 0.0,
    "epa": 0.0,
    "success": 0,
    "yardline_100": None,
    "third_down_converted": 0,
    "third_down_failed": 0,
    "fourth_down_converted": 0,
    "fourth_down_failed": 0,
    "two_point_attempt": 0,
    "two_point_conv_result": None,
    "td_team": None,
    "special": 0,
}

_FIXTURE_DTYPES: dict[str, Any] = {
    "down": pl.Int64,
    "yardline_100": pl.Float64,
    "yards_gained": pl.Float64,
    "epa": pl.Float64,
    "two_point_conv_result": pl.String,
    "td_team": pl.String,
}


def _play(**overrides: Any) -> dict[str, Any]:
    """Return a single fixture play row with the shared defaults applied."""
    return _PLAY_DEFAULTS | overrides


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a play-by-play frame with stable dtypes for the fixture rows."""
    return pl.DataFrame(rows, schema_overrides=_FIXTURE_DTYPES)


def _fixture_plays() -> pl.DataFrame:
    """Return the 16-play, two-game hand-built fixture.

    Game 1 is KC vs BUF and game 2 is SF vs DAL, all in season 2024 week 1.
    The per-play values below are the inputs for every hand-computed
    expectation in ``_EXPECTED``.

    KC offense (5 plays):
        K1 dropback, 1st down, 25 yards, epa 1.5, success, at the BUF 75.
        K2 rush, 2nd down, 14 yards, epa 0.8, success, at the BUF 50.
        K3 dropback, 3rd down, 18 yards, epa 2.0, success, at the BUF 18,
           third down converted, touchdown scored by KC.
        K4 kneel (kneel and rush flags both set), 1st down, -1 yards, epa -0.3.
        K5 dropback, 4th down, 0 yards, epa -1.2, fourth down failed.
    BUF offense (4 plays):
        B1 rush, 1st down, 3 yards, epa -0.1.
        B2 dropback, 2nd down, 30 yards, epa 2.2, success.
        B3 rush, 1st down, 0 yards, epa -0.5.
        B4 spike (spike and dropback flags both set), 2nd down, 0 yards, epa -0.4.
    Game 1 special teams (1 play):
        S1 KC punt, epa -0.6.

    SF offense (3 plays):
        F1 dropback, 3rd down, 12 yards, epa 0.6, success, at the DAL 25,
           third down converted.
        F2 rush, 1st down, 13 yards, epa 1.1, success, at the DAL 13.
        F3 rush two-point try, no down, 2 yards, epa 0.9, success, at the DAL 2,
           two-point result "success".
    DAL offense (2 plays):
        D1 dropback, 2nd down, 5 yards, epa 0.1, success, at the SF 80.
        D2 rush, 1st down, -3 yards, epa -1.0, at the SF 75.
    Game 2 special teams (1 play):
        S2 DAL field goal, epa 1.4.
    """
    return _frame(
        [
            # --- Game 1: KC offense -------------------------------------------------
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=1,
                yards_gained=25.0,
                epa=1.5,
                success=1,
                yardline_100=75.0,
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="run",
                rush=1,
                down=2,
                yards_gained=14.0,
                epa=0.8,
                success=1,
                yardline_100=50.0,
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=3,
                yards_gained=18.0,
                epa=2.0,
                success=1,
                yardline_100=18.0,
                third_down_converted=1,
                td_team="KC",
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="qb_kneel",
                qb_kneel=1,
                rush=1,
                down=1,
                yards_gained=-1.0,
                epa=-0.3,
                yardline_100=60.0,
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=4,
                yards_gained=0.0,
                epa=-1.2,
                yardline_100=45.0,
                fourth_down_failed=1,
            ),
            # --- Game 1: BUF offense ------------------------------------------------
            _play(
                posteam="BUF",
                defteam="KC",
                play_type="run",
                rush=1,
                down=1,
                yards_gained=3.0,
                epa=-0.1,
                yardline_100=70.0,
            ),
            _play(
                posteam="BUF",
                defteam="KC",
                play_type="pass",
                qb_dropback=1,
                down=2,
                yards_gained=30.0,
                epa=2.2,
                success=1,
                yardline_100=67.0,
            ),
            _play(
                posteam="BUF",
                defteam="KC",
                play_type="run",
                rush=1,
                down=1,
                yards_gained=0.0,
                epa=-0.5,
                yardline_100=37.0,
            ),
            _play(
                posteam="BUF",
                defteam="KC",
                play_type="qb_spike",
                qb_spike=1,
                qb_dropback=1,
                down=2,
                yards_gained=0.0,
                epa=-0.4,
                yardline_100=37.0,
            ),
            # --- Game 1: special teams ----------------------------------------------
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="punt",
                special=1,
                down=4,
                yards_gained=45.0,
                epa=-0.6,
                yardline_100=70.0,
            ),
            # --- Game 2: SF offense -------------------------------------------------
            _play(
                posteam="SF",
                defteam="DAL",
                play_type="pass",
                qb_dropback=1,
                down=3,
                yards_gained=12.0,
                epa=0.6,
                success=1,
                yardline_100=25.0,
                third_down_converted=1,
            ),
            _play(
                posteam="SF",
                defteam="DAL",
                play_type="run",
                rush=1,
                down=1,
                yards_gained=13.0,
                epa=1.1,
                success=1,
                yardline_100=13.0,
            ),
            _play(
                posteam="SF",
                defteam="DAL",
                play_type="run",
                rush=1,
                yards_gained=2.0,
                epa=0.9,
                success=1,
                yardline_100=2.0,
                two_point_attempt=1,
                two_point_conv_result="success",
            ),
            # --- Game 2: DAL offense ------------------------------------------------
            _play(
                posteam="DAL",
                defteam="SF",
                play_type="pass",
                qb_dropback=1,
                down=2,
                yards_gained=5.0,
                epa=0.1,
                success=1,
                yardline_100=80.0,
            ),
            _play(
                posteam="DAL",
                defteam="SF",
                play_type="run",
                rush=1,
                down=1,
                yards_gained=-3.0,
                epa=-1.0,
                yardline_100=75.0,
            ),
            # --- Game 2: special teams ----------------------------------------------
            _play(
                posteam="DAL",
                defteam="SF",
                play_type="field_goal",
                special=1,
                down=4,
                epa=1.4,
                yardline_100=20.0,
            ),
        ]
    )


# Hand-computed expectations for every produced column, derived from the play
# list documented in ``_fixture_plays``.
#
# Allowed denominators come from the opponent's plays in the same game:
#     KC faced BUF's 1 dropback and 2 carries; BUF faced KC's 3 dropbacks and
#     1 carry; SF faced DAL's 1 dropback and 1 carry; DAL faced SF's 1 dropback
#     and 2 carries.
#
# KC: snaps K1-K5 (5); dropbacks K1, K3, K5 (the kneel is excluded); carries K2
#     only (the kneel row sets the rush flag here, but a kneel is never a carry).
#     pass epa 1.5 + 2.0 - 1.2 = 2.3; rush epa 0.8.
#     explosive pass K1 (25 >= 20); explosive rush K2 (14 >= 12); no stuffed rush.
#     early downs K1, K2, K4 (3); early-down pass K1 (1).
#     red zone K3 (18 <= 20) with a KC touchdown; total plays K1-K5 (5).
#     Special teams: punt epa -0.6 credited for, 1 play.
# BUF: snaps B1-B4 (4); dropback B2 only (the spike is excluded); carries B1, B3.
#     pass epa 2.2; rush epa -0.1 - 0.5 = -0.6; explosive pass B2 (30 >= 20).
#     stuffed rush B3 (0 <= 0); early downs B1-B4 (4); early-down pass B2 (1).
#     Special teams: KC's punt is charged against BUF, 1 play.
# SF: snaps F1-F3 (3); dropback F1; carry F2 only. F3 is a two-point try, so it is a
#     scrimmage snap but not a carry, and its epa and success do not count.
#     pass epa 0.6; rush epa 1.1; explosive rush F2 (13 >= 12);
#     F1 gains 12 which is below the 20-yard explosive pass threshold.
#     early down F2 only (F1 is 3rd down, F3 has no down).
#     red zone F2 (13) and F3 (2) with no touchdown; two-point try F3 succeeded.
#     Special teams: DAL's field goal is charged against SF, 1 play.
# DAL: snaps D1, D2; dropback D1; carry D2; stuffed rush D2 (-3 <= 0).
#     early downs D1, D2; early-down pass D1.
#     Special teams: field goal epa 1.4 credited for, 1 play.
_EXPECTED: dict[str, dict[str, Any]] = {
    "KC": {
        "season": 2024,
        "week": 1,
        "team_abbr": "KC",
        "opponent_abbr": "BUF",
        "offensive_snaps": 5,
        "defensive_snaps": 4,
        "dropbacks": 3,
        "carries": 1,
        "dropbacks_allowed": 1,
        "carries_allowed": 2,
        "pass_epa_sum": 2.3,
        "rush_epa_sum": 0.8,
        "pass_epa_allowed_sum": 2.2,
        "rush_epa_allowed_sum": -0.6,
        "pass_success_count": 2,
        "rush_success_count": 1,
        "pass_success_allowed_count": 1,
        "rush_success_allowed_count": 0,
        "explosive_pass_count": 1,
        "explosive_rush_count": 1,
        "stuffed_rush_count": 0,
        "explosive_pass_allowed_count": 1,
        "explosive_rush_allowed_count": 0,
        "stuffed_rush_allowed_count": 1,
        "early_down_plays": 3,
        "early_down_passes": 1,
        "st_epa_for": -0.6,
        "st_epa_against": 0.0,
        "st_plays": 1,
        "third_down_conversions": 1,
        "third_down_fails": 0,
        "third_down_attempts": 1,
        "fourth_down_conversions": 0,
        "fourth_down_fails": 1,
        "fourth_down_attempts": 1,
        "red_zone_plays": 1,
        "red_zone_tds": 1,
        "two_point_attempts": 0,
        "two_point_successes": 0,
        "total_plays": 5,
    },
    "BUF": {
        "season": 2024,
        "week": 1,
        "team_abbr": "BUF",
        "opponent_abbr": "KC",
        "offensive_snaps": 4,
        "defensive_snaps": 5,
        "dropbacks": 1,
        "carries": 2,
        "dropbacks_allowed": 3,
        "carries_allowed": 1,
        "pass_epa_sum": 2.2,
        "rush_epa_sum": -0.6,
        "pass_epa_allowed_sum": 2.3,
        "rush_epa_allowed_sum": 0.8,
        "pass_success_count": 1,
        "rush_success_count": 0,
        "pass_success_allowed_count": 2,
        "rush_success_allowed_count": 1,
        "explosive_pass_count": 1,
        "explosive_rush_count": 0,
        "stuffed_rush_count": 1,
        "explosive_pass_allowed_count": 1,
        "explosive_rush_allowed_count": 1,
        "stuffed_rush_allowed_count": 0,
        "early_down_plays": 4,
        "early_down_passes": 1,
        "st_epa_for": 0.0,
        "st_epa_against": -0.6,
        "st_plays": 1,
        "third_down_conversions": 0,
        "third_down_fails": 0,
        "third_down_attempts": 0,
        "fourth_down_conversions": 0,
        "fourth_down_fails": 0,
        "fourth_down_attempts": 0,
        "red_zone_plays": 0,
        "red_zone_tds": 0,
        "two_point_attempts": 0,
        "two_point_successes": 0,
        "total_plays": 4,
    },
    "SF": {
        "season": 2024,
        "week": 1,
        "team_abbr": "SF",
        "opponent_abbr": "DAL",
        "offensive_snaps": 3,
        "defensive_snaps": 2,
        "dropbacks": 1,
        "carries": 1,
        "dropbacks_allowed": 1,
        "carries_allowed": 1,
        "pass_epa_sum": 0.6,
        "rush_epa_sum": 1.1,
        "pass_epa_allowed_sum": 0.1,
        "rush_epa_allowed_sum": -1.0,
        "pass_success_count": 1,
        "rush_success_count": 1,
        "pass_success_allowed_count": 1,
        "rush_success_allowed_count": 0,
        "explosive_pass_count": 0,
        "explosive_rush_count": 1,
        "stuffed_rush_count": 0,
        "explosive_pass_allowed_count": 0,
        "explosive_rush_allowed_count": 0,
        "stuffed_rush_allowed_count": 1,
        "early_down_plays": 1,
        "early_down_passes": 0,
        "st_epa_for": 0.0,
        "st_epa_against": 1.4,
        "st_plays": 1,
        "third_down_conversions": 1,
        "third_down_fails": 0,
        "third_down_attempts": 1,
        "fourth_down_conversions": 0,
        "fourth_down_fails": 0,
        "fourth_down_attempts": 0,
        "red_zone_plays": 2,
        "red_zone_tds": 0,
        "two_point_attempts": 1,
        "two_point_successes": 1,
        "total_plays": 3,
    },
    "DAL": {
        "season": 2024,
        "week": 1,
        "team_abbr": "DAL",
        "opponent_abbr": "SF",
        "offensive_snaps": 2,
        "defensive_snaps": 3,
        "dropbacks": 1,
        "carries": 1,
        "dropbacks_allowed": 1,
        "carries_allowed": 1,
        "pass_epa_sum": 0.1,
        "rush_epa_sum": -1.0,
        "pass_epa_allowed_sum": 0.6,
        "rush_epa_allowed_sum": 1.1,
        "pass_success_count": 1,
        "rush_success_count": 0,
        "pass_success_allowed_count": 1,
        "rush_success_allowed_count": 1,
        "explosive_pass_count": 0,
        "explosive_rush_count": 0,
        "stuffed_rush_count": 1,
        "explosive_pass_allowed_count": 0,
        "explosive_rush_allowed_count": 1,
        "stuffed_rush_allowed_count": 0,
        "early_down_plays": 2,
        "early_down_passes": 1,
        "st_epa_for": 1.4,
        "st_epa_against": 0.0,
        "st_plays": 1,
        "third_down_conversions": 0,
        "third_down_fails": 0,
        "third_down_attempts": 0,
        "fourth_down_conversions": 0,
        "fourth_down_fails": 0,
        "fourth_down_attempts": 0,
        "red_zone_plays": 0,
        "red_zone_tds": 0,
        "two_point_attempts": 0,
        "two_point_successes": 0,
        "total_plays": 2,
    },
}

_MIRRORED_COLUMNS = (
    ("pass_epa_allowed_sum", "pass_epa_sum"),
    ("rush_epa_allowed_sum", "rush_epa_sum"),
    ("pass_success_allowed_count", "pass_success_count"),
    ("rush_success_allowed_count", "rush_success_count"),
    ("explosive_pass_allowed_count", "explosive_pass_count"),
    ("explosive_rush_allowed_count", "explosive_rush_count"),
    ("stuffed_rush_allowed_count", "stuffed_rush_count"),
    ("defensive_snaps", "offensive_snaps"),
    ("dropbacks_allowed", "dropbacks"),
    ("carries_allowed", "carries"),
)


def _row_for(result: pl.DataFrame, team: str) -> dict[str, Any]:
    """Return the single team-game row for a team as a dictionary."""
    rows = result.filter(pl.col("team_abbr") == team).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_thresholds_are_named_constants() -> None:
    """Threshold constants hold the documented values for this repo."""
    assert pbp.EXPLOSIVE_PASS_YARDS == 20
    assert pbp.EXPLOSIVE_RUSH_YARDS == 12
    assert pbp.STUFFED_RUSH_YARDS == 0
    assert pbp.EARLY_DOWN_MAX == 2
    assert pbp.RED_ZONE_YARDLINE == 20


def test_team_game_columns_start_with_identity_keys() -> None:
    """The exported column order lists the identity keys before the stat columns."""
    assert pbp.PBP_TEAM_GAME_COLUMNS[:4] == ["season", "week", "team_abbr", "opponent_abbr"]
    assert len(set(pbp.PBP_TEAM_GAME_COLUMNS)) == len(pbp.PBP_TEAM_GAME_COLUMNS)


def test_aggregation_matches_hand_computed_values() -> None:
    """Every produced column matches the hand-computed value for all four team-games."""
    result = pbp.aggregate_pbp_team_game_stats(_fixture_plays())

    assert result.columns == pbp.PBP_TEAM_GAME_COLUMNS
    assert result.height == 4

    for team, expected in _EXPECTED.items():
        row = _row_for(result, team)
        for column in pbp.PBP_TEAM_GAME_COLUMNS:
            actual = row[column]
            wanted = expected[column]
            if isinstance(wanted, float):
                assert actual == pytest.approx(wanted), f"{team}.{column}"
            else:
                assert actual == wanted, f"{team}.{column}"


def test_allowed_columns_mirror_the_opponent_offense() -> None:
    """Each team's allowed columns equal its opponent's matching offensive columns."""
    result = pbp.aggregate_pbp_team_game_stats(_fixture_plays())

    for team, opponent in (("KC", "BUF"), ("BUF", "KC"), ("SF", "DAL"), ("DAL", "SF")):
        team_row = _row_for(result, team)
        opponent_row = _row_for(result, opponent)
        for allowed_column, offense_column in _MIRRORED_COLUMNS:
            assert team_row[allowed_column] == pytest.approx(opponent_row[offense_column]), (
                f"{team}.{allowed_column} should mirror {opponent}.{offense_column}"
            )


def test_kneels_and_spikes_are_snaps_but_not_dropbacks_or_carries() -> None:
    """A kneel and a spike each add an offensive snap without a dropback or carry."""
    plays = _frame(
        [
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="qb_kneel",
                qb_kneel=1,
                rush=1,
                down=1,
                yards_gained=-1.0,
                epa=-0.3,
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="qb_spike",
                qb_spike=1,
                qb_dropback=1,
                down=2,
                yards_gained=0.0,
                epa=-0.4,
            ),
        ]
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)
    row = _row_for(result, "KC")
    opponent_row = _row_for(result, "BUF")

    assert row["offensive_snaps"] == 2
    assert row["dropbacks"] == 0
    assert row["carries"] == 0
    # The same exclusions apply from the defending side.
    assert opponent_row["defensive_snaps"] == 2
    assert opponent_row["dropbacks_allowed"] == 0
    assert opponent_row["carries_allowed"] == 0
    assert row["pass_epa_sum"] == pytest.approx(0.0)
    assert row["rush_epa_sum"] == pytest.approx(0.0)
    assert row["stuffed_rush_count"] == 0
    assert row["early_down_plays"] == 2
    assert row["early_down_passes"] == 0
    assert row["total_plays"] == 2


def test_special_teams_play_credits_both_teams() -> None:
    """A special-teams play credits the possessing team and charges the other team."""
    plays = _frame(
        [
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="punt",
                special=1,
                epa=-0.6,
            ),
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=1,
                yards_gained=5.0,
                epa=0.2,
            ),
        ]
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)
    kc_row = _row_for(result, "KC")
    buf_row = _row_for(result, "BUF")

    assert kc_row["st_epa_for"] == pytest.approx(-0.6)
    assert kc_row["st_epa_against"] == pytest.approx(0.0)
    assert kc_row["st_plays"] == 1
    assert buf_row["st_epa_for"] == pytest.approx(0.0)
    assert buf_row["st_epa_against"] == pytest.approx(-0.6)
    assert buf_row["st_plays"] == 1
    assert kc_row["offensive_snaps"] == 1
    assert buf_row["defensive_snaps"] == 1


def test_special_teams_flag_column_prefers_the_first_candidate() -> None:
    """The special-teams flag resolves in candidate order and is None when absent."""
    assert pbp.special_teams_flag_column(["special", "special_teams_play"]) == "special"
    assert pbp.special_teams_flag_column(["special_teams_play"]) == "special_teams_play"
    assert pbp.special_teams_flag_column(["epa"]) is None


def test_empty_input_returns_typed_empty_frame() -> None:
    """Empty input returns a typed empty frame with exactly the expected columns."""
    result = pbp.aggregate_pbp_team_game_stats(pl.DataFrame())

    assert result.height == 0
    assert result.columns == pbp.PBP_TEAM_GAME_COLUMNS
    assert result.schema["season"] == pl.Int64
    assert result.schema["team_abbr"] == pl.String
    assert result.schema["pass_epa_sum"] == pl.Float64
    assert result.schema["offensive_snaps"] == pl.Int64
    assert result.schema["dropbacks_allowed"] == pl.Int64
    assert result.schema["carries_allowed"] == pl.Int64


def test_missing_identity_columns_raise() -> None:
    """Missing identity columns raise instead of producing unkeyed rows."""
    plays = pl.DataFrame({"season": [2024], "week": [1], "posteam": ["KC"]})

    with pytest.raises(ValueError, match="defteam"):
        pbp.aggregate_pbp_team_game_stats(plays)


def test_missing_optional_columns_use_documented_defaults() -> None:
    """Missing optional source columns fall back to documented defaults without crashing."""
    plays = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "posteam": ["KC", "KC"],
            "defteam": ["BUF", "BUF"],
            "qb_dropback": [1, 0],
            "rush": [0, 1],
        }
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)
    row = _row_for(result, "KC")

    assert result.columns == pbp.PBP_TEAM_GAME_COLUMNS
    assert row["offensive_snaps"] == 2
    assert row["dropbacks"] == 1
    assert row["carries"] == 1
    # BUF never has possession here, so KC faced no dropbacks and no carries.
    assert row["dropbacks_allowed"] == 0
    assert row["carries_allowed"] == 0
    # No epa column, so the EPA sums default to 0.0.
    assert row["pass_epa_sum"] == pytest.approx(0.0)
    assert row["rush_epa_sum"] == pytest.approx(0.0)
    # No success column, so no play counts as a success.
    assert row["pass_success_count"] == 0
    assert row["rush_success_count"] == 0
    # No yards_gained column, so yards default to 0.0: nothing is explosive and
    # every rush counts as stuffed.
    assert row["explosive_pass_count"] == 0
    assert row["explosive_rush_count"] == 0
    assert row["stuffed_rush_count"] == 1
    # No down column, so no play counts as an early down.
    assert row["early_down_plays"] == 0
    assert row["early_down_passes"] == 0
    # No play_type column, so the situational counts stay at zero.
    assert row["third_down_attempts"] == 0
    assert row["fourth_down_attempts"] == 0
    assert row["red_zone_plays"] == 0
    assert row["red_zone_tds"] == 0
    assert row["two_point_attempts"] == 0
    assert row["total_plays"] == 0


def test_missing_special_teams_flag_yields_null_special_teams_columns() -> None:
    """Without a special-teams flag column the special-teams stats are null, not zero."""
    plays = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "posteam": ["KC"],
            "defteam": ["BUF"],
            "qb_dropback": [1],
            "epa": [0.5],
        }
    )

    row = _row_for(pbp.aggregate_pbp_team_game_stats(plays), "KC")

    assert row["st_epa_for"] is None
    assert row["st_epa_against"] is None
    assert row["st_plays"] is None
    assert row["pass_epa_sum"] == pytest.approx(0.5)


def test_alternate_special_teams_flag_column_is_used() -> None:
    """The alternate special-teams flag name is honored when the primary name is absent."""
    plays = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "posteam": ["KC"],
            "defteam": ["BUF"],
            "special_teams_play": [1],
            "epa": [-0.9],
        }
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)

    assert _row_for(result, "KC")["st_epa_for"] == pytest.approx(-0.9)
    assert _row_for(result, "BUF")["st_epa_against"] == pytest.approx(-0.9)


def test_non_regular_season_rows_are_excluded() -> None:
    """Rows outside the regular season are dropped when season_type exists."""
    plays = _frame(
        [
            _play(
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=1,
                yards_gained=9.0,
                epa=0.4,
            ),
            _play(
                season_type="POST",
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                down=1,
                yards_gained=40.0,
                epa=3.0,
            ),
        ]
    )

    row = _row_for(pbp.aggregate_pbp_team_game_stats(plays), "KC")

    assert row["dropbacks"] == 1
    assert row["pass_epa_sum"] == pytest.approx(0.4)
    assert row["explosive_pass_count"] == 0


def test_all_postseason_rows_return_empty_frame() -> None:
    """A frame with no regular-season rows returns the typed empty frame."""
    plays = _frame(
        [
            _play(
                season_type="POST",
                posteam="KC",
                defteam="BUF",
                play_type="pass",
                qb_dropback=1,
                epa=3.0,
            )
        ]
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)

    assert result.height == 0
    assert result.columns == pbp.PBP_TEAM_GAME_COLUMNS


def test_rows_are_kept_when_season_type_column_is_absent() -> None:
    """Every row is kept when the frame has no season_type column."""
    plays = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "posteam": ["KC"],
            "defteam": ["BUF"],
            "qb_dropback": [1],
            "epa": [0.7],
        }
    )

    row = _row_for(pbp.aggregate_pbp_team_game_stats(plays), "KC")

    assert row["offensive_snaps"] == 1
    assert row["pass_epa_sum"] == pytest.approx(0.7)


def test_plays_without_both_teams_are_dropped() -> None:
    """Plays missing posteam or defteam cannot be credited and are dropped."""
    plays = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "posteam": ["KC", None, "KC"],
            "defteam": ["BUF", "BUF", None],
            "qb_dropback": [1, 1, 1],
            "epa": [0.5, 9.0, 9.0],
        }
    )

    result = pbp.aggregate_pbp_team_game_stats(plays)
    row = _row_for(result, "KC")

    assert result.height == 2
    assert row["offensive_snaps"] == 1
    assert row["pass_epa_sum"] == pytest.approx(0.5)


def test_boolean_and_integer_flag_dtypes_are_handled() -> None:
    """Boolean flag columns and integer yardage are cast rather than rejected."""
    plays = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "posteam": ["KC"],
            "defteam": ["BUF"],
            "qb_dropback": [True],
            "special": [False],
            "success": [True],
            "yards_gained": [22],
            "epa": [0.25],
        }
    )

    row = _row_for(pbp.aggregate_pbp_team_game_stats(plays), "KC")

    assert row["offensive_snaps"] == 1
    assert row["dropbacks"] == 1
    assert row["pass_epa_sum"] == pytest.approx(0.25)
    assert row["pass_success_count"] == 1
    assert row["explosive_pass_count"] == 1
    assert row["st_plays"] == 0


def test_blank_team_identifiers_are_dropped_like_nulls() -> None:
    """Plays with an empty-string team are dropped, not grouped under a blank team.

    Early nflverse seasons carry an empty string rather than a null for a missing
    possession team; grouping on it would invent a phantom team-game row and duplicate
    the `(season, week, team_abbr)` key the downstream join relies on.
    """
    plays = pl.DataFrame(
        {
            "season": [2001, 2001, 2001],
            "week": [1, 1, 1],
            "season_type": ["REG", "REG", "REG"],
            "posteam": ["AAA", "", "AAA"],
            "defteam": ["BBB", "BBB", ""],
            "qb_dropback": [1, 1, 1],
            "rush": [0, 0, 0],
            "qb_kneel": [0, 0, 0],
            "qb_spike": [0, 0, 0],
            "epa": [1.0, 5.0, 5.0],
            "yards_gained": [10.0, 10.0, 10.0],
            "success": [1, 1, 1],
            "down": [1, 1, 1],
        }
    )

    out = pbp.aggregate_pbp_team_game_stats(plays)

    assert out.height == 2, "expected exactly one row per real team"
    assert set(out["team_abbr"].to_list()) == {"AAA", "BBB"}
    assert "" not in out["team_abbr"].to_list()
    assert "" not in out["opponent_abbr"].to_list()
    # Only the single fully-identified play may contribute.
    aaa = out.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    assert aaa["dropbacks"] == 1
    assert aaa["pass_epa_sum"] == pytest.approx(1.0)


def test_two_point_attempts_are_not_dropbacks_or_carries() -> None:
    """Two-point conversion tries are excluded from dropbacks and carries.

    They are untimed conversion attempts rather than scrimmage downs, so counting them
    would contaminate the per-attempt EPA and success denominators.
    """
    plays = pl.DataFrame(
        {
            "season": [2001, 2001, 2001, 2001],
            "week": [1, 1, 1, 1],
            "season_type": ["REG"] * 4,
            "posteam": ["AAA"] * 4,
            "defteam": ["BBB"] * 4,
            "qb_dropback": [1, 1, 0, 0],
            "rush": [0, 0, 1, 1],
            "qb_kneel": [0] * 4,
            "qb_spike": [0] * 4,
            "two_point_attempt": [0, 1, 0, 1],
            "epa": [1.0, 9.0, 2.0, 9.0],
            "yards_gained": [30.0, 30.0, 15.0, 15.0],
            "success": [1, 1, 1, 1],
            "down": [1, 1, 1, 1],
        }
    )

    out = pbp.aggregate_pbp_team_game_stats(plays)
    row = out.filter(pl.col("team_abbr") == "AAA").row(0, named=True)

    assert row["dropbacks"] == 1
    assert row["carries"] == 1
    assert row["pass_epa_sum"] == pytest.approx(1.0)
    assert row["rush_epa_sum"] == pytest.approx(2.0)
    assert row["pass_success_count"] == 1
    assert row["rush_success_count"] == 1
    assert row["explosive_pass_count"] == 1
    assert row["explosive_rush_count"] == 1
    # The allowed side must exclude them identically.
    opp = out.filter(pl.col("team_abbr") == "BBB").row(0, named=True)
    assert opp["dropbacks_allowed"] == 1
    assert opp["carries_allowed"] == 1
    assert opp["pass_epa_allowed_sum"] == pytest.approx(1.0)
