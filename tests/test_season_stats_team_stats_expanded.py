"""Tests for the expanded team metrics derived from play-by-play."""

from pathlib import Path

import polars as pl

from nfl_predictor.season_stats import team_stats
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    is_rate_part,
    numerator_column,
    rate_parts,
)
from nfl_predictor.season_stats.team_stats_expanded import (
    TEAM_FORMULA_RATES,
    compute_expanded_team_game_stats,
)

_GAME = {
    "game_id": "2025_01_DEN_KC",
    "season": 2025,
    "season_type": "REG",
    "week": 1,
}


def _play(**overrides: object) -> dict[str, object]:
    """Return one synthetic DEN-offense play with quiet defaults."""
    play: dict[str, object] = {
        **_GAME,
        "posteam": "DEN",
        "defteam": "KC",
        "pass": 0,
        "rush": 0,
        "rush_attempt": 0,
        "qb_dropback": 0,
        "qb_scramble": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "pass_attempt": 0,
        "complete_pass": 0,
        "incomplete_pass": 0,
        "sack": 0,
        "interception": 0,
        "receiver_player_id": None,
        "fumble": 0,
        "fumble_lost": 0,
        "fumble_forced": 0,
        "fumbled_1_player_id": None,
        "fumbled_2_player_id": None,
        "tackled_for_loss": 0,
        "qb_hit": 0,
        "pass_defense_1_player_id": None,
        "two_point_attempt": 0,
        "two_point_conv_result": None,
        "defensive_two_point_conv": 0,
        "pass_touchdown": 0,
        "rush_touchdown": 0,
        "touchdown": 0,
        "return_touchdown": 0,
        "td_team": None,
        "return_yards": 0.0,
        "yards_gained": 0.0,
        "passing_yards": 0.0,
        "rushing_yards": 0.0,
        "air_yards": None,
        "yards_after_catch": None,
        "xyac_mean_yardage": None,
        "pass_length": None,
        "epa": 0.0,
        "wpa": 0.0,
        "success": 0,
        "cpoe": None,
        "pass_oe": None,
        "shotgun": 0,
        "no_huddle": 0,
        "down": 1,
        "ydstogo": 10,
        "goal_to_go": 0,
        "first_down": 0,
        "first_down_penalty": 0,
        "third_down_converted": 0,
        "third_down_failed": 0,
        "fourth_down_converted": 0,
        "fourth_down_failed": 0,
        "punt_attempt": 0,
        "field_goal_attempt": 0,
        "penalty": 0,
        "penalty_team": None,
        "penalty_type": None,
        "penalty_yards": None,
        "series": 1,
        "series_success": 0,
        "series_result": "Punt",
        "fixed_drive": 1,
        "fixed_drive_result": "Punt",
        "drive_play_count": 3,
        "drive_first_downs": 0,
        "drive_inside20": 0,
        "drive_ended_with_score": 0,
        "drive_time_of_possession": "2:00",
        "drive_start_yard_line": "DEN 25",
        "drive_yards_penalized": 0,
        "ydsnet": 0,
        "posteam_score": 0,
        "posteam_score_post": 0,
    }
    play.update(overrides)
    return play


def _row(frame: pl.DataFrame, team: str) -> dict[str, object]:
    """Return the single team-game row for one team as a dict."""
    return frame.filter(pl.col("team") == team).to_dicts()[0]


def _num(row: dict[str, object], key: str) -> float:
    """Return a row value as float, asserting it is numeric."""
    value = row[key]
    assert isinstance(value, int | float), key
    return float(value)


def test_passing_volume_and_efficiency_extras() -> None:
    """Verify attempts, dropbacks, air/YAC splits, and conventional rates.

    Fixture: 4 DEN dropbacks — deep 30-yard completion (20 air + 10 YAC),
    incompletion, sack losing 7 with a strip fumble lost, and a 9-yard scramble.
    """
    # Arrange
    plays = [
        _play(
            **{"pass": 1},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            passing_yards=30.0,
            yards_gained=30.0,
            air_yards=20.0,
            yards_after_catch=10.0,
            xyac_mean_yardage=6.0,
            pass_length="deep",
            epa=2.0,
            wpa=0.1,
            success=1,
            cpoe=10.0,
            pass_oe=5.0,
            first_down=1,
        ),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            incomplete_pass=1,
            qb_dropback=1,
            air_yards=8.0,
            pass_length="short",
            epa=-0.5,
            pass_oe=-3.0,
        ),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            sack=1,
            yards_gained=-7.0,
            fumble=1,
            fumble_lost=1,
            fumble_forced=1,
            epa=-2.5,
            qb_hit=1,
        ),
        _play(
            **{"pass": 1},
            qb_dropback=1,
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=9.0,
            yards_gained=9.0,
            epa=0.5,
            success=1,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")

    assert den["attempts"] == 2
    assert den["completions"] == 1
    assert den["completion_pct"] == 0.5
    assert den["dropbacks"] == 4
    assert den["sack_yards_lost"] == 7.0
    assert den["net_passing_yards"] == 23.0
    assert den["scrambles"] == 1
    assert den["scramble_yards"] == 9.0
    assert den["passing_air_yards"] == 28.0
    assert den["passing_yards_after_catch"] == 10.0
    assert den["air_yards_per_attempt"] == 14.0
    assert den["yac_per_completion"] == 10.0
    assert den["yards_per_attempt"] == 15.0
    # NY/A = (30 - 7) / (2 + 1); ANY/A adds nothing here (no TD/INT).
    assert abs(_num(den, "net_yards_per_attempt") - 23.0 / 3.0) < 1e-9
    assert abs(_num(den, "adjusted_net_yards_per_attempt") - 23.0 / 3.0) < 1e-9
    assert den["yards_per_dropback"] == 7.5
    assert abs(_num(den, "epa_per_dropback") - (-0.5 / 4.0)) < 1e-9
    assert den["pass_success_rate"] == 0.5
    assert den["explosive_pass_rate"] == 0.25
    assert den["deep_attempt_rate"] == 0.5
    assert den["longest_pass"] == 30.0
    assert den["sack_fumbles"] == 1
    assert den["sack_rate_per_dropback"] == 0.25
    assert den["xyac_per_completion"] == 6.0
    assert den["yac_over_expected_per_completion"] == 4.0
    assert den["offensive_wpa"] == 0.1
    assert abs(_num(den, "pass_rate_over_expected") - 1.0) < 1e-9

    kc = _row(result, "KC")
    assert kc["attempts_faced"] == 2
    assert kc["completions_allowed"] == 1
    assert kc["completion_pct_allowed"] == 0.5
    assert kc["net_passing_yards_allowed"] == 23.0
    assert kc["air_yards_allowed"] == 28.0
    assert kc["yac_allowed"] == 10.0
    assert kc["explosive_pass_rate_allowed"] == 0.25
    assert kc["deep_attempt_rate_faced"] == 0.5
    assert abs(_num(kc, "epa_per_dropback_allowed") - (-0.5 / 4.0)) < 1e-9
    assert abs(_num(kc, "any_a_allowed") - 23.0 / 3.0) < 1e-9
    assert kc["def_sack_yards"] == 7.0
    assert kc["def_sack_rate_per_dropback"] == 0.25
    # One sack plus one QB hit (on the sack play) over 4 dropbacks faced.
    assert kc["qb_pressure_events_rate"] == 0.5


def test_rushing_extras_and_run_defense_mirror() -> None:
    """Verify carries, designed splits, explosive/stuffed rates, and stuff rate.

    Fixture: 12-yard explosive run, -1-yard stuffed TFL run, kneel (a carry
    but not a stuff), and a 9-yard scramble (a carry but not a designed carry).
    """
    # Arrange
    plays = [
        _play(
            rush=1,
            rush_attempt=1,
            rushing_yards=12.0,
            yards_gained=12.0,
            epa=0.8,
            success=1,
            first_down=1,
        ),
        _play(
            rush=1,
            rush_attempt=1,
            rushing_yards=-1.0,
            yards_gained=-1.0,
            epa=-0.6,
            tackled_for_loss=1,
        ),
        _play(rush_attempt=1, qb_kneel=1, rushing_yards=-1.0, yards_gained=-1.0),
        _play(
            **{"pass": 1},
            qb_dropback=1,
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=9.0,
            yards_gained=9.0,
            epa=0.5,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")

    assert den["carries"] == 4
    assert den["designed_carries"] == 2
    assert abs(_num(den, "yards_per_carry") - 19.0 / 4.0) < 1e-9
    assert den["rush_success_rate"] == 0.5
    assert den["explosive_rush_rate"] == 0.25
    assert abs(_num(den, "stuffed_run_rate") - 1.0 / 3.0) < 1e-9
    assert den["longest_rush"] == 12.0

    kc = _row(result, "KC")
    assert kc["carries_faced"] == 4
    assert abs(_num(kc, "yards_per_carry_allowed") - 19.0 / 4.0) < 1e-9
    assert abs(_num(kc, "stuff_rate") - 1.0 / 3.0) < 1e-9
    assert kc["explosive_rush_rate_allowed"] == 0.25
    assert kc["rush_success_rate_allowed"] == 0.5


def test_downs_series_and_turnover_families() -> None:
    """Verify third/fourth-down, series, giveaway, and takeaway accounting."""
    # Arrange
    plays = [
        _play(
            **{"pass": 1},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            down=3,
            ydstogo=4,
            third_down_converted=1,
            first_down=1,
            passing_yards=6.0,
            yards_gained=6.0,
            series=1,
            series_success=1,
            series_result="First down",
        ),
        _play(
            rush=1,
            rush_attempt=1,
            down=3,
            ydstogo=8,
            third_down_failed=1,
            rushing_yards=2.0,
            yards_gained=2.0,
            series=2,
            series_success=0,
        ),
        _play(
            rush=1,
            rush_attempt=1,
            down=4,
            ydstogo=1,
            fourth_down_converted=1,
            first_down=1,
            rushing_yards=3.0,
            yards_gained=3.0,
            series=2,
            series_success=1,
            series_result="First down",
        ),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            down=4,
            ydstogo=2,
            fourth_down_failed=1,
            incomplete_pass=1,
            series=3,
            series_success=0,
            series_result="Turnover on downs",
        ),
        _play(punt_attempt=1, down=4, ydstogo=9, series=4, series_success=0),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            interception=1,
            epa=-4.0,
            return_yards=12.0,
            series=5,
            series_success=0,
            series_result="Interception",
        ),
        _play(
            rush=1,
            rush_attempt=1,
            fumble=1,
            fumble_lost=1,
            fumble_forced=1,
            epa=-3.0,
            rushing_yards=1.0,
            yards_gained=1.0,
            series=5,
            series_success=0,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")

    assert den["third_down_attempts"] == 2
    assert den["third_down_conversions"] == 1
    assert den["third_down_pct"] == 0.5
    assert den["third_down_avg_distance"] == 6.0
    assert den["fourth_down_attempts"] == 2
    assert den["fourth_down_conversions"] == 1
    assert den["fourth_down_pct"] == 0.5
    # Go rate: 2 go tries over 3 fourth downs faced (incl. the punt).
    assert abs(_num(den, "fourth_down_go_rate") - 2.0 / 3.0) < 1e-9
    # Fourth-and-short (<= 2): went on both such situations.
    assert den["fourth_down_aggressiveness"] == 1.0
    assert den["turnovers_on_downs"] == 1
    assert den["series"] == 5
    assert abs(_num(den, "series_conversion_rate") - 2.0 / 5.0) < 1e-9
    assert den["first_downs"] == 2
    assert den["giveaways"] == 2
    assert den["fumbles"] == 1
    assert den["fumbles_lost"] == 1
    assert den["turnover_epa"] == -7.0

    kc = _row(result, "KC")
    assert kc["third_down_pct_allowed"] == 0.5
    assert kc["fourth_down_pct_allowed"] == 0.5
    assert abs(_num(kc, "series_conversion_rate_allowed") - 2.0 / 5.0) < 1e-9
    assert kc["takeaways"] == 2
    assert kc["fumble_recovery_opp"] == 1
    assert kc["def_interception_yards"] == 12.0
    assert kc["takeaway_epa"] == 7.0
    assert kc["first_downs_allowed"] == 2


def test_drive_scoring_and_field_position_families() -> None:
    """Verify drive-level rates, red-zone accounting, and field position.

    Fixture: drive 1 is a 10-play touchdown drive reaching the red zone that
    starts at the DEN 40 (7 points scored); drive 2 is a three-and-out punt
    from the DEN 20.
    """
    # Arrange
    drive_one = {
        "fixed_drive": 1,
        "fixed_drive_result": "Touchdown",
        "drive_play_count": 10,
        "drive_first_downs": 4,
        "drive_inside20": 1,
        "drive_time_of_possession": "5:00",
        "drive_start_yard_line": "DEN 40",
        "drive_yards_penalized": 5,
        "ydsnet": 60,
        "posteam_score": 0,
        "posteam_score_post": 7,
        "series_result": "Touchdown",
    }
    drive_two = {
        "fixed_drive": 2,
        "fixed_drive_result": "Punt",
        "drive_play_count": 3,
        "drive_first_downs": 0,
        "drive_inside20": 0,
        "drive_time_of_possession": "1:30",
        "drive_start_yard_line": "DEN 20",
        "drive_yards_penalized": 0,
        "ydsnet": 4,
        "posteam_score": 7,
        "posteam_score_post": 7,
    }
    plays = [
        _play(
            rush=1,
            rush_attempt=1,
            rushing_yards=5.0,
            yards_gained=5.0,
            goal_to_go=1,
            series=1,
            series_success=1,
            **drive_one,
        ),
        _play(
            rush=1,
            rush_attempt=1,
            rush_touchdown=1,
            touchdown=1,
            td_team="DEN",
            rushing_yards=2.0,
            yards_gained=2.0,
            goal_to_go=1,
            series=1,
            series_success=1,
            **drive_one,
        ),
        _play(rush=1, rush_attempt=1, rushing_yards=4.0, yards_gained=4.0, series=2, **drive_two),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            incomplete_pass=1,
            qb_dropback=1,
            series=3,
            **drive_two,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")

    assert den["drives"] == 2
    assert den["yards_per_drive"] == 32.0
    assert den["plays_per_drive"] == 6.5
    assert den["time_per_drive"] == 195.0
    assert den["first_downs_per_drive"] == 2.0
    assert den["score_pct_per_drive"] == 0.5
    assert den["punt_pct_per_drive"] == 0.5
    assert den["turnover_pct_per_drive"] == 0.0
    assert den["three_and_out_rate"] == 0.5
    assert den["points_per_drive"] == 3.5
    assert den["red_zone_trips"] == 1
    assert den["red_zone_td_pct"] == 1.0
    assert den["points_per_red_zone_trip"] == 7.0
    assert den["avg_starting_field_position"] == 30.0
    # Only drive two started inside the 25 and it did not score.
    assert den["long_field_score_pct"] == 0.0
    assert den["drive_penalty_yards"] == 5.0
    assert den["goal_to_go_td_pct"] == 1.0
    assert den["total_tds"] == 1
    assert den["scrimmage_tds"] == 1

    kc = _row(result, "KC")
    assert kc["score_pct_per_drive_allowed"] == 0.5
    assert kc["punts_forced_pct"] == 0.5
    assert kc["three_and_outs_forced_rate"] == 0.5
    assert kc["points_per_drive_allowed"] == 3.5
    assert kc["red_zone_td_pct_allowed"] == 1.0
    assert kc["goal_to_go_td_pct_allowed"] == 1.0
    assert kc["avg_starting_field_position_allowed"] == 30.0


def test_epa_per_carry_and_stuff_rate_divide_over_their_own_plays() -> None:
    """Verify EPA per carry covers every carry and stuff rate leaves kneel-downs out.

    Fixture (nflverse flags): a 12-yard run, a run stopped for a 1-yard loss, a kneel-down, a
    9-yard scramble, and a run a penalty wiped out (a designed-run flag but no rush attempt).
    """
    # Arrange
    plays = [
        _play(rush=1, rush_attempt=1, rushing_yards=12.0, yards_gained=12.0, epa=0.8),
        _play(rush=1, rush_attempt=1, rushing_yards=-1.0, yards_gained=-1.0, epa=-0.6),
        _play(rush_attempt=1, qb_kneel=1, rushing_yards=-1.0, yards_gained=-1.0, epa=-0.4),
        _play(
            **{"pass": 1},
            qb_dropback=1,
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=9.0,
            yards_gained=9.0,
            epa=0.5,
        ),
        _play(rush=1, penalty=1, penalty_team="DEN", rushing_yards=None, epa=-0.6),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    assert abs(_num(den, "epa_per_carry") - (0.8 - 0.6 - 0.4 + 0.5) / 4) < 1e-9
    assert abs(_num(den, "stuffed_run_rate") - 1 / 3) < 1e-9
    assert abs(_num(_row(result, "KC"), "stuff_rate") - 1 / 3) < 1e-9


def test_a_scramble_without_the_dropback_flag_counts_as_a_dropback() -> None:
    """Verify a scramble counts as a dropback and a scrimmage snap however nflverse flags it.

    Before 2006 nflverse leaves ``qb_dropback`` at 0 on most scrambles, which it sets from 2006
    on. Fixture: a DEN pass with 0.5 EPA and a DEN scramble with 1.5 EPA coded the pre-2006 way.
    """
    # Arrange
    plays = [
        _play(play_type="pass", **{"pass": 1}, qb_dropback=1, pass_attempt=1, epa=0.5),
        _play(
            play_type="run",
            **{"pass": 1},
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=7.0,
            yards_gained=7.0,
            epa=1.5,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    assert (den["dropbacks"], den["scrambles"], den["offensive_epa"]) == (2, 1, 2.0)


def test_scrambles_and_designed_runs_leave_out_plays_wiped_out_by_penalty() -> None:
    """Verify nullified scrambles and designed runs are not counted as real plays.

    nflverse keeps the scramble and designed-run flags on a play a penalty wiped out, but sets
    ``rush_attempt`` only on plays that stood, as official carries do. Fixture: one real and one
    nullified designed run, one real and one nullified scramble, and a kneel-down (``rush`` is 0
    on scrambles and kneel-downs, as in nflverse).
    """
    # Arrange
    plays = [
        _play(rush=1, rush_attempt=1, rushing_yards=4.0, yards_gained=4.0, success=1),
        _play(rush=1, penalty=1, penalty_team="DEN", rushing_yards=None, epa=-0.6),
        _play(
            **{"pass": 1},
            qb_dropback=1,
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=9.0,
            yards_gained=9.0,
            success=1,
        ),
        _play(**{"pass": 1}, qb_scramble=1, penalty=1, penalty_team="DEN", rushing_yards=None),
        _play(rush_attempt=1, qb_kneel=1, rushing_yards=-1.0, yards_gained=-1.0),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    assert (den["carries"], den["designed_carries"], den["scrambles"]) == (3, 1, 1)
    assert den["scramble_yards"] == 9.0
    assert den["rush_success_rate"] == 1.0


def test_receiving_fumbles_count_only_the_receivers_fumbles() -> None:
    """Verify receiving fumbles are the receiver's, not any fumble on a completed pass.

    Fixture: three DEN completions, each with a fumble. The receiver fumbles and loses it on the
    first; the quarterback fumbles the snap and recovers before throwing on the second; the
    receiver laterals and the lateral's runner fumbles and loses it on the third.
    """
    # Arrange
    receiver, quarterback, runner = "00-0000001", "00-0000002", "00-0000003"
    catch = {
        "pass": 1,
        "pass_attempt": 1,
        "complete_pass": 1,
        "qb_dropback": 1,
        "receiver_player_id": receiver,
        "fumble": 1,
    }
    plays = [
        _play(**catch, fumble_lost=1, fumbled_1_player_id=receiver),
        _play(**catch, fumbled_1_player_id=quarterback),
        _play(**catch, fumble_lost=1, fumbled_1_player_id=runner),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    assert (den["receiving_fumbles"], den["receiving_fumbles_lost"]) == (1, 1)


def test_punt_muff_is_the_receiving_teams_giveaway() -> None:
    """A returner's muff the punting team recovers costs the receiving team, not the punter's.

    nflverse lists the punting team as ``posteam`` on a punt and flags ``fumble_lost`` for the
    returner's lost fumble too; ``fumbled_1_team`` names the receiving team. The play's EPA (+5.0
    for DEN, the punting team) is KC's giveaway cost and DEN's takeaway value.
    """
    # Arrange
    plays = [
        _play(
            punt_attempt=1,
            down=4,
            fumble=1,
            fumble_lost=1,
            fumbled_1_team="KC",
            epa=5.0,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den, kc = _row(result, "DEN"), _row(result, "KC")
    assert (den["turnover_epa"], den["takeaway_epa"]) == (0.0, 5.0)
    assert (kc["turnover_epa"], kc["takeaway_epa"]) == (-5.0, 0.0)


def test_interception_fumbled_back_is_only_the_passing_teams_giveaway() -> None:
    """An intercepting defender's fumble the offense recovers is not an offensive fumble lost.

    The interception is DEN's giveaway, so the play's EPA (-1.0 for DEN) counts once, as DEN's
    turnover EPA; KC's fumble on the return is not a DEN fumble lost or a KC fumble recovery.
    """
    # Arrange
    plays = [
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            interception=1,
            fumble=1,
            fumble_lost=1,
            fumbled_1_team="KC",
            epa=-1.0,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den, kc = _row(result, "DEN"), _row(result, "KC")
    assert (den["fumbles_lost"], den["giveaways"], den["turnover_epa"]) == (0, 1, -1.0)
    assert (kc["fumble_recovery_opp"], kc["takeaways"], kc["takeaway_epa"]) == (0, 1, 1.0)
    assert kc["turnover_epa"] == 0.0


def test_turnover_drive_rate_counts_giveaway_drives_by_nflverse_result() -> None:
    """Verify giveaway drives are nflverse "Turnover" drives plus giveaways returned for a score.

    nflverse labels a drive lost to an interception or fumble "Turnover"; it never uses
    "Interception" or "Fumble". "Opp touchdown" covers both giveaways returned for a touchdown
    and punt returns, so only those with an interception or lost fumble count. Fixture: DEN has
    five drives (interception, lost fumble, pick-six, punt returned for a touchdown, punt); KC has
    the one-play extra-point group after its pick-six, which nflverse also labels "Opp touchdown".
    """
    # Arrange
    drives = [
        {"fixed_drive": 1, "fixed_drive_result": "Turnover"},
        {"fixed_drive": 2, "fixed_drive_result": "Turnover"},
        {"fixed_drive": 3, "fixed_drive_result": "Opp touchdown"},
        {"fixed_drive": 4, "fixed_drive_result": "Opp touchdown"},
        {"fixed_drive": 5, "fixed_drive_result": "Punt"},
    ]
    plays = [
        _play(**{"pass": 1}, pass_attempt=1, qb_dropback=1, interception=1, **drives[0]),
        _play(rush=1, rush_attempt=1, fumble=1, fumble_lost=1, **drives[1]),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            interception=1,
            return_touchdown=1,
            touchdown=1,
            td_team="KC",
            **drives[2],
        ),
        _play(
            rush=1,
            rush_attempt=1,
            **drives[3],
        ),
        _play(
            punt_attempt=1,
            down=4,
            return_touchdown=1,
            touchdown=1,
            td_team="KC",
            **drives[3],
        ),
        _play(rush=1, rush_attempt=1, **drives[4]),
        _play(posteam="KC", defteam="DEN", **drives[2]),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    assert _row(result, "DEN")["turnover_pct_per_drive"] == 3 / 5
    assert _row(result, "KC")["turnover_pct_per_drive"] == 0.0


def test_penalty_families_track_both_sides() -> None:
    """Verify penalty counts, yards, splits, and the defensive mirror."""
    # Arrange
    plays = [
        _play(
            penalty=1,
            penalty_team="DEN",
            penalty_type="False Start",
            penalty_yards=5,
        ),
        _play(
            penalty=1,
            penalty_team="DEN",
            penalty_type="Offensive Holding",
            penalty_yards=10,
        ),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            incomplete_pass=1,
            qb_dropback=1,
            penalty=1,
            penalty_team="KC",
            penalty_type="Defensive Pass Interference",
            penalty_yards=23,
            first_down=1,
            first_down_penalty=1,
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    kc = _row(result, "KC")

    assert den["penalties"] == 2
    assert den["penalty_yards"] == 15.0
    assert den["offensive_penalties"] == 2
    assert den["offensive_penalty_yards"] == 15.0
    assert den["first_downs_penalty"] == 1
    assert den["penalty_differential"] == -1
    assert den["penalty_yards_differential"] == 8.0

    assert kc["penalties"] == 1
    assert kc["penalty_yards"] == 23.0
    assert kc["defensive_penalties"] == 1
    assert kc["defensive_penalty_yards"] == 23.0
    assert kc["defensive_pass_interference"] == 1
    assert kc["penalty_first_downs_allowed"] == 1
    assert kc["penalty_differential"] == 1


def test_two_point_and_defensive_playmaking() -> None:
    """Verify two-point tries, defensive scores, and havoc inputs."""
    # Arrange
    plays = [
        _play(
            two_point_attempt=1,
            two_point_conv_result="success",
            **{"pass": 1},
            pass_attempt=1,
        ),
        _play(two_point_attempt=1, two_point_conv_result="failure", rush=1, rush_attempt=1),
        _play(
            **{"pass": 1},
            pass_attempt=1,
            qb_dropback=1,
            interception=1,
            return_touchdown=1,
            touchdown=1,
            td_team="KC",
            return_yards=40.0,
            epa=-6.0,
            pass_defense_1_player_id="00-001",
        ),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    kc = _row(result, "KC")

    assert den["two_pt_attempts"] == 2
    assert den["two_pt_conversions"] == 1
    assert den["two_pt_conversion_rate"] == 0.5
    assert kc["two_pt_conversion_rate_allowed"] == 0.5
    assert kc["def_tds"] == 1
    assert kc["total_tds"] == 1
    assert den["total_tds"] == 0


def test_fields_the_season_lacks_give_blank_stats_not_zero() -> None:
    """Verify stats built on a field the season's play-by-play lacks are null, not zero.

    Fixture: a 2004-style game, where the loader leaves air yards, yards after catch, their EPA
    splits, pass depth, QB hits, and (as in 1999-2002) the no-huddle flag and drive penalty yards
    null on every play.
    """
    # Arrange
    missing = {
        "air_yards": None,
        "air_epa": None,
        "yac_epa": None,
        "yards_after_catch": None,
        "pass_length": None,
        "qb_hit": None,
        "no_huddle": None,
        "drive_yards_penalized": None,
    }
    plays = [
        _play(
            **{"pass": 1, **missing},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            passing_yards=15.0,
            yards_gained=15.0,
        ),
        _play(**{"pass": 1, **missing}, pass_attempt=1, qb_dropback=1, sack=1, yards_gained=-6.0),
    ]

    # Act
    result = compute_expanded_team_game_stats(pl.DataFrame(plays))

    # Assert
    den = _row(result, "DEN")
    kc = _row(result, "KC")
    blank_offense = [
        "passing_air_yards",
        "air_yards_per_attempt",
        "passing_yards_after_catch",
        "yac_per_completion",
        "air_epa_total",
        "yac_epa_total",
        "deep_attempt_rate",
        "drive_penalty_yards",
        "no_huddle_rate",
    ]
    blank_defense = [
        "air_yards_allowed",
        "yac_allowed",
        "deep_attempt_rate_faced",
        "qb_pressure_events_rate",
    ]
    assert {column: den[column] for column in blank_offense} == dict.fromkeys(blank_offense)
    assert {column: kc[column] for column in blank_defense} == dict.fromkeys(blank_defense)
    assert den["sack_rate_per_dropback"] == 0.5
    assert kc["def_sack_rate_per_dropback"] == 0.5


def _game_rows() -> pl.DataFrame:
    """Return one game's team rows built from varied DEN and KC plays, as the loader builds them.

    DEN: a completion, an incompletion, a sack, a scramble, an interception, designed runs, a
    kneel-down, a third-down conversion, a fourth-down try, a two-point pass, and a penalty;
    KC: a completion and a run.
    """
    den_drive = {"fixed_drive": 1, "fixed_drive_result": "Touchdown", "drive_inside20": 1}
    kc_drive = {
        "posteam": "KC",
        "defteam": "DEN",
        "fixed_drive": 2,
        "drive_start_yard_line": "KC 30",
    }
    plays = [
        _play(
            **{"pass": 1, **den_drive},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            passing_yards=25.0,
            yards_gained=25.0,
            air_yards=15.0,
            yards_after_catch=10.0,
            xyac_mean_yardage=4.0,
            pass_length="deep",
            receiver_player_id="WR1",
            epa=1.5,
            success=1,
            cpoe=20.0,
            pass_oe=8.0,
            first_down=1,
            down=3,
            ydstogo=5,
            third_down_converted=1,
            shotgun=1,
        ),
        _play(
            **{"pass": 1, **den_drive},
            pass_attempt=1,
            incomplete_pass=1,
            qb_dropback=1,
            air_yards=7.0,
            pass_length="short",
            receiver_player_id="WR1",
            epa=-0.6,
            cpoe=-30.0,
            pass_oe=-2.0,
        ),
        _play(
            **{"pass": 1, **den_drive},
            pass_attempt=1,
            qb_dropback=1,
            sack=1,
            qb_hit=1,
            yards_gained=-6.0,
            epa=-1.8,
        ),
        _play(
            **{"pass": 1, **den_drive},
            qb_dropback=1,
            qb_scramble=1,
            rush_attempt=1,
            rushing_yards=11.0,
            yards_gained=11.0,
            epa=0.9,
            success=1,
        ),
        _play(
            **{"pass": 1, **den_drive},
            pass_attempt=1,
            qb_dropback=1,
            interception=1,
            air_yards=20.0,
            pass_length="deep",
            receiver_player_id="WR2",
            epa=-3.5,
            down=4,
            ydstogo=2,
            fourth_down_failed=1,
        ),
        _play(rush=1, rush_attempt=1, rushing_yards=4.0, yards_gained=4.0, epa=0.2, **den_drive),
        _play(rush=1, rush_attempt=1, rushing_yards=-2.0, yards_gained=-2.0, epa=-0.7, **den_drive),
        _play(rush_attempt=1, qb_kneel=1, rushing_yards=-1.0, yards_gained=-1.0, **den_drive),
        _play(
            **{"pass": 1, **den_drive},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            two_point_attempt=1,
            two_point_conv_result="success",
            down=0,
        ),
        _play(penalty=1, penalty_team="DEN", penalty_type="False Start", penalty_yards=5.0),
        _play(
            **{"pass": 1, **kc_drive},
            pass_attempt=1,
            complete_pass=1,
            qb_dropback=1,
            passing_yards=8.0,
            yards_gained=8.0,
            air_yards=3.0,
            yards_after_catch=5.0,
            xyac_mean_yardage=6.0,
            pass_length="short",
            receiver_player_id="WR9",
            epa=0.3,
            cpoe=10.0,
            pass_oe=1.0,
        ),
        _play(rush=1, rush_attempt=1, rushing_yards=3.0, yards_gained=3.0, epa=0.1, **kc_drive),
    ]
    player_stats = pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "def_sacks": [0.0, 1.0],
            "def_qb_hits": [0, 1],
            "def_tackles_for_loss": [1, 0],
            "def_fumbles_forced": [0, 1],
            "def_interceptions": [1, 0],
            "def_pass_defended": [2, 1],
            "def_safeties": [0, 0],
        }
    )
    schedule = pl.DataFrame(
        {
            "game_id": [_GAME["game_id"]],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["DEN"],
            "home_score": [3],
            "away_score": [8],
        }
    )
    return team_stats.compute_team_game_stats_from_pbp(pl.DataFrame(plays), player_stats, schedule)


def _rate_columns(columns: list[str]) -> list[str]:
    """Return the columns the registry calls a rate or a mean, per-snap suffix columns included."""
    registry = get_registry()
    rates: list[str] = []
    for column in columns:
        resolved = registry.resolve_column(column)
        if resolved is None or is_rate_part(column):
            continue
        per_snap = column.endswith(("_per_offensive_snap", "_per_defensive_snap"))
        if resolved.base.shape in {"rate", "avg"} or per_snap:
            rates.append(column)
    return rates


def test_every_game_rate_carries_what_pools_it_over_games() -> None:
    # Arrange
    games = _game_rows()
    # A game counts once toward a team's win share, so its per-game mean is already pooled.
    averaged_over_games = {"win_value"}

    # Act
    pooled = set(rate_parts(games.columns))

    # Assert
    unpooled = [
        column
        for column in _rate_columns(games.columns)
        if column not in pooled
        and column not in averaged_over_games
        and not (
            column in TEAM_FORMULA_RATES and pooled.issuperset(TEAM_FORMULA_RATES[column].inputs)
        )
    ]
    assert unpooled == []


def test_every_game_rate_equals_its_numerator_over_its_denominator() -> None:
    # Arrange
    games = _game_rows()

    # Act
    mismatches = [
        (rate, row["team"], row[rate], row[numerator_column(rate)], row[denominator_column(rate)])
        for rate in rate_parts(games.columns)
        if rate in games.columns
        for row in games.iter_rows(named=True)
        if not _matches_parts(row[rate], row[numerator_column(rate)], row[denominator_column(rate)])
    ]

    # Assert
    assert mismatches == []


def _matches_parts(value: object, numerator: object, denominator: object) -> bool:
    """Return whether a game rate is its numerator over its denominator (null without one)."""
    if not isinstance(denominator, int | float) or denominator <= 0:
        return value is None
    assert isinstance(numerator, int | float), numerator
    assert isinstance(value, int | float), value
    return abs(value - numerator / denominator) <= 1e-12


def test_the_rate_guard_builds_every_rate_the_published_game_logs_carry() -> None:
    """The fixture holds the published team game logs of five games, every column."""
    # Arrange
    published = Path(__file__).parent / "fixtures" / "sos_season_stats_parity"
    built = set(_rate_columns(_game_rows().columns))

    # Act
    carried = _rate_columns(
        list(pl.read_parquet_schema(published / "games_team_game_logs.parquet"))
    )

    # Assert
    assert carried
    assert sorted(set(carried) - built) == []
