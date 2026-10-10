"""Tests for nfl_predictor.season_stats.qb_stats."""

import polars as pl
import pytest

from nfl_predictor.season_stats import qb_stats
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    numerator_column,
)


def test_compute_qb_season_stats_includes_volume_and_eligibility() -> None:
    """Verify QB season aggregation computes averages, volume, and eligibility flags."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B", "QB_C"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B", "QB C"],
            "qb_attempts": [30, 28, 20, 22, 18],
            "qb_passer_rating": [100.0, 102.0, 92.0, 95.0, 90.0],
            "qb_completion_percentage_above_expectation": [2.0, 3.0, 0.0, 0.5, -1.0],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3],
            "points_for": [24, 17, 21, 20, 14],
            "points_allowed": [17, 20, 14, 21, 14],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(
        qb_df,
        weekly_df=weekly_df,
        min_games=2,
        min_attempts=55,
    )

    # Assert
    qb_a = result.filter(pl.col("qb_id") == "QB_A")
    qb_b = result.filter(pl.col("qb_id") == "QB_B")

    assert qb_a.select("qb_games_played").item() == 2
    assert qb_a.select("qb_attempts_total").item() == 58
    assert qb_a.select("qb_is_eligible").item() is True
    assert qb_a.select("qb_passer_rating").item() == 101.0
    assert qb_a.select("qb_win_pct").item() == 0.5

    assert qb_b.select("qb_games_played").item() == 2
    assert qb_b.select("qb_attempts_total").item() == 42
    assert qb_b.select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_assigns_results_to_primary_qb_only() -> None:
    """Verify QB wins are assigned only to the primary QB for each team-week."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 1],
            "qb_id": ["QB_A", "QB_B"],
            "qb_name": ["QB A", "QB B"],
            "qb_attempts": [24, 8],
            "qb_dropbacks": [26, 10],
            "qb_offense_snaps": [48, 12],
            "qb_passer_rating": [100.0, 80.0],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN"],
            "week": [1],
            "points_for": [24],
            "points_allowed": [17],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_win_pct").item() == 1.0
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_wins").item() == 1
    # QB_B was never the primary passer, so he has no decisions and no win percentage.
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_win_pct").item() is None
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_wins").item() == 0
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_losses").item() == 0


def test_compute_qb_season_stats_sums_late_game_totals() -> None:
    """Verify season QB summaries expose summed 4QC and GWD totals."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [20, 22],
            "qb_fourth_quarter_comeback": [1, 0],
            "qb_game_winning_drive": [1, 0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_fourth_quarter_comebacks").item() == 1
    assert result.select("qb_game_winning_drives").item() == 1


def test_compute_qb_season_stats_exposes_explicit_per_game_and_total_columns() -> None:
    """Verify season summaries do not mix per-game averages under raw season column names."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [20, 30],
            "qb_completions": [10, 20],
            "qb_pass_yards": [120.0, 210.0],
            "qb_pass_touchdowns": [1.0, 2.0],
            "qb_interceptions": [0.0, 1.0],
            "qb_dropbacks": [22, 33],
            "qb_offense_snaps": [40, 50],
            "qb_sacks": [2.0, 3.0],
            "qb_sack_yards_lost": [10.0, 15.0],
            "qb_sack_fumbles_lost": [0.0, 1.0],
            "qb_passing_epa": [4.0, 6.0],
            "qb_fourth_quarter_comeback": [1, 0],
            "qb_game_winning_drive": [1, 0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert "qb_attempts" not in result.columns
    assert "qb_completions" not in result.columns
    assert "qb_pass_yards" not in result.columns
    assert result.select("qb_attempts_per_game").item() == 25.0
    assert result.select("qb_completions_per_game").item() == 15.0
    assert result.select("qb_pass_yards_per_game").item() == 165.0
    assert result.select("qb_completions_total").item() == 30
    assert result.select("qb_attempts_total").item() == 50
    assert result.select("qb_pass_yards_total").item() == 330.0


def test_compute_qb_season_stats_handles_missing_attempts_column() -> None:
    """Verify missing qb_attempts defaults attempts total to zero and marks ineligible."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_passer_rating": [100.0, 98.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(
        qb_df,
        min_games=2,
        min_attempts=1,
    )

    # Assert
    assert result.select("qb_attempts_total").item() == 0
    assert result.select("qb_is_eligible").item() is False
    assert result.select("qb_games_played").item() == 2


def test_compute_qb_season_stats_defaults_to_238_attempt_threshold() -> None:
    """Verify default QB eligibility requires a full-season 238-attempt threshold."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "KC", "KC"],
            "week": [1, 2, 1, 2],
            "qb_id": ["QB_A", "QB_A", "QB_B", "QB_B"],
            "qb_name": ["QB A", "QB A", "QB B", "QB B"],
            "qb_attempts": [119, 119, 119, 118],
            "qb_passer_rating": [100.0, 101.0, 99.0, 98.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, min_games=2)

    # Assert
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_is_eligible").item() is True
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_uses_16_game_qualifier_before_2021() -> None:
    """Verify pre-2021 seasons use the 16-game, 224-attempt qualifier.

    Weekly team data should drive the default season-length-based eligibility threshold.
    """
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["SF"] * 16 + ["JAX"] * 16,
            "week": list(range(1, 17)) + list(range(1, 17)),
            "qb_id": ["QB_A"] * 16 + ["QB_B"] * 16,
            "qb_name": ["QB A"] * 16 + ["QB B"] * 16,
            "qb_attempts": [14] * 16 + ([14] * 15) + [13],
            "qb_passer_rating": [95.0] * 32,
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["SF"] * 16 + ["JAX"] * 16,
            "week": list(range(1, 17)) + list(range(1, 17)),
            "points_for": [20] * 32,
            "points_allowed": [17] * 32,
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_attempts_total").item() == 224
    assert result.filter(pl.col("qb_id") == "QB_A").select("qb_is_eligible").item() is True
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_attempts_total").item() == 223
    assert result.filter(pl.col("qb_id") == "QB_B").select("qb_is_eligible").item() is False


def test_compute_qb_season_stats_default_eligibility_is_attempt_based() -> None:
    """Verify default qualification does not add a separate games-played cutoff."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN"] * 7,
            "week": list(range(1, 8)),
            "qb_id": ["QB_A"] * 7,
            "qb_name": ["QB A"] * 7,
            "qb_attempts": [34] * 7,
            "qb_passer_rating": [100.0] * 7,
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_games_played").item() == 7
    assert result.select("qb_attempts_total").item() == 238
    assert result.select("qb_is_eligible").item() is True


def test_compute_qb_season_stats_derives_attempt_normalized_rates() -> None:
    """Verify season efficiency rates use total attempts instead of game-average volume."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [10, 30],
            "qb_pass_yards": [100.0, 150.0],
            "qb_pass_touchdowns": [2.0, 1.0],
            "qb_interceptions": [0.0, 2.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_yards_per_attempt").item() == 6.25
    assert result.select("qb_touchdown_rate").item() == 0.075
    assert result.select("qb_interception_rate").item() == 0.05


def test_compute_qb_season_stats_derives_dropback_metrics_and_totals() -> None:
    """Verify season QB summaries expose dropback-based totals and advanced rates."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [10, 30],
            "qb_dropbacks": [12, 34],
            "qb_offense_snaps": [60, 65],
            "qb_pass_yards": [100.0, 150.0],
            "qb_pass_touchdowns": [2.0, 1.0],
            "qb_interceptions": [0.0, 2.0],
            "qb_sacks": [2.0, 1.0],
            "qb_sack_yards_lost": [12.0, 7.0],
            "qb_passing_epa": [5.0, 3.0],
            "qb_spike_epa": [0.0, 0.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_dropbacks_total").item() == 46
    assert result.select("qb_offense_snaps_total").item() == 125
    assert result.select("qb_td_int_differential").item() == 1.0
    assert result.select("qb_epa_per_dropback").item() == pytest.approx(8.0 / 46.0)
    assert result.select("qb_any_a").item() == pytest.approx(201.0 / 43.0)
    assert result.select("qb_pass_yards_per_dropback").item() == pytest.approx(250.0 / 46.0)
    assert result.select("qb_sack_rate").item() == pytest.approx(3.0 / 46.0)
    assert result.select("qb_td_int_margin_rate").item() == pytest.approx(1.0 / 46.0)


def test_compute_qb_season_stats_leaves_win_pct_empty_when_results_unavailable() -> None:
    """Verify qb_win_pct is empty, not NaN or a made-up .500, when team/week score joins fail."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["LAC", "LAC"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [30, 28],
            "qb_passer_rating": [90.0, 92.0],
        }
    )
    # Intentionally mismatched team code so joined points are unavailable.
    weekly_df = pl.DataFrame(
        {
            "team": ["SD", "SD"],
            "week": [1, 2],
            "points_for": [24, 21],
            "points_allowed": [20, 17],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.select("qb_win_pct").item() is None


def test_compute_qb_season_stats_leaves_win_pct_empty_without_team_results() -> None:
    """Verify qb_win_pct is empty when no weekly team results are given."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [30, 28],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_win_pct").item() is None
    assert result.schema["qb_win_pct"] == pl.Float64


def test_compute_qb_game_volumes_from_pbp_combines_dropbacks_and_snap_counts() -> None:
    """Verify QB game volumes union PBP dropbacks with QB snap-count rows."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 4,
            "week": [1] * 4,
            "posteam": ["DEN", "DEN", "DEN", "KC"],
            "passer_player_id": ["GSIS_A", "GSIS_A", "GSIS_B", "GSIS_C"],
            "passer_player_name": ["Starter QB", "Starter QB", "Backup QB", "KC QB"],
            "qb_dropback": [1, 1, 1, 1],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": [
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
                "2025_01_DEN_KC",
            ],
            "week": [1, 1, 1, 1],
            "team": ["DEN", "DEN", "DEN", "KC"],
            "player": ["Starter QB", "Backup QB", "Runner QB", "KC QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_D", "PFR_C"],
            "position": ["QB", "QB", "QB", "QB"],
            "offense_snaps": [42.0, 16.0, 3.0, 58.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_volumes_from_pbp(pbp, snap_counts).sort(
        ["team_abbr", "qb_name"]
    )

    # Assert
    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Backup QB",
            "qb_id": "GSIS_B",
            "snap_player_id": "PFR_B",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 16,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Runner QB",
            "qb_id": None,
            "snap_player_id": "PFR_D",
            "qb_dropbacks": 0,
            "qb_offense_snaps": 3,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Starter QB",
            "qb_id": "GSIS_A",
            "snap_player_id": "PFR_A",
            "qb_dropbacks": 2,
            "qb_offense_snaps": 42,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "KC",
            "qb_name": "KC QB",
            "qb_id": "GSIS_C",
            "snap_player_id": "PFR_C",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 58,
        },
    ]


def test_compute_qb_game_stats_from_pbp_derives_dropback_metrics() -> None:
    """Verify PBP-derived QB game stats carry dropbacks, snaps, and core passing metrics.

    CPOE carries its hidden parts: the CPOE summed over the passer's plays that have one, and
    their count.
    """
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 4,
            "week": [1] * 4,
            "posteam": ["DEN", "DEN", "DEN", "DEN"],
            "passer_player_id": ["GSIS_A", "GSIS_A", "GSIS_A", "GSIS_B"],
            "passer_player_name": ["Starter QB", "Starter QB", "Starter QB", "Backup QB"],
            "qb_dropback": [1, 1, 1, 1],
            "pass": [1, 0, 1, 1],
            "complete_pass": [1, 0, 0, 0],
            "passing_yards": [20.0, 0.0, 0.0, 0.0],
            "yards_gained": [20.0, -7.0, 0.0, 0.0],
            "pass_touchdown": [1, 0, 0, 0],
            "interception": [0, 0, 0, 1],
            "sack": [0, 1, 0, 0],
            "fumble_lost": [0, 1, 0, 0],
            "qb_epa": [2.0, -1.0, -0.5, -2.0],
            "cpoe": [5.0, None, -3.0, -6.0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "team": ["DEN", "DEN", "DEN"],
            "player": ["Starter QB", "Backup QB", "Runner QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_D"],
            "position": ["QB", "QB", "QB"],
            "offense_snaps": [42.0, 16.0, 3.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts).sort("qb_name")

    # Assert
    assert result.to_dicts() == [
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Backup QB",
            "qb_id": "GSIS_B",
            "snap_player_id": "PFR_B",
            "qb_dropbacks": 1,
            "qb_offense_snaps": 16,
            "qb_attempts": 1,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 1,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": -2.0,
            "qb_spike_epa": 0.0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": -2.0,
            "qb_pass_yards_per_dropback": 0.0,
            "qb_td_int_margin_rate": -1.0,
            "qb_sack_rate": 0.0,
            "qb_any_a": -45.0,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": -6.0,
            numerator_column(qb_stats.CPOE_COLUMN): -6.0,
            denominator_column(qb_stats.CPOE_COLUMN): 1,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Runner QB",
            "qb_id": None,
            "snap_player_id": "PFR_D",
            "qb_dropbacks": 0,
            "qb_offense_snaps": 3,
            "qb_attempts": 0,
            "qb_completions": 0,
            "qb_pass_yards": 0.0,
            "qb_pass_touchdowns": 0,
            "qb_interceptions": 0,
            "qb_sacks": 0,
            "qb_sack_yards_lost": 0.0,
            "qb_sack_fumbles_lost": 0,
            "qb_passing_epa": 0.0,
            "qb_spike_epa": 0.0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": None,
            "qb_pass_yards_per_dropback": None,
            "qb_td_int_margin_rate": None,
            "qb_sack_rate": None,
            "qb_any_a": None,
            "qb_scramble_rate": None,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": None,
            numerator_column(qb_stats.CPOE_COLUMN): None,
            denominator_column(qb_stats.CPOE_COLUMN): None,
        },
        {
            "game_id": "2025_01_DEN_KC",
            "week": 1,
            "team_abbr": "DEN",
            "qb_name": "Starter QB",
            "qb_id": "GSIS_A",
            "snap_player_id": "PFR_A",
            "qb_dropbacks": 3,
            "qb_offense_snaps": 42,
            "qb_attempts": 2,
            "qb_completions": 1,
            "qb_pass_yards": 20.0,
            "qb_pass_touchdowns": 1,
            "qb_interceptions": 0,
            "qb_sacks": 1,
            "qb_sack_yards_lost": 7.0,
            "qb_sack_fumbles_lost": 1,
            "qb_passing_epa": 0.5,
            "qb_spike_epa": 0.0,
            "qb_designed_carries": 0,
            "qb_designed_rush_yards": 0.0,
            "qb_designed_rush_epa": 0.0,
            "qb_scrambles": 0,
            "qb_scramble_yards": 0.0,
            "qb_kneels": 0,
            "qb_epa_per_dropback": 1 / 6,
            "qb_pass_yards_per_dropback": 20.0 / 3.0,
            "qb_td_int_margin_rate": 1 / 3,
            "qb_sack_rate": 1 / 3,
            "qb_any_a": 11.0,
            "qb_scramble_rate": 0.0,
            "qb_yards_per_scramble": None,
            "qb_designed_yards_per_carry": None,
            "qb_designed_epa_per_carry": None,
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_percentage_above_expectation": 1.0,
            numerator_column(qb_stats.CPOE_COLUMN): 2.0,
            denominator_column(qb_stats.CPOE_COLUMN): 2,
        },
    ]


def _pass_sack_and_spike_pbp() -> pl.DataFrame:
    """Return one DEN passer's pass (+1.2 EPA), sack (-0.4), and spike (-0.2) in one game."""
    return pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 3,
            "week": [1, 1, 1],
            "posteam": ["DEN"] * 3,
            "defteam": ["KC"] * 3,
            "passer_player_id": ["GSIS_A"] * 3,
            "passer_player_name": ["Starter QB"] * 3,
            "play_type": ["pass", "pass", "qb_spike"],
            "qb_dropback": [1, 1, 0],
            "qb_spike": [0, 0, 1],
            "pass": [1, 0, 1],
            "complete_pass": [1, 0, 0],
            "passing_yards": [18.0, 0.0, 0.0],
            "pass_touchdown": [0, 0, 0],
            "interception": [0, 0, 0],
            "sack": [0, 1, 0],
            "yards_gained": [18.0, -6.0, 0.0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [1.2, -0.4, -0.2],
        }
    )


def test_compute_qb_game_stats_from_pbp_leaves_spikes_out_of_epa_per_dropback() -> None:
    """A spike is a called incompletion to stop the clock, so the rating's EPA leaves it out.

    Passing EPA keeps it, as nflverse's official passing EPA does: 1.2 - 0.4 - 0.2 = 0.6 over the
    passer's plays, of which -0.2 is the spike, so EPA per dropback is (0.6 + 0.2) / 2 = 0.4.
    """
    # Arrange
    pbp = _pass_sack_and_spike_pbp()

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp)

    # Assert
    row = result.row(0, named=True)
    assert (row["qb_dropbacks"], row["qb_passing_epa"], row["qb_spike_epa"]) == pytest.approx(
        (2, 0.6, -0.2)
    )
    assert row["qb_epa_per_dropback"] == pytest.approx(0.4)


def test_compute_qb_season_stats_leaves_spikes_out_of_epa_per_dropback() -> None:
    """The season rate pools passing EPA less spike EPA over dropbacks: (3.0 + 0.5) / 50."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [20, 25],
            "qb_dropbacks": [22, 28],
            "qb_passing_epa": [2.0, 1.0],
            "qb_spike_epa": [-0.2, -0.3],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert result.select("qb_epa_per_dropback").item() == pytest.approx(3.5 / 50)
    assert result.select("qb_spike_epa_total").item() == pytest.approx(-0.5)


def test_compute_qb_game_stats_from_pbp_counts_only_the_offenses_sack_fumbles_lost() -> None:
    """A strip-sack the defense recovers and fumbles back is not the quarterback's lost fumble.

    The second sack's ``fumble_lost`` belongs to KC (``fumbled_1_team``): its defender lost the
    ball he had recovered. Only the first sack's fumble is DEN's.
    """
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC"] * 2,
            "week": [1, 1],
            "posteam": ["DEN", "DEN"],
            "defteam": ["KC", "KC"],
            "passer_player_id": ["GSIS_A", "GSIS_A"],
            "passer_player_name": ["Starter QB", "Starter QB"],
            "qb_dropback": [1, 1],
            "pass": [0, 0],
            "complete_pass": [0, 0],
            "passing_yards": [0.0, 0.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [1, 1],
            "yards_gained": [-5.0, -6.0],
            "fumble_lost": [1, 1],
            "fumbled_1_team": ["DEN", "KC"],
            "qb_epa": [-1.0, -1.5],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp)

    # Assert
    assert result.get_column("qb_sack_fumbles_lost").to_list() == [1]


def test_compute_qb_game_stats_from_pbp_splits_designed_runs_scrambles_and_kneels() -> None:
    """Verify QB rushing PBP splits exclude scrambles and kneels from designed-run value."""
    # Arrange
    # Rows: a completion, a spike, a designed run, a scramble, and a kneel. As in nflverse, the
    # scramble and the kneel have rush = 0 and a rusher but no passer, and the scramble is a pass.
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_04_BUF_MIA"] * 5,
            "week": [4] * 5,
            "posteam": ["BUF"] * 5,
            "passer_player_id": ["GSIS_A", "GSIS_A", None, None, None],
            "passer_player_name": ["Dual Threat QB", "Dual Threat QB", None, None, None],
            "rusher_player_id": [None, None, "GSIS_A", "GSIS_A", "GSIS_A"],
            "qb_dropback": [1, 1, 0, 1, 0],
            "pass": [1, 0, 0, 1, 0],
            "complete_pass": [1, 0, 0, 0, 0],
            "passing_yards": [15.0, 0.0, 0.0, 0.0, 0.0],
            "yards_gained": [15.0, 0.0, 8.0, 12.0, -1.0],
            "pass_touchdown": [0, 0, 0, 0, 0],
            "interception": [0, 0, 0, 0, 0],
            "sack": [0, 0, 0, 0, 0],
            "fumble_lost": [0, 0, 0, 0, 0],
            "qb_epa": [1.5, 0.0, 1.2, 0.8, -0.9],
            "cpoe": [4.0, None, None, None, None],
            "rush": [0, 0, 1, 0, 0],
            "qb_scramble": [0, 0, 0, 1, 0],
            "qb_kneel": [0, 0, 0, 0, 1],
            "qb_spike": [0, 1, 0, 0, 0],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_04_BUF_MIA"],
            "week": [4],
            "team": ["BUF"],
            "player": ["Dual Threat QB"],
            "pfr_player_id": ["PFR_A"],
            "position": ["QB"],
            "offense_snaps": [58.0],
        }
    )
    qb_identity = pl.DataFrame(
        {
            "qb_id": ["GSIS_A"],
            "snap_player_id": ["PFR_A"],
            "qb_name": ["Dual Threat QB"],
            "qb_position": ["QB"],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts, qb_identity)

    # Assert
    row = result.row(0, named=True)

    # Dropbacks and passing EPA count only plays with a passer, so the scramble is in neither.
    assert row["qb_dropbacks"] == 2
    assert row["qb_passing_epa"] == pytest.approx(1.5)
    assert row["qb_designed_carries"] == 1
    assert row["qb_designed_rush_yards"] == 8.0
    assert row["qb_designed_rush_epa"] == pytest.approx(1.2)
    assert row["qb_scrambles"] == 1
    assert row["qb_scramble_yards"] == 12.0
    assert row["qb_kneels"] == 1
    assert row["qb_scramble_rate"] == pytest.approx(1.0 / 2.0)
    assert row["qb_yards_per_scramble"] == pytest.approx(12.0)
    assert row["qb_designed_yards_per_carry"] == pytest.approx(8.0)
    assert row["qb_designed_epa_per_carry"] == pytest.approx(1.2)


def test_compute_qb_season_stats_reconciles_designed_rush_components_for_multi_team_qb() -> None:
    """Verify designed-run splits aggregate by QB across teams and reconcile to official carries."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "week": [1, 2, 3],
            "team_abbr": ["BAL", "BAL", "HOU"],
            "qb_id": ["qb-1", "qb-1", "qb-1"],
            "qb_name": ["Dual Threat QB", "Dual Threat QB", "Dual Threat QB"],
            "qb_dropbacks": [30, 25, 20],
            "qb_attempts": [28, 23, 18],
            "qb_carries": [5, 4, 6],
            "qb_rushing_yards": [31.0, 25.0, 40.0],
            "qb_designed_carries": [2, 1, 3],
            "qb_designed_rush_yards": [14.0, 7.0, 18.0],
            "qb_designed_rush_epa": [1.0, 0.4, 1.6],
            "qb_scrambles": [2, 2, 2],
            "qb_scramble_yards": [18.0, 19.0, 24.0],
            "qb_kneels": [1, 1, 1],
        }
    )

    # Act
    season = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    row = season.to_dicts()[0]

    assert season.height == 1
    assert row["team"] == "BAL"
    assert row["qb_carries_total"] == 15
    assert row["qb_designed_carries_total"] == 6
    assert row["qb_scrambles_total"] == 6
    assert row["qb_kneels_total"] == 3
    assert (
        row["qb_designed_carries_total"] + row["qb_scrambles_total"] + row["qb_kneels_total"]
        == row["qb_carries_total"]
    )
    assert row["qb_designed_rush_yards_total"] == 39.0
    assert row["qb_designed_rush_epa_total"] == pytest.approx(3.0)
    assert row["qb_designed_carries_per_game"] == pytest.approx(2.0)
    assert row["qb_designed_rush_yards_per_game"] == pytest.approx(13.0)
    assert row["qb_designed_yards_per_carry"] == pytest.approx(6.5)
    assert row["qb_designed_epa_per_carry"] == pytest.approx(0.5)
    assert row["qb_scramble_rate"] == pytest.approx(6.0 / 75.0)
    assert row["qb_yards_per_scramble"] == pytest.approx(61.0 / 6.0)


def test_compute_qb_game_stats_from_pbp_assigns_4qc_and_gwd_to_primary_qb() -> None:
    """Verify late-game comeback flags are assigned only to the primary QB row."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "posteam": ["DEN", "DEN", "KC"],
            "passer_player_id": ["GSIS_B", "GSIS_A", "GSIS_C"],
            "passer_player_name": ["Backup QB", "Starter QB", "KC QB"],
            "qb_dropback": [1, 1, 1],
            "pass": [1, 1, 1],
            "complete_pass": [0, 1, 0],
            "passing_yards": [0.0, 30.0, 0.0],
            "yards_gained": [0.0, 30.0, 0.0],
            "pass_touchdown": [0, 1, 0],
            "interception": [0, 0, 0],
            "sack": [0, 0, 0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [-0.5, 4.0, -1.0],
            "cpoe": [-2.0, 7.0, -3.0],
            "qtr": [2, 4, 4],
            "score_differential": [0, -3, -4],
            "score_differential_post": [0, 4, -4],
            "posteam_score": [7, 17, 20],
            "posteam_score_post": [7, 24, 20],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_01_DEN_KC", "2025_01_DEN_KC", "2025_01_DEN_KC"],
            "week": [1, 1, 1],
            "team": ["DEN", "DEN", "KC"],
            "player": ["Starter QB", "Backup QB", "KC QB"],
            "pfr_player_id": ["PFR_A", "PFR_B", "PFR_C"],
            "position": ["QB", "QB", "QB"],
            "offense_snaps": [50.0, 10.0, 60.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts).sort("qb_name")

    # Assert
    assert (
        result.filter(pl.col("qb_name") == "Starter QB").select("qb_fourth_quarter_comeback").item()
        == 1
    )
    assert (
        result.filter(pl.col("qb_name") == "Starter QB").select("qb_game_winning_drive").item() == 1
    )
    assert (
        result.filter(pl.col("qb_name") == "Backup QB").select("qb_fourth_quarter_comeback").item()
        == 0
    )
    assert (
        result.filter(pl.col("qb_name") == "Backup QB").select("qb_game_winning_drive").item() == 0
    )
    assert (
        result.filter(pl.col("qb_name") == "KC QB").select("qb_fourth_quarter_comeback").item() == 0
    )
    assert result.filter(pl.col("qb_name") == "KC QB").select("qb_game_winning_drive").item() == 0


def test_compute_qb_game_stats_from_pbp_does_not_assign_4qc_or_gwd_in_loss() -> None:
    """Verify late-game flags are cleared if the team ultimately loses after a late lead."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2025_03_NYJ_NE", "2025_03_NYJ_NE"],
            "week": [3, 3],
            "posteam": ["NYJ", "NE"],
            "passer_player_id": ["GSIS_TY", "GSIS_NE"],
            "passer_player_name": ["Tyrod Taylor", "NE QB"],
            "qb_dropback": [1, 1],
            "pass": [1, 1],
            "complete_pass": [1, 1],
            "passing_yards": [20.0, 18.0],
            "yards_gained": [20.0, 18.0],
            "pass_touchdown": [1, 1],
            "interception": [0, 0],
            "sack": [0, 0],
            "fumble_lost": [0, 0],
            "qb_epa": [2.0, 1.5],
            "cpoe": [4.0, 3.0],
            "qtr": [4, 4],
            "score_differential": [-3, -4],
            "score_differential_post": [4, 2],
            "posteam_score": [20, 23],
            "posteam_score_post": [27, 29],
        }
    )
    snap_counts = pl.DataFrame(
        {
            "game_id": ["2025_03_NYJ_NE", "2025_03_NYJ_NE"],
            "week": [3, 3],
            "team": ["NYJ", "NE"],
            "player": ["Tyrod Taylor", "NE QB"],
            "pfr_player_id": ["TaylTy00", "NEQb00"],
            "position": ["QB", "QB"],
            "offense_snaps": [60.0, 58.0],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp, snap_counts)

    # Assert
    tyrod_row = result.filter(pl.col("qb_name") == "Tyrod Taylor")

    assert tyrod_row.select("qb_fourth_quarter_comeback").item() == 0
    assert tyrod_row.select("qb_game_winning_drive").item() == 0


def test_compute_qb_season_stats_aggregates_rushing_and_completion_rates() -> None:
    """Verify season totals, per-game fields, and rates for the rushing family."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "week": [1, 2],
            "team_abbr": ["DEN", "DEN"],
            "qb_id": ["qb-1", "qb-1"],
            "qb_name": ["John Doe", "John Doe"],
            "qb_dropbacks": [30, 34],
            "qb_attempts": [28, 32],
            "qb_completions": [20, 25],
            "qb_pass_yards": [220.0, 260.0],
            "qb_pass_touchdowns": [2, 1],
            "qb_interceptions": [0, 1],
            "qb_sacks": [1, 1],
            "qb_sack_yards_lost": [7.0, 6.0],
            "qb_passing_epa": [5.0, 4.0],
            "qb_carries": [6, 4],
            "qb_rushing_yards": [42.0, 18.0],
            "qb_rushing_tds": [1, 0],
            "qb_rushing_first_downs": [3, 1],
            "qb_rushing_epa": [2.0, 0.5],
            "qb_rushing_fumbles": [0, 1],
            "qb_rushing_fumbles_lost": [0, 1],
            "qb_rushing_2pt_conversions": [0, 0],
        }
    )

    # Act
    season = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    row = season.to_dicts()[0]

    assert row["qb_carries_total"] == 10
    assert row["qb_rushing_yards_total"] == 60.0
    assert row["qb_rushing_tds_total"] == 1
    assert row["qb_carries_per_game"] == 5.0
    assert row["qb_rushing_yards_per_game"] == 30.0
    assert row["qb_completion_pct"] == 45.0 / 60.0
    assert row["qb_yards_per_carry"] == 6.0
    assert row["qb_epa_per_carry"] == 0.25


def test_compute_qb_season_stats_empty_input_returns_a_typed_empty_frame() -> None:
    # Arrange
    qb_df = pl.DataFrame()

    # Act
    season = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert season.is_empty()
    assert season.schema["qb_is_eligible"] == pl.Boolean


def test_compute_qb_season_stats_without_identity_columns_groups_by_team() -> None:
    # Arrange
    qb_df = pl.DataFrame({"team_abbr": ["DEN", "DEN"], "week": [1, 2], "qb_attempts": [30, 20]})

    # Act
    season = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    assert season.select("qb_id", "qb_attempts_total").rows() == [("DEN", 50)]


@pytest.mark.parametrize(
    "qb_df",
    [
        pl.DataFrame({"qb_name": ["A"], "qb_attempts": [10]}),
        pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_name": ["A"]}),
    ],
)
def test_select_primary_qb_rows_returns_rows_it_cannot_rank(qb_df: pl.DataFrame) -> None:
    # Act
    selected = qb_stats.select_primary_qb_rows(qb_df)

    # Assert
    assert selected.equals(qb_df)


def test_compute_qb_game_stats_from_pbp_without_plays_returns_zeroed_columns() -> None:
    # Act
    games = qb_stats.compute_qb_game_stats_from_pbp(pl.DataFrame())

    # Assert
    assert games.is_empty()
    assert {"qb_attempts", "qb_completions"} <= set(games.columns)


def test_canonicalize_qb_rows_empty_input_adds_a_position_column() -> None:
    # Act
    rows = qb_stats._canonicalize_qb_rows(pl.DataFrame({"qb_id": []}), None, join_key="qb_id")

    # Assert
    assert "qb_position" in rows.columns


def test_canonicalize_qb_rows_fills_missing_identity_columns() -> None:
    # Arrange
    rows = pl.DataFrame({"team_abbr": ["DEN"], "week": [1]})

    # Act
    canonical = qb_stats._canonicalize_qb_rows(rows, None, join_key="qb_id")

    # Assert
    assert {"qb_id", "snap_player_id", "qb_name"} <= set(canonical.columns)


@pytest.mark.parametrize("weekly_df", [None, pl.DataFrame()])
def test_default_qb_attempt_qualifier_without_weekly_data_uses_seventeen_games(
    weekly_df: pl.DataFrame | None,
) -> None:
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN"],
            "week": [1],
            "qb_id": ["QB_A"],
            "qb_name": ["QB A"],
            "qb_attempts": [30],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.get_column("qb_attempt_qualifier").to_list() == [238]


def test_compute_qb_game_volumes_from_pbp_without_dropbacks_or_qb_snaps_is_empty() -> None:
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "posteam": ["DEN"],
            "passer_player_id": [None],
            "passer_player_name": [None],
            "qb_dropback": [0],
        },
        schema_overrides={"passer_player_id": pl.String, "passer_player_name": pl.String},
    )
    snaps = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "team": ["DEN"],
            "player": ["Some Tackle"],
            "pfr_player_id": ["TackXx00"],
            "position": ["T"],
            "offense_snaps": [60.0],
        }
    )

    # Act
    volumes = qb_stats.compute_qb_game_volumes_from_pbp(pbp, snaps)

    # Assert
    assert volumes.is_empty()


def _one_dropback_pbp() -> pl.DataFrame:
    """Return play-by-play with a single completed pass, without a ``cpoe`` column."""
    return pl.DataFrame(
        {
            "game_id": ["2005_01_DEN_KC"],
            "week": [1],
            "posteam": ["DEN"],
            "passer_player_id": ["00-0031234"],
            "passer_player_name": ["J.Doe"],
            "qb_dropback": [1],
            "pass": [1],
            "complete_pass": [1],
            "passing_yards": [12.0],
            "pass_touchdown": [0],
            "interception": [0],
            "sack": [0],
            "fumble_lost": [0],
            "qb_epa": [0.8],
        }
    )


def test_compute_qb_game_stats_from_pbp_without_plays_has_the_full_schema() -> None:
    # Arrange
    full = qb_stats.compute_qb_game_stats_from_pbp(_one_dropback_pbp())

    # Act
    empty = qb_stats.compute_qb_game_stats_from_pbp(pl.DataFrame())

    # Assert
    assert empty.schema == full.schema


def test_compute_qb_game_stats_from_pbp_without_cpoe_leaves_it_null() -> None:
    # Act
    games = qb_stats.compute_qb_game_stats_from_pbp(_one_dropback_pbp())

    # Assert
    row = games.row(0, named=True)
    assert row["qb_completion_percentage_above_expectation"] is None
    assert row["qb_epa_per_dropback"] == pytest.approx(0.8)


# The typed empty frame the loader returns for a season before snap counts exist.
_EMPTY_SNAP_COUNTS = pl.DataFrame(
    schema={
        "game_id": pl.String,
        "week": pl.Int64,
        "team": pl.String,
        "player": pl.String,
        "pfr_player_id": pl.String,
        "position": pl.String,
        "offense_snaps": pl.Float64,
    }
)


@pytest.mark.parametrize("snap_counts", [None, _EMPTY_SNAP_COUNTS], ids=["absent", "empty"])
def test_compute_qb_game_stats_from_pbp_leaves_snaps_empty_without_snap_counts(
    snap_counts: pl.DataFrame | None,
) -> None:
    """Verify a season without snap-count data has unknown QB snaps, not zero snaps."""
    # Act
    games = qb_stats.compute_qb_game_stats_from_pbp(_one_dropback_pbp(), snap_counts)

    # Assert
    row = games.row(0, named=True)
    assert row["qb_offense_snaps"] is None
    assert row["qb_dropbacks"] == 1
    assert games.schema["qb_offense_snaps"] == pl.Int64


def test_compute_qb_season_stats_leaves_snap_totals_empty_when_snaps_are_unknown() -> None:
    """Verify unknown game snaps give an unknown season snap total, not zero."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN"],
            "week": [1, 2],
            "qb_id": ["QB_A", "QB_A"],
            "qb_name": ["QB A", "QB A"],
            "qb_attempts": [30, 28],
            "qb_dropbacks": [33, 31],
            "qb_offense_snaps": pl.Series([None, None], dtype=pl.Int64),
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df)

    # Assert
    row = result.row(0, named=True)
    assert row["qb_offense_snaps_total"] is None
    assert row["qb_offense_snaps_per_game"] is None
    assert row["qb_dropbacks_total"] == 64


def test_compute_qb_season_stats_picks_the_primary_qb_by_dropbacks_when_snaps_are_unknown() -> None:
    """Verify unknown snaps leave the primary-QB pick to dropbacks, as zero snaps did."""
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["ATL", "ATL"],
            "week": [6, 6],
            "qb_id": ["QB_A", "QB_B"],
            "qb_name": ["QB A", "QB B"],
            "qb_offense_snaps": pl.Series([None, None], dtype=pl.Int64),
            "qb_dropbacks": [5, 30],
            "qb_attempts": [4, 28],
        }
    )
    weekly_df = pl.DataFrame(
        {"team": ["ATL"], "week": [6], "points_for": [24], "points_allowed": [17]}
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.filter(pl.col("qb_wins") == 1).get_column("qb_id").to_list() == ["QB_B"]


def _tied_starters() -> pl.DataFrame:
    """Return one team-game where two passers tie on snaps, dropbacks, and attempts."""
    return pl.DataFrame(
        {
            "team_abbr": ["ATL", "ATL"],
            "week": [6, 6],
            "qb_id": ["QB_B", "QB_A"],
            "qb_name": ["QB B", "QB A"],
            "qb_offense_snaps": [0, 0],
            "qb_dropbacks": [20, 20],
            "qb_attempts": [18, 18],
        }
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_compute_qb_season_stats_breaks_a_primary_qb_tie_the_same_way_in_any_row_order(
    *, reverse: bool
) -> None:
    # Arrange
    qb_df = _tied_starters().reverse() if reverse else _tied_starters()
    weekly_df = pl.DataFrame(
        {"team": ["ATL"], "week": [6], "points_for": [24], "points_allowed": [17]}
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    assert result.filter(pl.col("qb_wins") == 1).get_column("qb_id").to_list() == ["QB_A"]


@pytest.mark.parametrize("reverse", [False, True])
def test_compute_qb_season_stats_gives_a_traded_qb_his_latest_team_on_a_games_tie(
    *, reverse: bool
) -> None:
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["BUF", "BUF", "JAX", "JAX"],
            "week": [1, 2, 5, 6],
            "qb_id": ["QB_A"] * 4,
            "qb_name": ["QB A"] * 4,
            "qb_attempts": [20, 20, 20, 20],
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df.reverse() if reverse else qb_df)

    # Assert
    assert result.get_column("team").to_list() == ["JAX"]


def test_compute_qb_season_stats_qualifies_against_the_qbs_own_team_games() -> None:
    # Arrange
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["SEA", "SEA", "KC", "KC", "KC", "KC"],
            "week": [1, 2, 1, 2, 3, 4],
            "qb_id": ["QB_SEA", "QB_SEA", "QB_KC", "QB_KC", "QB_KC", "QB_KC"],
            "qb_name": ["SEA QB", "SEA QB", "KC QB", "KC QB", "KC QB", "KC QB"],
            "qb_attempts": [24, 24, 12, 12, 12, 12],
        }
    )
    weekly_df = pl.DataFrame(
        {
            "team": ["SEA"] * 3 + ["KC"] * 4,
            "week": [1, 2, 3, 1, 2, 3, 4],
            "points_for": [20] * 7,
            "points_allowed": [17] * 7,
        }
    )

    # Act
    result = qb_stats.compute_qb_season_stats(qb_df, weekly_df=weekly_df)

    # Assert
    eligible = dict(zip(result["qb_id"], result["qb_is_eligible"], strict=True))
    assert eligible == {"QB_SEA": True, "QB_KC": False}


def test_late_game_flags_ignore_rows_without_a_real_team() -> None:
    # Arrange
    plays = pl.DataFrame(
        {
            "game_id": ["g1"] * 4,
            "posteam": ["ATL", "DAL", "", "ATL"],
            "qtr": [4, 4, 4, 4],
            "score_differential": [-10, 10, 0, -17],
            "score_differential_post": [-10, 17, 0, -17],
            "posteam_score": [0, 10, 0, 7],
            "posteam_score_post": [0, 17, 0, 7],
        }
    )

    # Act
    flags = qb_stats._compute_team_late_game_flags_from_pbp(plays)

    # Assert
    assert flags.sort("team_abbr").select("team_abbr", "qb_fourth_quarter_comeback").to_dicts() == [
        {"team_abbr": "ATL", "qb_fourth_quarter_comeback": 0},
        {"team_abbr": "DAL", "qb_fourth_quarter_comeback": 0},
    ]


def test_compute_qb_game_stats_from_pbp_keeps_one_row_when_a_passers_name_varies() -> None:
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": ["2010_09_NO_CAR"] * 3,
            "week": [9] * 3,
            "posteam": ["CAR"] * 3,
            "passer_player_id": ["GSIS_P"] * 3,
            "passer_player_name": ["T.Pike", "T.Pike (3rd QB)", "T.Pike (3rd QB)"],
            "qb_dropback": [1, 1, 1],
            "pass": [1, 1, 1],
            "complete_pass": [1, 0, 1],
            "passing_yards": [12.0, 0.0, 8.0],
            "yards_gained": [12.0, 0.0, 8.0],
            "pass_touchdown": [0, 0, 0],
            "interception": [0, 0, 0],
            "sack": [0, 0, 0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [0.5, -0.4, 0.2],
            "cpoe": [None, None, None],
        }
    )

    # Act
    result = qb_stats.compute_qb_game_stats_from_pbp(pbp)

    # Assert
    assert result.select("qb_id", "qb_attempts").to_dicts() == [
        {"qb_id": "GSIS_P", "qb_attempts": 3}
    ]


def _two_uneven_games() -> pl.DataFrame:
    """Return one passer's 10-attempt and 40-attempt games, with CPOE's hidden parts.

    Game 1: 9 of 10, 100 yards, 1 TD, CPOE +5.0 over 8 plays with a completion probability.
    Game 2: 20 of 40, 300 yards, 1 TD, 3 INT, CPOE -1.25 over 32 such plays.
    """
    cpoe = "qb_completion_percentage_above_expectation"
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "week": [1, 2],
            "team_abbr": ["DEN", "DEN"],
            "qb_id": ["qb-1", "qb-1"],
            "qb_name": ["John Doe", "John Doe"],
            "qb_dropbacks": [11, 42],
            "qb_attempts": [10, 40],
            "qb_completions": [9, 20],
            "qb_pass_yards": [100.0, 300.0],
            "qb_pass_touchdowns": [1, 1],
            "qb_interceptions": [0, 3],
            "qb_sacks": [1, 2],
            "qb_sack_yards_lost": [5.0, 12.0],
            "qb_passing_epa": [6.0, -8.0],
            cpoe: [5.0, -1.25],
            "qb_passer_rating": [135.4, 51.0],
            numerator_column(cpoe): [40.0, -40.0],
            denominator_column(cpoe): [8, 32],
        }
    )


def test_compute_qb_season_stats_rates_the_season_totals_not_the_mean_game() -> None:
    # Arrange
    qb_df = _two_uneven_games()

    # Act
    row = qb_stats.compute_qb_season_stats(qb_df).row(0, named=True)

    # Assert
    # Passer rating of 29 of 50, 400 yards, 2 TD, 3 INT, not the mean of 135.4 and 51.0.
    completion = min(max((29 / 50 - 0.3) * 5, 0.0), 2.375)
    yards = min(max((400 / 50 - 3.0) * 0.25, 0.0), 2.375)
    touchdowns = min(max(2 / 50 * 20, 0.0), 2.375)
    interceptions = min(max(2.375 - 3 / 50 * 25, 0.0), 2.375)
    expected = round((completion + yards + touchdowns + interceptions) / 6 * 100, 1)
    assert row["qb_passer_rating"] == pytest.approx(expected)
    # CPOE over all 40 plays with a completion probability: (40 - 40) / 40.
    assert row["qb_completion_percentage_above_expectation"] == pytest.approx(0.0)
    assert not [column for column in row if column.startswith("_")]


# Passing lines whose passer rating is exactly a tie at one decimal, with the rating to show:
# 1999 Testaverde (78.75) and the 1999 Bills passers IND faced (82.75) compute just below the tie in
# floating point, depending on how many rows Polars evaluates together; 81.25 is exact, where
# half-to-even rounding would give 81.2.
_RATING_TIES = (
    ((10, 15, 96, 1, 1), 78.8),
    ((276, 475, 3116, 17, 8), 82.8),
    ((1, 4, 14, 1, 0), 81.3),
)
_PASSING_NAMES = ("completions", "attempts", "yards", "touchdowns", "interceptions")


def test_qb_passer_rating_rounds_an_exact_tie_up() -> None:
    # Arrange
    passing = pl.DataFrame(
        [totals for totals, _ in _RATING_TIES], schema=list(_PASSING_NAMES), orient="row"
    ).cast(pl.Float64)

    # Act
    result = passing.select(
        qb_stats.qb_passer_rating_expr(*(pl.col(name) for name in _PASSING_NAMES))
    )

    # Assert
    assert result.to_series().to_list() == [rating for _, rating in _RATING_TIES]


def test_compute_qb_season_stats_rounds_a_season_rating_tie_up() -> None:
    # Arrange
    games = [
        {
            "game_id": f"g{index}-{half}",
            "week": half + 1,
            "team_abbr": "NYJ",
            "qb_id": f"qb-{index}",
            "qb_name": f"Passer {index}",
            # Each passer's line, split over two games.
            **{
                f"qb_{name}": value / 2 if half else value - value / 2
                for name, value in zip(
                    ("completions", "attempts", "pass_yards", "pass_touchdowns", "interceptions"),
                    totals,
                    strict=True,
                )
            },
        }
        for index, (totals, _) in enumerate(_RATING_TIES)
        for half in (0, 1)
    ]

    # Act
    season = qb_stats.compute_qb_season_stats(pl.DataFrame(games)).sort("qb_id")

    # Assert
    assert season.get_column("qb_passer_rating").to_list() == [rating for _, rating in _RATING_TIES]


def test_every_qb_stat_rebuilt_from_totals_for_a_defense_is_a_rate() -> None:
    """Verify QB_RATES holds rates only, as the QB opponent profiles publish them per game."""
    # Arrange
    registry = get_registry()

    # Act
    shapes = {name: registry.metrics[name].shape for name in qb_stats.QB_RATES}

    # Assert
    assert {name: shape for name, shape in shapes.items() if shape not in {"rate", "avg"}} == {}
