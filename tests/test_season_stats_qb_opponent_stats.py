"""Tests for the head-to-head-excluded QB opponent profiles."""

import polars as pl
import pytest

from nfl_predictor.season_stats import qb_opponent_stats
from nfl_predictor.season_stats.pooled_rates import denominator_column, numerator_column
from nfl_predictor.season_stats.qb_stats import CPOE_COLUMN


def test_compute_qb_opponent_profiles_excludes_head_to_head() -> None:
    """Verify QB opponent profiles exclude games against the evaluated QB's team."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["KC", "KC", "LAC", "LAC", "DEN", "DEN", "BUF", "LV"],
            "opponent_team": ["DEN", "BUF", "DEN", "LV", "KC", "LAC", "KC", "LAC"],
            "week": [1, 2, 1, 2, 1, 1, 2, 2],
            "points_allowed": [24, 14, 20, 21, 17, 20, 14, 21],
            "def_sacks": [2, 4, 1, 3, 3, 2, 4, 3],
            "def_interceptions": [0, 2, 1, 1, 1, 1, 2, 1],
            "def_pass_defended": [4, 7, 3, 5, 6, 5, 7, 5],
            "def_tackles_for_loss": [5, 8, 4, 6, 7, 6, 8, 6],
            "def_qb_hits": [6, 9, 5, 7, 8, 7, 9, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "BUF", "LV"],
            "week": [1, 2, 2],
            "qb_passer_rating": [100.0, 85.0, 90.0],
            "qb_completion_percentage_above_expectation": [2.0, -1.0, 0.5],
            "qb_aggressiveness": [11.0, 8.5, 9.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
            "qb_passer_rating": [100.0],
        }
    )

    qb_df = qb_df.with_columns(pl.col("team_abbr").replace({"DEN": "QB_DEN"}).alias("qb_id"))
    qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_name"))

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.columns[0] == "qb_id"
    assert "qopp_points_allowed" in profiles.columns
    assert "qopp_qb_passer_rating" in profiles.columns
    assert profiles.select("qopp_points_allowed").item() == 17.5
    assert profiles.select("qopp_qb_passer_rating").item() == 87.5
    assert details["QB_DEN"][0]["games_included"] == 1


def test_compute_qb_opponent_profiles_counts_actual_games_faced() -> None:
    """Verify repeated opponents are deduplicated before QB opponent averaging."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "DEN", "MIA", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "KC", "BUF", "KC", "BUF", "MIA", "MIA"],
            "week": [1, 2, 3, 4, 5, 4, 5],
            "points_allowed": [20, 24, 21, 30, 10, 14, 21],
            "def_sacks": [2, 3, 1, 5, 1, 4, 2],
            "def_interceptions": [1, 1, 0, 3, 0, 2, 1],
            "def_pass_defended": [4, 5, 3, 8, 2, 7, 4],
            "def_tackles_for_loss": [5, 6, 4, 9, 3, 8, 5],
            "def_qb_hits": [6, 7, 5, 10, 4, 9, 6],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "DEN", "MIA", "MIA"],
            "week": [1, 2, 3, 4, 5],
            "qb_id": ["QB_DEN", "QB_DEN", "QB_DEN", "QB_MIA", "QB_MIA"],
            "qb_name": ["Denver QB", "Denver QB", "Denver QB", "Miami QB", "Miami QB"],
            "qb_passer_rating": [100.0, 99.0, 98.0, 120.0, 60.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("qopp_qb_passer_rating").item() == 90.0
    assert [row["opponent"] for row in details["QB_DEN"]] == ["KC", "BUF"]


def test_compute_qb_opponent_profiles_skips_qb_without_reconstructable_opponents() -> None:
    """Verify QB opponent profiles do not fabricate schedules from team schedules."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "DEN", "KC", "KC", "KC", "BUF", "BUF", "BUF"],
            "opponent_team": ["KC", "KC", "BUF", "DEN", "DEN", "BUF", "KC", "KC", "DEN"],
            "week": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "points_allowed": [20, 24, 21, 30, 10, 14, 21, 17, 28],
            "def_sacks": [2, 3, 1, 5, 1, 4, 2, 3, 2],
            "def_interceptions": [1, 1, 0, 3, 0, 2, 1, 1, 1],
            "def_pass_defended": [4, 5, 3, 8, 2, 7, 4, 5, 5],
            "def_tackles_for_loss": [5, 6, 4, 9, 3, 8, 5, 6, 6],
            "def_qb_hits": [6, 7, 5, 10, 4, 9, 6, 7, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["BUF", "BUF", "KC"],
            "week": [1, 2, 3],
            "qb_id": ["QB_BUF1", "QB_BUF2", "QB_KC"],
            "qb_name": ["Buffalo 1", "Buffalo 2", "KC QB"],
            "qb_passer_rating": [80.0, 90.0, 70.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN"],
            "qb_name": ["Denver QB"],
            "team": ["DEN"],
            "qb_passer_rating": [100.0],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is None
    assert details["QB_DEN"] == []


def test_compute_qb_opponent_profiles_uses_each_qbs_actual_opponents() -> None:
    """Verify same-team QBs are profiled from the opponents each QB actually faced."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "BUF", "LV", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "BUF", "DEN", "DEN", "KC", "BUF", "LV", "MIA"],
            "week": [1, 2, 1, 2, 3, 3, 3, 3],
            "points_allowed": [20, 24, 21, 17, 14, 28, 14, 28],
            "def_sacks": [2, 3, 1, 4, 5, 2, 5, 2],
            "def_interceptions": [1, 1, 0, 2, 3, 1, 3, 1],
            "def_pass_defended": [4, 5, 3, 7, 8, 5, 8, 5],
            "def_tackles_for_loss": [5, 6, 4, 8, 9, 6, 9, 6],
            "def_qb_hits": [6, 7, 5, 9, 10, 7, 10, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "LV", "MIA"],
            "week": [1, 2, 3, 3],
            "qb_id": ["QB_DEN_1", "QB_DEN_2", "QB_LV", "QB_MIA"],
            "qb_name": ["Denver QB 1", "Denver QB 2", "Raiders QB", "Miami QB"],
            "qb_passer_rating": [100.0, 95.0, 70.0, 110.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_DEN_1", "QB_DEN_2"],
            "qb_name": ["Denver QB 1", "Denver QB 2"],
            "team": ["DEN", "DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    profile_values = profiles.sort("qb_id").select("qopp_qb_passer_rating").to_series().to_list()
    assert profile_values == [70.0, 110.0]
    assert [row["opponent"] for row in details["QB_DEN_1"]] == ["KC"]
    assert [row["opponent"] for row in details["QB_DEN_2"]] == ["BUF"]


def test_compute_qb_opponent_profiles_uses_majority_snap_games_only() -> None:
    """Verify QB opponent profiles use only the games each QB primarily played."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "DEN", "KC", "BUF", "LV", "MIA", "KC", "BUF"],
            "opponent_team": ["KC", "BUF", "DEN", "DEN", "KC", "BUF", "LV", "MIA"],
            "week": [1, 2, 1, 2, 3, 3, 3, 3],
            "points_allowed": [20, 24, 21, 17, 14, 28, 14, 28],
            "def_sacks": [2, 3, 1, 4, 5, 2, 5, 2],
            "def_interceptions": [1, 1, 0, 2, 3, 1, 3, 1],
            "def_pass_defended": [4, 5, 3, 7, 8, 5, 8, 5],
            "def_tackles_for_loss": [5, 6, 4, 8, 9, 6, 9, 6],
            "def_qb_hits": [6, 7, 5, 9, 10, 7, 10, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "DEN", "DEN", "BUF", "LV", "MIA"],
            "week": [1, 1, 2, 2, 3, 3],
            "qb_id": ["QB_A", "QB_B", "QB_B", "QB_BUF", "QB_LV", "QB_MIA"],
            "qb_name": ["QB A", "QB B", "QB B", "Buffalo QB", "Raiders QB", "Miami QB"],
            "qb_offense_snaps": [45, 10, 55, 60, 60, 60],
            "qb_dropbacks": [22, 5, 25, 30, 20, 18],
            "qb_passer_rating": [100.0, 80.0, 95.0, 110.0, 70.0, 105.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_A", "QB_B"],
            "qb_name": ["QB A", "QB B"],
            "team": ["DEN", "DEN"],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    profile_values = profiles.sort("qb_id").select("qopp_qb_passer_rating").to_series().to_list()
    assert profile_values == [70.0, 105.0]
    assert [row["opponent"] for row in details["QB_A"]] == ["KC"]
    assert [row["opponent"] for row in details["QB_B"]] == ["BUF"]


def test_compute_qb_opponent_profiles_handles_rams_alias_mismatch() -> None:
    """Verify LA/LAR source mismatches still produce Rams QB opponent profiles."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["LA", "SEA", "ARI", "SEA", "SF"],
            "opponent_team": ["SEA", "LA", "SEA", "ARI", "ARI"],
            "week": [1, 1, 2, 2, 2],
            "points_allowed": [17, 24, 14, 21, 21],
            "def_sacks": [3, 1, 4, 2, 2],
            "def_interceptions": [1, 0, 2, 1, 1],
            "def_pass_defended": [6, 3, 7, 5, 5],
            "def_tackles_for_loss": [7, 4, 8, 6, 6],
            "def_qb_hits": [8, 5, 9, 7, 7],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["LAR", "ARI"],
            "week": [1, 2],
            "qb_id": ["QB_LAR", "QB_ARI"],
            "qb_name": ["Rams QB", "Cardinals QB"],
            "qb_passer_rating": [105.0, 88.0],
        }
    )
    qb_season_df = pl.DataFrame(
        {
            "qb_id": ["QB_LAR"],
            "qb_name": ["Rams QB"],
            "team": ["LAR"],
            "qb_passer_rating": [105.0],
        }
    )

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("team").item() == "LAR"
    assert profiles.select("qopp_qb_passer_rating").item() == 88.0
    assert details["QB_LAR"] == [{"opponent": "SEA", "division": True, "games_included": 1}]


def test_compute_qb_opponent_profiles_derives_allowed_efficiency_rates() -> None:
    """Verify opponent QB context includes attempt-normalized allowed production rates."""
    # Arrange
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "LV", "MIA", "KC", "KC"],
            "opponent_team": ["KC", "KC", "KC", "LV", "MIA"],
            "week": [1, 2, 3, 2, 3],
            "points_allowed": [20, 24, 21, 24, 21],
            "def_sacks": [2, 3, 1, 3, 1],
            "def_interceptions": [1, 1, 0, 1, 0],
            "def_pass_defended": [4, 5, 3, 5, 3],
            "def_tackles_for_loss": [5, 6, 4, 6, 4],
            "def_qb_hits": [6, 7, 5, 7, 5],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "LV", "MIA"],
            "week": [1, 2, 3],
            "qb_id": ["QB_DEN", "QB_LV", "QB_MIA"],
            "qb_name": ["Denver QB", "Raiders QB", "Miami QB"],
            "qb_attempts": [25, 40, 20],
            "qb_dropbacks": [28, 45, 22],
            "qb_pass_yards": [200.0, 200.0, 160.0],
            "qb_pass_touchdowns": [1.0, 1.0, 2.0],
            "qb_interceptions": [0.0, 1.0, 0.0],
            "qb_sacks": [1.0, 2.0, 1.0],
            "qb_sack_yards_lost": [8.0, 12.0, 8.0],
            "qb_passing_epa": [3.0, 10.0, 4.4],
            "qb_spike_epa": [0.0, 0.0, 0.0],
            "qb_epa_per_dropback": [3.0 / 28.0, 10.0 / 45.0, 4.4 / 22.0],
            "qb_pass_yards_per_dropback": [200.0 / 28.0, 200.0 / 45.0, 160.0 / 22.0],
            "qb_td_int_margin_rate": [1.0 / 28.0, 0.0, 2.0 / 22.0],
            "qb_sack_rate": [1.0 / 28.0, 2.0 / 45.0, 1.0 / 22.0],
            "qb_any_a": [
                (200.0 + 20.0 - 0.0 - 8.0) / 26.0,
                (200.0 + 20.0 - 45.0 - 12.0) / 42.0,
                (160.0 + 40.0 - 0.0 - 8.0) / 21.0,
            ],
        }
    )
    qb_season_df = pl.DataFrame({"qb_id": ["QB_DEN"], "qb_name": ["Denver QB"], "team": ["DEN"]})

    # Act
    profiles, _ = qb_opponent_stats.compute_qb_opponent_profiles(
        weekly_df,
        qb_df,
        qb_season_df,
    )

    # Assert
    assert profiles is not None
    assert profiles.select("qopp_qb_yards_per_attempt").item() == 6.0
    assert profiles.select("qopp_qb_touchdown_rate").item() == 0.05
    assert profiles.select("qopp_qb_interception_rate").item() == 1 / 60
    assert profiles.select("qopp_qb_epa_per_dropback").item() == pytest.approx(14.4 / 67.0)
    assert profiles.select("qopp_qb_any_a").item() == pytest.approx(355.0 / 63.0)
    assert profiles.select("qopp_qb_sack_rate").item() == pytest.approx(3.0 / 67.0)
    assert profiles.select("qopp_qb_td_int_margin_rate").item() == pytest.approx(2.0 / 67.0)


def _weekly(rows: list[tuple[str, str, int]]) -> pl.DataFrame:
    """Return team-game rows from (team, opponent, week) with one defensive stat."""
    return pl.DataFrame(
        {
            "team": [team for team, _, _ in rows],
            "opponent_team": [opponent for _, opponent, _ in rows],
            "week": [week for _, _, week in rows],
            "points_allowed": [20] * len(rows),
        }
    )


# KC's defense, profiled without its games against DEN.
_KC_WITHOUT_DEN = pl.DataFrame({"team": ["KC"], "excluded_opponent": ["DEN"]})


def test_allowed_stats_for_a_defense_without_other_games_is_empty() -> None:
    # Arrange
    weekly = _weekly([("DEN", "KC", 1), ("KC", "DEN", 1)])
    qb_df = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_attempts": [30]})

    # Act
    allowed = qb_opponent_stats._qb_allowed_stats_by_pair(weekly, qb_df, _KC_WITHOUT_DEN)

    # Assert
    assert allowed.is_empty()


def test_allowed_stats_without_matching_qb_rows_is_empty() -> None:
    # Arrange
    weekly = _weekly([("BUF", "KC", 2), ("KC", "BUF", 2)])
    qb_df = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_attempts": [30]})

    # Act
    allowed = qb_opponent_stats._qb_allowed_stats_by_pair(weekly, qb_df, _KC_WITHOUT_DEN)

    # Assert
    assert allowed.is_empty()


def test_allowed_stats_without_numeric_qb_columns_is_empty() -> None:
    # Arrange
    weekly = _weekly([("BUF", "KC", 2), ("KC", "BUF", 2)])
    qb_df = pl.DataFrame({"team_abbr": ["BUF"], "week": [2], "qb_name": ["Bills QB"]})

    # Act
    allowed = qb_opponent_stats._qb_allowed_stats_by_pair(weekly, qb_df, _KC_WITHOUT_DEN)

    # Assert
    assert allowed.is_empty()


@pytest.mark.parametrize(
    "qb_df",
    [
        pl.DataFrame({"qb_name": ["A"], "qb_attempts": [10]}),
        pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_name": ["A"]}),
    ],
)
def test_select_primary_qb_games_returns_rows_it_cannot_rank(qb_df: pl.DataFrame) -> None:
    # Act
    selected = qb_opponent_stats._select_primary_qb_games(qb_df)

    # Assert
    assert selected.equals(qb_df)


def test_details_key_without_identity_values_uses_the_team_label() -> None:
    # Act
    key = qb_opponent_stats._details_key({"qb_id": None}, ["qb_id"], "DEN")

    # Assert
    assert key == "DEN"


def test_identity_filter_without_identity_values_matches_nothing() -> None:
    # Arrange
    rows = pl.DataFrame({"qb_id": ["a", "b"]})

    # Act
    matched = rows.filter(qb_opponent_stats._qb_identity_filter({"qb_id": None}, ["qb_id"]))

    # Assert
    assert matched.is_empty()


def test_compute_qb_opponent_profiles_without_qb_identity_raises_value_error() -> None:
    # Arrange
    weekly = _weekly([("DEN", "KC", 1), ("KC", "DEN", 1)])
    qb_df = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_attempts": [30]})
    qb_season_df = pl.DataFrame({"team": ["DEN"]})

    # Act & Assert
    with pytest.raises(ValueError, match="qb_id or qb_name"):
        qb_opponent_stats.compute_qb_opponent_profiles(weekly, qb_df, qb_season_df)


def test_compute_qb_opponent_profiles_skips_opponents_without_other_games() -> None:
    # Arrange
    weekly = _weekly([("DEN", "KC", 1), ("KC", "DEN", 1)])
    qb_df = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_id": ["q1"], "qb_attempts": [30]})
    qb_season_df = pl.DataFrame({"qb_id": ["q1"], "team": ["DEN"]})

    # Act
    profiles, details = qb_opponent_stats.compute_qb_opponent_profiles(weekly, qb_df, qb_season_df)

    # Assert
    assert profiles is None
    assert details["q1"] == [{"opponent": "KC", "division": True, "games_included": 0}]


def test_compute_qb_opponent_profiles_without_context_columns_returns_none() -> None:
    # Arrange
    weekly = pl.DataFrame(
        {
            "team": ["DEN", "KC", "KC", "BUF"],
            "opponent_team": ["KC", "DEN", "BUF", "KC"],
            "week": [1, 1, 2, 2],
        }
    )
    qb_df = pl.DataFrame({"team_abbr": ["DEN"], "week": [1], "qb_id": ["q1"]})
    qb_season_df = pl.DataFrame({"qb_id": ["q1"], "team": ["DEN"]})

    # Act
    profiles, _ = qb_opponent_stats.compute_qb_opponent_profiles(weekly, qb_df, qb_season_df)

    # Assert
    assert profiles is None


def test_allowed_rate_exprs_skip_numerators_that_are_absent() -> None:
    # Arrange
    allowed = pl.DataFrame({"qb_attempts": [20, 30], "qb_pass_yards": [150.0, 250.0]})

    # Act
    exprs = qb_opponent_stats._qb_allowed_rate_exprs(set(allowed.columns))

    # Assert
    assert allowed.select(exprs).columns == ["qopp_qb_yards_per_attempt"]


@pytest.mark.parametrize("reverse", [False, True])
def test_select_primary_qb_games_breaks_a_tie_the_same_way_in_any_row_order(
    *, reverse: bool
) -> None:
    # Arrange
    tied = pl.DataFrame(
        {
            "team_abbr": ["ATL", "ATL"],
            "week": [6, 6],
            "qb_id": ["QB_B", "QB_A"],
            "qb_offense_snaps": [0, 0],
            "qb_dropbacks": [20, 20],
            "qb_attempts": [18, 18],
        }
    )

    # Act
    selected = qb_opponent_stats._select_primary_qb_games(tied.reverse() if reverse else tied)

    # Assert
    assert selected.get_column("qb_id").to_list() == ["QB_A"]


def test_compute_qb_opponent_profiles_pools_every_rate_over_the_defenses_games() -> None:
    """Verify each faced defense's rates are pooled over its other games, not averaged per game.

    KC (DEN's opponent) faced LV's passer (5 of 10, 2 of 2 carries for 10 yards, CPOE +6.0 over 9
    plays) and MIA's (24 of 40, 3 of 6 carries for 9 yards, CPOE -1.5 over 36 plays).
    """
    # Arrange
    cpoe = CPOE_COLUMN
    weekly_df = pl.DataFrame(
        {
            "team": ["DEN", "LV", "MIA", "KC", "KC"],
            "opponent_team": ["KC", "KC", "KC", "LV", "MIA"],
            "week": [1, 2, 3, 2, 3],
            "points_allowed": [20, 24, 21, 24, 21],
        }
    )
    qb_df = pl.DataFrame(
        {
            "team_abbr": ["DEN", "LV", "MIA"],
            "week": [1, 2, 3],
            "qb_id": ["QB_DEN", "QB_LV", "QB_MIA"],
            "qb_name": ["Denver QB", "Raiders QB", "Miami QB"],
            "qb_attempts": [25, 10, 40],
            "qb_completions": [15, 5, 24],
            "qb_pass_yards": [200.0, 60.0, 340.0],
            "qb_pass_touchdowns": [1, 0, 3],
            "qb_interceptions": [0, 1, 1],
            "qb_completion_pct": [0.6, 0.5, 0.6],
            "qb_passer_rating": [95.0, 29.6, 99.4],
            "qb_carries": [3, 2, 6],
            "qb_rushing_yards": [12.0, 10.0, 9.0],
            "qb_yards_per_carry": [4.0, 5.0, 1.5],
            cpoe: [1.0, 6.0, -1.5],
            numerator_column(cpoe): [10.0, 54.0, -54.0],
            denominator_column(cpoe): [10, 9, 36],
        }
    )
    qb_season_df = pl.DataFrame({"qb_id": ["QB_DEN"], "qb_name": ["Denver QB"], "team": ["DEN"]})

    # Act
    profiles, _ = qb_opponent_stats.compute_qb_opponent_profiles(weekly_df, qb_df, qb_season_df)

    # Assert
    assert profiles is not None
    row = profiles.row(0, named=True)
    assert row["qopp_qb_completion_pct"] == pytest.approx(29 / 50)
    assert row["qopp_qb_yards_per_carry"] == pytest.approx(19 / 8)
    assert row[f"qopp_{cpoe}"] == pytest.approx(0.0)
    # Passer rating of 29 of 50, 400 yards, 3 TD, 2 INT.
    completion = min(max((29 / 50 - 0.3) * 5, 0.0), 2.375)
    yards = min(max((400 / 50 - 3.0) * 0.25, 0.0), 2.375)
    touchdowns = min(max(3 / 50 * 20, 0.0), 2.375)
    interceptions = min(max(2.375 - 2 / 50 * 25, 0.0), 2.375)
    rating = round((completion + yards + touchdowns + interceptions) / 6 * 100, 1)
    assert row["qopp_qb_passer_rating"] == pytest.approx(rating)
    assert not [column for column in row if "_num_" in column or "_den_" in column]
