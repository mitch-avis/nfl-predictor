"""Tests for Polars-based utility transforms."""

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import teamrankings


def test_combine_stats_creates_combined_and_drops() -> None:
    """Derived columns are computed and source columns are dropped."""
    df = pl.DataFrame(
        {
            "sack_fumbles": [1],
            "rushing_fumbles": [2],
            "receiving_fumbles": [3],
            "sack_fumbles_lost": [1],
            "rushing_fumbles_lost": [1],
            "receiving_fumbles_lost": [0],
            "passing_first_downs": [5],
            "rushing_first_downs": [4],
            "passing_2pt_conversions": [1],
            "rushing_2pt_conversions": [0],
            "receiving_2pt_conversions": [1],
            "fumble_recovery_own": [1],
            "fumble_recovery_opp": [2],
            "def_interceptions": [1],
            "interceptions_thrown": [2],
            "pass_yards": [250],
            "rush_yards": [100],
            "yards_lost_from_sacks": [10],
        }
    )

    combined = polars_utils.combine_stats(df)

    assert combined.select("fumbles").item() == 6
    assert combined.select("fumbles_lost").item() == 2
    assert combined.select("first_downs").item() == 9
    assert combined.select("2pt_conversions").item() == 2
    assert combined.select("fumble_recoveries").item() == 3
    assert combined.select("turnover_margin").item() == -1
    assert combined.select("total_yards").item() == 340

    for col in (
        "sack_fumbles",
        "rushing_fumbles",
        "receiving_fumbles",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "passing_first_downs",
        "rushing_first_downs",
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fumble_recovery_own",
        "fumble_recovery_opp",
    ):
        assert col not in combined.columns


def test_add_scoring_data_to_team_stats() -> None:
    """Schedule scores are joined into per-team stats."""
    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [24],
            "home_score": [17],
        }
    )

    scored = polars_utils.add_scoring_data_to_team_stats(team_stats, schedule)

    away_row = scored.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    home_row = scored.filter(pl.col("team_abbr") == "KC").row(0, named=True)

    assert away_row["points_scored"] == 24
    assert away_row["points_allowed"] == 17
    assert away_row["scoring_margin"] == 7
    assert home_row["points_scored"] == 17
    assert home_row["points_allowed"] == 24
    assert home_row["scoring_margin"] == -7


def test_add_per_game_opponent_stats() -> None:
    """Opponent stats are attached per game based on opponent_abbr."""
    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [275, 310],
        }
    )

    result = polars_utils.add_per_game_opponent_stats(team_stats)

    away_row = result.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    home_row = result.filter(pl.col("team_abbr") == "KC").row(0, named=True)

    assert away_row["opponent_pass_yards"] == 310
    assert home_row["opponent_pass_yards"] == 275


def test_aggregate_team_stats_to_week_regular() -> None:
    """Rolling averages aggregate correctly up to a target week."""
    team_stats = pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "team_abbr": ["BUF", "BUF", "KC", "KC"],
            "opponent_abbr": ["KC", "KC", "BUF", "BUF"],
            "pass_yards": [200, 300, 250, 350],
            "points_scored": [20, 30, 24, 28],
            "points_allowed": [17, 21, 20, 24],
        }
    )

    agg = polars_utils.aggregate_team_stats_to_week(team_stats, target_week=3, season=2024)

    buf_row = agg.filter(pl.col("team_abbr") == "BUF").row(0, named=True)
    assert buf_row["games_played"] == 2
    assert buf_row["pass_yards"] == pytest.approx(250.0)
    assert buf_row["points_scored"] == pytest.approx(25.0)


def test_calculate_stat_differentials_skips_non_numeric() -> None:
    """Stat differentials are only computed for numeric columns."""
    df = pl.DataFrame(
        {
            "away_elo_pre": [1500.0],
            "home_elo_pre": [1400.0],
            "away_qb": ["QB1"],
            "home_qb": ["QB2"],
        }
    )

    result = polars_utils.calculate_stat_differentials(df, ["elo_pre", "qb"])

    assert "elo_pre_diff" in result.columns
    assert result.select("elo_pre_diff").item() == pytest.approx(100.0)
    assert "qb_diff" not in result.columns


def test_build_final_column_order_metadata_first() -> None:
    """Final column ordering starts with metadata columns."""
    final_order = polars_utils.build_final_column_order()
    assert final_order[: len(constants.METADATA_COLUMNS)] == constants.METADATA_COLUMNS


def test_build_final_column_order_has_no_duplicates_and_includes_lines_results() -> None:
    """Final column ordering has no duplicates and includes lines/result columns."""
    final_order = polars_utils.build_final_column_order()
    assert len(final_order) == len(set(final_order))

    for col in (*constants.LINES_COLUMNS, *constants.RESULT_COLUMNS):
        assert col in final_order


def test_get_pbp_columns_returns_a_copy_of_the_published_contract() -> None:
    """Play-by-play stat accessor returns a defensive copy of the constant."""
    cols = polars_utils.get_pbp_columns()

    assert cols == constants.PBP_STATS
    cols.append("mutated")
    assert "mutated" not in constants.PBP_STATS


def test_build_final_column_order_publishes_pbp_stats_without_opponent_mirrors() -> None:
    """Every play-by-play stat is published per team and as a diff, with no generic mirror."""
    final_order = polars_utils.build_final_column_order()
    ordered = set(final_order)

    for stat in constants.PBP_STATS:
        assert f"away_{stat}" in ordered
        assert f"home_{stat}" in ordered
        assert f"{stat}_diff" in ordered
        # The allowed columns are explicit, so the generic opponent mirror must not exist.
        assert f"away_opponent_{stat}" not in ordered
        assert f"home_opponent_{stat}" not in ordered
        assert f"opponent_{stat}_diff" not in ordered

    assert len(final_order) == len(set(final_order))


def test_get_stats_for_diff_includes_pbp_stats() -> None:
    """Differentials are calculated for the published play-by-play stats."""
    stats = polars_utils.get_stats_for_diff()

    for stat in constants.PBP_STATS:
        assert stat in stats
    assert len(stats) == len(set(stats))


def test_pbp_derived_rates_are_ratios_of_sums() -> None:
    """Play-by-play rates divide season-to-date mean counts, which equals a ratio of sums.

    Aggregation stores the mean per-game count, so mean(numerator) / mean(denominator)
    equals sum(numerator) / sum(denominator) because the game count cancels.
    """
    # Two games: 60 and 40 offensive snaps, 6.0 and 2.0 pass EPA.
    # Ratio of sums = 8.0 / 100 = 0.08; the aggregated means are 50 snaps and 4.0 EPA.
    agg = pl.DataFrame(
        {
            "team_abbr": ["AAA"],
            "offensive_snaps": [50.0],
            "defensive_snaps": [60.0],
            "dropbacks": [20.0],
            "carries": [25.0],
            "dropbacks_allowed": [30.0],
            "carries_allowed": [20.0],
            "pass_epa_sum": [4.0],
            "rush_epa_sum": [-2.0],
            "pass_epa_allowed_sum": [3.0],
            "rush_epa_allowed_sum": [-1.5],
            "pass_success_count": [9.0],
            "rush_success_count": [10.0],
            "pass_success_allowed_count": [12.0],
            "rush_success_allowed_count": [8.0],
            "explosive_pass_count": [2.0],
            "explosive_rush_count": [1.0],
            "explosive_pass_allowed_count": [3.0],
            "explosive_rush_allowed_count": [2.0],
            "stuffed_rush_count": [5.0],
            "stuffed_rush_allowed_count": [4.0],
            "early_down_plays": [30.0],
            "early_down_passes": [15.0],
            "st_epa_for": [1.5],
            "st_epa_against": [0.5],
            "st_plays": [10.0],
        }
    )

    out = teamrankings._compute_derived_metrics(agg)
    row = out.row(0, named=True)

    assert row["off_pass_epa_per_snap"] == pytest.approx(4.0 / 50.0)
    assert row["off_rush_epa_per_snap"] == pytest.approx(-2.0 / 50.0)
    assert row["def_pass_epa_allowed_per_snap"] == pytest.approx(3.0 / 60.0)
    assert row["def_rush_epa_allowed_per_snap"] == pytest.approx(-1.5 / 60.0)
    assert row["epa_per_dropback"] == pytest.approx(4.0 / 20.0)
    assert row["epa_per_carry"] == pytest.approx(-2.0 / 25.0)
    assert row["epa_per_dropback_allowed"] == pytest.approx(3.0 / 30.0)
    assert row["epa_per_carry_allowed"] == pytest.approx(-1.5 / 20.0)
    assert row["pass_success_rate"] == pytest.approx(9.0 / 20.0)
    assert row["rush_success_rate"] == pytest.approx(10.0 / 25.0)
    assert row["pass_success_rate_allowed"] == pytest.approx(12.0 / 30.0)
    assert row["rush_success_rate_allowed"] == pytest.approx(8.0 / 20.0)
    assert row["success_rate"] == pytest.approx(19.0 / 45.0)
    assert row["success_rate_allowed"] == pytest.approx(20.0 / 50.0)
    assert row["explosive_pass_rate"] == pytest.approx(2.0 / 20.0)
    assert row["explosive_rush_rate"] == pytest.approx(1.0 / 25.0)
    assert row["explosive_pass_rate_allowed"] == pytest.approx(3.0 / 30.0)
    assert row["explosive_rush_rate_allowed"] == pytest.approx(2.0 / 20.0)
    assert row["stuffed_rush_rate"] == pytest.approx(5.0 / 25.0)
    assert row["stuffed_rush_rate_allowed"] == pytest.approx(4.0 / 20.0)
    assert row["early_down_pass_rate"] == pytest.approx(15.0 / 30.0)
    assert row["epa_margin_per_play"] == pytest.approx((4.0 - 2.0) / 50.0 - (3.0 - 1.5) / 60.0)
    assert row["st_epa_margin_per_play"] == pytest.approx((1.5 - 0.5) / 10.0)


def test_pbp_derived_rates_are_null_on_zero_denominators() -> None:
    """A zero denominator yields a null rate rather than a divide-by-zero or a fake zero."""
    agg = pl.DataFrame(
        {
            "team_abbr": ["AAA"],
            "offensive_snaps": [0.0],
            "defensive_snaps": [0.0],
            "dropbacks": [0.0],
            "carries": [0.0],
            "dropbacks_allowed": [0.0],
            "carries_allowed": [0.0],
            "pass_epa_sum": [0.0],
            "rush_epa_sum": [0.0],
            "pass_epa_allowed_sum": [0.0],
            "rush_epa_allowed_sum": [0.0],
            "pass_success_count": [0.0],
            "rush_success_count": [0.0],
            "pass_success_allowed_count": [0.0],
            "rush_success_allowed_count": [0.0],
            "explosive_pass_count": [0.0],
            "explosive_rush_count": [0.0],
            "explosive_pass_allowed_count": [0.0],
            "explosive_rush_allowed_count": [0.0],
            "stuffed_rush_count": [0.0],
            "stuffed_rush_allowed_count": [0.0],
            "early_down_plays": [0.0],
            "early_down_passes": [0.0],
            "st_epa_for": [0.0],
            "st_epa_against": [0.0],
            "st_plays": [0.0],
        }
    )

    out = teamrankings._compute_derived_metrics(agg)
    row = out.row(0, named=True)

    for stat in constants.PBP_STATS:
        if stat in ("offensive_snaps", "defensive_snaps"):
            continue
        assert row[stat] is None, f"{stat} should be null when its denominator is zero"


def test_pbp_derived_rates_are_null_when_source_counts_are_missing() -> None:
    """A season without play-by-play still emits every rate column, as nulls."""
    agg = pl.DataFrame({"team_abbr": ["AAA"], "points_scored": [21.0]})

    out = teamrankings._compute_derived_metrics(agg)
    row = out.row(0, named=True)

    for stat in constants.PBP_STATS:
        assert stat in out.columns, f"{stat} missing from the invariant schema"
        assert row[stat] is None
