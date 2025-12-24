import polars as pl
import pytest

from nfl_predictor.data_collection_polars import _merge_team_rankings, process_week


def test_process_week_uses_fallback_stats_for_week1() -> None:
    schedule_df = pl.DataFrame(
        {
            "season": [2007],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    team_stats_df = pl.DataFrame(
        {
            "season": [2006, 2006],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [300, 200],
            "points_scored": [24, 17],
            "points_allowed": [17, 24],
        }
    )

    result = process_week(
        season=2007,
        week=1,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["away_pass_yards"] == pytest.approx(283.333, rel=1e-3)
    assert row["home_pass_yards"] == pytest.approx(216.667, rel=1e-3)
    assert row["pass_yards_diff"] == pytest.approx(66.666, rel=1e-3)


def test_merge_team_rankings_week1_uses_prev() -> None:
    merged = pl.DataFrame({"away_abbr": ["BUF"], "home_abbr": ["KC"]})
    prev_tr_df = pl.DataFrame(
        {
            "team_abbr": ["BUF", "KC"],
            "week": [18, 18],
            "predictive_rating": [5.0, 4.0],
        }
    )

    result = _merge_team_rankings(
        merged=merged,
        season=2007,
        week=1,
        tr_df=None,
        prev_tr_df=prev_tr_df,
    )

    row = result.row(0, named=True)
    assert row["away_predictive_rating"] == pytest.approx(5.0)
    assert row["home_predictive_rating"] == pytest.approx(4.0)
