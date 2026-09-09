"""Tests for the Polars data collection pipeline."""

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.data_collection import _merge_team_rankings, process_week
from nfl_predictor.utils import polars_utils


def test_process_week_uses_fallback_stats_for_week1() -> None:
    """Week 1 uses prior-season fallbacks when no prior games exist."""
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
        min_season=2006,
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
    """Week 1 TeamRankings merge can fall back to prior-season week 18."""
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


def _pbp_play(
    season: int,
    week: int,
    posteam: str,
    defteam: str,
    *,
    epa: float,
    dropback: int = 0,
    rush: int = 0,
    yards: float = 0.0,
    success: int = 0,
    down: int = 1,
) -> dict[str, object]:
    """Build one synthetic scrimmage play row for play-by-play fixtures."""
    return {
        "season": season,
        "week": week,
        "season_type": "REG",
        "posteam": posteam,
        "defteam": defteam,
        "qb_dropback": dropback,
        "rush": rush,
        "qb_kneel": 0,
        "qb_spike": 0,
        "epa": epa,
        "success": success,
        "yards_gained": yards,
        "down": down,
        "play_type": "pass" if dropback else "run",
        "yardline_100": 50,
        "special": 0,
    }


def _three_week_pbp(epa_scale_after_week_1: float) -> pl.DataFrame:
    """Build three weeks of synthetic play-by-play for two teams.

    Weeks 2 and 3 are scaled by `epa_scale_after_week_1` so a perturbation of the
    later weeks can be tested against week-2 features, which must not see them.
    """
    rows: list[dict[str, object]] = []
    for week in (1, 2, 3):
        scale = 1.0 if week == 1 else epa_scale_after_week_1
        for posteam, defteam, base in (("AAA", "BBB", 0.4), ("BBB", "AAA", -0.2)):
            rows.append(
                _pbp_play(
                    2007,
                    week,
                    posteam,
                    defteam,
                    epa=base * scale,
                    dropback=1,
                    yards=25.0,
                    success=1,
                )
            )
            rows.append(
                _pbp_play(
                    2007, week, posteam, defteam, epa=-base * scale, rush=1, yards=2.0, down=2
                )
            )
    return pl.DataFrame(rows)


def _week_two_features(pbp_df: pl.DataFrame) -> dict[str, object]:
    """Run the play-by-play chain end to end and return week-2 feature values."""
    team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
    team_stats_df = pl.DataFrame(
        {
            "season": [2007] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team_abbr": ["AAA", "BBB"] * 3,
            "opponent_abbr": ["BBB", "AAA"] * 3,
            "pass_yards": [250.0, 180.0] * 3,
        }
    )
    team_stats_df = data_collection._join_pbp_team_game_stats(team_stats_df, team_games)

    schedule_df = pl.DataFrame(
        {"season": [2007], "week": [2], "away_abbr": ["AAA"], "home_abbr": ["BBB"]}
    )
    result = process_week(
        season=2007,
        week=2,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )
    assert result.height == 1
    row = result.row(0, named=True)
    return {
        col: row[col]
        for col in row
        if any(marker in col for marker in constants.FEATURE_GROUP_COLUMN_MARKERS["pbp"])
    }


def test_future_week_plays_do_not_change_earlier_week_features() -> None:
    """Perturbing week 2 and week 3 plays leaves week-2 features untouched.

    Week-N features may only use games strictly before week N of the season, so
    rewriting every play from week 2 onward must not move a single week-2 value.
    """
    baseline = _week_two_features(_three_week_pbp(1.0))
    perturbed = _week_two_features(_three_week_pbp(10.0))

    assert baseline, "expected play-by-play features on the week-2 row"
    assert baseline == perturbed


def test_week1_fallback_regresses_pbp_rates_toward_the_league_mean() -> None:
    """Week 1 falls back to the regressed prior season for the play-by-play family.

    Rates are ratios of regressed sums, so each regressed rate is a weighted mediant of
    the team's own rate and the league mean rate and must land between the two.
    """
    prior = pl.DataFrame(
        {
            "season": [2006, 2006],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [300.0, 200.0],
            "points_scored": [24.0, 17.0],
            "points_allowed": [17.0, 24.0],
            # BUF: 12.0 EPA over 30 dropbacks = 0.40. KC: 3.0 over 30 = 0.10.
            "dropbacks": [30.0, 30.0],
            "pass_epa_sum": [12.0, 3.0],
            "offensive_snaps": [60.0, 60.0],
        }
    )
    schedule_df = pl.DataFrame(
        {"season": [2007], "week": [1], "away_abbr": ["BUF"], "home_abbr": ["KC"]}
    )

    result = process_week(
        season=2007,
        week=1,
        schedule_df=schedule_df,
        team_stats_df=prior,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )

    row = result.row(0, named=True)
    league_mean_rate = 15.0 / 60.0  # (12.0 + 3.0) EPA over (30 + 30) dropbacks
    assert league_mean_rate < row["away_epa_per_dropback"] < 0.40
    assert 0.10 < row["home_epa_per_dropback"] < league_mean_rate
