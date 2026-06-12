"""Tests for TeamRankings helper utilities."""

from __future__ import annotations

from datetime import date

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import teamrankings


def test_get_team_name_resolves_canonical_and_alias_inputs() -> None:
    """Team names resolve for both canonical and alias abbreviations."""
    assert teamrankings.get_team_name("KC") == constants.TEAM_MAPPING["KC"]["name"]
    assert teamrankings.get_team_name("KAN") == constants.TEAM_MAPPING["KC"]["name"]
    assert teamrankings.get_team_name("NOT_A_TEAM") is None


def test_required_and_validate_tr_columns() -> None:
    """Required TR columns are identified and validation works."""
    required = teamrankings._get_required_tr_columns()
    assert "team_abbr" in required
    assert "week" in required

    empty_valid, missing = teamrankings._validate_tr_dataframe(pl.DataFrame())
    assert not empty_valid
    assert "team_abbr" in missing

    payload: dict[str, list[object]] = {}
    for col in required:
        if col == "team_abbr":
            payload[col] = ["AAA"]
        elif col == "week":
            payload[col] = [1]
        else:
            payload[col] = [0]
    valid, missing = teamrankings._validate_tr_dataframe(pl.DataFrame(payload))
    assert valid
    assert not missing


def test_normalize_tr_dataframe(monkeypatch) -> None:
    """TR dataframe normalization works as expected."""
    monkeypatch.setattr(teamrankings, "normalize_team_column", lambda df, _col: df)

    df = pl.DataFrame({"": [1], "abbr": ["AAA"]})
    normalized = teamrankings._normalize_tr_dataframe(df)
    assert "" not in normalized.columns
    assert "team_abbr" in normalized.columns

    df = pl.DataFrame({"abbr": ["AAA"], "team_abbr": ["BBB"]})
    normalized = teamrankings._normalize_tr_dataframe(df)
    assert "abbr" not in normalized.columns


def test_get_latest_team_rankings() -> None:
    """Latest week per team is selected correctly."""
    df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "AAA", "BBB"],
            "week": [1, 2, 2],
            "predictive_rating": [1.0, 3.0, 2.0],
        }
    )

    latest = teamrankings.get_latest_team_rankings(df)

    assert latest.filter(pl.col("team_abbr") == "AAA")["predictive_rating"][0] == 3.0


def test_compute_derived_metrics() -> None:
    """Derived metrics are computed correctly."""
    df = pl.DataFrame(
        {
            "team_abbr": ["AAA"],
            "total_yards": [300],
            "points_scored": [30],
            "opponent_total_yards": [250],
            "points_allowed": [20],
            "pass_attempts": [30],
            "rush_attempts": [20],
            "times_sacked": [2],
            "opponent_total_plays": [50],
            "penalty_yards": [40],
            "penalties": [5],
            "opponent_penalty_yards": [30],
            "opponent_penalties": [3],
        }
    )

    out = teamrankings._compute_derived_metrics(df)

    assert "yards_per_point" in out.columns
    assert "opponent_yards_per_point" in out.columns
    assert "points_per_play" in out.columns
    assert "penalty_yards_per_penalty" in out.columns


def test_aggregate_team_stats_to_week() -> None:
    """Rolling averages aggregate correctly up to a target week."""
    stats = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "team_abbr": ["AAA", "AAA"],
            "pass_yards": [200, 300],
            "rush_yards": [100, 150],
        }
    )

    out = teamrankings.aggregate_team_stats_to_week(stats, target_week=3, season=2023)

    assert out.height == 1
    assert "games_played" in out.columns


def test_calculate_league_means_and_regress() -> None:
    """League means are calculated and regression to mean works."""
    stats = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "points_scored": [20, 30],
        }
    )

    means = teamrankings.calculate_league_means(stats, season=2023)
    assert means["points_scored"] == 25.0

    regressed = teamrankings.regress_to_mean(
        pl.DataFrame({"team_abbr": ["AAA"], "points_scored": [20.0], "games_played": [1]}),
        means,
        regression_factor=0.5,
    )
    assert regressed["points_scored"][0] == 22.5


def test_merge_schedule_and_stat_differentials() -> None:
    """Schedule is merged with team stats and differentials are calculated."""
    schedule = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    agg = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "points_scored": [20, 10],
        }
    )

    merged = teamrankings.merge_schedule_with_team_stats(schedule, agg)
    assert "away_points_scored" in merged.columns
    assert "home_points_scored" in merged.columns

    with_diff = teamrankings.calculate_stat_differentials(merged, ["points_scored"])
    assert "points_scored_diff" in with_diff.columns


def test_get_tr_columns() -> None:
    """TR columns include expected ratings and stats."""
    cols = teamrankings.get_tr_columns()
    assert constants.TR_RATINGS[0] in cols
    assert constants.TR_STATS[0] in cols


def test_load_team_rankings_partial_full_and_future(tmp_path, monkeypatch) -> None:
    """TeamRankings are loaded correctly for past and future weeks."""
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        teamrankings,
        "_get_required_tr_columns",
        lambda: {"team_abbr", "week", "predictive_rating"},
    )

    week1_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [1, 1],
            "predictive_rating": [1.0, 2.0],
        }
    )
    week1_df.write_csv(season_dir / f"{season}_week_01_team_rankings.csv")

    # Seed cached current week and a future week with stale values.
    week2_cached = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [2, 2],
            "predictive_rating": [0.5, 0.6],
        }
    )
    week2_cached.write_csv(season_dir / f"{season}_week_02_team_rankings.csv")

    week3_cached = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [3, 3],
            "predictive_rating": [9.9, 9.8],
        }
    )
    week3_cached.write_csv(season_dir / f"{season}_week_03_team_rankings.csv")

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(constants, "get_regular_season_weeks", lambda _season: 3)
    monkeypatch.setattr(teamrankings, "get_week_date", lambda _season, _week: date(2023, 9, 1))

    scraped_weeks: list[int] = []

    def fake_scrape(week, _week_date, **_kwargs):
        """Fake scrape function returning dummy data."""
        scraped_weeks.append(int(week))
        return pl.DataFrame(
            {
                "team_abbr": ["AAA", "BBB"],
                "week": [week, week],
                "predictive_rating": [1.1, 2.2],
            }
        )

    monkeypatch.setattr(teamrankings, "scrape_team_rankings_for_week", fake_scrape)

    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
    )

    # Only the current week should be scraped; future weeks should be filled from it.
    assert scraped_weeks == [2]

    weeks = sorted(combined["week"].unique().to_list())
    assert 1 in weeks
    assert 2 in weeks
    assert 3 in weeks

    # Current week should reflect the fresh scrape (not the seeded cached values).
    w2 = combined.filter(pl.col("week") == 2).sort("team_abbr")
    assert w2["predictive_rating"].to_list() == [1.1, 2.2]

    # Future week should be overwritten to match current week's freshly scraped values.
    w3 = combined.filter(pl.col("week") == 3).sort("team_abbr")
    assert w3["predictive_rating"].to_list() == [1.1, 2.2]

    # And the cached file should be overwritten on disk as well.
    week3_path = season_dir / f"{season}_week_03_team_rankings.csv"
    reloaded_week3 = pl.read_csv(week3_path).sort("team_abbr")
    assert reloaded_week3["predictive_rating"].to_list() == [1.1, 2.2]


def test_load_team_rankings_min_week_skips_early_week(tmp_path, monkeypatch) -> None:
    """Min week setting skips known-missing early weeks."""
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(constants, "get_regular_season_weeks", lambda _season: 3)
    monkeypatch.setattr(teamrankings, "get_week_date", lambda _season, _week: date(2023, 9, 1))
    monkeypatch.setattr(teamrankings, "update_season_team_rankings", lambda _season: None)

    scraped_weeks: list[int] = []

    def fake_scrape(week, _week_date, **_kwargs):
        """Fake scrape function returning dummy data."""
        scraped_weeks.append(int(week))
        return pl.DataFrame(
            {
                "team_abbr": ["AAA", "BBB"],
                "week": [week, week],
                "predictive_rating": [1.1, 2.2],
            }
        )

    monkeypatch.setattr(teamrankings, "scrape_team_rankings_for_week", fake_scrape)

    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
        min_week=2,
    )

    weeks = sorted(combined["week"].unique().to_list())
    assert 1 not in weeks
    assert 2 in weeks
    assert all(week >= 2 for week in scraped_weeks)


def test_load_team_rankings_skips_pre_min_season(tmp_path, monkeypatch) -> None:
    """Seasons before TR availability return empty data."""
    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))

    season = constants.TEAMRANKINGS_MIN_SEASON - 1
    combined = teamrankings.load_team_rankings(
        season,
        current_season=constants.TEAMRANKINGS_MIN_SEASON,
        current_week=2,
    )

    assert combined.height == 0
