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


def test_calculate_league_means_and_regress_handle_empty_cases() -> None:
    """Empty seasons and empty regression means leave data unchanged."""
    stats = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team_abbr": ["AAA"],
            "points_scored": [20],
        }
    )

    assert teamrankings.calculate_league_means(stats, season=2024) == {}
    assert teamrankings.regress_to_mean(stats, {}).to_dicts() == stats.to_dicts()


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


def test_calculate_stat_differentials_skips_non_numeric_pairs() -> None:
    """Differentials are only added for numeric away/home stat pairs."""
    df = pl.DataFrame(
        {
            "away_status": ["healthy"],
            "home_status": ["healthy"],
            "away_points": [20],
            "home_points": [17],
        }
    )

    out = teamrankings.calculate_stat_differentials(df, ["status", "points"])

    assert "status_diff" not in out.columns
    assert out["points_diff"][0] == 3


def test_get_tr_columns() -> None:
    """TR columns include expected ratings and stats."""
    cols = teamrankings.get_tr_columns()
    assert constants.TR_RATINGS[0] in cols
    assert constants.TR_STATS[0] in cols


def test_helper_accessors_and_dataframe_converters(monkeypatch) -> None:
    """Column accessors and simple dataframe conversion helpers behave as expected."""
    assert teamrankings.get_stat_columns() == constants.NFLREADPY_STATS
    assert teamrankings.get_elo_columns() == constants.ELO_COLUMNS

    with_diffs = pl.DataFrame({"team_abbr": ["AAA"], "rating_diff": [1.0], "rating": [2.0]})
    without_diffs = teamrankings.remove_diff_columns(with_diffs)
    assert without_diffs.columns == ["team_abbr", "rating"]

    monkeypatch.setattr(
        pl.DataFrame,
        "to_pandas",
        lambda self: {"team_abbr": self["team_abbr"].to_list(), "rating": self["rating"].to_list()},
    )
    monkeypatch.setattr(teamrankings.pl, "from_pandas", lambda df: pl.DataFrame(df))

    pandas_df = teamrankings.polars_to_pandas(without_diffs)
    round_trip = teamrankings.pandas_to_polars(pandas_df)
    assert round_trip.to_dicts() == without_diffs.to_dicts()


def test_calculate_game_result_and_schedule_filters() -> None:
    """Game-result helpers and schedule filters classify completed and upcoming rows."""
    assert teamrankings.calculate_game_result({"away_score": 21, "home_score": 17}) == 1.0
    assert teamrankings.calculate_game_result({"away_score": 17, "home_score": 21}) == 0.0
    assert teamrankings.calculate_game_result({"away_score": 20, "home_score": 20}) == 0.5
    assert teamrankings.calculate_game_result({"away_score": None, "home_score": 20}) is None

    schedule = pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [2, 3, 3],
            "away_score": [17, None, 10],
            "home_score": [20, None, None],
        }
    )

    completed = teamrankings.filter_completed_games(schedule)
    upcoming = teamrankings.filter_upcoming_games(schedule, season=2023, week=3)

    assert completed.height == 1
    assert upcoming.height == 2


def test_aggregate_team_stats_to_week_handles_week_one_and_playoff_paths(monkeypatch) -> None:
    """Week-one and playoff aggregation paths use the intended prior-game window."""
    monkeypatch.setattr(constants, "get_regular_season_weeks", lambda _season: 18)

    stats = pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 18, 19],
            "team_abbr": ["AAA", "AAA", "AAA"],
            "pass_yards": [200, 300, 500],
            "rush_yards": [100, 150, 250],
        }
    )

    assert teamrankings.aggregate_team_stats_to_week(stats, target_week=1, season=2023).height == 0

    playoff = teamrankings.aggregate_team_stats_to_week(stats, target_week=19, season=2023)
    row = playoff.row(0, named=True)
    assert row["games_played"] == 2
    assert row["pass_yards"] == 250.0


def test_get_stats_for_diff_deduplicates_and_includes_expected_groups() -> None:
    """Diff-stat selection includes the expected source groups without duplicates."""
    stats = teamrankings.get_stats_for_diff()

    assert len(stats) == len(set(stats))
    assert constants.ELO_COLUMNS[0] in stats
    assert constants.TR_RATINGS[0] in stats
    assert constants.TREND_FEATURE_COLUMNS[0] in stats


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


def test_load_team_rankings_returns_empty_for_future_season(tmp_path, monkeypatch) -> None:
    """Future seasons short-circuit because there is no cached or scrapeable data yet."""
    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))

    combined = teamrankings.load_team_rankings(2027, current_season=2026, current_week=2)

    assert combined.height == 0


def test_load_team_rankings_uses_cached_fallback_when_refresh_fails(tmp_path, monkeypatch) -> None:
    """A current-week cache is reused when the forced refresh scrape returns nothing."""
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(constants, "get_regular_season_weeks", lambda _season: 3)
    monkeypatch.setattr(teamrankings, "get_week_date", lambda _season, _week: date(2023, 9, 1))
    monkeypatch.setattr(teamrankings, "_validate_tr_dataframe", lambda _df: (True, []))

    week2_cached = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [2, 2],
            "predictive_rating": [0.5, 0.6],
        }
    )
    week2_cached.write_csv(season_dir / f"{season}_week_02_team_rankings.csv")

    monkeypatch.setattr(
        teamrankings,
        "scrape_team_rankings_for_week",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
        min_week=2,
    )

    week2 = combined.filter(pl.col("week") == 2).sort("team_abbr")
    assert week2["predictive_rating"].to_list() == [0.5, 0.6]


def test_load_team_rankings_keeps_existing_partial_week_when_scrape_fails(
    tmp_path,
    monkeypatch,
) -> None:
    """Partial-scrape failures keep the cached week data instead of dropping it."""
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(constants, "get_regular_season_weeks", lambda _season: 1)
    monkeypatch.setattr(teamrankings, "get_week_date", lambda _season, _week: date(2023, 9, 1))
    monkeypatch.setattr(
        teamrankings,
        "_get_required_tr_columns",
        lambda: {"team_abbr", "week", "predictive_rating", "third_down_pct"},
    )
    monkeypatch.setattr(
        teamrankings,
        "get_missing_tr_columns",
        lambda _df: ({"stat": "third_down_pct"}, {}),
    )
    monkeypatch.setattr(
        teamrankings,
        "scrape_team_rankings_for_week",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    partial = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [1, 1],
            "predictive_rating": [1.0, 2.0],
        }
    )
    partial.write_csv(season_dir / f"{season}_week_01_team_rankings.csv")

    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
    )

    week1 = combined.filter(pl.col("week") == 1).sort("team_abbr")
    assert week1.to_dicts() == partial.sort("team_abbr").to_dicts()
