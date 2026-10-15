"""Tests for TeamRankings helper utilities."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import teamrankings
from tests.fakes import returning

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("abbr", "expected"),
    [
        pytest.param("KC", constants.TEAM_MAPPING["KC"]["name"], id="canonical"),
        pytest.param("KAN", constants.TEAM_MAPPING["KC"]["name"], id="alias"),
        pytest.param("NOT_A_TEAM", None, id="unknown"),
    ],
)
def test_get_team_name_resolves_canonical_and_alias_inputs(abbr: str, expected: str | None) -> None:
    """Team names resolve for both canonical and alias abbreviations."""
    # Act
    name = teamrankings.get_team_name(abbr)

    # Assert
    assert name == expected
    if expected is None:
        assert name is None


def test_required_and_validate_tr_columns() -> None:
    """Required TR columns are identified."""
    # Act
    required = teamrankings._get_required_tr_columns()

    # Assert
    assert "team_abbr" in required
    assert "week" in required


def test_validate_tr_dataframe_rejects_an_empty_frame() -> None:
    """Validation fails on an empty frame and names the missing columns."""
    # Act
    empty_valid, missing = teamrankings._validate_tr_dataframe(pl.DataFrame())

    # Assert
    assert not empty_valid
    assert "team_abbr" in missing


def test_validate_tr_dataframe_accepts_every_required_column() -> None:
    """Validation passes once every required column is present."""
    # Arrange
    payload: dict[str, list[object]] = {}
    for col in teamrankings._get_required_tr_columns():
        if col == "team_abbr":
            payload[col] = ["AAA"]
        elif col == "week":
            payload[col] = [1]
        else:
            payload[col] = [0]

    # Act
    valid, missing = teamrankings._validate_tr_dataframe(pl.DataFrame(payload))

    # Assert
    assert valid
    assert not missing


def _patch_passthrough_team_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make team-column normalization return its frame unchanged."""

    def passthrough(df: pl.DataFrame, _col: str) -> pl.DataFrame:
        return df

    monkeypatch.setattr(teamrankings, "normalize_team_column", passthrough)


def test_normalize_tr_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """TR dataframe normalization works as expected."""
    # Arrange
    _patch_passthrough_team_normalization(monkeypatch)
    df = pl.DataFrame({"": [1], "abbr": ["AAA"]})

    # Act
    normalized = teamrankings._normalize_tr_dataframe(df)

    # Assert
    assert "" not in normalized.columns
    assert "team_abbr" in normalized.columns


def test_normalize_tr_dataframe_drops_abbr_when_team_abbr_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``abbr`` column is dropped when ``team_abbr`` is already present."""
    # Arrange
    _patch_passthrough_team_normalization(monkeypatch)
    df = pl.DataFrame({"abbr": ["AAA"], "team_abbr": ["BBB"]})

    # Act
    normalized = teamrankings._normalize_tr_dataframe(df)

    # Assert
    assert "abbr" not in normalized.columns


def test_get_latest_team_rankings() -> None:
    """Latest week per team is selected correctly."""
    # Arrange
    df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "AAA", "BBB"],
            "week": [1, 2, 2],
            "predictive_rating": [1.0, 3.0, 2.0],
        }
    )

    # Act
    latest = teamrankings.get_latest_team_rankings(df)

    # Assert
    assert latest.filter(pl.col("team_abbr") == "AAA")["predictive_rating"][0] == 3.0


def test_compute_derived_metrics() -> None:
    """Derived metrics are computed correctly."""
    # Arrange
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

    # Act
    out = teamrankings._compute_derived_metrics(df)

    # Assert
    assert "yards_per_point" in out.columns
    assert "opponent_yards_per_point" in out.columns
    assert "points_per_play" in out.columns
    assert "penalty_yards_per_penalty" in out.columns


def test_compute_derived_metrics_opponent_points_per_play_uses_mirrored_plays() -> None:
    """opponent_points_per_play = points_allowed / (opp passes + opp rushes + opp sacks taken)."""
    # Arrange
    df = pl.DataFrame(
        {
            "team_abbr": ["AAA"],
            "points_scored": [30],
            "points_allowed": [20],
            "pass_attempts": [30],
            "rush_attempts": [20],
            "times_sacked": [2],
            "opponent_pass_attempts": [25],
            "opponent_rush_attempts": [20],
            "opponent_times_sacked": [5],
        }
    )

    # Act
    out = teamrankings._compute_derived_metrics(df)

    # Assert
    assert out["opponent_points_per_play"][0] == pytest.approx(20 / 50)
    assert out["points_per_play_margin"][0] == pytest.approx(30 / 52 - 20 / 50)


def test_aggregate_team_stats_to_week() -> None:
    """Rolling averages aggregate correctly up to a target week."""
    # Arrange
    stats = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "team_abbr": ["AAA", "AAA"],
            "pass_yards": [200, 300],
            "rush_yards": [100, 150],
        }
    )

    # Act
    out = teamrankings.aggregate_team_stats_to_week(stats, target_week=3, season=2023)

    # Assert
    assert out.height == 1
    assert "games_played" in out.columns


def _two_team_points_stats() -> pl.DataFrame:
    """Return one week of points scored for two teams."""
    return pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "points_scored": [20, 30],
        }
    )


def test_calculate_league_means_and_regress() -> None:
    """League means are calculated."""
    # Arrange
    stats = _two_team_points_stats()

    # Act
    means = teamrankings.calculate_league_means(stats, season=2023)

    # Assert
    assert means["points_scored"] == 25.0


def test_regress_to_mean_moves_values_toward_the_league_mean() -> None:
    """Regression to mean works."""
    # Arrange
    means = teamrankings.calculate_league_means(_two_team_points_stats(), season=2023)

    # Act
    regressed = teamrankings.regress_to_mean(
        pl.DataFrame({"team_abbr": ["AAA"], "points_scored": [20.0], "games_played": [1]}),
        means,
        regression_factor=0.5,
    )

    # Assert
    assert regressed["points_scored"][0] == 22.5


def _one_team_points_stats() -> pl.DataFrame:
    """Return one 2023 game of points scored for one team."""
    return pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team_abbr": ["AAA"],
            "points_scored": [20],
        }
    )


def test_calculate_league_means_and_regress_handle_empty_cases() -> None:
    """Empty seasons leave no league means."""
    # Arrange
    stats = _one_team_points_stats()

    # Act
    means = teamrankings.calculate_league_means(stats, season=2024)

    # Assert
    assert means == {}


def test_regress_to_mean_with_empty_means_leaves_data_unchanged() -> None:
    """Empty regression means leave data unchanged."""
    # Arrange
    stats = _one_team_points_stats()

    # Act
    regressed = teamrankings.regress_to_mean(stats, {})

    # Assert
    assert regressed.to_dicts() == stats.to_dicts()


def _merge_schedule_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a one-game schedule and the two teams' aggregated stats."""
    schedule = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    agg = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "points_scored": [20, 10],
        }
    )
    return schedule, agg


def test_merge_schedule_and_stat_differentials() -> None:
    """Schedule is merged with team stats."""
    # Arrange
    schedule, agg = _merge_schedule_inputs()

    # Act
    merged = teamrankings.merge_schedule_with_team_stats(schedule, agg)

    # Assert
    assert "away_points_scored" in merged.columns
    assert "home_points_scored" in merged.columns


def test_calculate_stat_differentials_on_a_merged_schedule() -> None:
    """Differentials are calculated on the merged schedule."""
    # Arrange
    merged = teamrankings.merge_schedule_with_team_stats(*_merge_schedule_inputs())

    # Act
    with_diff = teamrankings.calculate_stat_differentials(merged, ["points_scored"])

    # Assert
    assert "points_scored_diff" in with_diff.columns


def test_calculate_stat_differentials_skips_non_numeric_pairs() -> None:
    """Differentials are only added for numeric away/home stat pairs."""
    # Arrange
    df = pl.DataFrame(
        {
            "away_status": ["healthy"],
            "home_status": ["healthy"],
            "away_points": [20],
            "home_points": [17],
        }
    )

    # Act
    out = teamrankings.calculate_stat_differentials(df, ["status", "points"])

    # Assert
    assert "status_diff" not in out.columns
    assert out["points_diff"][0] == 3


def test_get_tr_columns() -> None:
    """TR columns include expected ratings and stats."""
    # Act
    cols = teamrankings.get_tr_columns()

    # Assert
    assert constants.TR_RATINGS[0] in cols
    assert constants.TR_STATS[0] in cols


def test_helper_accessors() -> None:
    """The stat column accessor returns the nflreadpy stat columns."""
    # Act
    columns = teamrankings.get_stat_columns()

    # Assert
    assert columns == constants.NFLREADPY_STATS


def test_get_elo_columns_returns_the_elo_columns() -> None:
    """The Elo column accessor returns the published Elo columns."""
    # Act
    columns = teamrankings.get_elo_columns()

    # Assert
    assert columns == constants.ELO_COLUMNS


def test_remove_diff_columns_drops_the_diff_columns() -> None:
    """Diff columns are dropped and the rest keep their order."""
    # Arrange
    with_diffs = pl.DataFrame({"team_abbr": ["AAA"], "rating_diff": [1.0], "rating": [2.0]})

    # Act
    without_diffs = teamrankings.remove_diff_columns(with_diffs)

    # Assert
    assert without_diffs.columns == ["team_abbr", "rating"]


@pytest.mark.parametrize(
    ("game", "expected"),
    [
        pytest.param({"away_score": 21, "home_score": 17}, 1.0, id="away_win"),
        pytest.param({"away_score": 17, "home_score": 21}, 0.0, id="home_win"),
        pytest.param({"away_score": 20, "home_score": 20}, 0.5, id="tie"),
        pytest.param({"away_score": None, "home_score": 20}, None, id="unplayed"),
    ],
)
def test_calculate_game_result_and_schedule_filters(
    game: dict[str, int | None], expected: float | None
) -> None:
    """Game-result helpers classify completed and unplayed games."""
    # Act
    result = teamrankings.calculate_game_result(game)

    # Assert
    assert result == expected
    if expected is None:
        assert result is None


def _three_game_schedule() -> pl.DataFrame:
    """Return one completed week 2 game and two unplayed week 3 games."""
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [2, 3, 3],
            "away_score": [17, None, 10],
            "home_score": [20, None, None],
        }
    )


def test_filter_completed_games_keeps_only_scored_games() -> None:
    """The completed-games filter keeps rows with both scores."""
    # Arrange
    schedule = _three_game_schedule()

    # Act
    completed = teamrankings.filter_completed_games(schedule)

    # Assert
    assert completed.height == 1


def test_filter_upcoming_games_keeps_the_unplayed_week() -> None:
    """The upcoming-games filter keeps the target week's unplayed rows."""
    # Arrange
    schedule = _three_game_schedule()

    # Act
    upcoming = teamrankings.filter_upcoming_games(schedule, season=2023, week=3)

    # Assert
    assert upcoming.height == 2


def _eighteen_week_stats(monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    """Fix an 18-week regular season and return week 1, 18 and 19 stats for one team."""
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(18))
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 18, 19],
            "team_abbr": ["AAA", "AAA", "AAA"],
            "pass_yards": [200, 300, 500],
            "rush_yards": [100, 150, 250],
        }
    )


def test_aggregate_team_stats_to_week_handles_week_one_and_playoff_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Week one has no prior-game window."""
    # Arrange
    stats = _eighteen_week_stats(monkeypatch)

    # Act
    week_one = teamrankings.aggregate_team_stats_to_week(stats, target_week=1, season=2023)

    # Assert
    assert week_one.height == 0


def test_aggregate_team_stats_to_week_uses_the_regular_season_for_playoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The playoff path aggregates the prior regular-season games, not the playoff game."""
    # Arrange
    stats = _eighteen_week_stats(monkeypatch)

    # Act
    playoff = teamrankings.aggregate_team_stats_to_week(stats, target_week=19, season=2023)

    # Assert
    row = playoff.row(0, named=True)
    assert row["games_played"] == 2
    assert row["pass_yards"] == 250.0


def test_get_stats_for_diff_deduplicates_and_includes_expected_groups() -> None:
    """Diff-stat selection includes the expected source groups without duplicates."""
    # Act
    stats = teamrankings.get_stats_for_diff()

    # Assert
    assert len(stats) == len(set(stats))
    assert constants.ELO_COLUMNS[0] in stats
    assert constants.TR_RATINGS[0] in stats
    assert constants.TREND_FEATURE_COLUMNS[0] in stats


def test_load_team_rankings_partial_full_and_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TeamRankings are loaded correctly for past and future weeks."""
    # Arrange
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

    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(3))
    monkeypatch.setattr(teamrankings, "get_week_date", returning(date(2023, 9, 1)))

    scraped_weeks: list[int] = []

    def fake_scrape(week: int, _week_date: date, **_kwargs: object) -> pl.DataFrame:
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

    # Act
    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
    )

    # Assert
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


def test_load_team_rankings_min_week_skips_early_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Min week setting skips known-missing early weeks."""
    # Arrange
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(3))
    monkeypatch.setattr(teamrankings, "get_week_date", returning(date(2023, 9, 1)))
    monkeypatch.setattr(teamrankings, "update_season_team_rankings", returning(None))

    scraped_weeks: list[int] = []

    def fake_scrape(week: int, _week_date: date, **_kwargs: object) -> pl.DataFrame:
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

    # Act
    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
        min_week=2,
    )

    # Assert
    weeks = sorted(combined["week"].unique().to_list())
    assert 1 not in weeks
    assert 2 in weeks
    assert all(week >= 2 for week in scraped_weeks)


def test_load_team_rankings_skips_pre_min_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seasons before TR availability return empty data."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    season = constants.TEAMRANKINGS_MIN_SEASON - 1

    # Act
    combined = teamrankings.load_team_rankings(
        season,
        current_season=constants.TEAMRANKINGS_MIN_SEASON,
        current_week=2,
    )

    # Assert
    assert combined.height == 0


def test_load_team_rankings_returns_empty_for_future_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Future seasons short-circuit because there is no cached or scrapeable data yet."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    # Act
    combined = teamrankings.load_team_rankings(2027, current_season=2026, current_week=2)

    # Assert
    assert combined.height == 0


def test_load_team_rankings_uses_cached_fallback_when_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current-week cache is reused when the forced refresh scrape returns nothing."""
    # Arrange
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(3))
    monkeypatch.setattr(teamrankings, "get_week_date", returning(date(2023, 9, 1)))

    def always_valid(_df: pl.DataFrame) -> tuple[bool, list[str]]:
        return True, []

    monkeypatch.setattr(teamrankings, "_validate_tr_dataframe", always_valid)

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
        returning(pl.DataFrame()),
    )

    # Act
    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
        min_week=2,
    )

    # Assert
    week2 = combined.filter(pl.col("week") == 2).sort("team_abbr")
    assert week2["predictive_rating"].to_list() == [0.5, 0.6]


def test_load_team_rankings_keeps_existing_partial_week_when_scrape_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial-scrape failures keep the cached week data instead of dropping it."""
    # Arrange
    season = 2023
    season_dir = tmp_path / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(1))
    monkeypatch.setattr(teamrankings, "get_week_date", returning(date(2023, 9, 1)))
    monkeypatch.setattr(
        teamrankings,
        "_get_required_tr_columns",
        lambda: {"team_abbr", "week", "predictive_rating", "third_down_pct"},
    )

    def missing_one_rating(_df: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
        return {"stat": "third_down_pct"}, {}

    monkeypatch.setattr(teamrankings, "get_missing_tr_columns", missing_one_rating)
    monkeypatch.setattr(
        teamrankings,
        "scrape_team_rankings_for_week",
        returning(pl.DataFrame()),
    )

    partial = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [1, 1],
            "predictive_rating": [1.0, 2.0],
        }
    )
    partial.write_csv(season_dir / f"{season}_week_01_team_rankings.csv")

    # Act
    combined = teamrankings.load_team_rankings(
        season,
        current_season=season,
        current_week=2,
    )

    # Assert
    week1 = combined.filter(pl.col("week") == 1).sort("team_abbr")
    assert week1.to_dicts() == partial.sort("team_abbr").to_dicts()
