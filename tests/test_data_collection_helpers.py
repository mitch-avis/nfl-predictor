"""Tests for data collection helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils


def test_prefix_team_records_and_invalid_side() -> None:
    """Team records are prefixed correctly for away/home sides, error on invalid side."""

    base_cols = [
        col[len("away_") :] for col in constants.RECORD_FEATURE_COLUMNS if col.startswith("away_")
    ]
    records_df = pl.DataFrame({"team_abbr": ["AAA"], **{col: [1] for col in base_cols}})

    away = data_collection._prefix_team_records(records_df, "away")
    assert "away_abbr" in away.columns
    assert "away_wins" in away.columns

    with pytest.raises(ValueError):
        data_collection._prefix_team_records(records_df, "bad")


def test_resolve_seasons_rejects_pre_nflreadpy() -> None:
    """min_season before nflreadpy availability raises a ValueError."""

    with pytest.raises(ValueError):
        data_collection._resolve_seasons(constants.NFLREADPY_MIN_SEASON - 1, 2000)


def test_merge_team_rankings_week_specific() -> None:
    """TeamRankings are merged correctly for the given week."""

    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [3, 3],
            "predictive_rating": [1.0, 2.0],
        }
    )

    out = data_collection._merge_team_rankings(
        merged,
        season=2023,
        week=3,
        tr_df=tr_df,
        prev_tr_df=None,
    )

    assert "away_predictive_rating" in out.columns
    assert "home_predictive_rating" in out.columns


def test_merge_team_rankings_week1_prev() -> None:
    """For week 1, previous season's TeamRankings are merged."""

    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    prev_tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "AAA", "BBB"],
            "week": [1, 2, 2],
            "predictive_rating": [1.0, 3.0, 2.5],
        }
    )

    out = data_collection._merge_team_rankings(
        merged,
        season=2023,
        week=1,
        tr_df=None,
        prev_tr_df=prev_tr_df,
    )

    assert out["away_predictive_rating"][0] == 3.0
    assert out["home_predictive_rating"][0] == 2.5


def test_save_and_load_dataframe(tmp_path: Path, monkeypatch) -> None:
    """DataFrame is saved and loaded correctly."""

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))

    df = pl.DataFrame({"a": [1], "b": [2]})
    data_collection.save_dataframe(df, "unit_test")

    loaded = data_collection.load_dataframe("unit_test")
    assert loaded is not None
    assert loaded.height == 1


def test_load_dataframe_missing(tmp_path: Path, monkeypatch) -> None:
    """Loading missing DataFrame returns None."""

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    assert data_collection.load_dataframe("missing") is None


def test_determine_nfl_week_branches() -> None:
    """NFL week is determined correctly for various dates."""

    assert data_collection._determine_nfl_week(date(2024, 7, 1)) == 1

    week = data_collection._determine_nfl_week(date(2024, 2, 1))
    assert 1 <= week <= 22


def test_collect_all_data_minimal(monkeypatch) -> None:
    """Data collection works end-to-end for minimal data."""

    season = constants.MIN_SEASON + 1
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "date": [date(2023, 9, 1)],
            "game_id": ["game1"],
        }
    )
    prev_schedule = schedule_df.with_columns(pl.lit(season - 1, dtype=pl.Int64).alias("season"))
    team_stats_df = pl.DataFrame(
        {
            "season": [season - 1],
            "week": [1],
            "team_abbr": ["AAA"],
        }
    )

    def fake_load_schedule(seasons, **_kwargs):
        """Fake schedule loader for testing."""
        if seasons == [season - 1]:
            return prev_schedule
        return schedule_df

    monkeypatch.setattr(polars_utils, "load_schedule", fake_load_schedule)
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_args, **_kwargs: team_stats_df)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _sched: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())

    def fake_load_raw_elo_data() -> pl.DataFrame:
        """Fake raw Elo loader for testing."""

        return pl.DataFrame()

    monkeypatch.setattr(polars_utils, "load_raw_elo_data", fake_load_raw_elo_data)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (season, 1))
    monkeypatch.setattr(
        polars_utils, "load_team_rankings", lambda *_args, **_kwargs: pl.DataFrame()
    )
    monkeypatch.setattr(data_collection, "process_season", lambda *_args, **_kwargs: schedule_df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    combined = data_collection.collect_all_data([season])
    assert combined.height == 1
    assert combined["game_id"][0] == "game1"


def test_collect_all_data_reuses_team_rankings_cache(monkeypatch) -> None:
    """TeamRankings loads once per season within a collection run."""

    season_one = constants.MIN_SEASON + 1
    season_two = season_one + 1
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season_one, season_two], dtype=pl.Int64),
            "week": [1, 1],
            "away_abbr": ["AAA", "CCC"],
            "home_abbr": ["BBB", "DDD"],
            "date": [date(2023, 9, 1), date(2024, 9, 1)],
            "game_id": ["game1", "game2"],
        }
    )
    prev_schedule = pl.DataFrame(
        {
            "season": pl.Series([season_one - 1], dtype=pl.Int64),
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "date": [date(2022, 9, 1)],
            "game_id": ["game0"],
        }
    )
    team_stats_df = pl.DataFrame(
        {
            "season": [season_one - 1],
            "week": [1],
            "team_abbr": ["AAA"],
        }
    )
    tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA"],
            "week": [1],
            "predictive_rating": [1.0],
        }
    )
    tr_calls: dict[int, int] = {}

    def fake_load_schedule(seasons, **_kwargs):
        """Fake schedule loader for testing."""
        if seasons == [season_one - 1]:
            return prev_schedule
        return schedule_df

    def fake_load_team_rankings(season, *_args, **_kwargs):
        """Track TeamRankings loads by season."""
        tr_calls[season] = tr_calls.get(season, 0) + 1
        return tr_df

    def fake_process_season(season, schedule_df, *_args, **_kwargs):
        """Return season-specific schedule rows."""
        return schedule_df.filter(pl.col("season") == season)

    monkeypatch.setattr(polars_utils, "load_schedule", fake_load_schedule)
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_args, **_kwargs: team_stats_df)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _sched: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", lambda: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (season_two, 1))
    monkeypatch.setattr(polars_utils, "load_team_rankings", fake_load_team_rankings)
    monkeypatch.setattr(data_collection, "process_season", fake_process_season)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    combined = data_collection.collect_all_data([season_one, season_two])

    assert combined.height == 2
    assert tr_calls == {}


def test_process_week_fallback(monkeypatch) -> None:
    """Week processing falls back gracefully when lookahead/motivation features fail."""

    schedule_df = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_type": ["REG"],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_score": [None],
            "home_score": [None],
        }
    )

    team_stats_df = pl.DataFrame(
        {
            "season": [2022, 2022],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "pass_yards": [200, 180],
            "rush_yards": [100, 90],
        }
    )

    elo_df = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_elo_pre": [1500],
            "home_elo_pre": [1400],
            "away_qb_value_pre": [1.0],
            "home_qb_value_pre": [1.1],
            "away_qb_elo_pre": [1300],
            "home_qb_elo_pre": [1250],
        }
    )

    tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [1, 1],
            "predictive_rating": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(polars_utils, "add_divisional_matchup_feature", lambda df: df)

    def raise_value_error(*_args, **_kwargs):
        """Raise ValueError for testing fallback."""

        raise ValueError("incomplete schedule")

    monkeypatch.setattr(polars_utils, "add_lookahead_features", raise_value_error)
    monkeypatch.setattr(polars_utils, "add_motivation_features", raise_value_error)
    monkeypatch.setattr(polars_utils, "get_stats_for_diff", lambda: [])
    monkeypatch.setattr(polars_utils, "calculate_stat_differentials", lambda df, _stats: df)

    out = data_collection.process_week(
        season=2023,
        week=1,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        min_season=2022,
        elo_df=elo_df,
        tr_df=tr_df,
        prev_tr_df=None,
    )

    assert "away_elo_pre" in out.columns
    assert "home_elo_pre" in out.columns
    assert "away_predictive_rating" in out.columns
    assert "home_predictive_rating" in out.columns
    assert "away_wins" in out.columns
