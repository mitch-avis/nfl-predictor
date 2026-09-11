"""Tests for data collection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils


def test_current_nfl_season_and_default_max_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Season helpers should respect the offseason boundary and explicit dates."""
    assert data_collection._current_nfl_season(date(2024, 9, 1)) == 2024
    assert data_collection._current_nfl_season(date(2024, 2, 1)) == 2023

    class _FrozenDate(date):
        """Date test double with a stable today() implementation."""

        @classmethod
        def today(cls) -> _FrozenDate:
            """Return a fixed offseason date."""
            return cls(2025, 2, 1)

    monkeypatch.setattr(data_collection, "date", _FrozenDate)

    assert data_collection._default_max_season() == 2024
    assert data_collection._default_max_season(date(2025, 10, 1)) == 2025


def test_configure_logging_only_enables_debug_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug logging should only mutate logger state when explicitly enabled."""
    handler_levels: list[int] = []
    debug_messages: list[str] = []
    logger = SimpleNamespace(
        handlers=[SimpleNamespace(setLevel=lambda level: handler_levels.append(level))],
        setLevel=lambda level: handler_levels.append(level),
        debug=lambda message: debug_messages.append(message),
    )
    monkeypatch.setattr(data_collection, "log", logger)

    data_collection._configure_logging(False)
    assert handler_levels == []

    data_collection._configure_logging(True)
    assert handler_levels == [10, 10]
    assert debug_messages == ["Debug logging enabled for data collection."]


def test_parse_args_uses_defaults_and_boolean_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI parsing should combine default toggles with explicit argv overrides."""
    monkeypatch.setattr(data_collection, "DEFAULT_MIN_SEASON", 2001)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_TIMING", True)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_DEBUG", False)
    monkeypatch.setattr(data_collection, "FORCE_REFRESH_NFLREADPY", True)
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    default_config = data_collection._parse_args([])
    assert default_config == data_collection.DataCollectionConfig(
        enable_timing=True,
        enable_debug=False,
        force_refresh_nflreadpy=True,
        min_season=2001,
        max_season=2025,
    )

    explicit_config = data_collection._parse_args(
        [
            "--min-season",
            "2020",
            "--max-season",
            "2021",
            "--no-timing",
            "--debug-logs",
            "--no-refresh-nflreadpy",
        ]
    )
    assert explicit_config == data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=True,
        force_refresh_nflreadpy=False,
        min_season=2020,
        max_season=2021,
    )


def test_parse_args_reads_the_stat_prior_blend_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stat prior blend is on at the shared K by default and can be tuned or ablated."""
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    default_config = data_collection._parse_args([])
    assert default_config.blend_stat_prior is True
    assert default_config.stat_prior_blend_games == constants.PRIOR_BLEND_GAMES

    tuned = data_collection._parse_args(["--stat-prior-blend-games", "6"])
    assert tuned.blend_stat_prior is True
    assert tuned.stat_prior_blend_games == pytest.approx(6.0)

    ablated = data_collection._parse_args(["--no-stat-prior-blend"])
    assert ablated.blend_stat_prior is False


def test_parse_args_rejects_a_non_positive_stat_prior_blend_games() -> None:
    """A zero or negative K would divide by zero for a team with no games, so it is refused."""
    with pytest.raises(SystemExit):
        data_collection._parse_args(["--stat-prior-blend-games", "0"])


def test_resolve_config_uses_defaults_or_parsed_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config resolution should either materialize defaults or delegate to _parse_args."""
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_TIMING", False)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_DEBUG", True)
    monkeypatch.setattr(data_collection, "FORCE_REFRESH_NFLREADPY", False)
    monkeypatch.setattr(data_collection, "DEFAULT_MIN_SEASON", 2002)
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2026)

    default_config = data_collection._resolve_config(None)
    assert default_config == data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=True,
        force_refresh_nflreadpy=False,
        min_season=2002,
        max_season=2026,
    )

    sentinel = data_collection.DataCollectionConfig(True, False, True, 2020, 2021)
    monkeypatch.setattr(data_collection, "_parse_args", lambda argv: sentinel)
    assert data_collection._resolve_config(["--min-season", "2020"]) is sentinel


def test_resolve_seasons_success_and_invalid_order() -> None:
    """Season range resolution should support inclusive bounds and reject reversed ranges."""
    assert data_collection._resolve_seasons(constants.NFLREADPY_MIN_SEASON, 2001) == [
        constants.NFLREADPY_MIN_SEASON,
        2000,
        2001,
    ]

    with pytest.raises(ValueError, match="max_season must be >= min_season"):
        data_collection._resolve_seasons(2024, 2023)


def test_timed_step_logs_elapsed_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timed steps should avoid perf-counter work when disabled.

    When enabled, they should emit the elapsed timing log.
    """
    perf_values = iter([1.0, 3.5])
    info_messages: list[str] = []

    monkeypatch.setattr(data_collection.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        data_collection.log,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    with data_collection._timed_step("disabled", False):
        pass
    assert info_messages == []

    with data_collection._timed_step("enabled", True):
        pass
    assert info_messages == ["Timing: enabled took 2.50s"]


def test_timed_substep_accumulates_totals_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed substeps should accumulate elapsed time into the provided totals mapping."""
    perf_values = iter([10.0, 12.25, 20.0, 21.0])
    monkeypatch.setattr(data_collection.time, "perf_counter", lambda: next(perf_values))

    totals: dict[str, float] = {"existing": 1.0}
    with data_collection._timed_substep("load", False, totals):
        pass
    assert totals == {"existing": 1.0}

    with data_collection._timed_substep("load", True, totals):
        pass
    assert totals["load"] == pytest.approx(2.25)

    with data_collection._timed_substep("skip", True, None):
        pass


def test_log_df_stats_respects_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """DataFrame stats should only be logged when debug output is enabled."""
    messages: list[str] = []
    monkeypatch.setattr(
        data_collection.log,
        "debug",
        lambda message, *args: messages.append(message % args if args else message),
    )

    df = pl.DataFrame({"a": [1], "b": [2]})
    data_collection._log_df_stats("disabled", df, False)
    assert messages == []

    data_collection._log_df_stats("enabled", df, True)
    assert messages == ["enabled: 1 rows, 2 cols"]


def test_main_orchestrates_collection_and_output_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main should resolve config, collect data, and write all expected output datasets."""
    config = data_collection.DataCollectionConfig(
        enable_timing=True,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=2023,
        max_season=2024,
    )
    all_data = pl.DataFrame({"game_id": ["g1"], "season": [2024], "week": [3]})
    no_diff = pl.DataFrame({"game_id": ["g1"]})
    completed = pl.DataFrame({"game_id": ["g1"]})
    upcoming = pl.DataFrame({"game_id": ["g2"]})
    saved: list[tuple[pl.DataFrame, str]] = []
    timed_labels: list[tuple[str, bool]] = []
    info_messages: list[str] = []

    @contextmanager
    def fake_timed_step(label: str, enabled: bool):
        """Record timed-step usage and behave like a no-op context manager."""
        timed_labels.append((label, enabled))
        yield

    monkeypatch.setattr(data_collection, "_resolve_config", lambda _argv: config)
    monkeypatch.setattr(
        data_collection,
        "_configure_logging",
        lambda enabled: timed_labels.append(("debug", enabled)),
    )
    monkeypatch.setattr(data_collection, "_determine_nfl_week", lambda _today: 3)
    monkeypatch.setattr(data_collection, "_resolve_seasons", lambda start, end: [start, end])
    monkeypatch.setattr(data_collection, "collect_all_data", lambda seasons, config: all_data)
    monkeypatch.setattr(data_collection, "_log_df_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_collection, "_timed_step", fake_timed_step)
    monkeypatch.setattr(
        data_collection, "save_dataframe", lambda df, name: saved.append((df, name))
    )
    monkeypatch.setattr(
        data_collection.polars_utils,
        "remove_diff_columns",
        lambda df: no_diff if df is all_data else completed,
    )
    monkeypatch.setattr(
        data_collection.polars_utils, "filter_completed_games", lambda _df: completed
    )
    monkeypatch.setattr(
        data_collection.polars_utils,
        "filter_upcoming_games",
        lambda _df, season, week: upcoming if (season, week) == (2024, 3) else pl.DataFrame(),
    )
    monkeypatch.setattr(
        data_collection.log,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    class _FrozenDate(date):
        """Date test double with a stable today() implementation."""

        @classmethod
        def today(cls) -> _FrozenDate:
            """Return a fixed in-season date."""
            return cls(2024, 9, 18)

    monkeypatch.setattr(data_collection, "date", _FrozenDate)

    data_collection.main([])

    assert timed_labels[0] == ("debug", False)
    assert ("collect_all_data", True) in timed_labels
    assert [name for _df, name in saved] == [
        "all_data_ml",
        "all_data",
        "completed_games_ml",
        "completed_games",
        "predict/week_03_games_to_predict",
    ]
    assert any(message == "Data collection complete." for message in info_messages)


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


def test_collect_all_data_handles_current_min_season_and_empty_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect-all-data should skip prior-season loading when already at the nflreadpy floor."""
    season = constants.NFLREADPY_MIN_SEASON
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    team_stats_df = pl.DataFrame({"season": [season], "week": [1], "team_abbr": ["AAA"]})
    warnings: list[str] = []
    scoring_inputs: list[pl.DataFrame] = []

    monkeypatch.setattr(constants, "TEAMRANKINGS_MIN_SEASON", season + 1)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (season, 2))
    monkeypatch.setattr(
        polars_utils,
        "load_schedule",
        lambda seasons, **_kwargs: schedule_df,
    )

    def _load_team_stats(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        """Assert that prior-season stats are not requested at the floor season."""
        assert seasons == [season]
        return team_stats_df

    monkeypatch.setattr(polars_utils, "load_team_stats", _load_team_stats)

    def _add_scoring(df: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
        """Capture the schedule frame used for scoring enrichment."""
        scoring_inputs.append(sched)
        return df

    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", _add_scoring)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", lambda: pl.DataFrame())
    monkeypatch.setattr(
        polars_utils,
        "load_team_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("TeamRankings should skip")),
    )
    monkeypatch.setattr(data_collection, "process_season", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr(
        data_collection.log,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    combined = data_collection.collect_all_data(
        [season],
        config=data_collection.DataCollectionConfig(False, False, False, season, season),
    )

    assert combined.height == 0
    assert scoring_inputs == [schedule_df]
    assert warnings == ["No ELO ratings loaded"]


def test_collect_all_data_dedupes_without_date_or_game_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect-all-data should fall back to matchup-based deduping when date/game_id are absent."""
    season = constants.NFLREADPY_MIN_SEASON
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    team_stats_df = pl.DataFrame({"season": [season], "week": [1], "team_abbr": ["AAA"]})
    season_data = pl.DataFrame(
        {
            "season": [season, season],
            "week": [2, 2],
            "away_abbr": ["AAA", "AAA"],
            "home_abbr": ["BBB", "BBB"],
        }
    )

    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (season, 2))
    monkeypatch.setattr(polars_utils, "load_schedule", lambda *_args, **_kwargs: schedule_df)
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_args, **_kwargs: team_stats_df)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _sched: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", lambda: pl.DataFrame())
    monkeypatch.setattr(
        polars_utils, "load_team_rankings", lambda *_args, **_kwargs: pl.DataFrame()
    )
    monkeypatch.setattr(data_collection, "process_season", lambda *_args, **_kwargs: season_data)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    combined = data_collection.collect_all_data(
        [season],
        config=data_collection.DataCollectionConfig(False, False, False, season, season),
    )

    assert combined.height == 1
    assert combined.columns == season_data.columns


def test_process_season_handles_empty_schedule_and_timing_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-season should return empty without schedule data.

    When timing is enabled and work occurs, it should also log a timing summary.
    """
    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(
        data_collection.log,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        data_collection.log,
        "info",
        lambda message, *args: infos.append(message % args if args else message),
    )

    empty = data_collection.process_season(
        2024,
        pl.DataFrame({"season": [2023], "week": [1]}),
        pl.DataFrame(),
        min_season=2023,
    )
    assert empty.height == 0
    assert warnings == ["No schedule data for season 2024"]

    schedule_df = pl.DataFrame({"season": [2024, 2024], "week": [2, 3]})
    trend_df = pl.DataFrame({"season": [2024], "week": [2], "team_abbr": ["AAA"], "x": [1.0]})
    qb_df = pl.DataFrame({"season": [2024], "week": [2], "qb_name": ["QB"], "x": [1.0]})
    coach_df = pl.DataFrame(
        {"season": [2024], "week": [2], "team_abbr": ["AAA"], "games_prior": [3]}
    )
    seen: list[tuple[int, int]] = []

    monkeypatch.setattr(polars_utils, "build_team_elo_trends", lambda *_args, **_kwargs: trend_df)
    monkeypatch.setattr(polars_utils, "build_qb_trends", lambda *_args, **_kwargs: qb_df)
    monkeypatch.setattr(polars_utils, "build_team_stat_trends", lambda *_args, **_kwargs: trend_df)
    monkeypatch.setattr(polars_utils, "build_coach_features", lambda *_args, **_kwargs: coach_df)

    def _fake_process_week(
        season: int,
        week: int,
        _schedule_df: pl.DataFrame,
        _team_stats_df: pl.DataFrame,
        **kwargs: object,
    ) -> pl.DataFrame:
        """Record timing usage and return one row per week."""
        timing_totals = kwargs.get("timing_totals")
        if isinstance(timing_totals, dict):
            typed_totals = cast(dict[str, float], timing_totals)
            typed_totals["merge"] = typed_totals.get("merge", 0.0) + 1.0
        assert kwargs["team_elo_trends"] is trend_df
        assert kwargs["qb_trends"] is qb_df
        assert kwargs["team_stat_trends"] is trend_df
        assert kwargs["coach_features"] is coach_df
        seen.append((season, week))
        return pl.DataFrame({"season": [season], "week": [week]})

    monkeypatch.setattr(data_collection, "process_week", _fake_process_week)

    season_out = data_collection.process_season(
        2024,
        schedule_df,
        pl.DataFrame({"season": [2024], "week": [1]}),
        min_season=2023,
        timing_enabled=True,
        elo_df=pl.DataFrame({"season": [2024], "week": [1]}),
    )

    assert seen == [(2024, 2), (2024, 3)]
    assert season_out.height == 2
    assert any(message.startswith("Timing summary season 2024:") for message in infos)


def test_process_week_early_exit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process-week should handle week-one, missing-week, and no-aggregate paths cleanly."""
    info_messages: list[str] = []
    debug_messages: list[str] = []
    monkeypatch.setattr(
        data_collection.log,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )
    monkeypatch.setattr(
        data_collection.log,
        "debug",
        lambda message, *args: debug_messages.append(message % args if args else message),
    )

    schedule_df = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    assert (
        data_collection.process_week(
            2023,
            1,
            schedule_df,
            pl.DataFrame(),
            min_season=2023,
        ).height
        == 0
    )
    assert any("Skipping season 2023 week 1" in message for message in info_messages)

    no_week = data_collection.process_week(
        2024,
        2,
        schedule_df,
        pl.DataFrame(),
        min_season=2023,
    )
    assert no_week.height == 0

    monkeypatch.setattr(
        polars_utils,
        "aggregate_team_stats_to_week",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    no_stats_schedule = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    no_stats = data_collection.process_week(
        2024,
        2,
        no_stats_schedule,
        pl.DataFrame({"season": [2024], "week": [1], "team_abbr": ["AAA"]}),
        min_season=2023,
    )
    assert no_stats.height == 0
    assert any("No aggregated stats available" in message for message in debug_messages)


def test_merge_helpers_and_team_rankings_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local merge helpers should support no-op guards and normal away/home merges."""
    merged = pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_qb": ["QB1"],
            "home_qb": ["QB2"],
        }
    )

    assert data_collection._merge_team_trends(merged, None).equals(merged)
    assert data_collection._merge_team_trends(merged, pl.DataFrame()).equals(merged)
    assert data_collection._merge_team_trends(merged, pl.DataFrame({"season": [2024]})).equals(
        merged
    )

    team_trends = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "team_abbr": ["AAA", "BBB"],
            "elo_trend": [1.0, 2.0],
        }
    )
    team_out = data_collection._merge_team_trends(merged, team_trends)
    assert team_out["away_elo_trend"][0] == 1.0
    assert team_out["home_elo_trend"][0] == 2.0

    assert data_collection._merge_qb_trends(
        merged.drop(["away_qb", "home_qb"]), team_trends
    ).equals(merged.drop(["away_qb", "home_qb"]))
    qb_trends = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "qb_name": ["QB1", "QB2"],
            "qb_form": [0.1, 0.2],
        }
    )
    qb_out = data_collection._merge_qb_trends(merged, qb_trends)
    assert qb_out["away_qb_form"][0] == pytest.approx(0.1)
    assert qb_out["home_qb_form"][0] == pytest.approx(0.2)

    coach_df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "team_abbr": ["AAA", "BBB"],
            "coach_name": ["A", "B"],
            "coach_games_prior": [3, 4],
        }
    )
    coach_out = data_collection._merge_coach_features(merged, coach_df)
    assert coach_out["away_coach_games_prior"][0] == 3
    assert coach_out["home_coach_games_prior"][0] == 4

    monkeypatch.setattr(
        polars_utils, "get_latest_team_rankings", lambda df: df.tail(1).drop("week")
    )
    monkeypatch.setattr(polars_utils, "get_tr_columns", lambda: ["predictive_rating"])

    playoff_tr = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [18, 18],
            "predictive_rating": [1.2, 2.4],
        }
    )
    playoff_out = data_collection._merge_team_rankings(
        merged.select(["away_abbr", "home_abbr"]),
        season=2024,
        week=20,
        tr_df=playoff_tr,
        prev_tr_df=None,
    )
    assert "away_predictive_rating" in playoff_out.columns

    warnings: list[str] = []
    monkeypatch.setattr(
        data_collection.log,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    no_expected_cols = pl.DataFrame({"team_abbr": ["AAA"], "week": [2], "other": [1.0]})
    unchanged = data_collection._merge_team_rankings(
        merged.select(["away_abbr", "home_abbr"]),
        season=2024,
        week=2,
        tr_df=no_expected_cols,
        prev_tr_df=None,
    )
    assert unchanged.columns == ["away_abbr", "home_abbr"]
    assert any("TR data has no expected columns" in warning for warning in warnings)


def _team_stats_stub() -> pl.DataFrame:
    """Build a minimal two-team, one-week team-stats frame."""
    return pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "pass_yards": [250.0, 180.0],
        }
    )


def test_join_pbp_team_game_stats_attaches_counts_by_team_week() -> None:
    """Play-by-play counts join onto the matching team-week rows."""
    pbp_team_games = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "offensive_snaps": [64],
            "pass_epa_sum": [5.5],
        }
    )

    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), pbp_team_games)

    assert out.height == 2
    aaa = out.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    bbb = out.filter(pl.col("team_abbr") == "BBB").row(0, named=True)
    assert aaa["offensive_snaps"] == 64
    assert aaa["pass_epa_sum"] == pytest.approx(5.5)
    # The unmatched team keeps a null rather than a fabricated zero.
    assert bbb["offensive_snaps"] is None
    # The join must not duplicate the identity columns already on team stats.
    assert out.columns.count("opponent_abbr") == 1
    assert "opponent_abbr_right" not in out.columns


def test_join_pbp_team_game_stats_emits_nulls_when_no_play_by_play() -> None:
    """A season with no play-by-play still gets every count column, as nulls."""
    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), pl.DataFrame())

    for col in constants.PBP_COUNT_COLUMNS:
        assert col in out.columns, f"{col} missing from the invariant schema"
        assert out.select(pl.col(col).is_null().all()).item() is True
    assert out.height == 2


def test_join_pbp_team_game_stats_never_multiplies_rows() -> None:
    """Duplicate play-by-play keys are collapsed instead of multiplying team-stat rows.

    A malformed source that produces two rows for one `(season, week, team_abbr)` would
    otherwise silently inflate the team-stats frame and corrupt every downstream mean.
    """
    duplicated = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "AAA"],
            "opponent_abbr": ["BBB", ""],
            "offensive_snaps": [64, 0],
        }
    )

    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), duplicated)

    assert out.height == 2, "the join must not multiply team-stat rows"
    assert out.filter(pl.col("team_abbr") == "AAA")["offensive_snaps"][0] == 64
