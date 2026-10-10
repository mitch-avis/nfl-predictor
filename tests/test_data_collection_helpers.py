"""Tests for data collection helpers."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import clock, polars_utils
from nfl_predictor.utils.polars import loaders, nflverse_sources, ratings_build, week_rows


@pytest.mark.parametrize(("today", "season"), [(date(2024, 9, 1), 2024), (date(2024, 2, 1), 2023)])
def test_current_nfl_season_respects_the_offseason_boundary(today: date, season: int) -> None:
    """A date before the season ends belongs to the previous season."""
    # Act
    result = clock.nfl_season(today)

    # Assert
    assert result == season


def test_default_max_season_uses_today_in_the_offseason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a date the default max season comes from today, a fixed offseason date here."""
    # Arrange
    monkeypatch.setattr(clock, "local_today", lambda: date(2025, 2, 1))

    # Act
    result = data_collection._default_max_season()

    # Assert
    assert result == 2024


def test_default_max_season_uses_an_explicit_date() -> None:
    """An explicit in-season date sets the default max season."""
    # Act
    result = data_collection._default_max_season(date(2025, 10, 1))

    # Assert
    assert result == 2025


def _recording_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[str]]:
    """Replace the module logger with one that records levels and debug messages."""
    handler_levels: list[int] = []
    debug_messages: list[str] = []
    logger = SimpleNamespace(
        handlers=[SimpleNamespace(setLevel=handler_levels.append)],
        setLevel=handler_levels.append,
        debug=debug_messages.append,
    )
    monkeypatch.setattr(data_collection, "log", logger)
    return handler_levels, debug_messages


def test_configure_logging_leaves_the_logger_alone_without_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug logging should only mutate logger state when explicitly enabled."""
    # Arrange
    handler_levels, _ = _recording_logger(monkeypatch)

    # Act
    data_collection._configure_logging(enable_debug=False)

    # Assert
    assert handler_levels == []


def test_configure_logging_enables_debug_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With debug on, the logger and its handlers drop to DEBUG and say so."""
    # Arrange
    handler_levels, debug_messages = _recording_logger(monkeypatch)

    # Act
    data_collection._configure_logging(enable_debug=True)

    # Assert
    assert handler_levels == [10, 10]
    assert debug_messages == ["Debug logging enabled for data collection."]


def _fixed_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the module's default toggles and season window."""
    monkeypatch.setattr(data_collection, "DEFAULT_MIN_SEASON", 2001)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_TIMING", True)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_DEBUG", False)
    monkeypatch.setattr(data_collection, "FORCE_REFRESH_NFLREADPY", True)
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)


def test_parse_args_uses_the_default_toggles(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI parsing with no arguments takes the module's default toggles."""
    # Arrange
    _fixed_defaults(monkeypatch)

    # Act
    default_config = data_collection._parse_args([])

    # Assert
    assert default_config == data_collection.DataCollectionConfig(
        enable_timing=True,
        enable_debug=False,
        force_refresh_nflreadpy=True,
        min_season=2001,
        max_season=2025,
    )


def test_parse_args_applies_boolean_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit argv overrides replace the default toggles."""
    # Arrange
    _fixed_defaults(monkeypatch)

    # Act
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

    # Assert
    assert explicit_config == data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=True,
        force_refresh_nflreadpy=False,
        min_season=2020,
        max_season=2021,
    )


@pytest.mark.parametrize(
    ("argv", "data_dir"), [([], None), (["--data-dir", "/srv/nfl/data"], Path("/srv/nfl/data"))]
)
def test_parse_args_reads_the_data_dir_override(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], data_dir: Path | None
) -> None:
    """The data directory defaults to the packaged one and can be pointed elsewhere."""
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    config = data_collection._parse_args(argv)

    # Assert
    assert config.data_dir == data_dir


@pytest.mark.parametrize(
    ("argv", "blend", "games"),
    [
        ([], True, constants.PRIOR_BLEND_GAMES),
        (["--stat-prior-blend-games", "6"], True, 6.0),
        (["--no-stat-prior-blend"], False, constants.PRIOR_BLEND_GAMES),
    ],
)
def test_parse_args_reads_the_stat_prior_blend_switches(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], *, blend: bool, games: float
) -> None:
    """The stat prior blend is on at the shared K by default and can be tuned or ablated."""
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    config = data_collection._parse_args(argv)

    # Assert
    assert config.blend_stat_prior is blend
    assert config.stat_prior_blend_games == pytest.approx(games)


@pytest.mark.parametrize(
    ("argv", "team_source", "tr_source"),
    [
        ([], "pbp", "pbp"),
        (["--team-stats-source", "nflverse", "--tr-stats-source", "scrape"], "nflverse", "scrape"),
    ],
)
def test_parse_args_reads_the_team_and_tr_source_switches(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], team_source: str, tr_source: str
) -> None:
    """The ETL exposes explicit source switches for box-score and situational stat families.

    Both default to "pbp" since 2026-09-21 (verified against nflverse team stats and the
    reviewed no-breakage walk-forward arm); "nflverse"/"scrape" remain selectable explicitly.
    """
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    config = data_collection._parse_args(argv)

    # Assert
    assert config.team_stats_source == team_source
    assert config.tr_stats_source == tr_source


def test_parse_args_rejects_a_non_positive_stat_prior_blend_games() -> None:
    """A zero or negative K would divide by zero for a team with no games, so it is refused."""
    # Act & Assert
    with pytest.raises(SystemExit):
        data_collection._parse_args(["--stat-prior-blend-games", "0"])


def test_parse_args_defaults_the_strength_prior_blend_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strength blend's K defaults to the shared constant."""
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    default_config = data_collection._parse_args([])

    # Assert
    assert default_config.strength_prior_blend_games == constants.PRIOR_BLEND_GAMES


def test_resolve_config_defaults_the_strength_prior_blend_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config built without arguments uses the shared constant too."""
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    config = data_collection._resolve_config(None)

    # Assert
    assert config.strength_prior_blend_games == constants.PRIOR_BLEND_GAMES


def test_parse_args_reads_the_strength_prior_blend_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strength blend's K is set on its own, leaving the stat blend alone."""
    # Arrange
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    # Act
    tuned = data_collection._parse_args(["--strength-prior-blend-games", "2"])

    # Assert
    assert tuned.strength_prior_blend_games == pytest.approx(2.0)
    assert tuned.stat_prior_blend_games == constants.PRIOR_BLEND_GAMES
    assert tuned.blend_strength_prior is True


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_args_rejects_a_non_positive_strength_prior_blend_games(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A zero or negative K would divide by zero for a team with no games, so it is refused."""
    # Act & Assert
    with pytest.raises(SystemExit):
        data_collection._parse_args([f"--strength-prior-blend-games={value}"])

    assert (
        "--strength-prior-blend-games must be a finite positive number" in capsys.readouterr().err
    )


@pytest.mark.parametrize("option", ["--strength-prior-blend-games", "--stat-prior-blend-games"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_parse_args_rejects_a_non_finite_prior_blend_games(
    option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A NaN or infinite K would turn every blended value into NaN or the bare prior."""
    # Act & Assert
    with pytest.raises(SystemExit):
        data_collection._parse_args([f"{option}={value}"])

    assert f"{option} must be a finite positive number" in capsys.readouterr().err


def test_resolve_config_materializes_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without arguments, config resolution builds the module defaults."""
    # Arrange
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_TIMING", False)
    monkeypatch.setattr(data_collection, "ENABLE_DATA_COLLECTION_DEBUG", True)
    monkeypatch.setattr(data_collection, "FORCE_REFRESH_NFLREADPY", False)
    monkeypatch.setattr(data_collection, "DEFAULT_MIN_SEASON", 2002)
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2026)

    # Act
    default_config = data_collection._resolve_config(None)

    # Assert
    assert default_config == data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=True,
        force_refresh_nflreadpy=False,
        min_season=2002,
        max_season=2026,
    )


def test_resolve_config_delegates_arguments_to_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """With arguments, config resolution returns what _parse_args builds."""
    # Arrange
    sentinel = data_collection.DataCollectionConfig(
        enable_timing=True,
        enable_debug=False,
        force_refresh_nflreadpy=True,
        min_season=2020,
        max_season=2021,
    )
    monkeypatch.setattr(data_collection, "_parse_args", lambda argv: sentinel)

    # Act
    resolved = data_collection._resolve_config(["--min-season", "2020"])

    # Assert
    assert resolved is sentinel


def test_resolve_seasons_includes_both_bounds() -> None:
    """Season range resolution supports inclusive bounds."""
    # Act
    seasons = data_collection._resolve_seasons(constants.NFLREADPY_MIN_SEASON, 2001)

    # Assert
    assert seasons == [constants.NFLREADPY_MIN_SEASON, 2000, 2001]


def test_resolve_seasons_rejects_a_reversed_range() -> None:
    """A max season before the min season is refused."""
    # Act & Assert
    with pytest.raises(ValueError, match="max_season must be >= min_season"):
        data_collection._resolve_seasons(2024, 2023)


def _recorded_info(monkeypatch: pytest.MonkeyPatch, perf_values: list[float]) -> list[str]:
    """Feed perf_counter the given values and return the info messages logged."""
    values = iter(perf_values)
    info_messages: list[str] = []
    monkeypatch.setattr(data_collection.time, "perf_counter", lambda: next(values))
    monkeypatch.setattr(
        data_collection.log,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )
    return info_messages


def test_timed_step_logs_nothing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timed steps should avoid perf-counter work and logging when disabled."""
    # Arrange
    info_messages = _recorded_info(monkeypatch, [])

    # Act
    with data_collection._timed_step("disabled", enabled=False):
        pass

    # Assert
    assert info_messages == []


def test_timed_step_logs_elapsed_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When enabled, a timed step emits the elapsed timing log."""
    # Arrange
    info_messages = _recorded_info(monkeypatch, [1.0, 3.5])

    # Act
    with data_collection._timed_step("enabled", enabled=True):
        pass

    # Assert
    assert info_messages == ["Timing: enabled took 2.50s"]


@pytest.mark.parametrize(
    ("label", "enabled", "totals", "expected"),
    [
        ("load", False, {"existing": 1.0}, {"existing": 1.0}),
        ("load", True, {"existing": 1.0}, {"existing": 1.0, "load": 2.25}),
        ("skip", True, None, None),
    ],
)
def test_timed_substep_accumulates_totals_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    *,
    enabled: bool,
    totals: dict[str, float] | None,
    expected: dict[str, float] | None,
) -> None:
    """Timed substeps accumulate elapsed time into the provided totals mapping when enabled."""
    # Arrange
    perf_values = iter([10.0, 12.25])
    monkeypatch.setattr(week_rows.time, "perf_counter", lambda: next(perf_values))

    # Act
    with week_rows._timed_substep(label, enabled=enabled, totals=totals):
        pass

    # Assert
    assert totals == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("label", "enabled", "expected"),
    [("disabled", False, []), ("enabled", True, ["enabled: 1 rows, 2 cols"])],
)
def test_log_df_stats_respects_debug_flag(
    monkeypatch: pytest.MonkeyPatch, label: str, *, enabled: bool, expected: list[str]
) -> None:
    """DataFrame stats should only be logged when debug output is enabled."""
    # Arrange
    messages: list[str] = []
    monkeypatch.setattr(
        data_collection.log,
        "debug",
        lambda message, *args: messages.append(message % args if args else message),
    )
    df = pl.DataFrame({"a": [1], "b": [2]})

    # Act
    data_collection._log_df_stats(label, df, enabled=enabled)

    # Assert
    assert messages == expected


def test_main_orchestrates_collection_and_output_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main should resolve config, collect data, and write all expected output datasets."""
    # Arrange
    config = data_collection.DataCollectionConfig(
        enable_timing=True,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=2023,
        max_season=2024,
        data_dir=tmp_path / "configured",
    )
    all_data = pl.DataFrame({"game_id": ["g1"], "season": [2024], "week": [3]})
    no_diff = pl.DataFrame({"game_id": ["g1"]})
    completed = pl.DataFrame({"game_id": ["g1"]})
    upcoming = pl.DataFrame({"game_id": ["g2"]})
    saved: list[tuple[pl.DataFrame, str, Path | None]] = []
    timed_labels: list[tuple[str, bool]] = []
    info_messages: list[str] = []

    @contextmanager
    def fake_timed_step(label: str, *, enabled: bool):
        """Record timed-step usage and behave like a no-op context manager."""
        timed_labels.append((label, enabled))
        yield

    monkeypatch.setattr(data_collection, "_resolve_config", lambda _argv: config)
    monkeypatch.setattr(
        data_collection,
        "_configure_logging",
        lambda *, enable_debug: timed_labels.append(("debug", enable_debug)),
    )
    monkeypatch.setattr(clock, "nfl_week", lambda _today: 3)
    monkeypatch.setattr(data_collection, "_resolve_seasons", lambda start, end: [start, end])
    monkeypatch.setattr(
        data_collection,
        "collect_all_data",
        lambda seasons, config, records: all_data,
    )
    monkeypatch.setattr(data_collection, "_log_df_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_collection, "_timed_step", fake_timed_step)
    monkeypatch.setattr(
        data_collection,
        "save_dataframe",
        lambda df, name, data_dir=None: saved.append((df, name, data_dir)),
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

    # Today is a fixed in-season date.
    monkeypatch.setattr(clock, "local_today", lambda: date(2024, 9, 18))

    # Act
    data_collection.main([])

    # Assert
    assert timed_labels[0] == ("debug", False)
    assert ("collect_all_data", True) in timed_labels
    assert [name for _df, name, _dir in saved] == [
        "all_data_ml",
        "all_data",
        "completed_games_ml",
        "completed_games",
        "strength_snapshots",
        "predict/week_03_games_to_predict",
    ]
    assert any(message == "Data collection complete." for message in info_messages)
    assert {data_dir for _df, _name, data_dir in saved} == {config.data_dir}


def _records_frame() -> pl.DataFrame:
    """Return one team's unprefixed record columns."""
    base_cols = [
        col[len("away_") :] for col in constants.RECORD_FEATURE_COLUMNS if col.startswith("away_")
    ]
    return pl.DataFrame({"team_abbr": ["AAA"], **{col: [1] for col in base_cols}})


def test_prefix_team_records_prefixes_the_side() -> None:
    """Team records are prefixed correctly for the away side."""
    # Arrange
    records_df = _records_frame()

    # Act
    away = week_rows._prefix_team_records(records_df, "away")

    # Assert
    assert "away_abbr" in away.columns
    assert "away_wins" in away.columns


def test_prefix_team_records_rejects_an_invalid_side() -> None:
    """A side other than away or home is refused."""
    # Arrange
    records_df = _records_frame()

    # Act & Assert
    with pytest.raises(ValueError, match="team_side must be 'away' or 'home'"):
        week_rows._prefix_team_records(records_df, "bad")


def test_resolve_seasons_rejects_pre_nflreadpy() -> None:
    """min_season before nflreadpy availability raises a ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="min_season must be >="):
        data_collection._resolve_seasons(constants.NFLREADPY_MIN_SEASON - 1, 2000)


def test_merge_team_rankings_week_specific() -> None:
    """TeamRankings are merged correctly for the given week."""
    # Arrange
    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [3, 3],
            "predictive_rating": [1.0, 2.0],
        }
    )

    # Act
    out = week_rows._merge_team_rankings(merged, 2023, 3, week_rows.TeamRankingsFrames(tr_df, None))

    # Assert
    assert "away_predictive_rating" in out.columns
    assert "home_predictive_rating" in out.columns


def test_merge_team_rankings_week1_prev() -> None:
    """For week 1, previous season's TeamRankings are merged."""
    # Arrange
    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    prev_tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "AAA", "BBB"],
            "week": [1, 2, 2],
            "predictive_rating": [1.0, 3.0, 2.5],
        }
    )

    # Act
    out = week_rows._merge_team_rankings(
        merged, 2023, 1, week_rows.TeamRankingsFrames(None, prev_tr_df)
    )

    # Assert
    assert out["away_predictive_rating"][0] == 3.0
    assert out["home_predictive_rating"][0] == 2.5


def test_save_and_load_dataframe(tmp_path: Path, monkeypatch) -> None:
    """DataFrame is saved and loaded correctly."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    df = pl.DataFrame({"a": [1], "b": [2]})
    data_collection.save_dataframe(df, "unit_test")

    # Act
    loaded = data_collection.load_dataframe("unit_test")

    # Assert
    assert loaded is not None
    assert loaded.height == 1


def test_save_dataframe_uses_an_explicit_data_dir(tmp_path: Path, monkeypatch) -> None:
    """An explicit data directory wins over the packaged default path when saving."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path / "unused")
    target = tmp_path / "configured"
    df = pl.DataFrame({"a": [1], "b": [2]})

    # Act
    data_collection.save_dataframe(df, "unit_test", target)

    # Assert
    assert (target / "unit_test.csv").is_file()
    assert not (tmp_path / "unused").exists()


def test_load_dataframe_uses_an_explicit_data_dir(tmp_path: Path, monkeypatch) -> None:
    """An explicit data directory wins over the packaged default path when loading."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path / "unused")
    target = tmp_path / "configured"
    data_collection.save_dataframe(pl.DataFrame({"a": [1], "b": [2]}), "unit_test", target)

    # Act
    loaded = data_collection.load_dataframe("unit_test", target)

    # Assert
    assert loaded is not None
    assert loaded.height == 1


def test_load_dataframe_types_a_column_first_filled_after_a_hundred_rows(tmp_path: Path) -> None:
    """A column that is empty for the first 150 rows still loads with its numeric type."""
    # Arrange
    rows = 160
    pl.DataFrame(
        {"season": [2020] * rows, "late": [None] * 150 + [1.5] * (rows - 150)},
        schema={"season": pl.Int64, "late": pl.Float64},
    ).write_csv(tmp_path / "unit_test.csv")

    # Act
    loaded = data_collection.load_dataframe("unit_test", tmp_path)

    # Assert
    assert loaded is not None
    assert loaded.schema["late"] == pl.Float64
    assert loaded["late"].drop_nulls().to_list() == [1.5] * (rows - 150)


def test_load_dataframe_missing(tmp_path: Path, monkeypatch) -> None:
    """Loading missing DataFrame returns None."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    # Act
    loaded = data_collection.load_dataframe("missing")

    # Assert
    assert loaded is None


def test_determine_nfl_week_before_the_season_is_week_one() -> None:
    """A summer date before the season resolves to week 1."""
    # Act
    week = clock.nfl_week(date(2024, 7, 1))

    # Assert
    assert week == 1


def test_determine_nfl_week_in_the_offseason_stays_in_range() -> None:
    """A late-winter date resolves to a valid week number."""
    # Act
    week = clock.nfl_week(date(2024, 2, 1))

    # Assert
    assert 1 <= week <= 22


def _no_play_by_play(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve an empty play-by-play frame, so a collection test never downloads or caches one."""
    monkeypatch.setattr(
        polars_utils, "load_pbp", lambda *_args, **_kwargs: loaders._empty_pbp_frame()
    )


def _stored_lines_config(seasons: list[int]) -> data_collection.DataCollectionConfig:
    """Return the default collection config over ``seasons`` with the stored lines.

    The collection tests stub the stored-line fills; the pick-time source would download
    nfelo's lines and the earlier seasons' schedules.
    """
    return dataclasses.replace(
        data_collection._default_config(seasons), line_source=constants.LINE_SOURCE_STORED
    )


def test_collect_all_data_minimal(monkeypatch) -> None:
    """Data collection works end-to-end for minimal data."""
    # Arrange
    _no_play_by_play(monkeypatch)
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
            "opponent_abbr": ["BBB"],
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

    # Act
    combined = data_collection.collect_all_data([season], config=_stored_lines_config([season]))

    # Assert
    assert combined.height == 1
    assert combined["game_id"][0] == "game1"


def test_collect_all_data_reuses_team_rankings_cache(monkeypatch) -> None:
    """TeamRankings loads once per season within a collection run."""
    # Arrange
    _no_play_by_play(monkeypatch)
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
            "opponent_abbr": ["BBB"],
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
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (season_two, 1))
    monkeypatch.setattr(polars_utils, "load_team_rankings", fake_load_team_rankings)
    monkeypatch.setattr(data_collection, "process_season", fake_process_season)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    # Act
    combined = data_collection.collect_all_data(
        [season_one, season_two], config=_stored_lines_config([season_one, season_two])
    )

    # Assert
    assert combined.height == 2
    assert tr_calls == {}


def test_process_week_fallback(monkeypatch) -> None:
    """Week processing falls back gracefully when lookahead/motivation features fail."""
    # Arrange
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
        msg = "incomplete schedule"
        raise ValueError(msg)

    monkeypatch.setattr(polars_utils, "add_lookahead_features", raise_value_error)
    monkeypatch.setattr(polars_utils, "add_motivation_features", raise_value_error)
    monkeypatch.setattr(polars_utils, "get_stats_for_diff", list)
    monkeypatch.setattr(polars_utils, "calculate_stat_differentials", lambda df, _stats: df)

    # Act
    out = week_rows.process_week(
        2023,
        1,
        schedule_df,
        team_stats_df,
        data_collection.SeasonInputs(min_season=2022, elo_df=elo_df, tr_df=tr_df, prev_tr_df=None),
    )

    # Assert
    assert "away_elo_pre" in out.columns
    assert "home_elo_pre" in out.columns
    assert "away_predictive_rating" in out.columns
    assert "home_predictive_rating" in out.columns
    assert "away_wins" in out.columns


def test_collect_all_data_handles_current_min_season_and_empty_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect-all-data should skip prior-season loading when already at the nflreadpy floor."""
    # Arrange
    _no_play_by_play(monkeypatch)
    season = constants.NFLREADPY_MIN_SEASON
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    team_stats_df = pl.DataFrame(
        {"season": [season], "week": [1], "team_abbr": ["AAA"], "opponent_abbr": ["BBB"]}
    )
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
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
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

    # Act
    combined = data_collection.collect_all_data(
        [season],
        config=data_collection.DataCollectionConfig(
            enable_timing=False,
            enable_debug=False,
            force_refresh_nflreadpy=False,
            min_season=season,
            max_season=season,
            line_source=constants.LINE_SOURCE_STORED,
        ),
    )

    # Assert
    assert combined.height == 0
    assert scoring_inputs == [schedule_df]
    assert warnings == [
        "No play-by-play team-game rows available; emitting null count columns.",
        "No ELO ratings loaded",
        "No play-by-play for the ratings; their columns stay null",
    ]


def test_collect_all_data_dedupes_without_date_or_game_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect-all-data should fall back to matchup-based deduping when date/game_id are absent."""
    # Arrange
    _no_play_by_play(monkeypatch)
    season = constants.NFLREADPY_MIN_SEASON
    schedule_df = pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    team_stats_df = pl.DataFrame(
        {"season": [season], "week": [1], "team_abbr": ["AAA"], "opponent_abbr": ["BBB"]}
    )
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
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
    monkeypatch.setattr(
        polars_utils, "load_team_rankings", lambda *_args, **_kwargs: pl.DataFrame()
    )
    monkeypatch.setattr(data_collection, "process_season", lambda *_args, **_kwargs: season_data)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    # Act
    combined = data_collection.collect_all_data(
        [season],
        config=data_collection.DataCollectionConfig(
            enable_timing=False,
            enable_debug=False,
            force_refresh_nflreadpy=False,
            min_season=season,
            max_season=season,
            line_source=constants.LINE_SOURCE_STORED,
        ),
    )

    # Assert
    assert combined.height == 1
    assert combined.columns == season_data.columns


def _recorded_logs(monkeypatch: pytest.MonkeyPatch, *levels: str) -> dict[str, list[str]]:
    """Record the module logger's messages at each named level."""
    recorded: dict[str, list[str]] = {level: [] for level in levels}
    for level, messages in recorded.items():
        monkeypatch.setattr(
            data_collection.log,
            level,
            lambda message, *args, sink=messages: sink.append(message % args if args else message),
        )
    return recorded


def test_process_season_returns_empty_without_schedule_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-season should return empty, with a warning, without schedule data."""
    # Arrange
    logs = _recorded_logs(monkeypatch, "warning")

    # Act
    empty = data_collection.process_season(
        2024,
        pl.DataFrame({"season": [2023], "week": [1]}),
        pl.DataFrame(),
        data_collection.SeasonInputs(min_season=2023),
    )

    # Assert
    assert empty.height == 0
    assert logs["warning"] == ["No schedule data for season 2024"]


def test_process_season_logs_a_timing_summary_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When timing is enabled and work occurs, process-season logs a timing summary."""
    # Arrange
    logs = _recorded_logs(monkeypatch, "info")
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
        inputs: data_collection.SeasonInputs,
    ) -> pl.DataFrame:
        """Record timing usage and return one row per week."""
        if inputs.timing_totals is not None:
            inputs.timing_totals["merge"] = inputs.timing_totals.get("merge", 0.0) + 1.0
        assert inputs.team_elo_trends is trend_df
        assert inputs.qb_trends is qb_df
        assert inputs.team_stat_trends is trend_df
        assert inputs.coach_features is coach_df
        seen.append((season, week))
        return pl.DataFrame({"season": [season], "week": [week]})

    monkeypatch.setattr(data_collection, "process_week", _fake_process_week)

    # Act
    season_out = data_collection.process_season(
        2024,
        schedule_df,
        pl.DataFrame({"season": [2024], "week": [1]}),
        data_collection.SeasonInputs(
            min_season=2023,
            timing_enabled=True,
            elo_df=pl.DataFrame({"season": [2024], "week": [1]}),
        ),
    )

    # Assert
    assert seen == [(2024, 2), (2024, 3)]
    assert season_out.height == 2
    assert any(message.startswith("Timing summary season 2024:") for message in logs["info"])


def _one_game_schedule(season: int, week: int) -> pl.DataFrame:
    """Return a one-game schedule for the given week."""
    return pl.DataFrame(
        {"season": [season], "week": [week], "away_abbr": ["AAA"], "home_abbr": ["BBB"]}
    )


def test_process_week_skips_week_one_of_the_first_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Week 1 of the run's first season has no history and is skipped with a message."""
    # Arrange
    logs = _recorded_logs(monkeypatch, "info")

    # Act
    rows = week_rows.process_week(
        2023,
        1,
        _one_game_schedule(2023, 1),
        pl.DataFrame(),
        data_collection.SeasonInputs(min_season=2023),
    )

    # Assert
    assert rows.height == 0
    assert any("Skipping season 2023 week 1" in message for message in logs["info"])


def test_process_week_returns_empty_for_a_week_missing_from_the_schedule() -> None:
    """A week the schedule does not hold yields no rows."""
    # Act
    no_week = week_rows.process_week(
        2024,
        2,
        _one_game_schedule(2023, 1),
        pl.DataFrame(),
        data_collection.SeasonInputs(min_season=2023),
    )

    # Assert
    assert no_week.height == 0


def test_process_week_returns_empty_without_aggregated_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A week whose team stats aggregate to nothing yields no rows and says why."""
    # Arrange
    logs = _recorded_logs(monkeypatch, "debug")
    monkeypatch.setattr(
        polars_utils,
        "aggregate_team_stats_to_week",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    # Act
    no_stats = week_rows.process_week(
        2024,
        2,
        _one_game_schedule(2024, 2),
        pl.DataFrame({"season": [2024], "week": [1], "team_abbr": ["AAA"]}),
        data_collection.SeasonInputs(min_season=2023),
    )

    # Assert
    assert no_stats.height == 0
    assert any("No aggregated stats available" in message for message in logs["debug"])


def _merged_game() -> pl.DataFrame:
    """Return one merged game row with both teams and quarterbacks."""
    return pl.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_qb": ["QB1"],
            "home_qb": ["QB2"],
        }
    )


def _team_trends() -> pl.DataFrame:
    """Return week-2 trend values for both teams."""
    return pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "team_abbr": ["AAA", "BBB"],
            "elo_trend": [1.0, 2.0],
        }
    )


@pytest.mark.parametrize(
    "trends",
    [None, pl.DataFrame(), pl.DataFrame({"season": [2024]})],
    ids=["none", "empty", "bare"],
)
def test_merge_team_trends_without_usable_trends_is_a_no_op(trends: pl.DataFrame | None) -> None:
    """Local merge helpers support no-op guards."""
    # Arrange
    merged = _merged_game()

    # Act
    out = week_rows._merge_team_trends(merged, trends)

    # Assert
    assert out.equals(merged)


def test_merge_team_trends_merges_both_sides() -> None:
    """Team trends join onto the away and home sides."""
    # Arrange
    merged = _merged_game()

    # Act
    team_out = week_rows._merge_team_trends(merged, _team_trends())

    # Assert
    assert team_out["away_elo_trend"][0] == 1.0
    assert team_out["home_elo_trend"][0] == 2.0


def test_merge_qb_trends_without_quarterback_columns_is_a_no_op() -> None:
    """Rows without quarterback names come back unchanged."""
    # Arrange
    without_qbs = _merged_game().drop(["away_qb", "home_qb"])

    # Act
    out = week_rows._merge_qb_trends(without_qbs, _team_trends())

    # Assert
    assert out.equals(without_qbs)


def test_merge_qb_trends_merges_both_quarterbacks() -> None:
    """Quarterback trends join onto each side's expected starter."""
    # Arrange
    qb_trends = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "qb_name": ["QB1", "QB2"],
            "qb_form": [0.1, 0.2],
        }
    )

    # Act
    qb_out = week_rows._merge_qb_trends(_merged_game(), qb_trends)

    # Assert
    assert qb_out["away_qb_form"][0] == pytest.approx(0.1)
    assert qb_out["home_qb_form"][0] == pytest.approx(0.2)


def test_merge_coach_features_merges_both_sides() -> None:
    """Coach features join onto the away and home sides."""
    # Arrange
    coach_df = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [2, 2],
            "team_abbr": ["AAA", "BBB"],
            "coach_name": ["A", "B"],
            "coach_games_prior": [3, 4],
        }
    )

    # Act
    coach_out = week_rows._merge_coach_features(_merged_game(), coach_df)

    # Assert
    assert coach_out["away_coach_games_prior"][0] == 3
    assert coach_out["home_coach_games_prior"][0] == 4


def test_merge_team_rankings_uses_the_latest_week_in_the_playoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A playoff week takes the latest regular-season TeamRankings values."""
    # Arrange
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

    # Act
    playoff_out = week_rows._merge_team_rankings(
        _merged_game().select(["away_abbr", "home_abbr"]),
        2024,
        20,
        week_rows.TeamRankingsFrames(playoff_tr, None),
    )

    # Assert
    assert "away_predictive_rating" in playoff_out.columns


def test_merge_team_rankings_without_expected_columns_warns_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TeamRankings rows without the expected columns leave the games alone, with a warning."""
    # Arrange
    monkeypatch.setattr(polars_utils, "get_tr_columns", lambda: ["predictive_rating"])
    logs = _recorded_logs(monkeypatch, "warning")
    no_expected_cols = pl.DataFrame({"team_abbr": ["AAA"], "week": [2], "other": [1.0]})

    # Act
    unchanged = week_rows._merge_team_rankings(
        _merged_game().select(["away_abbr", "home_abbr"]),
        2024,
        2,
        week_rows.TeamRankingsFrames(no_expected_cols, None),
    )

    # Assert
    assert unchanged.columns == ["away_abbr", "home_abbr"]
    assert any("TR data has no expected columns" in warning for warning in logs["warning"])


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


def _one_team_penalties(season: int) -> pl.DataFrame:
    """Return a game's penalties, every one charged to the visitor, as nflverse writes 2001's."""
    half = constants.ONE_TEAM_GAME_MIN_VALUES // 2
    return pl.DataFrame(
        [
            {
                "game_id": f"{season}_01_WAS_JAX",
                "season": season,
                "week": 1,
                "season_type": "REG",
                "posteam": offense,
                "defteam": defense,
                "play_type": "no_play",
                "penalty": 1.0,
                "penalty_yards": 5.0,
                "penalty_team": "WSH",
                "penalty_player_id": f"{side}_{index}",
            }
            for side, offense, defense in (("was", "WSH", "JAX"), ("jax", "JAX", "WSH"))
            for index in range(half)
        ]
    )


def test_load_team_game_frame_repairs_player_teams_in_one_team_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Penalties in a game nflverse credits to one team go to the team of each penalized player."""
    # Arrange
    half = constants.ONE_TEAM_GAME_MIN_VALUES // 2
    rosters = pl.DataFrame(
        {
            "season": [2001] * (2 * half),
            "team": ["WAS"] * half + ["JAX"] * half,
            "gsis_id": [f"was_{i}" for i in range(half)] + [f"jax_{i}" for i in range(half)],
        }
    ).with_columns(pl.col("season").cast(pl.Int32))
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_a, **_k: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_pbp", lambda *_a, **_k: _one_team_penalties(2001))
    monkeypatch.setattr(nflverse_sources.nfl, "load_rosters", lambda **_k: rosters)
    monkeypatch.setattr(data_collection, "_build_team_game_frame", lambda stats, *_a: stats)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _sched: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)

    # Act
    team_stats, _plays = data_collection._load_team_game_frame(
        [2001], pl.DataFrame(), _stored_lines_config([2001]), 2026
    )

    # Assert
    penalties = team_stats.select("team_abbr", "penalties").sort("team_abbr").rows()
    assert penalties == [("JAX", float(half)), ("WSH", float(half))]


def _empty_upstream(monkeypatch: pytest.MonkeyPatch, requested: list[list[int]]) -> None:
    """Serve no upstream rows to ``_load_sources`` and record each play-by-play request."""

    def fake_load_pbp(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        """Record the requested seasons and return no plays."""
        requested.append(list(seasons))
        return loaders._empty_pbp_frame()

    monkeypatch.setattr(polars_utils, "load_pbp", fake_load_pbp)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (2026, 1))
    monkeypatch.setattr(polars_utils, "load_schedule", lambda *_a, **_k: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_a, **_k: pl.DataFrame())
    monkeypatch.setattr(data_collection, "_build_team_game_frame", lambda stats, *_a: stats)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _sched: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)


def test_load_sources_records_every_play_by_play_season_it_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sources name the seasons the play-by-play load asked for, with or without plays.

    The ratings' own requests are covered in ``test_ratings_build_sources.py``.
    """
    # Arrange
    requested: list[list[int]] = []
    _empty_upstream(monkeypatch, requested)
    monkeypatch.setattr(ratings_build, "load_ratings_sources", lambda *_a, **_k: None)

    # Act
    sources = data_collection._load_sources([2025, 2026], _stored_lines_config([2025, 2026]))

    # Assert
    assert requested == [[2024, 2025, 2026]]
    assert sources.pbp_seasons == (2024, 2025, 2026)


def test_load_sources_keeps_the_ratings_player_sources_for_the_quarterback_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The players file and weekly rosters the ratings load are the ones every identity reads."""
    # Arrange
    _empty_upstream(monkeypatch, [])
    loaded = ratings_build.LoadedSources(
        plays=pl.DataFrame({"season": [2025]}),
        schedule=pl.DataFrame(),
        player_stats=pl.DataFrame(),
        snap_counts=pl.DataFrame(),
        players=pl.DataFrame({"gsis_id": ["00-1"]}),
        weekly_rosters=pl.DataFrame({"season": [2025], "gsis_id": ["00-1"]}),
        starter_identity=pl.DataFrame(),
    )
    monkeypatch.setattr(ratings_build, "load_ratings_sources", lambda *_a, **_k: loaded)
    monkeypatch.setattr(ratings_build, "ratings_inputs", lambda sources: sources)
    monkeypatch.setattr(ratings_build, "RatingsBuild", lambda inputs, **_kwargs: inputs)

    # Act
    sources = data_collection._load_sources([2025, 2026], _stored_lines_config([2025, 2026]))

    # Assert
    assert sources.qb_identity_sources is not None
    assert sources.qb_identity_sources.players is loaded.players
    assert sources.qb_identity_sources.weekly_rosters is loaded.weekly_rosters


def test_join_pbp_team_game_stats_attaches_counts_by_team_week() -> None:
    """Play-by-play counts join onto the matching team-week rows."""
    # Arrange
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

    # Act
    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), pbp_team_games)

    # Assert
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
    # Act
    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), pl.DataFrame())

    # Assert
    for col in constants.PBP_COUNT_COLUMNS:
        assert col in out.columns, f"{col} missing from the invariant schema"
        assert out.select(pl.col(col).is_null().all()).item() is True
    assert out.height == 2


def test_join_pbp_team_game_stats_never_multiplies_rows() -> None:
    """Duplicate play-by-play keys are collapsed instead of multiplying team-stat rows.

    A malformed source that produces two rows for one `(season, week, team_abbr)` would
    otherwise silently inflate the team-stats frame and corrupt every downstream mean.
    """
    # Arrange
    duplicated = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "AAA"],
            "opponent_abbr": ["BBB", ""],
            "offensive_snaps": [64, 0],
        }
    )

    # Act
    out = data_collection._join_pbp_team_game_stats(_team_stats_stub(), duplicated)

    # Assert
    assert out.height == 2, "the join must not multiply team-stat rows"
    assert out.filter(pl.col("team_abbr") == "AAA")["offensive_snaps"][0] == 64


def test_overlay_pbp_team_box_scores_prefers_pbp_and_keeps_nflverse_fallbacks() -> None:
    """The PBP team-stat source overrides derivable columns but keeps nflverse-only values."""
    # Arrange
    team_stats = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "season_type": ["REG"],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "pass_yards": [250.0],
            "penalties": [7.0],
            "def_qb_hits": [4.0],
        }
    )
    pbp_box = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "season_type": ["REG"],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "pass_yards": [240.0],
            "penalties": [None],
        }
    )

    # Act
    out = data_collection._overlay_pbp_team_box_scores(team_stats, pbp_box)
    row = out.row(0, named=True)

    # Assert
    assert row["pass_yards"] == pytest.approx(240.0)
    assert row["penalties"] == pytest.approx(7.0)
    assert row["def_qb_hits"] == pytest.approx(4.0)


def test_overlay_pbp_team_box_scores_keeps_pbp_only_rows() -> None:
    """A team-game found only in play-by-play survives the overlay for later skeleton joins."""
    # Arrange
    team_stats = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "season_type": ["REG"],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "pass_yards": [250.0],
        }
    )
    pbp_box = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "pass_yards": [240.0, 180.0],
        }
    )

    # Act
    out = data_collection._overlay_pbp_team_box_scores(team_stats, pbp_box)

    # Assert
    assert out.height == 2
    assert out.filter(pl.col("team_abbr") == "BBB")["pass_yards"][0] == pytest.approx(180.0)


def test_merge_team_rankings_skips_scraped_situational_columns_when_pbp_is_selected() -> None:
    """The pbp situational source still joins ratings but leaves existing values alone."""
    # Arrange
    merged = pl.DataFrame(
        {
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_third_down_pct": [0.40],
            "home_third_down_pct": [0.55],
        }
    )
    tr_df = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "week": [2, 2],
            "predictive_rating": [1.2, 2.4],
            "third_down_pct": [0.10, 0.20],
        }
    )

    # Act
    out = week_rows._merge_team_rankings(
        merged, 2024, 2, week_rows.TeamRankingsFrames(tr_df, None), tr_stats_source="pbp"
    )

    # Assert
    assert out["away_predictive_rating"][0] == pytest.approx(1.2)
    assert out["home_predictive_rating"][0] == pytest.approx(2.4)
    assert out["away_third_down_pct"][0] == pytest.approx(0.40)
    assert out["home_third_down_pct"][0] == pytest.approx(0.55)


def test_build_team_game_frame_recovers_a_team_game_the_stats_source_misses() -> None:
    """A game the stat source omits keeps its play-by-play counts and its game count."""
    # Arrange
    schedule = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_type": ["REG"],
            "home_abbr": ["AAA"],
            "away_abbr": ["BBB"],
            "home_score": [21],
            "away_score": [17],
        }
    )
    team_stats = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team_abbr": ["BBB"],
            "opponent_abbr": ["AAA"],
            "pass_yards": [180.0],
        }
    )
    pbp_team_games = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "offensive_snaps": [64, 58],
        }
    )

    # Act
    out = data_collection._build_team_game_frame(team_stats, schedule, pbp_team_games)

    # Assert
    assert out.height == 2
    aaa = out.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    assert aaa["offensive_snaps"] == 64
    assert aaa["pass_yards"] is None
    assert aaa["opponent_abbr"] == "BBB"


def test_build_team_game_frame_keeps_points_and_plays_on_a_repaired_row() -> None:
    """Repairing a two-team box score leaves the per-team sources and the schema alone."""
    # Arrange
    schedule = pl.DataFrame(
        {
            "season": [2001],
            "week": [1],
            "game_type": ["REG"],
            "home_abbr": ["AAA"],
            "away_abbr": ["BBB"],
            "home_score": [20],
            "away_score": [10],
        }
    )
    # Only BBB has a row, and it carries both teams' yards.
    team_stats = pl.DataFrame(
        {
            "season": [2001],
            "week": [1],
            "team_abbr": ["BBB"],
            "opponent_abbr": ["AAA"],
            "total_yards": [580.0],
        }
    )
    pbp_team_games = pl.DataFrame(
        {
            "season": [2001, 2001],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "offensive_snaps": [60.0, 55.0],
        }
    )

    # Act
    frame = data_collection._build_team_game_frame(team_stats, schedule, pbp_team_games)
    frame = polars_utils.add_scoring_data_to_team_stats(frame, schedule)
    mirrored = polars_utils.add_per_game_opponent_stats(frame)

    # Assert
    assert set(team_stats.columns) <= set(frame.columns)
    bbb = frame.filter(pl.col("team_abbr") == "BBB").row(0, named=True)
    aaa = frame.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    # The two-team box score is gone from the row that could not own it.
    assert bbb["total_yards"] is None
    assert aaa["total_yards"] is None
    # Per-team sources are untouched.
    assert bbb["points_scored"] == 10
    assert aaa["points_scored"] == 20
    assert bbb["offensive_snaps"] == 55.0
    assert aaa["offensive_snaps"] == 60.0
    # The mirror publishes a null rather than the two-team total.
    assert mirrored.filter(pl.col("team_abbr") == "AAA")["opponent_total_yards"][0] is None


def _two_week_sources() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return a two-game schedule, box scores that miss AAA's first game, and full play-by-play."""
    schedule = pl.DataFrame(
        {
            "season": [2001, 2001],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "home_abbr": ["AAA", "BBB"],
            "away_abbr": ["BBB", "AAA"],
            "home_score": [20, 17],
            "away_score": [10, 14],
        }
    )
    # The box-score source covers only the second game for AAA.
    team_stats = pl.DataFrame(
        {
            "season": [2001, 2001, 2001],
            "week": [1, 2, 2],
            "team_abbr": ["BBB", "BBB", "AAA"],
            "opponent_abbr": ["AAA", "AAA", "BBB"],
            "penalties": [5.0, 7.0, 9.0],
            "penalty_yards": [40.0, 60.0, 81.0],
        }
    )
    # Play-by-play covers both games for both teams.
    pbp_team_games = pl.DataFrame(
        {
            "season": [2001, 2001, 2001, 2001],
            "week": [1, 1, 2, 2],
            "team_abbr": ["AAA", "BBB", "AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA", "BBB", "AAA"],
            "offensive_snaps": [60.0, 55.0, 70.0, 65.0],
            "pass_epa_sum": [6.0, 1.0, 7.0, 2.0],
        }
    )
    return schedule, team_stats, pbp_team_games


def _aggregated_week_three() -> pl.DataFrame:
    """Return the season-to-date aggregates entering week 3 from ``_two_week_sources``."""
    schedule, team_stats, pbp_team_games = _two_week_sources()
    frame = data_collection._build_team_game_frame(team_stats, schedule, pbp_team_games)
    frame = polars_utils.add_scoring_data_to_team_stats(frame, schedule)
    return polars_utils.aggregate_team_stats_to_week(frame, 3, 2001)


def test_season_to_date_denominators_follow_each_stat_source() -> None:
    """Each aggregated value divides by the games its own source covers.

    A team-game the box-score source does not cover still counts as a game played and
    still carries its points and its play-by-play counts. The box-score average stays an
    average over the games that have box scores, so it is not diluted by the games the
    source is missing, while points and play-by-play rates use every scheduled game.
    """
    # Arrange
    schedule, team_stats, pbp_team_games = _two_week_sources()
    frame = data_collection._build_team_game_frame(team_stats, schedule, pbp_team_games)
    frame = polars_utils.add_scoring_data_to_team_stats(frame, schedule)

    # Act
    agg = polars_utils.aggregate_team_stats_to_week(frame, 3, 2001)

    # Assert
    aaa = agg.filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    # The game count comes from the schedule, not from the box-score coverage.
    assert aaa["games_played"] == 2
    # A box-score average is the average over its one covered game, never halved.
    assert aaa["penalties"] == pytest.approx(9.0)
    assert aaa["penalty_yards"] == pytest.approx(81.0)
    assert aaa["penalty_yards_per_penalty"] == pytest.approx(81.0 / 9.0)
    # Points come from the schedule, so both games count.
    assert aaa["points_scored"] == pytest.approx((20.0 + 14.0) / 2.0)
    # A play-by-play rate is the ratio of sums over both games.
    assert aaa["off_pass_epa_per_snap"] == pytest.approx((6.0 + 7.0) / (60.0 + 70.0))


def test_prior_blend_weighs_the_in_season_side_by_the_scheduled_game_count() -> None:
    """The early-season blend weighs the in-season side by the scheduled game count.

    That holds even for a column whose average rests on fewer games than that.
    """
    # Arrange
    agg = _aggregated_week_three()
    prior = pl.DataFrame({"team_abbr": ["AAA"], "penalties": [3.0], "penalty_yards": [30.0]})
    blend_games = 4.0

    # Act
    blended = polars_utils.blend_with_prior_stats(
        agg.filter(pl.col("team_abbr") == "AAA"), prior, blend_games
    )

    # Assert
    weight = 2.0 / (2.0 + blend_games)
    assert blended["penalties"][0] == pytest.approx(weight * 9.0 + (1.0 - weight) * 3.0)
