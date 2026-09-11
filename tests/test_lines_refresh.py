"""Tests for the lines-only dataset refresh."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nfl_predictor import lines_refresh

DATASET_COLUMNS = [
    "game_id",
    "season",
    "week",
    "away_abbr",
    "home_abbr",
    "total_line",
    "away_spread",
    "home_spread",
    "away_moneyline",
    "home_moneyline",
    "away_elo_pre",
]

SCHEDULE_ROWS: list[dict[str, Any]] = [
    {
        "game_id": "2026_02_DEN_KC",
        "season": 2026,
        "week": 2,
        "away_abbr": "DEN",
        "home_abbr": "KC",
        "total_line": 45.5,
        "away_spread": 3.5,
        "home_spread": -3.5,
        "away_moneyline": 150,
        "home_moneyline": -180,
    },
    {
        "game_id": "2026_02_BUF_NYJ",
        "season": 2026,
        "week": 2,
        "away_abbr": "BUF",
        "home_abbr": "NYJ",
        "total_line": 41.0,
        "away_spread": -6.5,
        "home_spread": 6.5,
        "away_moneyline": None,
        "home_moneyline": None,
    },
]

DATASET_ROWS: list[dict[str, Any]] = [
    {
        "game_id": "2026_02_DEN_KC",
        "season": 2026,
        "week": 2,
        "away_abbr": "DEN",
        "home_abbr": "KC",
        "total_line": 44.0,
        "away_spread": 2.5,
        "home_spread": -2.5,
        "away_moneyline": 120,
        "home_moneyline": -140,
        "away_elo_pre": 1500.0,
    },
    {
        "game_id": "2026_02_BUF_NYJ",
        "season": 2026,
        "week": 2,
        "away_abbr": "BUF",
        "home_abbr": "NYJ",
        "total_line": None,
        "away_spread": None,
        "home_spread": None,
        "away_moneyline": None,
        "home_moneyline": None,
        "away_elo_pre": 1490.0,
    },
    {
        "game_id": "2026_03_GB_MIN",
        "season": 2026,
        "week": 3,
        "away_abbr": "GB",
        "home_abbr": "MIN",
        "total_line": 46.5,
        "away_spread": 1.0,
        "home_spread": -1.0,
        "away_moneyline": 105,
        "home_moneyline": -125,
        "away_elo_pre": 1520.0,
    },
    {
        # A stale row carrying a game id that also appears in the refreshed season: the season
        # guard must leave it alone.
        "game_id": "2026_02_DEN_KC",
        "season": 2025,
        "week": 2,
        "away_abbr": "DEN",
        "home_abbr": "KC",
        "total_line": 40.0,
        "away_spread": 7.0,
        "home_spread": -7.0,
        "away_moneyline": 250,
        "home_moneyline": -300,
        "away_elo_pre": 1400.0,
    },
]


def _schedule_frame() -> pl.DataFrame:
    """Return a schedule frame shaped like ``load_schedule`` output."""
    return pl.DataFrame(SCHEDULE_ROWS, schema_overrides={"away_moneyline": pl.Int64})


def _dataset_frame() -> pl.DataFrame:
    """Return a dataset frame with the project's line columns."""
    return pl.DataFrame(DATASET_ROWS).select(DATASET_COLUMNS)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write the three refreshable datasets plus an untouched completed-games file."""
    frame = _dataset_frame()
    (tmp_path / "predict").mkdir(parents=True, exist_ok=True)
    frame.write_csv(tmp_path / "all_data_ml.csv")
    frame.write_csv(tmp_path / "all_data.csv")
    frame.filter(pl.col("week") == 2).write_csv(
        tmp_path / "predict" / "week_02_games_to_predict.csv"
    )
    frame.write_csv(tmp_path / "completed_games_ml.csv")
    return tmp_path


@pytest.fixture
def fake_schedule(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``load_schedule`` with a stub and record how it was called."""
    calls: list[dict[str, Any]] = []

    def _load_schedule(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        calls.append({"seasons": seasons, **kwargs})
        return _schedule_frame()

    monkeypatch.setattr(lines_refresh.polars_utils, "load_schedule", _load_schedule)
    return calls


def _read(path: Path) -> pl.DataFrame:
    """Read a written dataset back."""
    return pl.read_csv(path)


def test_refresh_lines_updates_only_line_columns(data_dir: Path, fake_schedule: list) -> None:
    """Line columns take the schedule's values; every other column and the row order survive."""
    lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    frame = _read(data_dir / "all_data_ml.csv")
    assert frame.columns == DATASET_COLUMNS
    assert frame["game_id"].to_list() == [row["game_id"] for row in DATASET_ROWS]
    assert frame["away_elo_pre"].to_list() == [row["away_elo_pre"] for row in DATASET_ROWS]
    first = frame.row(0, named=True)
    assert first["total_line"] == 45.5
    assert first["home_spread"] == -3.5
    assert first["away_moneyline"] == 150


def test_refresh_lines_fills_missing_moneylines_from_spreads(
    data_dir: Path, fake_schedule: list
) -> None:
    """A schedule row with spreads but no moneylines is completed before the write."""
    lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    row = _read(data_dir / "all_data_ml.csv").row(1, named=True)
    assert row["home_spread"] == 6.5
    assert row["home_moneyline"] is not None
    assert row["away_moneyline"] is not None
    assert row["away_moneyline"] < 0 < row["home_moneyline"]


def test_refresh_lines_leaves_unmatched_rows_alone(data_dir: Path, fake_schedule: list) -> None:
    """Rows the schedule does not cover, and rows from another season, keep their lines."""
    lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    frame = _read(data_dir / "all_data_ml.csv")
    other_week = frame.filter(pl.col("game_id") == "2026_03_GB_MIN").row(0, named=True)
    assert other_week["total_line"] == 46.5
    other_season = frame.filter(pl.col("season") == 2025).row(0, named=True)
    assert other_season["total_line"] == 40.0
    assert other_season["home_moneyline"] == -300


def test_refresh_lines_touches_every_target_file(data_dir: Path, fake_schedule: list) -> None:
    """The week file and both all-data datasets are refreshed; completed games are not."""
    before = (data_dir / "completed_games_ml.csv").read_text(encoding="utf-8")
    result = lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    assert {path.name for path in result.written_paths} == {
        "all_data_ml.csv",
        "all_data.csv",
        "week_02_games_to_predict.csv",
    }
    assert _read(data_dir / "predict" / "week_02_games_to_predict.csv")["total_line"][0] == 45.5
    assert (data_dir / "completed_games_ml.csv").read_text(encoding="utf-8") == before


def test_refresh_lines_reports_per_file_counts(data_dir: Path, fake_schedule: list) -> None:
    """The result carries matched and changed row counts per file."""
    result = lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    assert result.season == 2026
    assert result.week == 2
    assert result.schedule_games == 2
    by_name = {refresh.path.name: refresh for refresh in result.files}
    assert by_name["all_data_ml.csv"].matched_rows == 2
    assert by_name["all_data_ml.csv"].changed_rows == 2
    assert by_name["all_data_ml.csv"].changed_columns["total_line"] == 2
    assert result.changed_rows == 6


def test_refresh_lines_rewrites_nothing_when_lines_are_unchanged(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schedule matching the datasets leaves every file byte-identical."""
    unchanged = (
        _dataset_frame()
        .filter(pl.col("season") == 2026)
        .select(["game_id", "season", "week", *lines_refresh.LINE_COLUMNS])
    )
    monkeypatch.setattr(
        lines_refresh.polars_utils, "load_schedule", lambda seasons, **kwargs: unchanged
    )
    before = (data_dir / "all_data_ml.csv").read_text(encoding="utf-8")

    result = lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    assert result.changed_rows == 0
    assert result.written_paths == ()
    assert (data_dir / "all_data_ml.csv").read_text(encoding="utf-8") == before


def test_refresh_lines_skips_files_that_do_not_exist(tmp_path: Path, fake_schedule: list) -> None:
    """Missing datasets are reported as skipped rather than raising."""
    _dataset_frame().write_csv(tmp_path / "all_data.csv")

    result = lines_refresh.refresh_lines(2026, 2, data_dir=tmp_path)

    by_name = {refresh.path.name: refresh for refresh in result.files}
    assert by_name["all_data_ml.csv"].exists is False
    assert by_name["all_data.csv"].exists is True


def test_refresh_lines_forces_a_schedule_refresh(data_dir: Path, fake_schedule: list) -> None:
    """The schedule is re-fetched for the target season rather than read from the cache."""
    lines_refresh.refresh_lines(2026, 2, data_dir=data_dir, cache_dir=data_dir / "cache")

    assert fake_schedule == [
        {
            "seasons": [2026],
            "cache_dir": data_dir / "cache",
            "force_refresh": True,
            "current_season": 2026,
        }
    ]


def test_refresh_lines_raises_when_the_schedule_has_no_games(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty schedule is an error, not a silent no-op."""
    monkeypatch.setattr(
        lines_refresh.polars_utils, "load_schedule", lambda seasons, **kwargs: pl.DataFrame()
    )

    with pytest.raises(ValueError, match="no games"):
        lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)


def test_main_refreshes_the_requested_week(
    data_dir: Path, fake_schedule: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI passes its arguments through to :func:`refresh_lines`."""
    exit_code = lines_refresh.main(["--season", "2026", "--week", "2", "--data-dir", str(data_dir)])

    assert exit_code == 0
    assert _read(data_dir / "all_data_ml.csv")["total_line"][0] == 45.5


def test_main_defaults_to_the_current_season_and_week(
    data_dir: Path, fake_schedule: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--season``/``--week`` the CLI uses the project's NFL calendar."""
    monkeypatch.setattr(lines_refresh.polars_utils, "get_current_nfl_week", lambda: (2026, 2))
    monkeypatch.setattr(lines_refresh, "DEFAULT_DATA_DIR", data_dir)

    assert lines_refresh.main([]) == 0
    assert _read(data_dir / "all_data_ml.csv")["total_line"][0] == 45.5


def test_join_keys_prefers_game_id_then_the_matchup() -> None:
    """``game_id`` wins when both sides have it; otherwise the season/week/team key is used."""
    lines = _schedule_frame()
    with_id = _dataset_frame()
    assert lines_refresh.join_keys(with_id, lines) == ["game_id"]
    without_id = with_id.drop("game_id")
    assert lines_refresh.join_keys(without_id, lines) == list(lines_refresh.MATCHUP_COLUMNS)
    assert lines_refresh.join_keys(without_id.drop("away_abbr"), lines) == []


def test_refresh_lines_matches_week_files_that_have_no_game_id(
    data_dir: Path, fake_schedule: list
) -> None:
    """The weekly games-to-predict files are keyed by season, week, and both teams."""
    week_file = data_dir / "predict" / "week_02_games_to_predict.csv"
    _read(week_file).drop("game_id").write_csv(week_file)

    result = lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    by_name = {refresh.path.name: refresh for refresh in result.files}
    assert by_name["week_02_games_to_predict.csv"].matched_rows == 2
    frame = _read(week_file)
    assert "game_id" not in frame.columns
    assert frame["total_line"].to_list() == [45.5, 41.0, 40.0]  # the 2025 row keeps its lines


def test_refresh_lines_skips_a_file_it_cannot_key(data_dir: Path, fake_schedule: list) -> None:
    """A dataset without a usable key is reported as unchanged rather than guessed at."""
    week_file = data_dir / "predict" / "week_02_games_to_predict.csv"
    _read(week_file).drop("game_id", "away_abbr").write_csv(week_file)

    result = lines_refresh.refresh_lines(2026, 2, data_dir=data_dir)

    by_name = {refresh.path.name: refresh for refresh in result.files}
    assert by_name["week_02_games_to_predict.csv"].changed_rows == 0
    assert by_name["week_02_games_to_predict.csv"].exists is True
