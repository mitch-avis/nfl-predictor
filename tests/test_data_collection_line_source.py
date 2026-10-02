"""Tests for the ETL's line-source option: stored lines by default, pick-time lines on request."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import clock, polars_utils
from nfl_predictor.utils.polars import moneyline_map, nfelo_lines

if TYPE_CHECKING:
    from pathlib import Path

_SEASON = 2015


def _schedule(season: int = _SEASON) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"{season}_01_AAA_BBB"],
            "season": pl.Series([season], dtype=pl.Int32),
            "week": pl.Series([1], dtype=pl.Int32),
            "date": [date(season, 9, 13)],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_score": pl.Series([17], dtype=pl.Int32),
            "home_score": pl.Series([20], dtype=pl.Int32),
            "away_spread": [3.0],
            "home_spread": [-3.0],
            "away_moneyline": pl.Series([140], dtype=pl.Int32),
            "home_moneyline": pl.Series([-160], dtype=pl.Int32),
            "total_line": [44.0],
        }
    )


def _price_history() -> pl.DataFrame:
    spreads = [-7.0, -3.0, 3.0, 7.0] * 60
    prices = {-7.0: (-320, 270), -3.0: (-165, 145), 3.0: (145, -165), 7.0: (270, -320)}
    return pl.DataFrame(
        {
            "season": pl.Series([_SEASON - 1] * len(spreads), dtype=pl.Int32),
            "home_spread": spreads,
            "home_moneyline": pl.Series([prices[s][0] for s in spreads], dtype=pl.Int32),
            "away_moneyline": pl.Series([prices[s][1] for s in spreads], dtype=pl.Int32),
        }
    )


def _snapshot() -> nfelo_lines.NfeloLinesSnapshot:
    frame = pl.DataFrame(
        {
            "season": [_SEASON],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "home_spread_open": [-6.0],
            "home_spread_last": [-3.0],
            "total_line_open": pl.Series([None], dtype=pl.Float64),
            "total_line_last": [44.0],
            "home_ml_open": pl.Series([None], dtype=pl.Int64),
            "away_ml_open": pl.Series([None], dtype=pl.Int64),
            "home_ml_last": pl.Series([-160], dtype=pl.Int64),
            "away_ml_last": pl.Series([140], dtype=pl.Int64),
        }
    )
    return nfelo_lines.NfeloLinesSnapshot(frame, "u", "download", "abc", None)


def _patch_sources(monkeypatch: pytest.MonkeyPatch, schedule_calls: list[list[int]]) -> None:
    """Replace every upstream loader with small in-memory frames."""

    def fake_load_schedule(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        schedule_calls.append(list(seasons))
        if seasons == [_SEASON]:
            return _schedule()
        if seasons == [_SEASON - 1]:
            return _schedule(_SEASON - 1)
        return _price_history()

    team_stats = pl.DataFrame(
        {"season": [_SEASON - 1], "week": [1], "team_abbr": ["AAA"], "opponent_abbr": ["BBB"]}
    )
    monkeypatch.setattr(polars_utils, "load_schedule", fake_load_schedule)
    monkeypatch.setattr(polars_utils, "load_team_stats", lambda *_a, **_k: team_stats)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", lambda df, _s: df)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", lambda df: df)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", lambda _seasons: pl.DataFrame())
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (_SEASON + 1, 1))
    monkeypatch.setattr(polars_utils, "load_team_rankings", lambda *_a, **_k: pl.DataFrame())
    monkeypatch.setattr(
        data_collection, "process_season", lambda _season, schedule, *_a, **_k: schedule
    )
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)


def _config(line_source: str) -> data_collection.DataCollectionConfig:
    return data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=_SEASON,
        max_season=_SEASON,
        line_source=line_source,
    )


def test_the_line_source_defaults_to_the_stored_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the option the ETL keeps today's stored lines."""
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    assert data_collection._parse_args([]).line_source == constants.LINE_SOURCE_STORED
    assert data_collection._resolve_config(None).line_source == constants.LINE_SOURCE_STORED


def test_the_line_source_option_accepts_pick_time_only_among_new_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--line-source pick_time`` selects the candidate; an unknown source is refused."""
    monkeypatch.setattr(data_collection, "_default_max_season", lambda: 2025)

    chosen = data_collection._parse_args(["--line-source", "pick_time"])

    assert chosen.line_source == constants.LINE_SOURCE_PICK_TIME
    with pytest.raises(SystemExit):
        data_collection._parse_args(["--line-source", "closing"])


def test_the_stored_source_never_reads_nfelo_and_fills_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path fetches nothing new and calls the line fills exactly as today."""
    schedule_calls: list[list[int]] = []
    _patch_sources(monkeypatch, schedule_calls)
    fills: list[str] = []

    def no_nfelo(**_kwargs: object) -> nfelo_lines.NfeloLinesSnapshot:
        msg = "the stored source must not read nfelo"
        raise AssertionError(msg)

    def future_fill(df: pl.DataFrame) -> pl.DataFrame:
        fills.append("future")
        return df

    def moneyline_fill(df: pl.DataFrame) -> pl.DataFrame:
        fills.append("moneyline")
        return df

    monkeypatch.setattr(nfelo_lines, "load_nfelo_lines", no_nfelo)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", future_fill)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", moneyline_fill)
    metadata: dict[str, object] = {}

    games = data_collection.collect_all_data(
        [_SEASON], config=_config("stored"), market_lines_metadata=metadata
    )

    assert fills == ["future", "moneyline"]
    assert games.select("home_spread", "home_moneyline").row(0) == (-3.0, -160)
    assert metadata == {}
    assert all(seasons in ([_SEASON], [_SEASON - 1]) for seasons in schedule_calls)


def test_the_pick_time_source_anchors_to_the_opener_and_records_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under pick_time the opener replaces the stored line, priced by the earlier seasons' map."""
    schedule_calls: list[list[int]] = []
    _patch_sources(monkeypatch, schedule_calls)
    monkeypatch.setattr(nfelo_lines, "load_nfelo_lines", lambda **_kwargs: _snapshot())
    monkeypatch.setattr(data_collection.game_utils, "scrape_survivor_grid_spreads", dict)
    metadata: dict[str, object] = {}

    games = data_collection.collect_all_data(
        [_SEASON], config=_config("pick_time"), market_lines_metadata=metadata
    )

    expected = moneyline_map.fit_spread_moneyline_map(_price_history())
    assert expected is not None
    row = games.row(0, named=True)
    assert (row["home_spread"], row["away_spread"]) == (-6.0, 6.0)
    assert (row["home_moneyline"], row["away_moneyline"]) == expected.moneylines(-6.0)
    assert row["total_line"] == pytest.approx(44.0)
    history_call = list(range(constants.NFLREADPY_MIN_SEASON, _SEASON))
    assert history_call in schedule_calls
    assert metadata["line_source"] == "pick_time"
    assert metadata["nfelo"] == _snapshot().metadata()
    seasons = metadata["seasons"]
    assert isinstance(seasons, dict)
    assert seasons[str(_SEASON)]["opener"] == 1
    assert seasons[str(_SEASON)]["moneylines_from_map"] == 1


def test_main_writes_the_line_record_for_every_build_so_none_goes_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every build writes its record, so a stored build replaces an earlier pick-time record."""
    monkeypatch.setattr(clock, "local_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(data_collection, "save_dataframe", lambda *_a, **_k: None)
    monkeypatch.setattr(polars_utils, "filter_completed_games", lambda df: df)
    monkeypatch.setattr(polars_utils, "filter_upcoming_games", lambda df, _s, _w: df)
    record = {"line_source": "pick_time", "nfelo": {"sha256": "abc"}}

    def fake_collect(
        _seasons: list[int],
        *,
        config: data_collection.DataCollectionConfig,
        strength_snapshots: list[pl.DataFrame],
        market_lines_metadata: dict[str, object],
    ) -> pl.DataFrame:
        del strength_snapshots
        if config.line_source == "pick_time":
            market_lines_metadata.update(record)
        return pl.DataFrame({"season": [2026], "week": [4], "away_score": [None]})

    monkeypatch.setattr(data_collection, "collect_all_data", fake_collect)
    path = tmp_path / f"{constants.MARKET_LINES_METADATA_NAME}.json"
    base = ["--min-season", "2026", "--max-season", "2026", "--data-dir", str(tmp_path)]

    data_collection.main(base)
    assert json.loads(path.read_text(encoding="utf-8")) == {"line_source": "stored"}

    data_collection.main([*base, "--line-source", "pick_time"])
    assert json.loads(path.read_text(encoding="utf-8")) == record

    data_collection.main(base)
    assert json.loads(path.read_text(encoding="utf-8")) == {"line_source": "stored"}
