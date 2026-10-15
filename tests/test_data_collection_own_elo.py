"""Tests for publishing both own Elo families during data collection."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, NoReturn

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import clock, game_utils, polars_utils
from nfl_predictor.utils.polars import elo_win_totals, loaders, qb_starters
from nfl_predictor.utils.polars.finalize import build_final_column_order
from tests.fakes import returning

_TEAMS = (("NE", "BOS00"), ("LAC", "LAX01"))


def _same_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` unchanged."""
    return df


def _same_frame_of_three(df: pl.DataFrame, _second: object, _third: object) -> pl.DataFrame:
    """Return ``df`` unchanged, ignoring the other two arguments."""
    return df


def _same_frame_of_two(df: pl.DataFrame, _other: object) -> pl.DataFrame:
    """Return ``df`` unchanged, ignoring the second argument."""
    return df


def _schedule(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Return NE and LAC hosting each other once per season, the later season's game upcoming."""
    rows = [
        {
            "game_id": f"{season}_{week:02d}_{away}_{home}",
            "season": season,
            "week": week,
            "game_type": "REG",
            "date": date(season, 9, 10) + timedelta(days=7 * (week - 1)),
            "home_abbr": home,
            "away_abbr": away,
            "home_score": 24 if season < max(seasons) else None,
            "away_score": 20 if season < max(seasons) else None,
            "neutral": 0,
            "stadium_id": stadium,
        }
        for season in seasons
        for week, ((home, stadium), (away, _stadium)) in enumerate(
            (_TEAMS, tuple(reversed(_TEAMS))), start=1
        )
    ]
    return pl.DataFrame(rows)


def _games(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return published game rows for ``schedule``, each side's starter named."""
    return schedule.select(
        "game_id",
        "season",
        "week",
        "away_abbr",
        "home_abbr",
        pl.lit("Tom Brady").alias("away_qb"),
        pl.lit("Josh Allen").alias("home_qb"),
    )


def _sources(schedule: pl.DataFrame) -> data_collection._EtlSources:
    return data_collection._EtlSources(
        current_season=max(schedule.get_column("season").to_list()),
        current_week=1,
        schedule_df=schedule,
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
    )


def _empty_history() -> qb_starters.QuarterbackHistory:
    return qb_starters.quarterback_history(pl.DataFrame(), qb_starters.season_positions({}))


def _win_totals(seasons: tuple[int, ...]) -> Any:
    """Return a win-total loader serving a snapshot that rates both teams in ``seasons``."""

    def load(**_kwargs: Any) -> elo_win_totals.WinTotalsSnapshot:
        frame = pl.DataFrame(
            [
                {"season": season, "team_abbr": team, "win_total_elo": 1530.0 - 50.0 * index}
                for season in seasons
                for index, (team, _stadium) in enumerate(_TEAMS)
            ],
            schema=elo_win_totals.WIN_TOTALS_SCHEMA,
        )
        return elo_win_totals.WinTotalsSnapshot(
            frame, "https://example.invalid/wt.csv", "download", "f00d", Path("wt.csv")
        )

    return load


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, *, win_total_seasons: tuple[int, ...]) -> None:
    """Serve the win totals and, for the seasons the run does not build, no history."""
    monkeypatch.setattr(elo_win_totals, "load_win_totals", _win_totals(win_total_seasons))
    monkeypatch.setattr(polars_utils, "load_schedule", returning(pl.DataFrame()))
    monkeypatch.setattr(polars_utils, "load_elo_ratings", returning(pl.DataFrame()))


def test_attach_own_elo_publishes_both_families_on_every_game_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    schedule = _schedule((2019, 2020))
    _patch_inputs(monkeypatch, win_total_seasons=(2019, 2020))
    record: dict[str, object] = {}

    # Act
    out = data_collection._attach_own_elo(
        _games(schedule),
        _sources(schedule),
        _empty_history(),
        max_season=2020,
        metadata=record,
    )

    # Assert
    columns = [*constants.OWN_ELO_COLUMNS, *constants.OWN_WT_ELO_COLUMNS]
    assert out.select(columns).null_count().row(0) == (0,) * len(columns)
    assert out.get_column("game_id").to_list() == schedule.get_column("game_id").to_list()
    assert json.loads(json.dumps(record))["win_total_file"]["sha256"] == "f00d"


def test_attach_own_elo_without_the_schedule_columns_it_reads_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    schedule = _schedule((2019, 2020)).drop("stadium_id")

    def load_win_totals(**_k: object) -> NoReturn:
        pytest.fail("loaded")

    monkeypatch.setattr(elo_win_totals, "load_win_totals", load_win_totals)
    games = _games(schedule)

    # Act
    out = data_collection._attach_own_elo(
        games, _sources(schedule), _empty_history(), max_season=2020, metadata={}
    )

    # Assert
    assert out.equals(games)


def test_attach_own_elo_fails_loudly_when_no_built_season_has_win_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    schedule = _schedule((2019, 2020))
    _patch_inputs(monkeypatch, win_total_seasons=(2003,))

    # Act & Assert
    with pytest.raises(ValueError, match="cover none of the build's seasons"):
        data_collection._attach_own_elo(
            _games(schedule), _sources(schedule), _empty_history(), max_season=2020, metadata={}
        )


def test_attach_own_elo_rates_the_seasons_a_partial_run_does_not_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    full = _schedule((2018, 2019, 2020))
    built = full.filter(pl.col("season") >= 2019)
    requested: list[list[int]] = []

    def load_schedule(seasons: list[int], **_kwargs: Any) -> pl.DataFrame:
        requested.append(list(seasons))
        return full.filter(pl.col("season").is_in(seasons))

    _patch_inputs(monkeypatch, win_total_seasons=(2018, 2019, 2020))
    monkeypatch.setattr(polars_utils, "load_schedule", load_schedule)
    monkeypatch.setattr(constants, "NFLREADPY_MIN_SEASON", 2018)
    whole = data_collection._attach_own_elo(
        _games(full), _sources(full), _empty_history(), max_season=2020, metadata={}
    )

    # Act
    partial = data_collection._attach_own_elo(
        _games(built), _sources(built), _empty_history(), max_season=2020, metadata={}
    )

    # Assert
    assert requested == [[2018]]
    columns = ["game_id", *constants.OWN_ELO_COLUMNS, *constants.OWN_WT_ELO_COLUMNS]
    assert partial.select(columns).equals(whole.filter(pl.col("season") >= 2019).select(columns))


def test_the_final_schema_carries_both_own_elo_families_once() -> None:
    # Act
    order = build_final_column_order()

    # Assert
    published = [*constants.OWN_ELO_COLUMNS, *constants.OWN_WT_ELO_COLUMNS]
    assert [column for column in order if column in published] == sorted(published, key=order.index)
    assert all(order.count(column) == 1 for column in published)


_RECORD = {"win_total_file": {"origin": "download", "sha256": "f00d"}, "seasons": [2019, 2020]}


def _patch_collection(monkeypatch: pytest.MonkeyPatch, schedule: pl.DataFrame) -> None:
    """Serve ``schedule`` as the only source and stub every step after the season builds."""
    monkeypatch.setattr(polars_utils, "load_pbp", returning(loaders._empty_pbp_frame()))
    monkeypatch.setattr(polars_utils, "load_schedule", returning(schedule))
    monkeypatch.setattr(polars_utils, "load_team_stats", returning(pl.DataFrame()))
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", _same_frame_of_two)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", _same_frame)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", returning(pl.DataFrame()))
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (2020, 1))
    monkeypatch.setattr(polars_utils, "load_team_rankings", returning(pl.DataFrame()))
    monkeypatch.setattr(polars_utils, "select_final_columns", _same_frame)
    monkeypatch.setattr(data_collection, "process_season", returning(_games(schedule)))
    monkeypatch.setattr(data_collection, "_quarterback_history", returning(None))
    monkeypatch.setattr(game_utils, "fill_future_qb_data", _same_frame_of_two)
    monkeypatch.setattr(game_utils, "fill_future_game_lines", _same_frame)
    monkeypatch.setattr(game_utils, "fill_missing_moneylines", _same_frame)


def test_collect_all_data_hands_its_record_to_the_own_elo_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _patch_collection(monkeypatch, _schedule((2020,)))

    def attach(games: pl.DataFrame, *_args: Any, metadata: dict[str, object], **_k: Any) -> Any:
        metadata.update(_RECORD)
        return games

    monkeypatch.setattr(data_collection, "_attach_own_elo", attach)
    config = data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=2020,
        max_season=2020,
        line_source=constants.LINE_SOURCE_STORED,
    )
    records = data_collection.CollectionRecords()

    # Act
    data_collection.collect_all_data([2020], config=config, records=records)

    # Assert
    assert records.own_elo_metadata == _RECORD


def _patch_main(monkeypatch: pytest.MonkeyPatch, record: dict[str, object]) -> None:
    """Run main without writing datasets; the collection fills in ``record`` as its own Elo's."""
    monkeypatch.setattr(clock, "local_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(data_collection, "save_dataframe", returning(None))
    monkeypatch.setattr(polars_utils, "filter_completed_games", _same_frame)
    monkeypatch.setattr(polars_utils, "filter_upcoming_games", _same_frame_of_three)

    def fake_collect(
        _seasons: list[int], *, records: data_collection.CollectionRecords, **_kwargs: Any
    ) -> pl.DataFrame:
        records.own_elo_metadata.update(record)
        return pl.DataFrame({"season": [2026], "week": [4], "away_score": [None]})

    monkeypatch.setattr(data_collection, "collect_all_data", fake_collect)


def _own_elo_record(data_dir: Path) -> object:
    path = data_dir / f"{constants.OWN_ELO_METADATA_NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("record", [_RECORD, {}], ids=["built", "not_built"])
def test_main_writes_the_own_elo_record_beside_the_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: dict[str, object]
) -> None:
    # Arrange
    _patch_main(monkeypatch, record)
    (tmp_path / f"{constants.OWN_ELO_METADATA_NAME}.json").write_text('{"stale": true}\n')

    # Act
    data_collection.main(
        ["--min-season", "2026", "--max-season", "2026", "--data-dir", str(tmp_path)]
    )

    # Assert
    assert _own_elo_record(tmp_path) == record
