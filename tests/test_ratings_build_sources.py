"""Tests for how the ETL's ratings build loads its sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, nflverse_sources, ratings_build

_FIRST = constants.NFLREADPY_MIN_SEASON


@dataclass
class _Calls:
    """Every loader call: the seasons asked for and the keyword arguments."""

    pbp: list[tuple[list[int], dict[str, Any]]] = field(default_factory=list)
    schedule: list[tuple[list[int], dict[str, Any]]] = field(default_factory=list)
    sources: dict[str, tuple[list[int], dict[str, Any]]] = field(default_factory=dict)
    players: list[dict[str, Any]] = field(default_factory=list)


def _seasons_frame(seasons: list[int], label: str) -> pl.DataFrame:
    """Return one row per season, labeled with where it came from."""
    return pl.DataFrame({"season": seasons, "source": [label] * len(seasons)})


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Replace every loader the build reads with one that records its call."""
    recorded = _Calls()

    def load_pbp(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        recorded.pbp.append((list(seasons), kwargs))
        return _seasons_frame(list(seasons), "cache")

    def load_schedule(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        recorded.schedule.append((list(seasons), kwargs))
        return _seasons_frame(list(seasons), "cache")

    def source(name: str) -> Any:
        def load(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
            recorded.sources[name] = (list(seasons), kwargs)
            return _seasons_frame(list(seasons), name)

        return load

    def load_players(**kwargs: Any) -> pl.DataFrame:
        recorded.players.append(kwargs)
        return pl.DataFrame({"gsis_id": ["00-1"]})

    monkeypatch.setattr(loaders, "load_pbp", load_pbp)
    monkeypatch.setattr(loaders, "load_schedule", load_schedule)
    monkeypatch.setattr(
        nflverse_sources, "load_weekly_player_stats", source("load_weekly_player_stats")
    )
    monkeypatch.setattr(nflverse_sources, "load_snap_counts", source("load_snap_counts"))
    monkeypatch.setattr(nflverse_sources, "load_weekly_rosters", source("load_weekly_rosters"))
    monkeypatch.setattr(nflverse_sources, "load_players", load_players)
    return recorded


def _load(max_season: int = 2005, current_season: int = 2005) -> ratings_build.LoadedSources:
    """Load the sources with the ETL's play-by-play and schedules for 2003 on."""
    return ratings_build.load_ratings_sources(
        max_season,
        pbp_df=_seasons_frame([2003, 2004, 2005], "etl"),
        schedule_df=_seasons_frame([2003, 2004, 2005], "etl"),
        current_season=current_season,
        force_refresh=True,
    )


def test_sources_reuse_the_etls_seasons_and_read_the_earlier_ones_unrefreshed(
    calls: _Calls,
) -> None:
    # Act
    sources = _load()

    # Assert
    assert calls.pbp == [
        ([1999, 2000, 2001, 2002], {"force_refresh": False, "current_season": 2005})
    ]
    assert sources.plays.sort("season").rows() == [
        *((season, "cache") for season in range(_FIRST, 2003)),
        *((season, "etl") for season in (2003, 2004, 2005)),
    ]


def test_sources_read_the_earlier_schedules_unrefreshed(calls: _Calls) -> None:
    # Act
    sources = _load()

    # Assert
    assert calls.schedule == [
        ([1999, 2000, 2001, 2002], {"force_refresh": False, "current_season": 2005})
    ]
    assert sorted(sources.schedule.get_column("season").to_list()) == list(range(_FIRST, 2006))


@pytest.mark.parametrize(
    ("loader", "first"),
    [
        ("load_weekly_player_stats", constants.NFLREADPY_PLAYER_STATS_MIN_SEASON),
        ("load_snap_counts", constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON),
        ("load_weekly_rosters", constants.NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON),
    ],
)
def test_sources_ask_each_player_source_for_the_seasons_it_publishes(
    calls: _Calls, loader: str, first: int
) -> None:
    # Act
    _load(max_season=2014, current_season=2014)

    # Assert
    seasons, kwargs = calls.sources[loader]
    assert seasons == list(range(max(first, _FIRST), 2015))
    assert kwargs == {"force_refresh": True, "current_season": 2014}


def test_sources_refresh_the_players_file_and_survive_its_outage_in_the_current_season(
    calls: _Calls,
) -> None:
    # Act
    _load(max_season=2005, current_season=2005)

    # Assert
    assert calls.players == [{"force_refresh": True, "degrade_on_failure": True}]


def test_sources_of_finished_seasons_keep_the_players_files_outage_fatal(calls: _Calls) -> None:
    # Act
    ratings_build.load_ratings_sources(
        2004,
        pbp_df=_seasons_frame([2003, 2004], "etl"),
        schedule_df=_seasons_frame([2003, 2004], "etl"),
        current_season=2006,
        force_refresh=False,
    )

    # Assert
    assert calls.players == [{"force_refresh": False, "degrade_on_failure": False}]


def test_sources_read_the_quarterback_identity_file_of_the_data_directory(calls: _Calls) -> None:
    # Arrange
    constants.DATA_PATH.mkdir(parents=True, exist_ok=True)
    (constants.DATA_PATH / f"{constants.QB_META_DATA_NAME}.csv").write_text(
        "name_id,gsis_id\nAlpha Passer,00-0000001\n", encoding="utf-8"
    )

    # Act
    sources = _load()

    # Assert
    assert sources.starter_identity.rows() == [("Alpha Passer", "00-0000001")]


def test_sources_load_every_season_when_the_etl_loaded_none(calls: _Calls) -> None:
    # Act
    ratings_build.load_ratings_sources(
        2001,
        pbp_df=pl.DataFrame(),
        schedule_df=pl.DataFrame(),
        current_season=2005,
        force_refresh=False,
    )

    # Assert
    assert [seasons for seasons, _ in calls.pbp] == [[1999, 2000, 2001]]
