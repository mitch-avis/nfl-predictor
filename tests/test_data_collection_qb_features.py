"""Tests for attaching the quarterback family during data collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils

if TYPE_CHECKING:
    from pathlib import Path


def _dropbacks(season: int, week: int, team: str, passer: str, count: int) -> list[dict[str, Any]]:
    """Return ``count`` completed-pass dropbacks by ``passer`` for ``team``."""
    return [
        {
            "season": season,
            "week": week,
            "season_type": "REG",
            "posteam": team,
            "defteam": "OPP",
            "qb_dropback": 1,
            "sack": 0,
            "complete_pass": 1,
            "yards_gained": 5.0,
            "qb_epa": 0.2,
            "passer_player_id": passer,
            "passer_player_name": None,
        }
        for _ in range(count)
    ]


def _identity_file(tmp_path: Path) -> Path:
    """Write a two-quarterback identity file and return its path."""
    path = tmp_path / "meta_data.csv"
    pl.DataFrame({"name_id": ["Tom Brady", "Josh Allen"], "gsis_id": ["TB", "JA"]}).write_csv(path)
    return path


def test_attach_qb_features_loads_missing_history_and_joins_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Seasons missing from memory are loaded so career rates span the whole history."""
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    history = pl.DataFrame(_dropbacks(2000, 5, "NE", "TB", 30))
    requested: list[list[int]] = []
    refreshed: list[bool] = []

    def fake_load_pbp(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        """Record the requested seasons and refresh flag, return the earlier history."""
        requested.append(list(seasons))
        refreshed.append(bool(kwargs.get("force_refresh")))
        return history

    monkeypatch.setattr(polars_utils, "load_pbp", fake_load_pbp)
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )

    out = data_collection._attach_qb_features(
        games,
        in_memory,
        max_season=2001,
        current_season=2026,
        family=data_collection.QbFamilyInputs(identity_path=_identity_file(tmp_path)),
    )

    assert requested == [list(range(constants.NFLREADPY_MIN_SEASON, 2001))]
    assert refreshed == [False], "history seasons always come from the cache"
    row = out.row(0, named=True)
    assert row["away_qb_history_dropbacks"] == 50
    assert row["home_qb_history_dropbacks"] == 12
    assert row["qb_history_dropbacks_diff"] == 38
    assert out["game_id"].to_list() == ["g1"]


def test_attach_qb_features_reads_the_nfeloqb_identity_file_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without an explicit path, the identity map comes from ``DATA_PATH/meta_data.csv``."""
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(polars_utils, "load_pbp", lambda *_args, **_kwargs: pl.DataFrame())
    _identity_file(tmp_path)
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )

    out = data_collection._attach_qb_features(
        games, in_memory, max_season=2001, current_season=2026
    )

    row = out.row(0, named=True)
    assert row["away_qb_history_dropbacks"] == 20
    assert row["home_qb_history_dropbacks"] == 12


def test_attach_qb_features_leaves_rows_without_quarterbacks_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rows without quarterback columns come back unchanged and trigger no loading."""

    def fail_load_pbp(*_args: Any, **_kwargs: Any) -> pl.DataFrame:
        """Fail if play-by-play is requested."""
        msg = "load_pbp should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(polars_utils, "load_pbp", fail_load_pbp)
    games = pl.DataFrame({"game_id": ["g1"], "season": [2001], "week": [1]})

    out = data_collection._attach_qb_features(
        games,
        pl.DataFrame(),
        max_season=2001,
        current_season=2026,
        family=data_collection.QbFamilyInputs(identity_path=tmp_path / "missing.csv"),
    )

    assert out.equals(games)


def test_attach_qb_features_adjusts_for_the_defense_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The faced defense's pre-game snapshot reaches the defense-adjusted rate."""
    monkeypatch.setattr(polars_utils, "load_pbp", lambda *_args, **_kwargs: pl.DataFrame())
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )
    # Every week-1 dropback is worth 0.2 EPA against a defense 0.1 better than average.
    snapshots = pl.DataFrame(
        {
            "season": [2001],
            "week": [1],
            "team_abbr": ["OPP"],
            constants.QB_DEF_ADJ_SOURCE_STAT: [0.1],
        }
    )

    out = data_collection._attach_qb_features(
        games,
        in_memory,
        max_season=2001,
        current_season=2026,
        family=data_collection.QbFamilyInputs(
            identity_path=_identity_file(tmp_path), defense_snapshots=snapshots
        ),
    )

    row = out.row(0, named=True)
    assert row["away_qb_dropback_epa"] == pytest.approx(0.2)
    assert row["away_qb_def_adj_epa"] == pytest.approx(0.3)
    assert row["home_qb_def_adj_epa_recent"] == pytest.approx(0.3)


def test_collect_all_data_hands_the_strength_snapshots_to_the_qb_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every processed week's snapshot reaches the quarterback family, without a caller's list."""
    season = 2001
    snapshot = pl.DataFrame(
        {
            "season": [season],
            "week": [1],
            "team_abbr": ["OPP"],
            constants.QB_DEF_ADJ_SOURCE_STAT: [0.1],
        }
    )
    sources = data_collection._EtlSources(
        current_season=2026,
        current_week=1,
        schedule_df=pl.DataFrame(),
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
    )

    def fake_process_season(
        _season: int, _schedule: pl.DataFrame, _stats: pl.DataFrame, inputs: Any
    ) -> pl.DataFrame:
        """Record one week's snapshot the way the week builder does and return one game."""
        assert inputs.strength_snapshots is not None
        inputs.strength_snapshots.append(snapshot)
        return pl.DataFrame({"game_id": ["g1"], "season": [season], "week": [2]})

    received: dict[str, Any] = {}

    def fake_attach(games: pl.DataFrame, *_args: Any, **kwargs: Any) -> pl.DataFrame:
        """Record the keyword arguments the family is called with."""
        received.update(kwargs)
        return games

    monkeypatch.setattr(data_collection, "_load_sources", lambda *_args: sources)
    monkeypatch.setattr(data_collection, "process_season", fake_process_season)
    monkeypatch.setattr(data_collection, "_attach_qb_features", fake_attach)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_qb_data", lambda df, _elo: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(data_collection.game_utils, "fill_missing_moneylines", lambda df: df)
    monkeypatch.setattr(polars_utils, "select_final_columns", lambda df: df)

    data_collection.collect_all_data([season])

    assert received["family"].defense_snapshots.equals(snapshot)
