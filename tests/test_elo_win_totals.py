"""Tests for the win-total ratings getter: download, cache, fallback and normalization."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import pytest
import requests

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_win_totals

if TYPE_CHECKING:
    from pathlib import Path

_HEADER = ",team,season,line_rating,wt_rating,wt_rating_elo,sos,line"


def _csv(*rows: str) -> bytes:
    return "\n".join([_HEADER, *rows, ""]).encode()


_FIRST = _csv(
    "0,ARI,2003,-3.3,-2.0,1392.85,-0.1,5.0",
    "1,OAK,2003,1.1,0.7,1543.33,0.0,9.5",
)
_SECOND = _csv("0,ARI,2003,-3.3,-2.0,1400.00,-0.1,5.0")


def _failing_fetch(_url: str) -> bytes:
    msg = "offline"
    raise requests.ConnectionError(msg)


def test_a_download_is_cached_and_kept_as_a_snapshot_named_by_its_hash(tmp_path: Path) -> None:
    # Arrange
    digest = hashlib.sha256(_FIRST).hexdigest()

    # Act
    snapshot = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    # Assert
    assert snapshot.origin == "download"
    assert snapshot.sha256 == digest
    assert snapshot.snapshot_path == tmp_path / "snapshots" / f"wt_ratings_{digest}.csv"
    assert (tmp_path / "snapshots" / f"wt_ratings_{digest}.csv").read_bytes() == _FIRST
    assert (tmp_path / "wt_ratings.csv").read_bytes() == _FIRST
    assert snapshot.frame.height == 2


def test_a_failed_download_falls_back_to_the_cached_copy(tmp_path: Path) -> None:
    # Arrange
    elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    # Act
    snapshot = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=_failing_fetch)

    # Assert
    assert snapshot.origin == "cache"
    assert snapshot.sha256 == hashlib.sha256(_FIRST).hexdigest()
    assert snapshot.frame.height == 2


def test_a_newer_download_replaces_the_cache_and_keeps_both_snapshots(tmp_path: Path) -> None:
    # Arrange
    first = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    # Act
    second = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=lambda _url: _SECOND)

    # Assert
    assert first.snapshot_path is not None
    assert second.snapshot_path is not None
    assert first.snapshot_path.read_bytes() == _FIRST
    assert second.snapshot_path.read_bytes() == _SECOND
    assert (tmp_path / "wt_ratings.csv").read_bytes() == _SECOND


def test_an_unreadable_download_keeps_the_cached_copy(tmp_path: Path) -> None:
    # Arrange
    elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    # Act
    snapshot = elo_win_totals.load_win_totals(
        cache_dir=tmp_path, fetch=lambda _url: b"<html>rate limited</html>"
    )

    # Assert
    assert snapshot.origin == "cache"
    assert (tmp_path / "wt_ratings.csv").read_bytes() == _FIRST


def test_an_unreadable_cache_with_no_download_degrades_to_an_empty_frame(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "wt_ratings.csv").write_bytes(b"not,the,file\n1,2,3\n")

    # Act
    snapshot = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=_failing_fetch)

    # Assert
    assert snapshot.origin == "unavailable"
    assert snapshot.frame.height == 0


def test_no_download_and_no_cache_degrades_to_an_empty_typed_frame(tmp_path: Path) -> None:
    # Act
    snapshot = elo_win_totals.load_win_totals(cache_dir=tmp_path, fetch=_failing_fetch)

    # Assert
    assert snapshot.origin == "unavailable"
    assert snapshot.sha256 is None
    assert snapshot.snapshot_path is None
    assert snapshot.frame.columns == list(elo_win_totals.WIN_TOTALS_SCHEMA)
    assert snapshot.frame.height == 0


def test_the_metadata_records_the_hash_origin_and_snapshot(tmp_path: Path) -> None:
    # Act
    snapshot = elo_win_totals.load_win_totals(
        cache_dir=tmp_path, url="https://example.test/wt.csv", fetch=lambda _url: _FIRST
    )

    # Assert
    assert snapshot.metadata() == {
        "url": "https://example.test/wt.csv",
        "origin": "download",
        "sha256": hashlib.sha256(_FIRST).hexdigest(),
        "snapshot": str(snapshot.snapshot_path),
        "rows": 2,
    }


def test_the_default_cache_is_a_win_totals_directory_beside_the_nfelo_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(constants, "NFELO_CACHE_DIR", tmp_path)

    # Act
    elo_win_totals.load_win_totals(fetch=lambda _url: _FIRST)

    # Assert
    assert (tmp_path / "win_totals" / "wt_ratings.csv").read_bytes() == _FIRST


def test_normalization_maps_team_codes_to_the_canonical_abbreviations() -> None:
    # Arrange
    raw = elo_win_totals.parse_win_totals(
        _csv("0,OAK,2019,0,0,1500.0,0,8.0", "1,WAS,2019,0,0,1490.0,0,7.0")
    )

    # Act
    teams = raw["team_abbr"].to_list()

    # Assert
    assert teams == ["LV", "WSH"]
    assert raw["win_total_elo"].to_list() == [1500.0, 1490.0]


def test_a_team_listed_twice_in_a_season_keeps_its_first_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)
    content = _csv("0,ARI,2019,0,0,1500.0,0,8.0", "1,ARI,2019,0,0,1600.0,0,9.0")

    # Act
    frame = elo_win_totals.parse_win_totals(content)

    # Assert
    assert frame["win_total_elo"].to_list() == [1500.0]
    assert any("more than once" in message for message in caplog.messages)


def test_parse_rejects_a_file_without_the_rating_column() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="wt_rating_elo"):
        elo_win_totals.parse_win_totals(b"team,season\nARI,2019\n")


def test_parse_rejects_bytes_that_are_not_a_csv() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="unreadable"):
        elo_win_totals.parse_win_totals(b"")


class _Response:
    """The parts of ``requests.Response`` the default fetch reads."""

    content = _FIRST

    def raise_for_status(self) -> None:
        """Report success."""


def test_the_default_fetch_gets_the_url_with_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    calls: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float) -> _Response:
        calls.append((url, timeout))
        return _Response()

    monkeypatch.setattr(elo_win_totals.requests, "get", fake_get)

    # Act
    snapshot = elo_win_totals.load_win_totals(cache_dir=tmp_path, url="https://example.test/wt")

    # Assert
    assert calls == [("https://example.test/wt", 30)]
    assert snapshot.origin == "download"
