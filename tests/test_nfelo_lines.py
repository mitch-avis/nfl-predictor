"""Tests for the nfelo market-lines getter: download, cache, fallback and normalization."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
import pytest
import requests

from nfl_predictor.utils.polars import nfelo_lines

if TYPE_CHECKING:
    from pathlib import Path

_HEADER = (
    ",game_id,season,week,home_team,away_team,home_spread_open,home_spread_last,"
    "home_ml_open,away_ml_open,home_ml_last,away_ml_last,total_line_open,total_line_last"
)


def _csv(*rows: str) -> bytes:
    return "\n".join([_HEADER, *rows, ""]).encode()


_FIRST = _csv(
    "0,2019_01_PIT_NE,2019,1,NE,PIT,-6.0,-5.5,,,-245.0,205.0,,49.0",
    "1,2019_01_OAK_WAS,2019,1,WAS,OAK,3.0,2.5,,,,,,44.5",
)
_SECOND = _csv("0,2019_01_PIT_NE,2019,1,NE,PIT,-6.5,-5.5,,,-245.0,205.0,,49.0")


def _failing_fetch(_url: str) -> bytes:
    msg = "offline"
    raise requests.ConnectionError(msg)


def test_a_download_is_cached_and_kept_as_a_snapshot_named_by_its_hash(tmp_path: Path) -> None:
    """A fetched file becomes the cached copy and a content-addressed snapshot."""
    snapshot = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    digest = hashlib.sha256(_FIRST).hexdigest()
    assert snapshot.origin == "download"
    assert snapshot.sha256 == digest
    assert snapshot.snapshot_path == tmp_path / "snapshots" / f"lines_{digest}.csv"
    assert (tmp_path / "snapshots" / f"lines_{digest}.csv").read_bytes() == _FIRST
    assert (tmp_path / "lines.csv").read_bytes() == _FIRST
    assert snapshot.frame.height == 2


def test_a_failed_download_falls_back_to_the_cached_copy(tmp_path: Path) -> None:
    """When the source is unreachable the last cached copy is used, without failing."""
    nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    snapshot = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=_failing_fetch)

    assert snapshot.origin == "cache"
    assert snapshot.sha256 == hashlib.sha256(_FIRST).hexdigest()
    assert snapshot.frame.height == 2


def test_a_newer_download_replaces_the_cache_and_keeps_both_snapshots(tmp_path: Path) -> None:
    """Every run's snapshot stays on disk, while the cached copy follows the latest download."""
    first = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=lambda _url: _FIRST)
    second = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=lambda _url: _SECOND)

    assert first.snapshot_path is not None
    assert second.snapshot_path is not None
    assert first.snapshot_path.read_bytes() == _FIRST
    assert second.snapshot_path.read_bytes() == _SECOND
    assert (tmp_path / "lines.csv").read_bytes() == _SECOND


def test_an_unreadable_download_keeps_the_cached_copy(tmp_path: Path) -> None:
    """A response that is not the lines file never overwrites the cache."""
    nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=lambda _url: _FIRST)

    snapshot = nfelo_lines.load_nfelo_lines(
        cache_dir=tmp_path, fetch=lambda _url: b"<html>rate limited</html>"
    )

    assert snapshot.origin == "cache"
    assert (tmp_path / "lines.csv").read_bytes() == _FIRST


def test_no_download_and_no_cache_degrades_to_an_empty_frame(tmp_path: Path) -> None:
    """Without any copy the getter reports the source unavailable and returns no rows."""
    snapshot = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=_failing_fetch)

    assert snapshot.origin == "unavailable"
    assert snapshot.sha256 is None
    assert snapshot.snapshot_path is None
    assert snapshot.frame.height == 0
    assert set(nfelo_lines.NFELO_LINE_COLUMNS) <= set(snapshot.frame.columns)


def test_the_metadata_records_the_hash_origin_and_snapshot(tmp_path: Path) -> None:
    """The run metadata names exactly which copy of the file the run used."""
    snapshot = nfelo_lines.load_nfelo_lines(
        cache_dir=tmp_path, url="https://example.test/lines.csv", fetch=lambda _url: _FIRST
    )

    assert snapshot.metadata() == {
        "url": "https://example.test/lines.csv",
        "origin": "download",
        "sha256": hashlib.sha256(_FIRST).hexdigest(),
        "snapshot": str(snapshot.snapshot_path),
        "rows": 2,
    }


def test_normalization_maps_team_codes_to_the_canonical_abbreviations() -> None:
    """nfelo's historical codes (OAK, WAS, STL) join the schedule's canonical teams."""
    raw = pl.read_csv(
        _csv(
            "0,2019_01_OAK_WAS,2019,1,WAS,OAK,3.0,2.5,,,,,,44.5",
            "1,1999_01_BAL_STL,1999,1,STL,BAL,0.0,0.0,,,,,,39.0",
        )
    )

    normalized = nfelo_lines.normalize_nfelo_lines(raw)

    assert normalized.select("season", "week", "away_abbr", "home_abbr").rows() == [
        (2019, 1, "LV", "WSH"),
        (1999, 1, "BAL", "LAR"),
    ]


def test_normalization_types_the_lines_and_keeps_the_first_duplicate() -> None:
    """Lines are floats, moneylines whole numbers, and a repeated game keeps its first row."""
    raw = pl.read_csv(
        _csv(
            "0,2019_01_PIT_NE,2019,1,NE,PIT,-6,-5.5,,,-245,205,,49",
            "1,2019_01_PIT_NE,2019,1,NE,PIT,-7,-7,,,-300,250,,50",
        )
    )

    normalized = nfelo_lines.normalize_nfelo_lines(raw)

    assert normalized.height == 1
    row = normalized.row(0, named=True)
    assert row["home_spread_open"] == pytest.approx(-6.0)
    assert row["home_ml_last"] == -245
    assert normalized.schema["home_spread_open"] == pl.Float64
    assert normalized.schema["home_ml_last"] == pl.Int64
    assert normalized.schema["season"] == pl.Int64


def test_an_unreadable_cache_is_treated_as_no_copy(tmp_path: Path) -> None:
    """A damaged cached file degrades to the unavailable result rather than failing the run."""
    (tmp_path / "lines.csv").write_bytes(b"not,the,file\n1,2,3\n")

    snapshot = nfelo_lines.load_nfelo_lines(cache_dir=tmp_path, fetch=_failing_fetch)

    assert snapshot.origin == "unavailable"
    assert snapshot.frame.height == 0
