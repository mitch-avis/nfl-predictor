"""Tests for the per-game pick-time line order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.utils.polars import moneyline_map, nfelo_lines, pick_time_lines

if TYPE_CHECKING:
    from pathlib import Path

_SCHEDULE_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int32,
    "week": pl.Int32,
    "away_abbr": pl.Utf8,
    "home_abbr": pl.Utf8,
    "away_score": pl.Int32,
    "home_score": pl.Int32,
    "away_spread": pl.Float64,
    "home_spread": pl.Float64,
    "away_moneyline": pl.Int32,
    "home_moneyline": pl.Int32,
    "total_line": pl.Float64,
}
_NFELO_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "away_abbr": pl.Utf8,
    "home_abbr": pl.Utf8,
    "home_spread_open": pl.Float64,
    "home_spread_last": pl.Float64,
    "total_line_open": pl.Float64,
    "total_line_last": pl.Float64,
    "home_ml_open": pl.Int64,
    "away_ml_open": pl.Int64,
    "home_ml_last": pl.Int64,
    "away_ml_last": pl.Int64,
}


def _game(
    game_id: str, season: int, *, played: bool = True, **changes: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
        "season": season,
        "week": 1,
        "away_abbr": f"A{game_id}",
        "home_abbr": f"H{game_id}",
        "away_score": 17 if played else None,
        "home_score": 20 if played else None,
        "away_spread": 3.0,
        "home_spread": -3.0,
        "away_moneyline": 140,
        "home_moneyline": -160,
        "total_line": 44.0,
    }
    row.update(changes)
    return row


def _line(game_id: str, season: int, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": season,
        "week": 1,
        "away_abbr": f"A{game_id}",
        "home_abbr": f"H{game_id}",
        "home_spread_open": -6.0,
        "home_spread_last": -7.0,
        "total_line_open": None,
        "total_line_last": 47.0,
        "home_ml_open": None,
        "away_ml_open": None,
        "home_ml_last": -300,
        "away_ml_last": 250,
    }
    row.update(changes)
    return row


def _apply(
    games: list[dict[str, object]], lines: list[dict[str, object]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    schedule = pl.DataFrame(games, schema=_SCHEDULE_SCHEMA)
    nfelo = pl.DataFrame(lines, schema=_NFELO_SCHEMA)
    return pick_time_lines.apply_pick_time_lines(schedule, nfelo)


def _lines_of(frame: pl.DataFrame, game_id: str) -> dict[str, object]:
    row = frame.filter(pl.col("game_id") == game_id).row(0, named=True)
    return {
        key: row[key]
        for key in ("home_spread", "away_spread", "home_moneyline", "away_moneyline", "total_line")
    }


def test_a_completed_game_with_a_real_opener_takes_the_opening_line() -> None:
    """The opening spread replaces the stored one; a missing opening moneyline is left to derive."""
    lines, _report = _apply([_game("1", 2015)], [_line("1", 2015, total_line_open=46.5)])

    assert _lines_of(lines, "1") == {
        "home_spread": -6.0,
        "away_spread": 6.0,
        "home_moneyline": None,
        "away_moneyline": None,
        "total_line": 46.5,
    }


def test_a_missing_opening_total_falls_back_to_the_stored_total() -> None:
    """Before nfelo published opening totals, the stored total stays."""
    lines, report = _apply([_game("1", 2015)], [_line("1", 2015)])

    assert _lines_of(lines, "1")["total_line"] == pytest.approx(44.0)
    assert report.row(0, named=True)["opener_with_stored_total"] == 1


def test_real_opening_moneylines_are_kept_as_a_pair() -> None:
    """From 2024 nfelo publishes opening moneylines, and they are used."""
    lines, _report = _apply(
        [_game("1", 2024)],
        [_line("1", 2024, home_ml_open=-250, away_ml_open=205, total_line_open=45.0)],
    )

    assert _lines_of(lines, "1")["home_moneyline"] == -250
    assert _lines_of(lines, "1")["away_moneyline"] == 205


def test_a_single_opening_moneyline_is_not_kept_without_its_pair() -> None:
    """A lone opening moneyline is dropped, so both sides derive from the opening spread."""
    line = _line("1", 2024, total_line_open=45.0, home_ml_open=-250)
    lines, _report = _apply([_game("1", 2024)], [line])

    assert _lines_of(lines, "1")["home_moneyline"] is None
    assert _lines_of(lines, "1")["away_moneyline"] is None


@pytest.mark.parametrize("season", [2005, 2022])
def test_a_season_without_real_openers_keeps_the_stored_line(season: int) -> None:
    """Before 2007 and in 2022 nfelo's openers are not real, so the stored line is kept."""
    lines, report = _apply([_game("1", season)], [_line("1", season)])

    assert _lines_of(lines, "1") == {
        "home_spread": -3.0,
        "away_spread": 3.0,
        "home_moneyline": -160,
        "away_moneyline": 140,
        "total_line": 44.0,
    }
    assert report.row(0, named=True)["stored_fallback"] == 1


def test_a_game_without_an_opener_or_an_nfelo_row_keeps_the_stored_line() -> None:
    """A null opener, or a game nfelo does not list, is a counted fallback row."""
    lines, report = _apply(
        [_game("1", 2024), _game("2", 2024)], [_line("1", 2024, home_spread_open=None)]
    )

    assert _lines_of(lines, "1")["home_spread"] == pytest.approx(-3.0)
    assert _lines_of(lines, "2")["home_spread"] == pytest.approx(-3.0)
    season = report.row(0, named=True)
    assert season["games"] == 2
    assert season["matched"] == 1
    assert season["opener"] == 0
    assert season["stored_fallback"] == 2


def test_an_upcoming_game_takes_nfelo_s_latest_line() -> None:
    """Before kickoff the latest nfelo line, moneylines and total are used."""
    lines, report = _apply([_game("1", 2026, played=False)], [_line("1", 2026)])

    assert _lines_of(lines, "1") == {
        "home_spread": -7.0,
        "away_spread": 7.0,
        "home_moneyline": -300,
        "away_moneyline": 250,
        "total_line": 47.0,
    }
    assert report.row(0, named=True)["upcoming_nfelo"] == 1


def test_an_upcoming_game_without_a_latest_line_uses_the_opener_then_nflverse() -> None:
    """nfelo's opener stands in for a missing latest line; without either, nflverse's line."""
    lines, report = _apply(
        [_game("1", 2026, played=False), _game("2", 2026, played=False)],
        [
            _line(
                "1",
                2026,
                home_spread_last=None,
                home_ml_last=None,
                away_ml_last=None,
                total_line_last=None,
                total_line_open=48.0,
            )
        ],
    )

    assert _lines_of(lines, "1") == {
        "home_spread": -6.0,
        "away_spread": 6.0,
        "home_moneyline": None,
        "away_moneyline": None,
        "total_line": 48.0,
    }
    assert _lines_of(lines, "2")["home_spread"] == pytest.approx(-3.0)
    assert _lines_of(lines, "2")["home_moneyline"] == -160
    season = report.row(0, named=True)
    assert season["upcoming_nfelo"] == 1
    assert season["upcoming_nflverse"] == 1


def test_an_upcoming_game_with_no_line_anywhere_stays_empty() -> None:
    """With no nfelo or nflverse line the game is left for SurvivorGrid, and counted."""
    lines, report = _apply(
        [
            _game(
                "1",
                2026,
                played=False,
                home_spread=None,
                away_spread=None,
                home_moneyline=None,
                away_moneyline=None,
                total_line=None,
            )
        ],
        [],
    )

    assert _lines_of(lines, "1")["home_spread"] is None
    assert report.row(0, named=True)["upcoming_without_line"] == 1


def test_the_schedule_keeps_its_rows_order_and_types() -> None:
    """Only the five line columns change; row order and column types stay."""
    games = [_game(str(index), 2015) for index in range(5)]
    schedule = pl.DataFrame(games, schema=_SCHEDULE_SCHEMA)
    nfelo = pl.DataFrame([_line("3", 2015), _line("1", 2015)], schema=_NFELO_SCHEMA)

    lines, _report = pick_time_lines.apply_pick_time_lines(schedule, nfelo)

    assert lines.schema == schedule.schema
    assert lines.get_column("game_id").to_list() == ["0", "1", "2", "3", "4"]
    assert lines.drop(pick_time_lines.LINE_COLUMNS).equals(
        schedule.drop(pick_time_lines.LINE_COLUMNS)
    )


def test_the_report_counts_each_season_separately() -> None:
    """Per-season counts: games, matches, openers and fallback rows."""
    _lines, report = _apply(
        [_game("1", 2005), _game("2", 2015), _game("3", 2015)],
        [_line("1", 2005), _line("2", 2015)],
    )

    assert report.select("season", "games", "matched", "opener", "stored_fallback").rows() == [
        (2005, 1, 1, 0, 1),
        (2015, 2, 1, 1, 1),
    ]


def _price_history() -> pl.DataFrame:
    spreads = [-7.0, -3.0, 3.0, 7.0] * 60
    return pl.DataFrame(
        {
            "season": [2014] * len(spreads),
            "home_spread": spreads,
            "home_moneyline": [{-7.0: -320, -3.0: -165, 3.0: 145, 7.0: 270}[s] for s in spreads],
            "away_moneyline": [{-7.0: 270, -3.0: 145, 3.0: -165, 7.0: -320}[s] for s in spreads],
        }
    )


def test_prepare_fits_the_maps_on_earlier_seasons_and_records_the_run(tmp_path: Path) -> None:
    """Preparation applies the line order, fits each season's map and records both."""
    schedule = pl.DataFrame([_game("1", 2015)], schema=_SCHEDULE_SCHEMA)
    snapshot = nfelo_lines.NfeloLinesSnapshot(
        frame=pl.DataFrame([_line("1", 2015)], schema=_NFELO_SCHEMA),
        url="https://example.test/lines.csv",
        origin="download",
        sha256="abc",
        snapshot_path=tmp_path / "lines_abc.csv",
    )

    lines, prepared = pick_time_lines.prepare_pick_time_lines(
        schedule, history=_price_history(), snapshot=snapshot
    )
    filled = prepared.fill_moneylines(lines)
    metadata = prepared.metadata()

    expected_map = moneyline_map.fit_spread_moneyline_map(_price_history())
    assert expected_map is not None
    assert filled.row(0, named=True)["home_moneyline"] == expected_map.moneylines(-6.0)[0]
    assert metadata["line_source"] == "pick_time"
    assert metadata["nfelo"] == snapshot.metadata()
    assert metadata["moneyline_maps"] == {"2015": expected_map.to_dict()}
    assert metadata["seasons"] == {
        "2015": {
            "games": 1,
            "matched": 1,
            "opener": 1,
            "opener_with_stored_total": 1,
            "stored_fallback": 0,
            "upcoming_nfelo": 0,
            "upcoming_nflverse": 0,
            "upcoming_without_line": 0,
            "opener_sign_flips": 0,
            "upcoming_nfelo_sign_flips": 0,
            "moneylines_from_map": 1,
            "moneylines_from_fixed_conversion": 0,
        }
    }


def test_prepare_uses_the_run_s_own_seasons_for_earlier_season_maps(tmp_path: Path) -> None:
    """The schedule's own prices count toward a later season's map, never toward its own."""
    history = _price_history().with_columns(pl.col("season").cast(pl.Int32))
    schedule = pl.concat(
        [
            history.with_columns(
                pl.lit("x").alias("game_id"),
                pl.lit(1, dtype=pl.Int32).alias("week"),
                pl.lit("A").alias("away_abbr"),
                pl.lit("H").alias("home_abbr"),
                pl.lit(17, dtype=pl.Int32).alias("away_score"),
                pl.lit(20, dtype=pl.Int32).alias("home_score"),
                (-pl.col("home_spread")).alias("away_spread"),
                pl.col("home_moneyline").cast(pl.Int32),
                pl.col("away_moneyline").cast(pl.Int32),
                pl.lit(44.0).alias("total_line"),
            ).select(list(_SCHEDULE_SCHEMA)),
            pl.DataFrame([_game("1", 2015)], schema=_SCHEDULE_SCHEMA),
        ]
    )
    snapshot = nfelo_lines.NfeloLinesSnapshot(
        frame=pl.DataFrame(schema=_NFELO_SCHEMA),
        url="u",
        origin="unavailable",
        sha256=None,
        snapshot_path=None,
    )

    _lines, prepared = pick_time_lines.prepare_pick_time_lines(
        schedule, history=pl.DataFrame(), snapshot=snapshot
    )

    assert prepared.maps[2014] is None
    assert prepared.maps[2015] is not None
    assert prepared.maps[2015].seasons == (2014,)


def test_the_report_counts_opener_sign_flips_against_the_stored_line() -> None:
    """An opener of at least 3 points on the other side of a stored line of 3+ is counted."""
    _lines, report = _apply(
        [
            _game("1", 2015, home_spread=-3.5, away_spread=3.5),
            _game("2", 2015, home_spread=-3.5, away_spread=3.5),
            _game("3", 2015, home_spread=-2.5, away_spread=2.5),
        ],
        [
            _line("1", 2015, home_spread_open=4.0),
            _line("2", 2015, home_spread_open=-6.0),
            _line("3", 2015, home_spread_open=4.0),
        ],
    )

    assert report.row(0, named=True)["opener_sign_flips"] == 1


def test_the_report_counts_upcoming_nfelo_sign_flips_against_nflverse() -> None:
    """An upcoming game whose nfelo line and nflverse line of 3+ disagree in sign is counted."""
    _lines, report = _apply(
        [
            _game("1", 2026, played=False, home_spread=-7.0, away_spread=7.0),
            _game("2", 2026, played=False, home_spread=-7.0, away_spread=7.0),
        ],
        [_line("1", 2026, home_spread_last=7.0), _line("2", 2026)],
    )

    row = report.row(0, named=True)
    assert row["upcoming_nfelo_sign_flips"] == 1
    assert row["opener_sign_flips"] == 0


def test_filling_twice_counts_each_derived_game_once(tmp_path: Path) -> None:
    """A game whose away side stays missing is not counted again by a second fill."""
    schedule = pl.DataFrame([_game("1", 2005)], schema=_SCHEDULE_SCHEMA)
    snapshot = nfelo_lines.NfeloLinesSnapshot(
        frame=pl.DataFrame(schema=_NFELO_SCHEMA),
        url="u",
        origin="unavailable",
        sha256=None,
        snapshot_path=tmp_path / "none.csv",
    )
    _lines, prepared = pick_time_lines.prepare_pick_time_lines(
        schedule, history=pl.DataFrame(), snapshot=snapshot
    )
    games = schedule.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("away_spread"),
        pl.lit(None, dtype=pl.Int32).alias("home_moneyline"),
        pl.lit(None, dtype=pl.Int32).alias("away_moneyline"),
    )

    prepared.fill_moneylines(prepared.fill_moneylines(games))

    assert prepared.derived[2005] == {
        "moneylines_from_map": 0,
        "moneylines_from_fixed_conversion": 1,
    }
