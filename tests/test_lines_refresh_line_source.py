"""Tests for the lines refresh following the line source the data was built with."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants, lines_refresh
from nfl_predictor.utils import game_utils
from nfl_predictor.utils.polars import moneyline_map, nfelo_lines

if TYPE_CHECKING:
    from pathlib import Path

SEASON = 2026
WEEK = 5
LINE_COLUMNS = ["total_line", "away_spread", "home_spread", "away_moneyline", "home_moneyline"]
# A map whose prices differ clearly from the fixed conversion's normal curve.
SEASON_MAP = moneyline_map.SpreadMoneylineMap(
    spreads=(0.0, 3.0, 7.0, 14.0),
    favorite_prob=(0.5238, 0.62, 0.75, 0.88),
    underdog_prob=(0.5238, 0.42, 0.29, 0.15),
    games=500,
    seasons=(2024, 2025),
)
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


def _schedule_game(
    away: str, home: str, *, week: int = WEEK, played: bool = False, **lines: object
) -> dict[str, object]:
    """Return one schedule row with the nflverse (stored) lines given, else none."""
    row: dict[str, object] = {
        "game_id": f"{SEASON}_{week:02d}_{away}_{home}",
        "season": SEASON,
        "week": week,
        "away_abbr": away,
        "home_abbr": home,
        "away_score": 20 if played else None,
        "home_score": 23 if played else None,
        "away_spread": None,
        "home_spread": None,
        "away_moneyline": None,
        "home_moneyline": None,
        "total_line": None,
    }
    row.update(lines)
    return row


def _nfelo_game(away: str, home: str, *, week: int = WEEK, **lines: object) -> dict[str, object]:
    """Return one nfelo row with the opening and latest lines given, else none."""
    row: dict[str, object] = dict(dict.fromkeys(_NFELO_SCHEMA))
    row.update({"season": SEASON, "week": week, "away_abbr": away, "home_abbr": home})
    row.update(lines)
    return row


SCHEDULE = [
    # nfelo has a latest line with both moneylines and a total.
    _schedule_game(
        "DEN",
        "KC",
        away_spread=3.5,
        home_spread=-3.5,
        away_moneyline=150,
        home_moneyline=-180,
        total_line=45.5,
    ),
    # nfelo has no row: the nflverse line stays, its missing moneylines come from the map.
    _schedule_game("BUF", "NYJ", away_spread=-6.5, home_spread=6.5, total_line=41.0),
    # Neither source has a line: SurvivorGrid's spread and the default total fill it.
    _schedule_game("SF", "SEA"),
    # nfelo has a latest spread but no latest moneylines.
    _schedule_game("MIA", "NE", away_spread=2.0, home_spread=-2.0, total_line=43.0),
    # A completed game with a real opener keeps the opener, never the stored line.
    _schedule_game(
        "LV",
        "LAC",
        week=4,
        played=True,
        away_spread=7.0,
        home_spread=-7.0,
        away_moneyline=260,
        home_moneyline=-320,
        total_line=40.0,
    ),
]
NFELO = [
    _nfelo_game(
        "DEN",
        "KC",
        home_spread_open=-3.0,
        home_spread_last=-5.5,
        total_line_last=47.0,
        home_ml_last=-230,
        away_ml_last=190,
    ),
    _nfelo_game("MIA", "NE", home_spread_open=-1.0, home_spread_last=-3.0, total_line_last=44.5),
    _nfelo_game(
        "LV",
        "LAC",
        week=4,
        home_spread_open=-4.5,
        home_spread_last=-7.0,
        total_line_open=42.5,
        home_ml_open=-200,
        away_ml_open=170,
    ),
]
SURVIVOR_GRID = {"SEA": {WEEK: -2.5}}


def _dataset_rows() -> list[dict[str, object]]:
    """Return the build's rows: every game present, its lines from an earlier build."""
    rows = []
    for game in SCHEDULE:
        row = {key: game[key] for key in ("game_id", "season", "week", "away_abbr", "home_abbr")}
        row.update(
            {
                "away_score": game["away_score"],
                "home_score": game["home_score"],
                "total_line": 44.0,
                "away_spread": 1.0,
                "home_spread": -1.0,
                "away_moneyline": 100,
                "home_moneyline": -120,
            }
        )
        rows.append(row)
    return rows


def _write_record(data_dir: Path, record: dict[str, object]) -> None:
    """Write the build's line record beside its datasets."""
    path = data_dir / f"{constants.MARKET_LINES_METADATA_NAME}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def _pick_time_record(maps: dict[str, object] | None = None) -> dict[str, object]:
    """Return a pick-time build's record carrying the season's moneyline map."""
    return {
        "line_source": constants.LINE_SOURCE_PICK_TIME,
        "moneyline_maps": {str(SEASON): SEASON_MAP.to_dict()} if maps is None else maps,
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Return a data directory holding the all-data dataset of the season."""
    pl.DataFrame(_dataset_rows()).write_csv(tmp_path / "all_data_ml.csv")
    return tmp_path


@pytest.fixture
def nfelo_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Serve the schedule, nfelo's lines and SurvivorGrid, and count the nfelo reads."""
    calls: list[int] = []

    def load_schedule(*_args: object, **_kwargs: object) -> pl.DataFrame:
        return pl.DataFrame(SCHEDULE, schema=_SCHEDULE_SCHEMA)

    def load_nfelo(**_kwargs: object) -> nfelo_lines.NfeloLinesSnapshot:
        calls.append(1)
        frame = pl.DataFrame(NFELO, schema=_NFELO_SCHEMA)
        return nfelo_lines.NfeloLinesSnapshot(frame, "url", "download", "abc123", None)

    monkeypatch.setattr(lines_refresh.polars_utils, "load_schedule", load_schedule)
    monkeypatch.setattr(lines_refresh.nfelo_lines, "load_nfelo_lines", load_nfelo)
    monkeypatch.setattr(game_utils, "scrape_survivor_grid_spreads", lambda: SURVIVOR_GRID)
    return calls


def _lines_by_game(data_dir: Path) -> dict[str, dict[str, object]]:
    """Return each game's five line columns as written in the all-data dataset."""
    frame = pl.read_csv(data_dir / "all_data_ml.csv")
    return {row["game_id"]: {c: row[c] for c in LINE_COLUMNS} for row in frame.to_dicts()}


def _expected(home_spread: float, total: float, moneylines: tuple[int, int]) -> dict[str, object]:
    """Return the five line columns of a game with these spread, total and moneylines."""
    home_moneyline, away_moneyline = moneylines
    return {
        "total_line": total,
        "away_spread": -home_spread,
        "home_spread": home_spread,
        "away_moneyline": away_moneyline,
        "home_moneyline": home_moneyline,
    }


def test_pick_time_build_takes_nfelos_latest_line_for_an_upcoming_game(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """An upcoming game takes nfelo's latest spread, moneylines and total, as the ETL does."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    assert nfelo_calls == [1]
    assert _lines_by_game(data_dir)["2026_05_DEN_KC"] == _expected(-5.5, 47.0, (-230, 190))


def test_pick_time_build_keeps_the_nflverse_line_when_nfelo_has_none(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """Without an nfelo line the nflverse line stays, priced with the build's fitted map."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    expected = _expected(6.5, 41.0, SEASON_MAP.moneylines(6.5))
    assert _lines_by_game(data_dir)["2026_05_BUF_NYJ"] == expected
    assert expected["home_moneyline"] != game_utils.spread_to_moneyline(6.5)


def test_pick_time_build_fills_a_game_without_any_line_from_survivorgrid(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """With no nfelo or nflverse line, SurvivorGrid's spread and the default total fill in."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    expected = _expected(-2.5, constants.DEFAULT_TOTAL_LINE, SEASON_MAP.moneylines(-2.5))
    assert _lines_by_game(data_dir)["2026_05_SF_SEA"] == expected


def test_pick_time_build_prices_a_latest_spread_without_moneylines_with_the_map(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """nfelo's latest spread without a moneyline pair is priced with the build's map."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    expected = _expected(-3.0, 44.5, SEASON_MAP.moneylines(-3.0))
    assert _lines_by_game(data_dir)["2026_05_MIA_NE"] == expected


def test_pick_time_build_keeps_a_completed_games_opener(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """A completed game takes its opener, as the build did, never the stored closing line."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    assert _lines_by_game(data_dir)["2026_04_LV_LAC"] == _expected(-4.5, 42.5, (-200, 170))


def test_stored_build_keeps_the_nflverse_lines_and_never_reads_nfelo(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """A stored build refreshes from the schedule with the fixed conversion, as before."""
    # Arrange
    _write_record(data_dir, {"line_source": constants.LINE_SOURCE_STORED})

    # Act
    result = lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    lines = _lines_by_game(data_dir)
    assert nfelo_calls == []
    assert result.line_source == constants.LINE_SOURCE_STORED
    assert lines["2026_05_DEN_KC"] == _expected(-3.5, 45.5, (-180, 150))
    fixed = (game_utils.spread_to_moneyline(6.5), game_utils.spread_to_moneyline(-6.5))
    assert lines["2026_05_BUF_NYJ"] == _expected(6.5, 41.0, fixed)
    assert lines["2026_04_LV_LAC"] == _expected(-7.0, 40.0, (-320, 260))


def test_missing_line_record_is_read_as_a_stored_build(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """A data directory without a line record predates pick-time builds, so it is stored."""
    # Act
    result = lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    assert nfelo_calls == []
    assert result.line_source == constants.LINE_SOURCE_STORED
    assert _lines_by_game(data_dir)["2026_05_DEN_KC"] == _expected(-3.5, 45.5, (-180, 150))


def test_refresh_result_names_the_pick_time_source(data_dir: Path, nfelo_calls: list[int]) -> None:
    """The result says which line source the refresh followed."""
    # Arrange
    _write_record(data_dir, _pick_time_record())

    # Act
    result = lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    assert result.line_source == constants.LINE_SOURCE_PICK_TIME


def test_unknown_line_source_stops_the_refresh(data_dir: Path, nfelo_calls: list[int]) -> None:
    """A record naming a source this code does not know stops before anything is written."""
    # Arrange
    _write_record(data_dir, {"line_source": "closing"})
    before = (data_dir / "all_data_ml.csv").read_bytes()

    # Act & Assert
    with pytest.raises(ValueError, match="closing"):
        lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)
    assert (data_dir / "all_data_ml.csv").read_bytes() == before


def test_pick_time_record_without_the_seasons_map_stops_the_refresh(
    data_dir: Path, nfelo_calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A season without a map entry stops the refresh before any download or write."""
    # Arrange
    _write_record(data_dir, _pick_time_record(maps={"2025": SEASON_MAP.to_dict()}))
    before = (data_dir / "all_data_ml.csv").read_bytes()
    schedule_loads: list[int] = []

    def load_schedule(*_args: object, **_kwargs: object) -> pl.DataFrame:
        schedule_loads.append(1)
        return pl.DataFrame(SCHEDULE, schema=_SCHEDULE_SCHEMA)

    monkeypatch.setattr(lines_refresh.polars_utils, "load_schedule", load_schedule)

    # Act & Assert
    with pytest.raises(ValueError, match="2026"):
        lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)
    assert (schedule_loads, nfelo_calls) == ([], [])
    assert (data_dir / "all_data_ml.csv").read_bytes() == before


def test_pick_time_record_with_no_map_for_the_season_uses_the_fixed_conversion(
    data_dir: Path, nfelo_calls: list[int]
) -> None:
    """A season the build priced with the fixed conversion (a null map) is priced the same."""
    # Arrange
    _write_record(data_dir, _pick_time_record(maps={str(SEASON): None}))

    # Act
    lines_refresh.refresh_lines(SEASON, WEEK, data_dir=data_dir)

    # Assert
    fixed = (game_utils.spread_to_moneyline(6.5), game_utils.spread_to_moneyline(-6.5))
    assert _lines_by_game(data_dir)["2026_05_BUF_NYJ"] == _expected(6.5, 41.0, fixed)


def test_moneyline_map_round_trips_through_its_record() -> None:
    """A map written to a build's record reads back as the same map."""
    # Arrange
    record = json.loads(json.dumps(SEASON_MAP.to_dict()))

    # Act
    restored = moneyline_map.SpreadMoneylineMap.from_dict(record)

    # Assert
    assert restored == SEASON_MAP


def test_line_record_that_is_not_json_stops_the_refresh(data_dir: Path) -> None:
    """An unreadable line record is an error, never a silent stored refresh."""
    # Arrange
    (data_dir / f"{constants.MARKET_LINES_METADATA_NAME}.json").write_text("{", encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="not valid JSON"):
        lines_refresh.read_build_lines(data_dir)


def test_pick_time_record_without_a_map_table_stops_the_refresh(data_dir: Path) -> None:
    """A pick-time record must carry the maps its moneylines were priced with."""
    # Arrange
    _write_record(data_dir, {"line_source": constants.LINE_SOURCE_PICK_TIME})

    # Act & Assert
    with pytest.raises(TypeError, match="moneyline map table"):
        lines_refresh.read_build_lines(data_dir)


@pytest.mark.parametrize(
    "change",
    [
        {"spreads": "0, 3"},
        {"favorite_prob": [0.5, "high"]},
        {"games": 500.5},
        {"seasons": [True]},
        {"underdog_prob": None},
    ],
)
def test_moneyline_map_record_with_a_bad_field_is_refused(change: dict[str, object]) -> None:
    """A record field of the wrong kind is refused with the reason."""
    # Arrange
    record = {**SEASON_MAP.to_dict(), **change}

    # Act & Assert
    with pytest.raises(ValueError, match="Not a spread-to-moneyline map record"):
        moneyline_map.SpreadMoneylineMap.from_dict(record)


def test_moneyline_map_record_without_a_field_is_refused() -> None:
    """A record missing a field is refused with the field's name."""
    # Arrange
    record = {key: value for key, value in SEASON_MAP.to_dict().items() if key != "games"}

    # Act & Assert
    with pytest.raises(ValueError, match="games"):
        moneyline_map.SpreadMoneylineMap.from_dict(record)
