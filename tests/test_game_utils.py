"""Tests for game utility helpers (spreads, moneylines, QB fills)."""

from datetime import datetime

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import game_utils


def test_spread_to_moneyline_signs() -> None:
    """Spread sign conventions map to moneyline sign conventions."""
    # Act
    moneyline = game_utils.spread_to_moneyline(-7.0)

    # Assert
    assert moneyline < 0


@pytest.mark.parametrize("spread", [7.0, 0.0])
def test_spread_to_moneyline_signs_for_an_underdog_or_pick_em(spread: float) -> None:
    """A positive or zero spread maps to a positive moneyline."""
    # Act
    moneyline = game_utils.spread_to_moneyline(spread)

    # Assert
    assert moneyline > 0


def test_spread_to_moneyline_pins_the_fixed_conversion() -> None:
    """The fixed normal-curve conversion with a flat 5% vig gives these exact moneylines."""
    # Arrange
    spreads = (-14.0, -7.0, -3.5, -3.0, 0.0, 3.0, 3.5, 7.0, 14.0)

    # Act
    moneylines = [game_utils.spread_to_moneyline(spread) for spread in spreads]

    # Assert
    assert moneylines == [-730, -261, -168, -158, 90, 129, 136, 206, 487]


def test_fill_missing_moneylines() -> None:
    """Missing moneylines are filled from spreads."""
    # Arrange
    df = pl.DataFrame(
        {
            "home_spread": [-3.5],
            "away_spread": [3.5],
            "home_moneyline": [None],
            "away_moneyline": [None],
        }
    )

    # Act
    filled = game_utils.fill_missing_moneylines(df)

    # Assert
    home_ml = filled.select("home_moneyline").item()
    away_ml = filled.select("away_moneyline").item()
    assert home_ml is not None
    assert away_ml is not None
    assert home_ml < 0
    assert away_ml > 0


def test_fill_missing_moneylines_passthrough_branches() -> None:
    """Moneyline filling is skipped when spreads or missing values are absent."""
    # Arrange
    no_spreads = pl.DataFrame({"week": [1]})

    # Act
    result = game_utils.fill_missing_moneylines(no_spreads)

    # Assert
    assert result is no_spreads


def test_fill_missing_moneylines_keeps_filled_moneylines() -> None:
    """Moneyline filling leaves rows that already carry moneylines unchanged."""
    # Arrange
    already_filled = pl.DataFrame(
        {
            "home_spread": [-3.5],
            "away_spread": [3.5],
            "home_moneyline": [-160],
            "away_moneyline": [140],
        }
    )

    # Act
    result = game_utils.fill_missing_moneylines(already_filled)

    # Assert
    assert result.to_dicts() == already_filled.to_dicts()


def test_fill_future_qb_data() -> None:
    """Future games use the latest available QB rows for missing QB fields."""
    # Arrange
    df = pl.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_qb": [None],
            "home_qb": [None],
            "away_qb_value_pre": [None],
            "away_qb_elo_pre": [None],
            "home_qb_value_pre": [None],
            "home_qb_elo_pre": [None],
        }
    )
    elo_df = pl.DataFrame(
        {
            "date": [datetime(2024, 9, 1), datetime(2024, 9, 1)],
            "team1": ["KC", "BUF"],
            "team2": ["BUF", "KC"],
            "qb1": ["P. Mahomes", "J. Allen"],
            "qb2": ["J. Allen", "P. Mahomes"],
            "qb1_value_pre": [3.0, 2.8],
            "qb2_value_pre": [2.8, 3.0],
            "qbelo1_pre": [200.0, 180.0],
            "qbelo2_pre": [180.0, 200.0],
        }
    )

    # Act
    filled = game_utils.fill_future_qb_data(df, elo_df)

    # Assert
    row = filled.row(0, named=True)
    assert row["away_qb"] == "J. Allen"
    assert row["home_qb"] == "P. Mahomes"
    assert row["away_qb_value_pre"] == pytest.approx(2.8)
    assert row["home_qb_value_pre"] == pytest.approx(3.0)
    assert row["away_qb_elo_pre"] == pytest.approx(180.0)
    assert row["home_qb_elo_pre"] == pytest.approx(200.0)


def test_get_latest_qb_by_team() -> None:
    """Latest QB per team is selected correctly."""
    # Arrange
    elo_df = pl.DataFrame(
        {
            "date": [datetime(2024, 9, 1), datetime(2024, 9, 2)],
            "team1": ["KC", "KC"],
            "team2": ["BUF", "BUF"],
            "qb1": ["P. Mahomes", "P. Mahomes"],
            "qb2": ["J. Allen", "J. Allen"],
            "qb1_value_pre": [3.0, 3.1],
            "qb2_value_pre": [2.8, 2.9],
            "qbelo1_pre": [200.0, 205.0],
            "qbelo2_pre": [180.0, 185.0],
        }
    )

    # Act
    latest = game_utils.get_latest_qb_by_team(elo_df)

    # Assert
    row = latest.filter(pl.col("team_abbr") == "KC").row(0, named=True)
    assert row["qb_name"] == "P. Mahomes"
    assert row["qb_value_pre"] == pytest.approx(3.1)


def test_get_latest_qb_by_team_returns_empty_without_required_columns() -> None:
    """Missing input columns short-circuit the latest-QB lookup."""
    # Arrange
    elo_df = pl.DataFrame({"team1": ["KC"]})

    # Act
    latest = game_utils.get_latest_qb_by_team(elo_df)

    # Assert
    assert latest.height == 0


def _two_qb_elo_frame() -> pl.DataFrame:
    """Return two dated ELO rows for the same pair of quarterbacks."""
    return pl.DataFrame(
        {
            "date": [datetime(2024, 9, 1), datetime(2024, 9, 2)],
            "qb1": ["QB1", "QB1"],
            "qb2": ["QB2", "QB2"],
            "qb1_value_pre": [2.0, 2.2],
            "qb2_value_pre": [1.5, 1.6],
            "qbelo1_pre": [150.0, 155.0],
            "qbelo2_pre": [140.0, 145.0],
        }
    )


@pytest.mark.parametrize(
    ("name", "key", "expected"),
    [
        ("QB1", "qb_value_pre", 2.2),
        ("QB2", "qb_elo_pre", 145.0),
    ],
)
def test_get_qb_elo_by_name(name: str, key: str, expected: float) -> None:
    """QB elo lookup by name returns latest values."""
    # Arrange
    elo_df = _two_qb_elo_frame()

    # Act
    result = game_utils.get_qb_elo_by_name(elo_df, name)

    # Assert
    assert result[key] == pytest.approx(expected)


def test_get_qb_elo_by_name_returns_empty_for_a_missing_name() -> None:
    """QB elo lookup by name returns empty for a name it does not know."""
    # Arrange
    elo_df = _two_qb_elo_frame()

    # Act & Assert
    assert not game_utils.get_qb_elo_by_name(elo_df, "MISSING")


@pytest.mark.parametrize(
    ("elo_df", "name"),
    [
        pytest.param(pl.DataFrame(), "QB1", id="empty_frame"),
        pytest.param(pl.DataFrame({"qb1": ["QB1"]}), "", id="blank_name"),
    ],
)
def test_get_qb_elo_by_name_returns_empty_for_blank_inputs(elo_df: pl.DataFrame, name: str) -> None:
    """Blank QB names and empty ELO frames return no lookup result."""
    # Act & Assert
    assert game_utils.get_qb_elo_by_name(elo_df, name) == {}


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.DataFrame({"away_qb": [None]}), id="missing_required"),
        pytest.param(
            pl.DataFrame(
                {
                    "away_abbr": ["BUF"],
                    "home_abbr": ["KC"],
                    "away_qb": [None],
                    "home_qb": [None],
                }
            ),
            id="no_latest_qbs",
        ),
    ],
)
def test_fill_future_qb_data_passthrough_branches(frame: pl.DataFrame) -> None:
    """QB filling returns the original frame when prerequisites or source data are missing."""
    # Act
    result = game_utils.fill_future_qb_data(frame, pl.DataFrame())

    # Assert
    assert result is frame


def test_fill_future_qb_data_adds_qb_name_columns_when_absent() -> None:
    """QB name columns are created when the input frame only has team abbreviations."""
    # Arrange
    df = pl.DataFrame({"away_abbr": ["BUF"], "home_abbr": ["KC"]})
    elo_df = pl.DataFrame(
        {
            "date": [datetime(2024, 9, 1), datetime(2024, 9, 1)],
            "team1": ["KC", "BUF"],
            "team2": ["BUF", "KC"],
            "qb1": ["P. Mahomes", "J. Allen"],
            "qb2": ["J. Allen", "P. Mahomes"],
            "qb1_value_pre": [3.0, 2.8],
            "qb2_value_pre": [2.8, 3.0],
            "qbelo1_pre": [200.0, 180.0],
            "qbelo2_pre": [180.0, 200.0],
        }
    )

    # Act
    filled = game_utils.fill_future_qb_data(df, elo_df)

    # Assert
    row = filled.row(0, named=True)
    assert row["away_qb"] == "J. Allen"
    assert row["home_qb"] == "P. Mahomes"


def test_fill_future_game_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future games are filled from SurvivorGrid spreads."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["game1"],
            "week": [16],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [None],
            "home_spread": [None],
            "away_spread": [None],
            "total_line": [None],
        }
    )
    monkeypatch.setattr(
        game_utils,
        "scrape_survivor_grid_spreads",
        lambda: {"KC": {16: -3.5}, "BUF": {16: 3.5}},
    )

    # Act
    filled = game_utils.fill_future_game_lines(df)

    # Assert
    row = filled.row(0, named=True)
    assert row["home_spread"] == pytest.approx(-3.5)
    assert row["away_spread"] == pytest.approx(3.5)
    assert row["total_line"] == pytest.approx(constants.DEFAULT_TOTAL_LINE)
    assert row["home_moneyline"] is not None


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.DataFrame({"week": [16], "home_abbr": ["KC"]}), id="missing_required"),
        pytest.param(
            pl.DataFrame(
                {
                    "game_id": ["game1"],
                    "week": [16],
                    "away_abbr": ["BUF"],
                    "home_abbr": ["KC"],
                    "away_score": [17],
                    "home_spread": [-3.5],
                }
            ),
            id="no_future_games",
        ),
    ],
)
def test_fill_future_game_lines_passthrough_branches(frame: pl.DataFrame) -> None:
    """Future-line filling skips frames without required columns or future rows."""
    # Act
    result = game_utils.fill_future_game_lines(frame)

    # Assert
    assert result is frame


def test_fill_future_game_lines_passes_through_without_spreads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-line filling skips future rows when SurvivorGrid has no spreads."""
    # Arrange
    future_without_spreads = pl.DataFrame(
        {
            "game_id": ["game2"],
            "week": [16],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "home_spread": [None],
        }
    )
    monkeypatch.setattr(game_utils, "scrape_survivor_grid_spreads", dict)

    # Act
    result = game_utils.fill_future_game_lines(future_without_spreads)

    # Assert
    assert result is future_without_spreads


def test_fill_future_game_lines_uses_away_perspective_when_home_spread_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Away-team SurvivorGrid spreads are negated into the home-team perspective."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["game3"],
            "week": [17],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "home_spread": [None],
        }
    )
    monkeypatch.setattr(
        game_utils,
        "scrape_survivor_grid_spreads",
        lambda: {"BUF": {17: 2.5}},
    )

    # Act
    filled = game_utils.fill_future_game_lines(df)

    # Assert
    row = filled.row(0, named=True)
    assert row["home_spread"] == pytest.approx(-2.5)
    assert row["away_spread"] == pytest.approx(2.5)


def test_fill_future_game_lines_derives_moneylines_with_the_given_filler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied filler, not the fixed conversion, prices the SurvivorGrid spreads."""
    # Arrange
    df = pl.DataFrame(
        {
            "game_id": ["game1"],
            "week": [16],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_score": [None],
            "home_spread": [None],
            "away_spread": [None],
            "total_line": [None],
        }
    )
    monkeypatch.setattr(game_utils, "scrape_survivor_grid_spreads", lambda: {"KC": {16: -3.5}})

    def filler(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(-999).alias("home_moneyline"), pl.lit(999).alias("away_moneyline")
        )

    # Act
    filled = game_utils.fill_future_game_lines(df, fill_moneylines=filler)

    # Assert
    row = filled.row(0, named=True)
    assert (row["home_moneyline"], row["away_moneyline"]) == (-999, 999)


_QB_VALUE_KEYS = ("qb_value_pre", "qb_elo_pre", "qb_adj")


def _latest_starters() -> pl.DataFrame:
    """Return one dated ELO game: J. Allen's BUF at P. Mahomes's KC."""
    return pl.DataFrame(
        {
            "date": [datetime(2024, 9, 1)],
            "team1": ["KC"],
            "team2": ["BUF"],
            "qb1": ["P. Mahomes"],
            "qb2": ["J. Allen"],
            "qb1_value_pre": [3.0],
            "qb2_value_pre": [2.8],
            "qbelo1_pre": [200.0],
            "qbelo2_pre": [180.0],
            "qb1_adj": [4.0],
            "qb2_adj": [-1.5],
        }
    )


def _bills_at_chiefs(away_qbs: list[str | None], away_scores: list[float | None]) -> pl.DataFrame:
    """Return BUF-at-KC rows, one per away starter, with no home starter or values."""
    rows = len(away_qbs)
    nulls: list[float | None] = [None] * rows
    schema: dict[str, type[pl.DataType]] = {
        "away_abbr": pl.String,
        "home_abbr": pl.String,
        "away_qb": pl.String,
        "home_qb": pl.String,
        "away_score": pl.Float64,
        "home_score": pl.Float64,
        **{f"{side}_{key}": pl.Float64 for side in ("away", "home") for key in _QB_VALUE_KEYS},
        **{f"{key}_diff": pl.Float64 for key in _QB_VALUE_KEYS},
    }
    data: dict[str, list[str | float | None]] = {
        "away_abbr": ["BUF"] * rows,
        "home_abbr": ["KC"] * rows,
        "away_qb": list(away_qbs),
        "home_qb": [None] * rows,
        "away_score": list(away_scores),
        "home_score": [None if score is None else 17.0 for score in away_scores],
        **{column: list(nulls) for column in schema if column.endswith(("_pre", "_adj", "_diff"))},
    }
    return pl.DataFrame(data, schema=schema)


def test_fill_future_qb_data_leaves_a_played_game_alone() -> None:
    """A game with both final scores keeps its missing starters and values missing."""
    # Arrange
    games = _bills_at_chiefs(away_qbs=[None, None], away_scores=[20.0, None])

    # Act
    filled = game_utils.fill_future_qb_data(games, _latest_starters())

    # Assert
    assert filled.row(0) == games.row(0)
    assert filled.row(1, named=True)["away_qb"] == "J. Allen"


def test_fill_future_qb_data_leaves_a_named_starters_missing_values_alone() -> None:
    """The latest starter's values never stand in for a different starter's.

    The second game has no starter, so the away side is filled in the same pass.
    """
    # Arrange
    games = _bills_at_chiefs(away_qbs=["M. Trubisky", None], away_scores=[None, None])

    # Act
    filled = game_utils.fill_future_qb_data(games, _latest_starters())

    # Assert
    row = filled.row(0, named=True)
    assert row["away_qb"] == "M. Trubisky"
    assert [row[f"away_{key}"] for key in _QB_VALUE_KEYS] == [None, None, None]
    assert row["home_qb"] == "P. Mahomes"
    assert [row[f"home_{key}"] for key in _QB_VALUE_KEYS] == [3.0, 200.0, 4.0]
    assert [row[f"{key}_diff"] for key in _QB_VALUE_KEYS] == [None, None, None]


def test_fill_future_qb_data_recomputes_the_diffs_of_the_filled_values() -> None:
    """Each filled value's diff is away minus home, as the week rows compute it."""
    # Arrange
    games = _bills_at_chiefs(away_qbs=[None], away_scores=[None])

    # Act
    filled = game_utils.fill_future_qb_data(games, _latest_starters())

    # Assert
    row = filled.row(0, named=True)
    assert [row[f"{key}_diff"] for key in _QB_VALUE_KEYS] == pytest.approx([-0.2, -20.0, -5.5])
