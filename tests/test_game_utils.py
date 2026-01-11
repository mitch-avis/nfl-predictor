"""Tests for game utility helpers (spreads, moneylines, QB fills)."""

from datetime import datetime

import polars as pl
import pytest

from nfl_predictor.utils import game_utils


def test_spread_to_moneyline_signs() -> None:
    """Spread sign conventions map to moneyline sign conventions."""

    assert game_utils.spread_to_moneyline(-7.0) < 0
    assert game_utils.spread_to_moneyline(7.0) > 0
    assert game_utils.spread_to_moneyline(0.0) > 0


def test_fill_missing_moneylines() -> None:
    """Missing moneylines are filled from spreads."""

    df = pl.DataFrame(
        {
            "home_spread": [-3.5],
            "away_spread": [3.5],
            "home_moneyline": [None],
            "away_moneyline": [None],
        }
    )

    filled = game_utils.fill_missing_moneylines(df)

    home_ml = filled.select("home_moneyline").item()
    away_ml = filled.select("away_moneyline").item()
    assert home_ml is not None
    assert away_ml is not None
    assert home_ml < 0
    assert away_ml > 0


def test_fill_future_qb_data() -> None:
    """Future games use the latest available QB rows for missing QB fields."""

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

    filled = game_utils.fill_future_qb_data(df, elo_df)

    row = filled.row(0, named=True)
    assert row["away_qb"] == "J. Allen"
    assert row["home_qb"] == "P. Mahomes"
    assert row["away_qb_value_pre"] == pytest.approx(2.8)
    assert row["home_qb_value_pre"] == pytest.approx(3.0)
    assert row["away_qb_elo_pre"] == pytest.approx(180.0)
    assert row["home_qb_elo_pre"] == pytest.approx(200.0)


def test_get_latest_qb_by_team() -> None:
    """Latest QB per team is selected correctly."""

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

    latest = game_utils.get_latest_qb_by_team(elo_df)
    row = latest.filter(pl.col("team_abbr") == "KC").row(0, named=True)
    assert row["qb_name"] == "P. Mahomes"
    assert row["qb_value_pre"] == pytest.approx(3.1)


def test_get_qb_elo_by_name() -> None:
    """QB elo lookup by name returns latest values and empty for missing."""

    elo_df = pl.DataFrame(
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

    qb1 = game_utils.get_qb_elo_by_name(elo_df, "QB1")
    assert qb1["qb_value_pre"] == pytest.approx(2.2)

    qb2 = game_utils.get_qb_elo_by_name(elo_df, "QB2")
    assert qb2["qb_elo_pre"] == pytest.approx(145.0)

    assert not game_utils.get_qb_elo_by_name(elo_df, "MISSING")


def test_fill_future_game_lines(monkeypatch) -> None:
    """Future games are filled from SurvivorGrid spreads."""

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

    filled = game_utils.fill_future_game_lines(df)

    row = filled.row(0, named=True)
    assert row["home_spread"] == pytest.approx(-3.5)
    assert row["away_spread"] == pytest.approx(3.5)
    assert row["total_line"] == pytest.approx(game_utils.constants.DEFAULT_TOTAL_LINE)
    assert row["home_moneyline"] is not None
