from datetime import datetime

import polars as pl
import pytest

from nfl_predictor.utils import game_utils


def test_spread_to_moneyline_signs() -> None:
    assert game_utils.spread_to_moneyline(-7.0) < 0
    assert game_utils.spread_to_moneyline(7.0) > 0
    assert game_utils.spread_to_moneyline(0.0) > 0


def test_fill_missing_moneylines() -> None:
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
