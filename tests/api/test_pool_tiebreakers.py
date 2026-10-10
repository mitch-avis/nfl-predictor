"""Tests for the tiebreaker card: the tiebreak games' entries and the week's extreme scores."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import polars as pl
import pytest

from nfl_predictor.api.pool.tiebreakers import build_tiebreakers
from nfl_predictor.api.readers import market

Z90 = NormalDist().inv_cdf(0.9)


def _frame(**overrides: list[Any]) -> pl.DataFrame:
    """Return a five-game week: Thursday night, two Sunday afternoons, Sunday and Monday night."""
    columns: dict[str, list[Any]] = {
        "game_id": ["2026_05_TB_DAL", "2026_05_CHI_GB", "2026_05_SF_SEA", "2026_05_BAL_ATL",
                    "2026_05_BUF_LA"],
        "away_abbr": ["TB", "CHI", "SF", "BAL", "BUF"],
        "home_abbr": ["DAL", "GB", "SEA", "ATL", "LAR"],
        "game_datetime": ["2026-10-08T20:15:00", "2026-10-11T13:00:00", "2026-10-11T16:25:00",
                          "2026-10-11T20:20:00", "2026-10-12T20:15:00"],
        "predicted_margin_raw": [10.8, -3.0, 2.7, 3.0, 1.4],
        "predicted_total_raw": [48.0, 47.2, 46.9, 45.0, 54.6],
        "predicted_margin_p10": [-6.9, -19.3, -13.8, -12.4, -13.7],
        "predicted_margin_p90": [25.4, 13.5, 19.1, 20.0, 19.4],
        "predicted_total_p10": [32.8, 30.9, 30.8, 28.2, 39.5],
        "predicted_total_p90": [66.2, 63.0, 63.7, 61.8, 72.8],
    }  # fmt: skip
    columns.update(overrides)
    return pl.DataFrame(columns, strict=False)


def test_tiebreak_defaults_are_the_sunday_and_monday_night_games() -> None:
    """The suggested tiebreak games are the latest Sunday and Monday kickoffs, in order."""
    # Act
    card = build_tiebreakers(_frame())

    # Assert
    assert card.default_game_ids == ["2026_05_BAL_ATL", "2026_05_BUF_LA"]


def test_tiebreak_defaults_fill_with_the_latest_kickoffs_without_a_monday_game() -> None:
    """A week without a Monday game suggests its two latest kickoffs."""
    # Arrange
    frame = _frame().filter(pl.col("game_id") != "2026_05_BUF_LA")

    # Act
    card = build_tiebreakers(frame)

    # Assert
    assert card.default_game_ids == ["2026_05_SF_SEA", "2026_05_BAL_ATL"]


def test_tiebreak_games_come_in_kickoff_order_with_night_slots() -> None:
    """Every game is listed by kickoff, and night games are named by their day."""
    # Arrange
    frame = _frame().reverse()

    # Act
    card = build_tiebreakers(frame)

    # Assert
    assert [(game.game_id, game.slot) for game in card.games] == [
        ("2026_05_TB_DAL", "Thursday night"),
        ("2026_05_CHI_GB", None),
        ("2026_05_SF_SEA", None),
        ("2026_05_BAL_ATL", "Sunday night"),
        ("2026_05_BUF_LA", "Monday night"),
    ]


def test_tiebreak_entry_keeps_the_rounded_total() -> None:
    """Whole-point entries keep the rounded total and split it by the predicted margin."""
    # Act
    card = build_tiebreakers(_frame())

    # Assert
    monday = card.games[-1]
    assert (monday.total_points, monday.home_points, monday.away_points) == (55, 28, 27)
    assert monday.home_raw == pytest.approx(28.0)
    assert monday.away_raw == pytest.approx(26.6)
    assert monday.total_raw == pytest.approx(54.6)


def test_tiebreak_bands_come_from_the_quantile_models() -> None:
    """The total band is the total's own p10 to p90; each team's spreads both heads' bands."""
    # Act
    card = build_tiebreakers(_frame())

    # Assert
    monday = card.games[-1]
    sigma_total = (72.8 - 39.5) / (2 * market.NORMAL_Z_P90)
    sigma_margin = (19.4 + 13.7) / (2 * market.NORMAL_Z_P90)
    sigma_team = math.hypot(sigma_total, sigma_margin) / 2
    assert (monday.total_p10, monday.total_p90) == (39.5, 72.8)
    assert monday.home_p10 == pytest.approx(28.0 - Z90 * sigma_team)
    assert monday.away_p90 == pytest.approx(26.6 + Z90 * sigma_team)


def test_tiebreak_bands_fall_back_without_quantiles() -> None:
    """Without quantile columns the bands use the betting page's default spreads."""
    # Arrange
    frame = _frame().drop(
        "predicted_margin_p10", "predicted_margin_p90", "predicted_total_p10", "predicted_total_p90"
    )

    # Act
    card = build_tiebreakers(frame)

    # Assert
    monday = card.games[-1]
    assert monday.total_p10 == pytest.approx(54.6 - Z90 * market.DEFAULT_SIGMA_TOTAL)
    assert monday.total_p90 == pytest.approx(54.6 + Z90 * market.DEFAULT_SIGMA_TOTAL)


def test_tiebreak_scores_of_two_equal_teams_match_their_order_statistics() -> None:
    """For two equal teams the median top score solves Phi(z) squared equals one half."""
    # Arrange
    frame = pl.DataFrame(
        {
            "game_id": ["g"],
            "away_abbr": ["A"],
            "home_abbr": ["H"],
            "game_datetime": ["2026-10-11T13:00:00"],
            "predicted_margin_raw": [0.0],
            "predicted_total_raw": [40.0],
            "predicted_margin_p10": [-10 * market.NORMAL_Z_P90],
            "predicted_margin_p90": [10 * market.NORMAL_Z_P90],
            "predicted_total_p10": [40 - 10 * market.NORMAL_Z_P90],
            "predicted_total_p90": [40 + 10 * market.NORMAL_Z_P90],
        }
    )
    sigma = math.hypot(10, 10) / 2
    z_median = NormalDist().inv_cdf(math.sqrt(0.5))

    # Act
    card = build_tiebreakers(frame)

    # Assert
    assert card.highest.median == pytest.approx(20 + z_median * sigma, abs=1e-6)
    assert card.lowest.median == pytest.approx(20 - z_median * sigma, abs=1e-6)
    assert (card.highest.points, card.lowest.points) == (24, 16)


def test_tiebreak_top_score_runs_past_the_best_projection() -> None:
    """The week's top score is expected well above any one team's projection."""
    # Act
    card = build_tiebreakers(_frame())

    # Assert
    highest, lowest = card.highest, card.lowest
    assert highest.p10 is not None
    assert highest.median is not None
    assert highest.p90 is not None
    assert lowest.median is not None
    assert (highest.projected_team, highest.projected_points) == ("DAL", pytest.approx(29.4))
    assert highest.p10 < highest.median < highest.p90
    assert highest.median > 29.4
    assert (lowest.projected_team, lowest.projected_points) == ("TB", pytest.approx(18.6))
    assert lowest.median < 18.6


def test_tiebreak_lowest_score_is_never_negative() -> None:
    """Spreads wide enough to reach below zero still give a lowest score of at least zero."""
    # Arrange
    frame = _frame(predicted_total_raw=[8.0, 8.0, 8.0, 8.0, 8.0], predicted_margin_raw=[0.0] * 5)

    # Act
    card = build_tiebreakers(frame)

    # Assert
    assert card.lowest.p10 == 0.0
    assert card.lowest.points == 0


def test_tiebreak_game_without_predictions_has_no_entry() -> None:
    """A game without a predicted margin or total is listed without an entry."""
    # Arrange
    frame = _frame(predicted_margin_raw=[10.8, None, 2.7, 3.0, 1.4])

    # Act
    card = build_tiebreakers(frame)

    # Assert
    chi = card.games[1]
    assert (chi.total_points, chi.home_points, chi.away_points) == (None, None, None)
    assert card.highest.projected_team == "DAL"


def test_tiebreak_card_reads_the_display_columns_without_raw_ones() -> None:
    """Files without the unrounded columns fall back to the display margin and total."""
    # Arrange
    frame = _frame().rename(
        {"predicted_margin_raw": "predicted_margin", "predicted_total_raw": "predicted_total"}
    )

    # Act
    card = build_tiebreakers(frame)

    # Assert
    assert card.games[-1].total_points == 55


def test_tiebreak_card_of_an_empty_week_has_no_estimates() -> None:
    """A week without games suggests nothing and estimates nothing."""
    # Act
    card = build_tiebreakers(_frame().head(0))

    # Assert
    assert card.games == []
    assert card.default_game_ids == []
    assert card.highest.points is None
    assert card.lowest.projected_team is None
