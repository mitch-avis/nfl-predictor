"""Tests for the spread-to-moneyline map fitted to the market's own prices."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import game_utils
from nfl_predictor.utils.polars import moneyline_map


def _moneyline(prob: float) -> int:
    return moneyline_map.probability_to_moneyline(prob)


def _pairs(
    rows: list[tuple[float, float, float]], *, copies: int = 1, season: int = 2010
) -> pl.DataFrame:
    """Price pairs: (home spread, home implied prob, away implied prob), each ``copies`` times."""
    expanded = [row for row in rows for _ in range(copies)]
    return pl.DataFrame(
        {
            "season": [season] * len(expanded),
            "home_spread": [row[0] for row in expanded],
            "home_moneyline": [_moneyline(row[1]) for row in expanded],
            "away_moneyline": [_moneyline(row[2]) for row in expanded],
        }
    )


# Home favorite by 2, 2.5, 3, 3.5 and 7, with a jump at the key number 3.
_KEY_NUMBER_ROWS = [
    (-2.0, 0.55, 0.48),
    (-2.5, 0.57, 0.46),
    (-3.0, 0.62, 0.41),
    (-3.5, 0.65, 0.38),
    (-7.0, 0.76, 0.27),
]


def _fit(
    rows: list[tuple[float, float, float]], copies: int = 50
) -> moneyline_map.SpreadMoneylineMap:
    fitted = moneyline_map.fit_spread_moneyline_map(_pairs(rows, copies=copies), min_games=1)
    assert fitted is not None
    return fitted


def test_probability_to_moneyline_uses_the_american_convention() -> None:
    """A favorite's price is negative, an underdog's positive, and even odds is +100."""
    assert moneyline_map.probability_to_moneyline(0.75) == -300
    assert moneyline_map.probability_to_moneyline(0.25) == 300
    assert moneyline_map.probability_to_moneyline(0.5) == 100


def test_implied_probability_inverts_the_moneyline() -> None:
    """The implied probability of a price includes the vig, as quoted."""
    assert moneyline_map.implied_probability(-300) == pytest.approx(0.75)
    assert moneyline_map.implied_probability(300) == pytest.approx(0.25)


def test_the_map_keeps_the_jump_at_the_key_number_three() -> None:
    """Each half point keeps the market's own price, so crossing 3 moves more than 2 to 2.5."""
    fitted = _fit(_KEY_NUMBER_ROWS)

    favorite = {spread: fitted.implied_probabilities(-spread)[0] for spread in (2.0, 2.5, 3.0)}

    assert favorite[3.0] == pytest.approx(0.62, abs=0.002)
    assert favorite[3.0] - favorite[2.5] > 2 * (favorite[2.5] - favorite[2.0])


def test_the_map_gives_the_underdog_the_mirrored_prices() -> None:
    """A home underdog by 3 gets the prices of an away favorite by 3."""
    fitted = _fit(_KEY_NUMBER_ROWS)

    home_favorite = fitted.implied_probabilities(-3.0)
    home_underdog = fitted.implied_probabilities(3.0)

    assert home_underdog == pytest.approx((home_favorite[1], home_favorite[0]))


def test_a_pick_em_prices_both_sides_alike() -> None:
    """With no favorite, both sides get the same price."""
    fitted = _fit(_KEY_NUMBER_ROWS)

    home, away = fitted.implied_probabilities(0.0)

    assert home == pytest.approx(away)
    assert 0.46 < home < 0.55


def test_the_map_interpolates_between_half_points_and_holds_beyond_the_last() -> None:
    """Unseen spreads interpolate linearly between fitted ones and hold past the largest."""
    fitted = _fit(_KEY_NUMBER_ROWS)

    between = fitted.implied_probabilities(-5.25)[0]
    beyond = fitted.implied_probabilities(-14.0)[0]

    assert between == pytest.approx((0.65 + 0.76) / 2, abs=0.002)
    assert beyond == pytest.approx(fitted.implied_probabilities(-7.0)[0])


def test_the_map_is_monotone_where_the_prices_are_noisy() -> None:
    """A sparse spread priced against the trend is pooled with its neighbor."""
    rows = [*_KEY_NUMBER_ROWS, (-8.0, 0.80, 0.24), (-8.5, 0.78, 0.26)]

    fitted = _fit(rows)
    favorite = [fitted.implied_probabilities(-spread)[0] for spread in (7.0, 8.0, 8.5)]

    assert favorite == sorted(favorite)
    assert favorite[1] == pytest.approx(favorite[2])


def test_the_map_returns_moneylines_for_both_sides() -> None:
    """The fitted prices convert back to a moneyline pair, favorite negative."""
    fitted = _fit(_KEY_NUMBER_ROWS)

    home, away = fitted.moneylines(-7.0)

    assert home == _moneyline(0.76)
    assert away == _moneyline(0.27)


def test_too_few_games_fits_no_map() -> None:
    """Below the minimum game count the fit returns None."""
    pairs = _pairs(_KEY_NUMBER_ROWS, copies=1)

    assert moneyline_map.fit_spread_moneyline_map(pairs, min_games=pairs.height + 1) is None
    assert moneyline_map.fit_spread_moneyline_map(pairs.clear(), min_games=1) is None


def test_price_pairs_keep_only_games_with_a_spread_and_both_moneylines() -> None:
    """A game missing its spread or either moneyline is not a price pair."""
    schedule = pl.DataFrame(
        {
            "season": [2010, 2010, 2010, 2010],
            "home_spread": [-3.0, None, -3.0, -3.0],
            "home_moneyline": [-150, -150, None, -150],
            "away_moneyline": [130, 130, 130, None],
            "home_score": [10, 20, 30, 40],
        }
    )

    pairs = moneyline_map.price_pairs(schedule)

    assert pairs.columns == ["season", "home_spread", "home_moneyline", "away_moneyline"]
    assert pairs.height == 1


def test_a_season_map_is_fitted_only_on_earlier_seasons() -> None:
    """The map for a season never sees that season's or any later season's prices."""
    early = _pairs(_KEY_NUMBER_ROWS, copies=10, season=2008)
    middle = _pairs(_KEY_NUMBER_ROWS, copies=10, season=2009)
    shifted = [(spread, home - 0.05, away + 0.05) for spread, home, away in _KEY_NUMBER_ROWS]
    late = _pairs(shifted, copies=10, season=2010)
    history = pl.concat([early, middle, late])

    maps = moneyline_map.fit_season_maps(history, [2008, 2009, 2010, 2011], min_games=40)

    assert maps[2008] is None
    assert maps[2009] is not None
    assert maps[2009].seasons == (2008,)
    assert maps[2010] is not None
    assert maps[2010].seasons == (2008, 2009)
    assert maps[2010].implied_probabilities(-3.0)[0] == pytest.approx(0.62, abs=0.002)
    assert maps[2011] is not None
    assert maps[2011].implied_probabilities(-3.0)[0] < maps[2010].implied_probabilities(-3.0)[0]


def test_the_map_record_lists_its_seasons_games_and_prices() -> None:
    """The metadata record carries enough to rebuild the map."""
    fitted = _fit(_KEY_NUMBER_ROWS, copies=2)

    record = fitted.to_dict()

    assert fitted.spreads[0] == 0.0
    assert record == {
        "games": 10,
        "seasons": [2010],
        "spreads": list(fitted.spreads),
        "favorite_prob": list(fitted.favorite_prob),
        "underdog_prob": list(fitted.underdog_prob),
    }


def test_filling_uses_the_season_map_and_the_fixed_conversion_without_one() -> None:
    """A season with a map gets fitted prices; one without keeps the fixed conversion."""
    fitted = _fit(_KEY_NUMBER_ROWS)
    games = pl.DataFrame(
        {
            "season": [2010, 2005, 2010, 2010],
            "home_spread": [-7.0, -7.0, -3.0, None],
            "away_spread": [7.0, 7.0, 3.0, None],
            "home_moneyline": pl.Series([None, None, -170, None], dtype=pl.Int32),
            "away_moneyline": pl.Series([None, None, 150, None], dtype=pl.Int32),
        }
    )

    filled = moneyline_map.fill_moneylines_from_maps(games, {2010: fitted, 2005: None})

    assert filled.schema["home_moneyline"] == pl.Int64
    assert filled.select("home_moneyline", "away_moneyline").rows() == [
        fitted.moneylines(-7.0),
        (game_utils.spread_to_moneyline(-7.0), game_utils.spread_to_moneyline(7.0)),
        (-170, 150),
        (None, None),
    ]


def test_filling_keeps_a_real_side_and_derives_only_the_missing_one() -> None:
    """Like the fixed fill, each side is filled only where it is missing."""
    fitted = _fit(_KEY_NUMBER_ROWS)
    games = pl.DataFrame(
        {
            "season": [2010],
            "home_spread": [-7.0],
            "away_spread": [7.0],
            "home_moneyline": [-400],
            "away_moneyline": pl.Series([None], dtype=pl.Int64),
        }
    )

    filled = moneyline_map.fill_moneylines_from_maps(games, {2010: fitted})

    assert filled.row(0, named=True)["home_moneyline"] == -400
    assert filled.row(0, named=True)["away_moneyline"] == fitted.moneylines(-7.0)[1]


def test_the_default_minimum_comes_from_the_constants() -> None:
    """Without an explicit minimum the fit needs MONEYLINE_MAP_MIN_GAMES games."""
    short = _pairs(_KEY_NUMBER_ROWS, copies=1)
    enough = _pairs(_KEY_NUMBER_ROWS, copies=constants.MONEYLINE_MAP_MIN_GAMES)

    assert moneyline_map.fit_spread_moneyline_map(short) is None
    assert moneyline_map.fit_spread_moneyline_map(enough) is not None
