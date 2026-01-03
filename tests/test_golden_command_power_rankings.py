"""Tests for golden command power rankings helpers."""

from __future__ import annotations

import pandas as pd

from scripts import golden_command

# pylint: disable=protected-access


def test_build_pregame_power_rankings_uses_postgame_for_prior_weeks() -> None:
    """Week w rankings should use postgame ratings for weeks < w."""

    df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "away_abbr": ["A", "A"],
            "home_abbr": ["B", "B"],
            "pregame_away_rating": [4.0, 5.0],
            "pregame_home_rating": [6.0, 5.0],
            "postgame_away_rating": [2.0, 1.0],
            "postgame_home_rating": [8.0, 9.0],
        }
    )

    rankings = golden_command._build_pregame_power_rankings(df)

    week2 = rankings[(rankings["season"] == 2025) & (rankings["week"] == 2)].copy()
    assert not week2.empty

    # Week 2 should reflect week 1 postgame rating influence.
    rating_a = float(week2.loc[week2["team"] == "A", "rating"].iloc[0])
    rating_b = float(week2.loc[week2["team"] == "B", "rating"].iloc[0])
    assert rating_b > rating_a

    rank_a = int(week2.loc[week2["team"] == "A", "rank"].iloc[0])
    rank_b = int(week2.loc[week2["team"] == "B", "rank"].iloc[0])
    assert rank_b == 1
    assert rank_a == 2
